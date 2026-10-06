"""Harness runner: load a model, score items, write predictions and a summary.

Readout protocol (fixed for every model, so baselines are comparable — ADVISORY T6):

* the item is rendered through the model's own chat template with a fixed system prompt
  ("answer with the letter of the correct option only") and thinking disabled;
* the distribution is read from the **next-token logits** at the answer position;
* option letters are scored as single tokens, using the ``" A"`` (leading space) variant when
  it is single-token for that tokenizer and ``"A"`` otherwise — which variant was used is
  recorded per model;
* no text is generated at eval time.

Per-item outputs: option probabilities (softmax over the option logits), the raw
full-vocabulary mass of the option tokens ("label mass"), argmax, expected score level for
``score`` items, and latency. Predictions are written as JSONL under
``outputs/bench_v0/<task>/preds/`` and never committed (R: public repo).
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from meddecide.bench.schema import Item, QuestionType
from meddecide.eval.metrics import accuracy, bootstrap_ci, brier_score, ece_detail, macro_accuracy
from meddecide.eval.readout import (
    LetterVariant,
    canonicalise_options,
    check_label_tokens,
    detect_variant,
    expected_level,
    key_token_id,
    pick_variant,
    read_option_probabilities,
    render_prompt,
)


@dataclass
class ModelSpec:
    """One model to evaluate: id, resolved revision, and the readout variant used."""

    model_id: str
    revision: str | None = None
    variant: LetterVariant | None = None
    dtype: str = "bfloat16"
    max_options: int = 32

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "revision": self.revision,
            "readout_variant": self.variant,
            "dtype": self.dtype,
            "max_options": self.max_options,
        }


@dataclass
class Prediction:
    """One model's answer to one item."""

    item_id: str
    model_id: str
    source: str
    template_id: str
    split: str
    qtype: str
    option_keys: list[str]
    option_probs: list[float]
    label_mass: float
    argmax_index: int
    gold_key: str
    correct: bool
    expected_level: float | None
    latency_s: float
    variant: str
    prompt_tokens: int
    # original source-vocabulary keys in display order (yes/no, 1..N, A..); defaults to the
    # rendered letters when a caller builds a Prediction by hand (unit tests)
    original_option_keys: list[str] = field(default_factory=list)
    vocab_argmax_is_option: bool = True
    n_tokens_above_best_option: int = 0
    best_option_in_top5: bool = False
    top1_over_option_mass: float = 0.0
    transform: dict[str, Any] = field(default_factory=dict)
    # Identifies the run that produced this row. Prediction files are append-only across
    # re-runs (bench_v0_fix0 P7), so a row count is not an item count: consumers dedupe by
    # (run_id, item_id). Rows written before loop 1 carry no run_id and dedupe on item_id.
    run_id: str = ""

    def to_json(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "model_id": self.model_id,
            "run_id": self.run_id,
            "source": self.source,
            "template_id": self.template_id,
            "split": self.split,
            "qtype": self.qtype,
            # option labels are deliberately absent: the repo is public and the item text
            # lives only in the gitignored data/ tree
            # `option_keys` are the letters actually rendered and scored; the original keys
            # (yes/no, 1..N for score levels, A.. for choice) are recorded so a consumer can
            # map a prediction back to the source vocabulary without re-deriving it.
            "option_keys": self.option_keys,
            "original_option_keys": self.original_option_keys,
            "option_probs": [round(p, 6) for p in self.option_probs],
            "label_mass": self.label_mass,
            "vocab_argmax_is_option": self.vocab_argmax_is_option,
            "n_tokens_above_best_option": self.n_tokens_above_best_option,
            "best_option_in_top5": self.best_option_in_top5,
            "top1_over_option_mass": round(self.top1_over_option_mass, 6),
            "argmax_key": self.option_keys[self.argmax_index],
            "gold_key": self.gold_key,
            "correct": self.correct,
            "expected_level": self.expected_level,
            "latency_s": round(self.latency_s, 5),
            "variant": self.variant,
            "prompt_tokens": self.prompt_tokens,
            "transform": self.transform,
        }


def plan_batches(
    prompt_token_counts: Sequence[int],
    *,
    batch_size: int,
    max_batch_tokens: int,
    max_open_batches: int = 16,
) -> list[list[int]]:
    """Group item positions into batches that respect **both** limits.

    First-fit-decreasing with a bounded number of open batches: a batch only accepts an item
    when adding it keeps the batch within ``batch_size`` **and** within ``max_batch_tokens``.
    A new batch is opened when nothing fits, and when ``max_open_batches`` is already open the
    longest remaining items each get their own batch (they are processed sequentially, never
    merged into an oversized batch).

    Every position appears exactly once. The hard guarantee is that no returned batch exceeds
    ``batch_size`` items or ``max_batch_tokens`` tokens of *real* (unpadded) prompt.
    """
    if batch_size < 1 or max_batch_tokens < 1:
        raise ValueError("batch_size and max_batch_tokens must be >= 1")
    if max_open_batches < 1:
        raise ValueError("max_open_batches must be >= 1")
    order = sorted(range(len(prompt_token_counts)), key=lambda i: -prompt_token_counts[i])
    batches: list[list[int]] = []
    tokens: list[int] = []
    for index in order:
        n = prompt_token_counts[index]
        placed = False
        for b, batch in enumerate(batches):
            if len(batch) < batch_size and tokens[b] + n <= max_batch_tokens:
                batch.append(index)
                tokens[b] += n
                placed = True
                break
        if placed:
            continue
        if len(batches) < max_open_batches:
            batches.append([index])
            tokens.append(n)
        else:
            # no room anywhere and no new batch allowed: give this item its own batch rather
            # than merging it into a full one (an oversized batch is what OOMed: 239 sequences
            # at ~1.9k tokens each exhausted a 95 GiB card).
            batches.append([index])
            tokens.append(n)
    return batches


def configure_truncation(tokenizer: Any) -> None:
    """Make truncation keep the **end** of a prompt: the question, options and instruction.

    The question is at the end and the state is what gets too long, so the tokenizer default
    (``truncation_side="right"``) silently removed the question from every over-long item — the
    model was then scored on a prompt that asked nothing. Measured on v0.2: 258 of 2,000
    ``fda_route_claim_noul_v1`` test items, 185 of 1,910 ``fda_boxed_warning_noul_v1`` items and
    137 of 3,236 ``fda_class_choice_v2`` items were over the 16,384-token cap (584 of 24,263
    fresh test items in total). The cap is unchanged; only which end survives it.
    """
    tokenizer.truncation_side = "left"


class Harness:
    """Loads one model and scores batches of items."""

    def __init__(
        self,
        spec: ModelSpec,
        *,
        batch_size: int = 8,
        device: str = "cuda:0",
        trust_remote_code: bool = False,
        max_prompt_tokens: int = 16384,
        max_batch_tokens: int = 65536,
        debug: bool = False,
        run_id: str = "",
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.spec = spec
        self.batch_size = batch_size
        self.device = device
        self.max_prompt_tokens = max_prompt_tokens
        # A fixed batch *size* is unsafe on the fresh tier: PubMed states run to many thousands of
        # tokens and 32 of them made torch try to allocate 104 GiB on a 95 GiB card. Batches are
        # therefore also capped by total prompt tokens.
        self.max_batch_tokens = max_batch_tokens
        self.debug = debug
        # stamped onto every Prediction this harness returns, so an append-only prediction
        # file can be deduplicated per run (S1)
        self.run_id = run_id
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(spec.model_id, revision=spec.revision,
                                                       trust_remote_code=trust_remote_code)
        # left padding: with a decoder-only model the answer position is the last token, so
        # padding on the right would move it
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            spec.model_id,
            revision=spec.revision,
            dtype=getattr(torch, spec.dtype),
            device_map=device,
            trust_remote_code=trust_remote_code,
        )
        self.model.eval()
        checks = check_label_tokens(self.tokenizer, spec.model_id, n_options=max(2, spec.max_options))
        self.label_check = pick_variant(checks)
        if spec.variant is not None:
            self.variant: LetterVariant = spec.variant
            self.variant_detection = {"detected": spec.variant, "method": "pinned by caller"}
        else:
            self.variant, self.variant_detection = detect_variant(self.model, self.tokenizer, checks)
        self.spec.variant = self.variant
        self._key_token_cache: dict[str, int] = {}

    # ---- scoring ------------------------------------------------------------
    def _next_token_logits(self, prompts: Sequence[str]) -> tuple[np.ndarray, list[int]]:
        """Next-token logits at the answer position for a batch of prompts.

        Padding is on the **left** (``padding_side="left"``), so for every row the next-token
        distribution is the logits at the final sequence position.
        """
        torch = self.torch
        configure_truncation(self.tokenizer)
        encoded = self.tokenizer(
            list(prompts), return_tensors="pt", padding=True, truncation=True,
            max_length=self.max_prompt_tokens,
        )
        if self.debug:
            print(
                f"[harness] batch={tuple(encoded['input_ids'].shape)} "
                f"max_id={int(encoded['input_ids'].max())} "
                f"real_tokens={int(encoded['attention_mask'].sum())} "
                f"vocab={int(getattr(self.model.config, 'vocab_size', 0) or 0)} "
                f"mem={torch.cuda.memory_allocated() / 1e9:.1f}GB",
                flush=True,
            )
        # Guard: an out-of-range token id makes torch.embedding try to allocate a table-sized
        # tensor for the *whole batch* (observed: "Tried to allocate 428.83 GiB" on a 95 GiB
        # card) instead of raising an index error. Check and report the offending token.
        vocab_size = int(getattr(self.model.config, "vocab_size", 0) or 0)
        if vocab_size:
            max_id = int(encoded["input_ids"].max())
            if max_id >= vocab_size:
                offenders = [
                    int(t)
                    for t in encoded["input_ids"][encoded["input_ids"] >= vocab_size].flatten().tolist()
                ]
                pieces = self.tokenizer.convert_ids_to_tokens(offenders[:5])
                raise ValueError(
                    f"token id {max_id} >= vocab_size {vocab_size} in a batch of "
                    f"{encoded['input_ids'].shape[0]} sequences "
                    f"(max len {encoded['input_ids'].shape[1]}); "
                    f"offending ids {offenders[:5]} decode to {pieces}"
                )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        n_tokens = attention_mask.sum(dim=1).tolist()
        with torch.inference_mode():
            output = self.model(input_ids=input_ids, attention_mask=attention_mask)
            # With left padding the next-token prediction sits at the LAST position of every
            # row, whose input token is the final real token. Indexing by the last non-pad
            # position instead (attention_mask.sum()-1) reads the logits *of* the final token,
            # i.e. the token before the answer position — a silent readout bug that produced
            # an argmax honouring the format only 4 % of the time.
            last = torch.full(
                (input_ids.shape[0],), input_ids.shape[1] - 1, device=input_ids.device
            )
            logits = output.logits[torch.arange(input_ids.shape[0]), last, :]
        return logits.float().cpu().numpy(), [int(n) for n in n_tokens]

    def _key_token(self, key: str) -> int:
        if key not in self._key_token_cache:
            self._key_token_cache[key] = key_token_id(key, self.tokenizer, self.variant)
        return self._key_token_cache[key]

    def _plan_batches(self, prompt_token_counts: list[int]) -> list[list[int]]:
        """Plan batches that respect both ``batch_size`` and ``max_batch_tokens``."""
        return plan_batches(
            prompt_token_counts, batch_size=self.batch_size, max_batch_tokens=self.max_batch_tokens
        )

    def score(
        self,
        items: Sequence[Item],
        *,
        transform: dict[str, Any] | None = None,
        sort_by_length: bool = True,
    ) -> list[Prediction]:
        """Score items; returns one :class:`Prediction` per item, in the input order.

        Prompt lengths are measured first so the batches respect the token budget as well as the
        batch size. This matters on the fresh tier: PubMed states run to many thousands of tokens
        and a fixed batch of 32 exhausted a 95 GiB card (torch tried to allocate 104 GiB).
        """
        # Every question type goes through the letter path: `canonicalise_options` is a no-op
        # for `choice` (already A/B/C) and converts `noul` (yes/no) and `score` (1..N) to
        # letters, which is what the prompt instructs the model to answer with. Before this,
        # `noul`/`score` prompts said "yes. Yes" while the instruction said "Respond with a
        # single letter", and the readout scored the yes/no tokens: label mass ~0.002.
        canonical_items: list[Item] = []
        original_keys: list[list[str]] = []
        for item in items:
            canonical, keys = canonicalise_options(item)
            canonical_items.append(canonical)
            original_keys.append(keys)
        items = canonical_items
        prompts = [render_prompt(item, self.tokenizer, self.variant) for item in items]
        prompt_token_counts = [
            len(self.tokenizer.encode(prompt, add_special_tokens=False)) for prompt in prompts
        ]
        if sort_by_length:
            batches = self._plan_batches(prompt_token_counts)
        else:  # pragma: no cover - escape hatch for order-sensitive callers
            batches = [
                list(range(start, min(start + self.batch_size, len(items))))
                for start in range(0, len(items), self.batch_size)
            ]
        by_position: dict[int, Prediction] = {}
        for indices in batches:
            batch_items = [items[i] for i in indices]
            batch_prompts = [prompts[i] for i in indices]
            scored = self._score_batch(batch_items, batch_prompts, transform=transform)
            for position, prediction in zip(indices, scored, strict=True):
                prediction.original_option_keys = original_keys[position]
                by_position[position] = prediction
        return [by_position[i] for i in range(len(items))]

    def score_prompts(
        self,
        prompts: Sequence[str],
        option_keys: Sequence[Sequence[str]],
        gold_keys: Sequence[str],
        qtypes: Sequence[str] | None = None,
        sources: Sequence[str] | None = None,
        template_ids: Sequence[str] | None = None,
        item_ids: Sequence[str] | None = None,
        *,
        transform: dict[str, Any] | None = None,
        original_keys: Sequence[Sequence[str]] | None = None,
    ) -> list[Prediction]:
        """Score already-rendered prompts with an explicit option-key list per prompt.

        This exists for reference validation: the *same* prompt text can be handed to this
        harness and to another tool, so a disagreement cannot come from a prompt-building
        difference. The caller supplies the option keys in display order (the letters the prompt
        shows) and the gold key.
        """
        n = len(prompts)
        for name, seq in (("option_keys", option_keys), ("gold_keys", gold_keys)):
            if len(seq) != n:
                raise ValueError(f"{name} must have one entry per prompt ({len(seq)} != {n})")
        qtypes = list(qtypes) if qtypes else ["choice"] * n
        sources = list(sources) if sources else ["reference"] * n
        template_ids = list(template_ids) if template_ids else ["reference_prompt"] * n
        item_ids = list(item_ids) if item_ids else [f"prompt-{i}" for i in range(n)]
        original_keys = list(original_keys) if original_keys else [list(k) for k in option_keys]
        token_ids = [[self._key_token(k) for k in keys] for keys in option_keys]
        token_counts = [
            len(self.tokenizer.encode(prompt, add_special_tokens=False)) for prompt in prompts
        ]
        predictions: list[Prediction] = []
        batches = self._plan_batches(token_counts)
        for indices in batches:
            batch_prompts = [prompts[i] for i in indices]
            t0 = time.perf_counter()
            logits, n_tokens = self._next_token_logits(batch_prompts)
            elapsed = time.perf_counter() - t0
            per_item = elapsed / max(1, len(indices))
            for row, position, prompt_tokens in zip(logits, indices, n_tokens, strict=True):
                keys = list(option_keys[position])
                readout = read_option_probabilities(row, token_ids[position])
                probs = [float(p) for p in readout["option_probs"]]
                argmax = int(np.argmax(readout["option_probs"]))
                gold_key = gold_keys[position]
                gold_index = keys.index(gold_key)
                predictions.append(
                    Prediction(
                        run_id=self.run_id,
                        item_id=item_ids[position],
                        model_id=self.spec.model_id,
                        source=sources[position],
                        template_id=template_ids[position],
                        split="test",
                        qtype=qtypes[position],
                        option_keys=keys,
                        original_option_keys=list(original_keys[position]),
                        option_probs=probs,
                        label_mass=float(readout["label_mass"]),
                        vocab_argmax_is_option=bool(readout["vocab_argmax_is_option"]),
                        n_tokens_above_best_option=int(readout["n_tokens_above_best_option"]),
                        best_option_in_top5=bool(readout["best_option_in_top5"]),
                        top1_over_option_mass=float(readout["top1_over_option_mass"]),
                        argmax_index=argmax,
                        gold_key=gold_key,
                        correct=argmax == gold_index,
                        expected_level=(
                            expected_level(probs) if qtypes[position] == "score" else None
                        ),
                        latency_s=per_item,
                        variant=self.variant,
                        prompt_tokens=int(prompt_tokens),
                        transform=transform or {},
                    )
                )
        return predictions

    def _score_batch(
        self,
        batch: list[Item],
        prompts: list[str],
        *,
        transform: dict[str, Any] | None = None,
    ) -> list[Prediction]:
        """Score one batch of already-rendered prompts."""
        predictions: list[Prediction] = []
        t0 = time.perf_counter()
        logits, n_tokens = self._next_token_logits(prompts)
        elapsed = time.perf_counter() - t0
        per_item = elapsed / max(1, len(batch))
        for row, item, prompt_tokens in zip(logits, batch, n_tokens, strict=True):
            if item.n_options > self.spec.max_options:
                raise ValueError(
                    f"{item.item_id}: {item.n_options} options exceeds "
                    f"max_options={self.spec.max_options} for {self.spec.model_id}"
                )
            token_ids = [self._key_token(k) for k in item.option_keys]
            readout = read_option_probabilities(row, token_ids)
            probs = [float(p) for p in readout["option_probs"]]
            argmax = int(np.argmax(readout["option_probs"]))
            predictions.append(
                Prediction(
                    run_id=self.run_id,
                    item_id=item.item_id,
                    model_id=self.spec.model_id,
                    source=item.source,
                    template_id=item.template_id,
                    split=str(item.split),
                    qtype=str(item.qtype),
                    option_keys=item.option_keys,
                    option_probs=probs,
                    label_mass=float(readout["label_mass"]),
                    vocab_argmax_is_option=bool(readout["vocab_argmax_is_option"]),
                    n_tokens_above_best_option=int(readout["n_tokens_above_best_option"]),
                    best_option_in_top5=bool(readout["best_option_in_top5"]),
                    top1_over_option_mass=float(readout["top1_over_option_mass"]),
                    argmax_index=argmax,
                    gold_key=item.gold,
                    correct=argmax == item.gold_index,
                    expected_level=(
                        # canonical options are ordered lowest level first, so the expected
                        # level (1-based) is sum_k (k+1) * p_k over the canonical order
                        expected_level(probs) if item.qtype is QuestionType.SCORE else None
                    ),
                    latency_s=per_item,
                    variant=self.variant,
                    prompt_tokens=int(prompt_tokens),
                    transform=transform or {},
                )
            )
        return predictions


def greedy_first_token(
    harness: Harness,
    prompts: Sequence[str],
    option_token_ids: Sequence[Sequence[int]],
) -> list[int]:
    """Greedily generate one token per prompt and return which option (if any) it was.

    Returns the option *index* whose token id equals the generated token id, or ``-1`` when the
    model generated something that is not one of the item's option tokens. This is the check the
    D12 gate needs: it compares the readout's argmax against what the model actually says, so a
    readout that is internally consistent but wrong cannot pass.
    """
    import torch

    results: list[int] = []
    for prompt, ids in zip(prompts, option_token_ids, strict=True):
        encoded = harness.tokenizer(prompt, return_tensors="pt")
        device = getattr(harness.model, "device", None)
        if device is not None:
            encoded = {k: v.to(device) for k, v in encoded.items()}
        with torch.inference_mode():
            generated = harness.model.generate(**encoded, max_new_tokens=1, do_sample=False)
        new_tokens = generated[0, encoded["input_ids"].shape[1] :]
        token_id = int(new_tokens[0]) if len(new_tokens) else -1
        results.append(ids.index(token_id) if token_id in ids else -1)
    return results


def summarise(predictions: Sequence[Prediction], *, n_bins: int = 15, n_resamples: int = 1000,
              seed: int = 0) -> dict[str, Any]:
    """Per (model, source, template, split) metrics with explicit denominators.

    Every accuracy is reported with its majority baseline, every mean with a bootstrap CI,
    and the arithmetic is checked before returning (R5).
    """
    groups: dict[tuple[str, str, str, str], list[Prediction]] = {}
    for pred in predictions:
        groups.setdefault((pred.model_id, pred.source, pred.template_id, pred.split), []).append(pred)

    out: dict[str, Any] = {"groups": {}, "n_predictions": len(predictions)}
    for key, group in sorted(groups.items()):
        model_id, source, template_id, split = key
        gold = [p.gold_key for p in group]
        predicted = [p.option_keys[p.argmax_index] for p in group]
        acc = accuracy(predicted, gold)
        probs = np.asarray([p.option_probs for p in group], dtype=np.float64)
        gold_idx = [p.option_keys.index(p.gold_key) for p in group]
        correct_flags = [1.0 if p.correct else 0.0 for p in group]
        _, lo, hi = bootstrap_ci(correct_flags, n_resamples=n_resamples, seed=seed)
        entry: dict[str, Any] = {
            "model_id": model_id,
            "source": source,
            "template_id": template_id,
            "split": split,
            "qtype": group[0].qtype,
            "n": acc.n,
            "n_correct": acc.n_correct,
            "accuracy": acc.micro,
            "accuracy_ci95": [lo, hi],  # percentile bootstrap, seeded
            "majority_baseline": acc.majority_baseline,
            "majority_label": acc.majority_label,
            "macro_accuracy": macro_accuracy(predicted, gold),
            "brier": brier_score(probs, gold_idx),
            "ece": ece_detail(probs, gold_idx, n_bins=n_bins)["ece"],
            "mean_label_mass": float(np.mean([p.label_mass for p in group])),
            "share_vocab_argmax_is_option": float(
                np.mean([1.0 if p.vocab_argmax_is_option else 0.0 for p in group])
            ),
            "share_best_option_in_vocab_top5": float(
                np.mean([1.0 if p.best_option_in_top5 else 0.0 for p in group])
            ),
            "median_n_tokens_above_best_option": float(
                np.median([p.n_tokens_above_best_option for p in group])
            ),
            "mean_top1_over_option_mass": float(
                np.mean([p.top1_over_option_mass for p in group])
            ),
            "p50_latency_s": float(np.percentile([p.latency_s for p in group], 50)),
            "p95_latency_s": float(np.percentile([p.latency_s for p in group], 95)),
            "variant": group[0].variant,
        }
        if group[0].qtype == "score":
            levels = [float(p.expected_level) for p in group if p.expected_level is not None]
            gold_levels = [float(p.option_keys.index(p.gold_key) + 1) for p in group]
            entry["mean_expected_level"] = float(np.mean(levels))
            entry["mean_gold_level"] = float(np.mean(gold_levels))
            entry["mean_level_abs_error"] = float(np.mean(np.abs(np.array(levels) - np.array(gold_levels))))
        if sum(1 for p in group) != acc.n:
            raise AssertionError("group size disagrees with accuracy denominator")
        out["groups"][f"{model_id}|{source}|{template_id}|{split}"] = entry
    return out


def write_predictions(path: Path, predictions: Sequence[Prediction]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for pred in predictions:
            fh.write(json.dumps(pred.to_json(), ensure_ascii=False) + "\n")
    return path


def write_summary(path: Path, summary: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2) + "\n")
    return path
