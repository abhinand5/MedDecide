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
    vocab_argmax_is_option: bool = True
    n_tokens_above_best_option: int = 0
    best_option_in_top5: bool = False
    top1_over_option_mass: float = 0.0
    transform: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "model_id": self.model_id,
            "source": self.source,
            "template_id": self.template_id,
            "split": self.split,
            "qtype": self.qtype,
            # option labels are deliberately absent: the repo is public and the item text
            # lives only in the gitignored data/ tree
            "option_keys": self.option_keys,
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
        encoded = self.tokenizer(
            list(prompts), return_tensors="pt", padding=True, truncation=True,
            max_length=self.max_prompt_tokens,
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

    def _plan_batches(self, prompt_token_counts: list[int]) -> list[tuple[list[int], int, int]]:
        """Plan batches that respect both ``batch_size`` and ``max_batch_tokens``.

        Items are ordered by prompt length first (then the original order is restored), so a
        handful of very long items do not force small batches on all the short ones.
        """
        order = sorted(range(len(prompt_token_counts)), key=lambda i: prompt_token_counts[i])
        batches: list[tuple[list[int], int, int]] = []
        current: list[int] = []
        tokens = 0
        for index in order:
            n = prompt_token_counts[index]
            if current and (len(current) >= self.batch_size or tokens + n > self.max_batch_tokens):
                batches.append((current, current[0], current[-1] + 1))
                current, tokens = [], 0
            current.append(index)
            tokens += n
        if current:
            batches.append((current, current[0], current[-1] + 1))
        return batches

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
        prompts = [render_prompt(item, self.tokenizer, self.variant) for item in items]
        prompt_token_counts = [
            len(self.tokenizer.encode(prompt, add_special_tokens=False)) for prompt in prompts
        ]
        if sort_by_length:
            batches = [(indices, start, end) for indices, start, end in self._plan_batches(prompt_token_counts)]
        else:  # pragma: no cover - escape hatch for order-sensitive callers
            batches = [
                (list(range(start, min(start + self.batch_size, len(items)))), start,
                 min(start + self.batch_size, len(items)))
                for start in range(0, len(items), self.batch_size)
            ]
        by_position: dict[int, Prediction] = {}
        for indices, _start, _end in batches:
            batch_items = [items[i] for i in indices]
            batch_prompts = [prompts[i] for i in indices]
            scored = self._score_batch(batch_items, batch_prompts, transform=transform)
            for position, prediction in zip(indices, scored, strict=True):
                by_position[position] = prediction
        return [by_position[i] for i in range(len(items))]

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
                        expected_level(probs) if item.qtype is QuestionType.SCORE else None
                    ),
                    latency_s=per_item,
                    variant=self.variant,
                    prompt_tokens=int(prompt_tokens),
                    transform=transform or {},
                )
            )
        return predictions


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
