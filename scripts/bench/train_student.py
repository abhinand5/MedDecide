#!/usr/bin/env python
"""S9: train MedDecide-0.8B (instruct base) — one pass over the S6 mix.

A thin CLI over :class:`meddecide.train.Trainer`. It adds exactly four things the library does
not do on its own, all of them about *this* run's bookkeeping:

1. the S9 recipe from ``configs/student_v0.yaml`` held fixed (LoRA r=16, LoRA lr 2e-4, batch 8,
   8k prompt cap, one epoch) with a wall-clock budget covering data loading and dev evals;
2. **dev evaluation every ``--eval-every`` steps** on a fixed, reproducible sample of
   ``data/train/student_v0/dev.jsonl`` stratified **by template** and at least
   ``--min-eval-items`` items (the sample's item ids and a digest are written beside the run, so
   the selection can be re-derived);
3. **best-checkpoint selection by dev macro accuracy, Brier as the tie-break** (ADVISORY S9),
   then the earlier step. A checkpoint is written every time the rule improves; the rule is
   quoted verbatim in ``best.json`` and ``run.json``;
4. a **planned-epoch pass** before training: the CLI encodes the epoch once
   (:func:`meddecide.train.data.plan_epoch_detail`), records the batch-count/token plan and the
   per-batch length series, checks that batch length does not trend with batch index (the
   accidental-curriculum regression test, on the real item set), and hands the planned batch
   count to the trainer as the total for **linear warmup (3 %) + cosine decay**;
5. after training, the **per-qtype temperature fit on dev** (never on test), written to
   ``temperature.json`` and embedded in the saved checkpoint's ``model.json`` so
   ``scripts/bench/run_student.py`` applies it without a second fit.

Usage (the real S9 run)::

    uv run python scripts/bench/train_student.py \\
        --max-seconds 9600 --eval-every 500 \\
        --out outputs/student_v0/S9

Training reads no test split: the loader refuses items whose ``split`` is not the expected one.
Predictions and item text stay in the gitignored ``outputs/``; only aggregates are committed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from collections import defaultdict
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr

from meddecide.eval.metrics import accuracy, brier_score, ece_from_confidence, macro_accuracy
from meddecide.model.meddecide_model import MedDecideModel, ScoredItems
from meddecide.train.config import DEFAULT_CONFIG_PATH, StudentConfig, load_config
from meddecide.train.data import (
    plan_epoch_detail,
    read_items,
    shuffled_order,
    stratified_sample_by,
)
from meddecide.train.temperature import fit_per_qtype
from meddecide.train.trainer import Trainer, provenance
from meddecide.utils.provenance import utcnow

# The S8 tool owns the loop's temperature-fit conventions (minimum item count, bound handling,
# `NOT FITTED` reasons). Reusing its function keeps this run's `temperature.json` exactly the
# shape `run_student.py` reads back, instead of a second, drifting implementation.
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
from run_student import (  # noqa: E402
    DEFAULT_MIN_TEMPERATURE_ITEMS,
    TEMPERATURE_BOUNDS,
    fit_temperatures,
)

DEFAULT_OUT = Path("outputs/student_v0/S9")
DEFAULT_MODEL_ID = "meddecide-0.8b-lora-pointer"
# selection rule, quoted verbatim into every artifact that reports a best step (ADVISORY S9:
# "keep the best dev checkpoint (dev macro-accuracy, Brier as tiebreak)")
SELECTION_RULE = (
    "highest dev macro accuracy on the fixed dev evaluation sample; ties (|dmacro| <= 1e-12) "
    "broken by lower dev Brier, then by the earlier step"
)
# how many dev items the *periodic* evaluation scores, per template, and the floor the sampled
# set must reach (the full dev split is used for the final fit and the per-template report)
DEFAULT_EVAL_PER_TEMPLATE = 150
MIN_EVAL_ITEMS = 2000
# forward-only dev scoring batches: bigger than training batches (no activations are kept); the
# same defaults the S8 evaluation path uses
DEFAULT_EVAL_BATCH_SIZE = 16
DEFAULT_EVAL_MAX_BATCH_TOKENS = 32768


# --------------------------------------------------------------------------- pure helpers
def selection_key(metrics: dict[str, Any]) -> tuple[float, float]:
    """Sort key whose minimum is the selected checkpoint: ``(-macro accuracy, Brier)``."""
    return (-float(metrics["macro_accuracy"]), float(metrics["brier"]))


def is_better(candidate: dict[str, Any], best: dict[str, Any] | None, *, tol: float = 1e-12) -> bool:
    """Whether ``candidate`` beats ``best`` under :data:`SELECTION_RULE` (strict improvement)."""
    if best is None:
        return True
    cand_macro, cand_brier = selection_key(candidate)
    best_macro, best_brier = selection_key(best)
    if cand_macro < best_macro - tol:  # -macro is smaller, i.e. macro accuracy is higher
        return True
    return abs(cand_macro - best_macro) <= tol and cand_brier < best_brier - tol


def length_trend(
    batch_tokens: Sequence[int], batch_mean_chars: Sequence[float], *, head: int = 200
) -> dict[str, Any]:
    """Spearman correlations between batch index and batch length (the anti-curriculum check).

    ``rho_tokens`` / ``rho_chars`` are taken over the whole epoch; the ``head_*`` numbers are the
    first and last 20 batches *of the first* ``head`` batches (the operator's sanity check). The
    accidental-curriculum bug produced rho near +1 with batches growing monotonically; a
    stationary plan gives rho near 0 and a head ratio near 1.
    """
    n = len(batch_tokens)
    if len(batch_mean_chars) != n:
        raise ValueError("batch_tokens and batch_mean_chars must be the same length")
    index = np.arange(n, dtype=np.float64)
    tokens = np.asarray(batch_tokens, dtype=np.float64)
    chars = np.asarray(batch_mean_chars, dtype=np.float64)
    head_n = int(min(head, n))
    first = slice(0, min(20, head_n))
    last = slice(max(0, head_n - 20), head_n)
    first_mean = float(tokens[first].mean()) if head_n else None
    last_mean = float(tokens[last].mean()) if head_n else None
    return {
        "n_batches": n,
        "rho_tokens": None if n < 3 else float(spearmanr(index, tokens).statistic),
        "rho_mean_chars": None if n < 3 else float(spearmanr(index, chars).statistic),
        "head_batches": head_n,
        "head_first20_mean_tokens": first_mean,
        "head_last20_mean_tokens": last_mean,
        "head_first20_mean_chars": float(chars[first].mean()) if head_n else None,
        "head_last20_mean_chars": float(chars[last].mean()) if head_n else None,
        "head_ratio_tokens": (
            float(last_mean / first_mean) if first_mean not in (None, 0.0) else None
        ),
        "head_ratio_chars": (
            float(chars[last].mean() / chars[first].mean())
            if head_n and chars[first].mean() > 0
            else None
        ),
    }


def best_eval(evals: Sequence[dict[str, Any]], *, tol: float = 1e-12) -> dict[str, Any] | None:
    """The eval row the rule selects from a trajectory (the first one on an exact tie)."""
    best: dict[str, Any] | None = None
    for row in evals:
        if is_better(row["metrics"], None if best is None else best["metrics"], tol=tol):
            best = row
    return best


def sample_digest(items: Sequence[Any]) -> str:
    """Deterministic digest of a sample's item ids (order-independent)."""
    joined = "\n".join(sorted(str(item.item_id) for item in items))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def check_split(items: Sequence[Any], expected: str, *, path: str) -> None:
    """Refuse a file that holds a split it should not (test data is evaluation-only, D13)."""
    bad = sorted({str(item.split) for item in items} - {expected})
    if bad:
        raise ValueError(
            f"{path}: expected only split={expected!r}, found {bad} — training never reads a "
            f"test split (AGENTS.md data rules)"
        )


def grouped_metrics(scored: ScoredItems, *, group_by: str = "template_id") -> dict[str, Any]:
    """Accuracy/macro/Brier/ECE per ``group_by`` value, for metrics that need explicit weights.

    Brier is computed per option-count group and combined with explicit weights, exactly as
    :meth:`ScoredItems.metrics` does: a template's items need not all offer the same number of
    options, and ``brier_score`` needs rectangular rows (R5: the parts sum to the total).
    """
    groups: dict[str, list[int]] = defaultdict(list)
    for i, item in enumerate(scored.items):
        if not hasattr(item, group_by):
            raise ValueError(f"items have no attribute {group_by!r}")
        groups[str(getattr(item, group_by))].append(i)
    out: dict[str, Any] = {}
    for name, idx in sorted(groups.items()):
        keys = [scored.option_keys[i] for i in idx]
        probs = [np.asarray(scored.probs[i], dtype=np.float64) for i in idx]
        gold_idx = [int(scored.gold_indices[i]) for i in idx]
        gold = [keys[j][gold_idx[j]] for j in range(len(idx))]
        predicted = [keys[j][int(np.argmax(probs[j]))] for j in range(len(idx))]
        correct = [predicted[j] == gold[j] for j in range(len(idx))]
        by_width: dict[int, list[int]] = defaultdict(list)
        for j, row in enumerate(probs):
            by_width[len(row)].append(j)
        weighted = 0.0
        for width, js in sorted(by_width.items()):
            stacked = np.stack([probs[j] for j in js])
            if stacked.shape[1] != width:  # pragma: no cover - shape guard
                raise AssertionError("option-count grouping is inconsistent")
            weighted += len(js) * brier_score(stacked, [gold_idx[j] for j in js])
        report = accuracy(predicted, gold)
        out[name] = {
            "n": len(idx),
            "n_correct": report.n_correct,
            "accuracy": report.micro,
            "majority_baseline": report.majority_baseline,
            "majority_label": report.majority_label,
            "macro_accuracy": macro_accuracy(predicted, gold),
            "brier": weighted / len(idx),
            "ece": ece_from_confidence([float(p.max()) for p in probs], correct),
            "mean_gold_prob": float(np.mean([probs[j][gold_idx[j]] for j in range(len(idx))])),
            "qtype": str(scored.items[idx[0]].qtype),
            "source": str(scored.items[idx[0]].source),
            "n_options_max": int(max(len(p) for p in probs)),
        }
    return out


def apply_temperature(scored: ScoredItems, temperature: dict[str, float]) -> ScoredItems:
    """A copy of ``scored`` with per-qtype temperatures applied (``{}`` = leave at T=1)."""
    if not temperature:
        return scored
    probs = []
    for logits, qtype in zip(scored.logits, scored.qtypes, strict=True):
        t = float(temperature.get(qtype, temperature.get("*", 1.0)))
        shifted = np.asarray(logits, dtype=np.float64) / t
        shifted = shifted - shifted.max()
        exp = np.exp(shifted)
        probs.append(exp / exp.sum())
    return ScoredItems(
        items=scored.items,
        option_keys=scored.option_keys,
        original_option_keys=scored.original_option_keys,
        logits=scored.logits,
        probs=probs,
        gold_indices=scored.gold_indices,
        prompt_tokens=scored.prompt_tokens,
        latency_s=scored.latency_s,
        wall_clock_s=scored.wall_clock_s,
        batch_sizes=scored.batch_sizes,
    )


def dev_report(scored: ScoredItems, *, temperature: dict[str, float] | None = None) -> dict[str, Any]:
    """Overall + per-qtype + per-template dev metrics for one scoring pass.

    ``temperature`` is applied to the logits first (``{}`` = deliberately uncalibrated), so the
    caller can report the calibrated and uncalibrated numbers from a single forward pass.
    """
    calibrated = apply_temperature(scored, dict(temperature or {}))
    return {
        "overall": calibrated.metrics(),
        "per_qtype": calibrated.per_qtype_metrics(),
        "per_template": grouped_metrics(calibrated, group_by="template_id"),
        "per_source": grouped_metrics(calibrated, group_by="source"),
    }


def compare_logits(alone: Sequence[Any], batched: Sequence[Any]) -> dict[str, Any]:
    """Max/mean absolute head-logit difference between two scorings of the same items.

    ``alone[i]`` and ``batched[i]`` are one item's logit vector from the single-item run and from
    the batched run; this is the arithmetic the padding probe is built on.
    """
    if len(alone) != len(batched):
        raise ValueError("the two scorings must hold the same number of items")
    if not alone:
        raise ValueError("nothing to compare")
    deltas = []
    for i, (a, b) in enumerate(zip(alone, batched, strict=True)):
        row_a = np.asarray(a, dtype=np.float64)
        row_b = np.asarray(b, dtype=np.float64)
        if row_a.shape != row_b.shape:
            raise ValueError(
                f"item {i}: {row_a.shape[0]} options in one scoring and {row_b.shape[0]} in the "
                f"other — the two scorings are not aligned"
            )
        deltas.append(float(np.max(np.abs(row_a - row_b))))
    return {
        "n_items": len(deltas),
        "max_abs_delta": max(deltas),
        "mean_abs_delta": float(np.mean(deltas)),
        "per_item_max_abs_delta": deltas,
    }


def padding_check(
    model: Any,
    items: Sequence[Any],
    *,
    n_items: int = 8,
    tolerance: float = 1e-3,
    max_prompt_tokens: int | None = None,
) -> dict[str, Any]:
    """Arm (c): is left padding sound for this architecture?

    The same item is scored (i) alone, (ii) inside a left-padded batch of items with very
    different lengths, and (iii) inside a *uniform-length* batch (the same item repeated), which
    isolates batch-shape numerical noise from padding leakage. Qwen3.5 mixes softmax and
    linear-attention layers; a masked softmax layer ignores pads, but a linear-attention mixer can
    carry pad state into the real tokens, in which case (ii) differs from (i) far above (iii).
    """
    if len(items) < 2:
        raise ValueError("the padding check needs at least two items")
    order = sorted(range(len(items)), key=lambda i: (len(items[i].state) + len(items[i].question)))
    picks = sorted({order[round(k * (len(order) - 1) / max(1, n_items - 1))] for k in range(n_items)})
    sample = [items[i] for i in picks]
    kwargs = {
        "max_prompt_tokens": max_prompt_tokens,
        "max_batch_tokens": 10**9,  # force one batch, so the only difference is the padding
    }
    alone = model.score_items(sample, batch_size=1, **kwargs)
    together = model.score_items(sample, batch_size=len(sample), **kwargs)
    longest = max(sample, key=lambda i: len(i.state))
    longest_index = sample.index(longest)
    uniform = model.score_items([longest] * len(sample), batch_size=len(sample), **kwargs)
    padded = compare_logits(alone.logits, together.logits)
    # the uniform batch holds one item repeated: every row is that item, so compare its alone
    # scoring with row 0 (this is the batch-shape noise floor, not a padding effect)
    control = compare_logits(
        [alone.logits[longest_index]], [uniform.logits[0]]
    )
    verdict = "PASS" if padded["max_abs_delta"] <= tolerance else "FAIL"
    return {
        "task": "S9-diag",
        "kind": "padding_check",
        "at_utc": utcnow(),
        "tolerance": tolerance,
        "n_items": len(sample),
        "items": [
            {
                "item_id": sample[j].item_id,
                "template_id": sample[j].template_id,
                "prompt_tokens": int(alone.prompt_tokens[j]),
                "max_abs_delta_padded": padded["per_item_max_abs_delta"][j],
            }
            for j in range(len(sample))
        ],
        "alone_vs_left_padded_batch": padded,
        "alone_vs_uniform_batch_control": control,
        "verdict": verdict,
        "interpretation": (
            "PASS: left padding does not change the head logits beyond the tolerance. FAIL: the "
            "architecture (linear-attention layers are the suspect) carries pad state into the "
            "real tokens, so every mixed-length batch is affected."
        ),
    }


def write_json(path: Path, payload: Any) -> Path:
    """Write JSON atomically (tmp + replace), so a killed process never leaves a half file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)
    return path


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def seed_everything(seed: int) -> None:
    """Same seeding the S7 smoke used, so an S9 rerun reproduces the batch plan."""
    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _field(record: Any, name: str) -> Any:
    """One step record's field, whether it is a StepRecord or a row read back from the log."""
    return record[name] if isinstance(record, dict) else getattr(record, name)


def throughput_of(history: Sequence[Any]) -> dict[str, Any]:
    """Items/s and tokens/s over the optimiser steps only (dev evals are not training time).

    Accepts :class:`~meddecide.train.trainer.StepRecord` objects and the JSON rows the run wrote,
    so a stopped run's log can be summarised with the same arithmetic.
    """
    if not history:
        return {"steps": 0}
    step_seconds = float(sum(_field(r, "elapsed_s") for r in history))
    items = sum(_field(r, "n_items") for r in history)
    real = sum(_field(r, "real_tokens") for r in history)
    padded = sum(_field(r, "padded_tokens") for r in history)
    elapsed = np.asarray([_field(r, "elapsed_s") for r in history], dtype=np.float64)
    return {
        "steps": len(history),
        "items": items,
        "real_tokens": real,
        "padded_tokens": padded,
        "step_seconds_total": step_seconds,
        "step_seconds_mean": float(elapsed.mean()),
        "step_seconds_p50": float(np.percentile(elapsed, 50)),
        "step_seconds_p95": float(np.percentile(elapsed, 95)),
        "items_per_s": items / step_seconds if step_seconds else None,
        "tokens_per_s": real / step_seconds if step_seconds else None,
        "padded_tokens_per_s": padded / step_seconds if step_seconds else None,
        "gpu_peak_memory_gb": max((_field(r, "gpu_peak_gb") for r in history), default=0.0),
    }


def applied_temperatures(verdict: dict[str, Any]) -> dict[str, float]:
    """The temperatures the S8 verdict says may be applied — never a ``NOT FITTED`` value."""
    return {
        qtype: float(entry["temperature"])
        for qtype, entry in sorted(verdict.get("per_qtype", {}).items())
        if entry.get("fitted") and entry.get("temperature") is not None
    }


def logits_by_qtype(scored: ScoredItems) -> dict[str, list[np.ndarray]]:
    out: dict[str, list[np.ndarray]] = defaultdict(list)
    for logits, qtype in zip(scored.logits, scored.qtypes, strict=True):
        out[str(qtype)].append(logits)
    return dict(out)


def gold_by_qtype(scored: ScoredItems) -> dict[str, list[int]]:
    out: dict[str, list[int]] = defaultdict(list)
    for gold, qtype in zip(scored.gold_indices, scored.qtypes, strict=True):
        out[str(qtype)].append(int(gold))
    return dict(out)


# --------------------------------------------------------------------------- the dev-eval hook
class DevSelector:
    """The training hook: dev-evaluate every ``eval_every`` steps, keep the best by the S9 rule.

    The rule (:data:`SELECTION_RULE`) is dev macro accuracy first, Brier as the tie-break — the
    ADVISORY's S9 wording. The Brier-first alternative is reported as additional analysis, since
    the earlier brief had it the other way round.

    ``trainer`` only has to provide ``evaluate(items)``, ``save_checkpoint(path)`` and a ``model``
    with ``train_mode()`` — which is what makes this testable without a GPU.
    """

    def __init__(
        self,
        trainer: Any,
        eval_items: Sequence[Any],
        *,
        eval_every: int,
        evals_path: Path,
        best_dir: Path,
        best_json_path: Path,
        eval_sample: dict[str, Any],
        model_id: str,
        save_every_eval: bool = False,
        checkpoints_dir: Path | None = None,
        print_fn: Callable[[str], None] = print,
    ) -> None:
        self.trainer = trainer
        self.eval_items = list(eval_items)
        self.eval_every = int(eval_every)
        self.evals_path = Path(evals_path)
        self.best_dir = Path(best_dir)
        self.best_json_path = Path(best_json_path)
        self.eval_sample = dict(eval_sample)
        self.model_id = model_id
        self.save_every_eval = bool(save_every_eval)
        self.checkpoints_dir = Path(checkpoints_dir) if checkpoints_dir else None
        self.print = print_fn
        self.eval_checkpoints: list[dict[str, Any]] = []
        self.evals: list[dict[str, Any]] = []
        self.best_step: int | None = None
        self.best_metrics: dict[str, Any] | None = None
        self.best_saved_at_utc: str | None = None
        self.save_error: str | None = None
        self.n_errors = 0

    # ---- the hook -----------------------------------------------------------
    def __call__(self, step: int) -> None:
        if self.eval_every and step % self.eval_every == 0:
            self.evaluate(step)

    def evaluate(self, step: int) -> dict[str, Any]:
        t_eval = time.perf_counter()
        try:
            metrics = self.trainer.evaluate(self.eval_items)
        except Exception as exc:  # a failed eval must not kill a 2.7 h training run
            self.n_errors += 1
            row = {
                "event": "dev_eval_error",
                "step": step,
                "at_utc": utcnow(),
                "error": f"{type(exc).__name__}: {exc}",
            }
            append_jsonl(self.evals_path, row)
            self.print(f"[S9] step {step}: dev eval FAILED — {row['error']}")
            self.trainer.model.train_mode()
            return row
        seconds = time.perf_counter() - t_eval
        improved = is_better(metrics, self.best_metrics)
        row = {
            "event": "dev_eval",
            "step": step,
            "at_utc": utcnow(),
            "seconds": seconds,
            "n_items": len(self.eval_items),
            "metrics": metrics,
            "selected": bool(improved),
            "selection_rule": SELECTION_RULE,
            "best_step_so_far": step if improved else self.best_step,
        }
        self.evals.append(row)
        if self.save_every_eval:
            # a checkpoint per dev eval, so a later reselection never needs a fresh run
            target = (self.checkpoints_dir or (self.best_dir.parent / "checkpoints")) / (
                f"step_{step}"
            )
            try:
                self.trainer.save_checkpoint(target)
                row["checkpoint"] = str(target)
                self.eval_checkpoints.append({"step": step, "path": str(target), "metrics": metrics})
            except Exception as exc:  # recorded, never fatal
                row["checkpoint_error"] = f"{type(exc).__name__}: {exc}"
        append_jsonl(self.evals_path, row)
        if improved:
            error = None
            try:
                self.trainer.save_checkpoint(self.best_dir)
            except Exception as exc:  # recorded, then the run continues
                error = f"{type(exc).__name__}: {exc}"
            self.best_step = step
            self.best_metrics = metrics
            self.best_saved_at_utc = utcnow()
            self.save_error = error
            append_jsonl(self.evals_path, {"event": "checkpoint_saved", "step": step,
                                           "checkpoint": str(self.best_dir), "save_error": error})
            write_json(self.best_json_path, self.best_payload())
        self.print(
            f"[S9] step {step}: n={metrics['n']} acc={metrics['accuracy']:.4f} "
            f"macro={metrics['macro_accuracy']:.4f} brier={metrics['brier']:.4f} "
            f"nll={metrics['mean_nll']:.4f} eval={seconds:.1f}s"
            + ("  <- best, checkpoint saved" if improved else "")
        )
        # the hook runs inside the optimiser loop: `evaluate` left the model in eval mode
        self.trainer.model.train_mode()
        return row

    # ---- artifacts ----------------------------------------------------------
    def brier_best(self) -> dict[str, Any] | None:
        """Additional analysis: the step a Brier-first rule would pick."""
        rows = [r for r in self.evals if r.get("event") == "dev_eval"]
        if not rows:
            return None
        best = min(rows, key=lambda r: (r["metrics"]["brier"], -r["metrics"]["macro_accuracy"]))
        return {
            "note": (
                "additional — a Brier-first rule (the superseded run-1 rule) would pick this step; "
                "reported, never used for S9's selection"
            ),
            "brier_best_step": int(best["step"]),
            "brier_best_brier": float(best["metrics"]["brier"]),
            "brier_best_macro_accuracy": float(best["metrics"]["macro_accuracy"]),
            "brier_best_accuracy": float(best["metrics"]["accuracy"]),
        }

    def best_payload(self) -> dict[str, Any]:
        return {
            "task": "S9",
            "kind": "best_checkpoint",
            "model_id": self.model_id,
            "checkpoint": str(self.best_dir),
            "step": self.best_step,
            "selection_rule": SELECTION_RULE,
            "dev_metrics_at_selection": self.best_metrics,
            "dev_eval": {
                "n_items": len(self.eval_items),
                "per_template": self.eval_sample.get("per_template"),
                "min_eval_items": self.eval_sample.get("min_eval_items"),
                "by_template": self.eval_sample.get("by_template"),
                "item_ids_path": self.eval_sample.get("path"),
                "sample_sha256": self.eval_sample.get("sha256"),
                "eval_every": self.eval_every,
            },
            "n_evals_so_far": len([r for r in self.evals if r.get("event") == "dev_eval"]),
            "written_at_utc": utcnow(),
            "save_error": self.save_error,
            "additional_analysis": self.brier_best(),
        }

    def trajectory(self) -> list[dict[str, Any]]:
        return [
            {
                "step": r["step"],
                "brier": r["metrics"]["brier"],
                "macro_accuracy": r["metrics"]["macro_accuracy"],
                "accuracy": r["metrics"]["accuracy"],
                "mean_nll": r["metrics"]["mean_nll"],
                "n": r["metrics"]["n"],
                "seconds": r["seconds"],
                "selected": r["selected"],
            }
            for r in self.evals
            if r.get("event") == "dev_eval"
        ]


# --------------------------------------------------------------------------- CLI
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="S9: train MedDecide-0.8B on the S6 mix")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--train-path", default=None, help="default: the config's train_path")
    parser.add_argument("--dev-path", default=None, help="default: the config's dev_path")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID,
                        help="id stamped into the run artifacts (not into any prediction row)")
    parser.add_argument("--max-seconds", type=float, default=None,
                        help="wall-clock budget for loading + training, measured from process "
                             "start; the run stops at the first step after it is spent")
    parser.add_argument("--steps", type=int, default=None,
                        help="hard step cap (default: one epoch, bounded only by --max-seconds)")
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--eval-per-template", type=int, default=DEFAULT_EVAL_PER_TEMPLATE,
                        help="dev items per template in the periodic evaluation")
    parser.add_argument("--min-eval-items", type=int, default=MIN_EVAL_ITEMS,
                        help="the periodic dev sample must reach at least this many items")
    parser.add_argument("--eval-batch-size", type=int, default=DEFAULT_EVAL_BATCH_SIZE,
                        help="forward-only dev scoring batch size (bigger than training's)")
    parser.add_argument("--eval-max-batch-tokens", type=int, default=DEFAULT_EVAL_MAX_BATCH_TOKENS,
                        help="forward-only dev scoring token cap per batch")
    parser.add_argument("--chunk-factor", type=int, default=None,
                        help="length-bucketing chunk size in units of batch_size "
                             "(default: the config's batch_chunk_factor, 100)")
    parser.add_argument("--dev-final-per-qtype", type=int, default=None,
                        help="cap per qtype for the final dev fit/report (default: every dev item)")
    parser.add_argument("--temperature-items-per-qtype", type=int, default=None,
                        help="cap per qtype for the temperature fit (default: every dev item)")
    parser.add_argument("--seed", type=int, default=None, help="default: the config's seed")
    parser.add_argument("--device", default=None, help="default: the config's device")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="default: the config's batch_size (8)")
    parser.add_argument("--train-limit", type=int, default=None,
                        help="debug: read only N training items (recorded in run.json)")
    parser.add_argument("--train-stride", type=int, default=1,
                        help="read every Nth training line; with --train-limit this is the fixed "
                             "subset the S9-diag arms share")
    parser.add_argument("--head-lr", type=float, default=None,
                        help="override the config's peak head LR (config.lr)")
    parser.add_argument("--lora-lr", type=float, default=None,
                        help="override the config's peak LoRA LR (config.lora_lr)")
    parser.add_argument("--lora-rank", type=int, default=None,
                        help="override LoRA r; alpha is scaled to keep the config's alpha/r ratio")
    parser.add_argument("--max-prompt-tokens", type=int, default=None,
                        help="override the prompt cap (config.max_prompt_tokens)")
    parser.add_argument("--save-every-eval", action="store_true",
                        help="write one checkpoint per dev eval into <out>/checkpoints/step_<n>")
    parser.add_argument("--diag-note", default=None,
                        help="free-text label recorded with the run (which diagnostic arm this is)")
    parser.add_argument("--padding-check", action="store_true",
                        help="arm (c): score items alone vs inside a left-padded batch, compare "
                             "the head logits, write padding_check.json and exit")
    parser.add_argument("--padding-check-from", type=Path, default=None,
                        help="with --padding-check: probe this saved checkpoint instead of a "
                             "fresh model (its weights are loaded, not retrained)")
    parser.add_argument("--padding-check-dtype", default=None,
                        help="with --padding-check: load the model in this dtype (e.g. float32) "
                             "to separate bf16 batch-shape noise from a real padding leak")
    parser.add_argument("--padding-check-out", type=Path, default=None,
                        help="with --padding-check: where to write the JSON (default "
                             "<out>/padding_check.json)")
    parser.add_argument("--log", type=Path, default=None,
                        help="per-step JSONL (default <out>/logs/train_steps.jsonl)")
    parser.add_argument("--dev-evals", type=Path, default=None,
                        help="per-eval JSONL (default <out>/logs/dev_evals.jsonl)")
    parser.add_argument("--best", type=Path, default=None,
                        help="best-checkpoint directory (default <out>/best)")
    parser.add_argument("--pidfile", type=Path, default=None,
                        help="default <out>/train.pid; removed on a clean exit")
    parser.add_argument("--finalise-from", type=Path, default=None,
                        help="write the post-training artifacts for an already-stopped run whose "
                             "checkpoint is at this path (nothing is trained)")
    parser.add_argument("--finalise-reason", default="stopped before the end of the pass",
                        help="recorded as the run's exit reason in --finalise-from mode")
    parser.add_argument("--dry-run", action="store_true",
                        help="load config + data, print the plan, and exit without touching the GPU")
    return parser.parse_args(argv)


def build_config(args: argparse.Namespace) -> StudentConfig:
    """The loaded config with the S9 recipe's run-level overrides applied."""
    from dataclasses import replace

    config = load_config(args.config)
    overrides: dict[str, Any] = {
        "train_path": args.train_path or config.train_path,
        "dev_path": args.dev_path or config.dev_path,
        "output_dir": str(args.out),
        "epochs": 1,
        "eval_every": args.eval_every,
        "eval_items": args.eval_per_template,
        "eval_batch_size": args.eval_batch_size,
        "eval_max_batch_tokens": args.eval_max_batch_tokens,
        "log_every": 1,  # the brief asks for a per-step log
        "temperature_per_qtype": True,
        "temperature_items_per_qtype": args.temperature_items_per_qtype,
        "fit_temperature": True,
        "max_seconds": args.max_seconds,
    }
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.device is not None:
        overrides["device"] = args.device
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size
    if args.chunk_factor is not None:
        overrides["batch_chunk_factor"] = args.chunk_factor
    if args.head_lr is not None:
        overrides["lr"] = float(args.head_lr)
    if args.lora_lr is not None:
        overrides["lora_lr"] = float(args.lora_lr)
    if args.max_prompt_tokens is not None:
        overrides["max_prompt_tokens"] = int(args.max_prompt_tokens)
    if args.lora_rank is not None:
        rank = int(args.lora_rank)
        if rank < 1:
            raise ValueError("--lora-rank must be >= 1")
        # keep alpha/r (the scaling that multiplies the adapter output) at the config's value:
        # comparing ranks must not also change the adapter's effective scale
        ratio = config.lora.alpha / config.lora.r
        overrides["lora"] = replace(
            config.lora, r=rank, alpha=max(1, round(rank * ratio))
        )
    return replace(config, **overrides)


def calibrate_and_report(
    best_model: Any,
    final_items: Sequence[Any],
    *,
    config: StudentConfig,
    out: Path,
    best_dir: Path,
    best_step: int,
    model_id: str,
    dev_items_available: int,
    limited_sample: bool,
) -> dict[str, Any]:
    """Temperature fit + per-template dev report + calibration embedded in the checkpoint.

    It scores the **saved** checkpoint (not whatever is in memory), so it is valid both at the end
    of a run and for a run that was stopped early (``--finalise-from``). Dev only; never a test
    split. Returns the payloads it wrote so the caller can put them in ``run.json``.
    """
    t0 = time.perf_counter()
    scored_final = best_model.score_items(
        final_items,
        batch_size=config.eval_batch_size or config.batch_size,
        max_batch_tokens=config.eval_max_batch_tokens or config.max_batch_tokens,
        max_prompt_tokens=config.max_prompt_tokens,
    )
    scored_seconds = time.perf_counter() - t0
    fits = fit_per_qtype(scored_final)
    # `fit_temperatures` (S8) owns the FITTED / NOT FITTED verdicts this loop uses
    verdict = fit_temperatures(
        split="dev",
        logits_by_qtype=logits_by_qtype(scored_final),
        gold_by_qtype=gold_by_qtype(scored_final),
        seed=config.seed,
        expected_qtypes=["choice", "noul", "score"],
    )
    applied = applied_temperatures(verdict)
    temperature_payload = {
        "task": "S9",
        "kind": "temperature_fit",
        "split": "dev",
        "fitted_at_utc": utcnow(),
        "seed": config.seed,
        "min_items": DEFAULT_MIN_TEMPERATURE_ITEMS,
        "bounds": list(TEMPERATURE_BOUNDS),
        "checkpoint": str(best_dir),
        "checkpoint_step": best_step,
        "model_id": model_id,
        "n_dev_items_scored": len(final_items),
        "dev_path": config.dev_path,
        "dev_split_sha256": sample_digest(final_items),
        "limited_sample": limited_sample,
        "fits": {k: v.to_dict() for k, v in sorted(fits.items())},
        "per_qtype": verdict["per_qtype"],
        "n_qtype_fitted": verdict["n_qtype_fitted"],
        "n_qtype_not_fitted": verdict["n_qtype_not_fitted"],
        "applied": applied,
    }
    write_json(out / "temperature.json", temperature_payload)

    calibrated_report = dev_report(scored_final, temperature=applied)
    uncalibrated_report = dev_report(scored_final, temperature={})
    dev_final = {
        "task": "S9",
        "kind": "dev_report",
        "split": "dev",
        "checkpoint": str(best_dir),
        "checkpoint_step": best_step,
        "dev_path": config.dev_path,
        "n_items": len(final_items),
        "n_items_available": dev_items_available,
        "sample_sha256": sample_digest(final_items),
        "scored_seconds": scored_seconds,
        "applied_temperatures": applied,
        "calibrated": calibrated_report,
        "uncalibrated": uncalibrated_report,
        "note": (
            "calibrated = per-qtype temperatures applied; uncalibrated = the same forward pass at "
            "T=1 (additional). A per-qtype temperature is monotone, so accuracy/macro accuracy "
            "are identical in both; Brier/ECE are not."
        ),
    }
    write_json(out / "dev_final.json", dev_final)

    # re-save the selected checkpoint with its calibration embedded (weights unchanged)
    best_model.calibration = applied
    best_model.save(best_dir)
    write_json(best_dir / "temperature.json", temperature_payload)
    reopened = json.loads((best_dir / "model.json").read_text(encoding="utf-8"))
    return {
        "temperature": temperature_payload,
        "dev_final": dev_final,
        "calibrated": calibrated_report,
        "uncalibrated": uncalibrated_report,
        "applied": applied,
        "calibration_embedded": reopened.get("calibration") or {},
        "scored_seconds": scored_seconds,
    }


def finalise_stopped_run(args: argparse.Namespace, out: Path) -> int:
    """Write the post-training artifacts for a run that was stopped before its own final phase.

    Everything it needs is on disk: `run_started.json` (the config actually used),
    `planned_epoch.json` (the anti-curriculum plan), `logs/train_steps.jsonl` (per-step records),
    `logs/dev_evals.jsonl` (the dev trajectory) and `best.json`/`best/` (the selected checkpoint).
    It trains nothing and reads no test split. The recorded `exit` names the stop reason.
    """
    best_dir = (
        Path(args.finalise_from)
        if args.finalise_from is not None
        else (Path(args.best) if args.best else out / "best")
    )
    started = json.loads((out / "run_started.json").read_text(encoding="utf-8"))
    config = StudentConfig.from_dict(started["config"])
    planned = json.loads((out / "planned_epoch.json").read_text(encoding="utf-8"))
    best = json.loads((out / "best.json").read_text(encoding="utf-8"))
    step_rows = [
        json.loads(line)
        for line in (out / "logs" / "train_steps.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    steps = [r for r in step_rows if r.get("event") == "step"]
    if not steps:
        print("BLOCKED — no step records to summarise", file=sys.stderr)
        return 4
    eval_rows = [
        json.loads(line)
        for line in (out / "logs" / "dev_evals.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    evals = [r for r in eval_rows if r.get("event") == "dev_eval"]
    rates = throughput_of(steps)
    items_seen = sum(r["n_items"] for r in steps)
    tokens_seen = sum(r["real_tokens"] for r in steps)
    train_items_total = int(started["data"]["train_items"])
    # the schedule was laid out over the planned batch count; the run never wrote its `end`
    # record, so reconstruct it here. The realised per-step lr_head/lr_lora in the step log are
    # the measurement; this is the specification it was applied from.
    total_steps = int(planned["plan"]["n_batches"])
    schedule = {
        "kind": "linear_warmup_cosine_decay",
        "total_steps": total_steps,
        "warmup_steps": round(float(config.warmup_fraction) * total_steps),
        "warmup_fraction": float(config.warmup_fraction),
        "peak_lr_by_group": [float(config.lr), float(config.lora_lr)],
        "source": (
            "reconstructed at finalisation from planned_epoch.json and the config; the per-step "
            "lr_head/lr_lora in logs/train_steps.jsonl are the realised values"
        ),
    }

    dev_items_all = read_items(config.dev_path)
    check_split(dev_items_all, "dev", path=config.dev_path)
    final_items = (
        dev_items_all
        if args.dev_final_per_qtype is None
        else stratified_sample_by(
            dev_items_all, key="qtype", per_group=args.dev_final_per_qtype, seed=config.seed
        )
    )
    best_model = MedDecideModel.load(best_dir, device=config.device)
    final = calibrate_and_report(
        best_model,
        final_items,
        config=config,
        out=out,
        best_dir=best_dir,
        best_step=int(best["step"]),
        model_id=args.model_id,
        dev_items_available=len(dev_items_all),
        limited_sample=args.dev_final_per_qtype is not None,
    )
    trajectory = [
        {
            "step": r["step"],
            "brier": r["metrics"]["brier"],
            "macro_accuracy": r["metrics"]["macro_accuracy"],
            "accuracy": r["metrics"]["accuracy"],
            "mean_nll": r["metrics"]["mean_nll"],
            "n": r["metrics"]["n"],
            "seconds": r["seconds"],
            "selected": r["selected"],
        }
        for r in evals
    ]
    run = {
        "task": "S9",
        "kind": "training_run",
        "model_id": args.model_id,
        "command": " ".join(sys.argv),
        "generated_at_utc": utcnow(),
        "finalised_because": args.finalise_reason,
        "provenance": started.get("provenance"),
        "config": config.to_dict(),
        "data": {
            "train_path": config.train_path,
            "dev_path": config.dev_path,
            "train_items": train_items_total,
            "dev_items": len(dev_items_all),
            "read_seconds": started["data"].get("read_seconds"),
            "plan_seconds": planned.get("seconds"),
            "items_seen": items_seen,
            "fraction_of_train_seen": items_seen / train_items_total,
            "tokens_seen": tokens_seen,
            "fraction_of_planned_tokens": (
                tokens_seen / planned["plan"]["real_tokens"]
                if planned.get("plan", {}).get("real_tokens")
                else None
            ),
        },
        "training": {
            "steps": steps[-1]["step"],
            "epochs_completed": steps[-1]["epoch"] + 1,
            "stopped_early": f"stopped: {args.finalise_reason}",
            "pass_completed": False,
            "step_seconds": rates.get("step_seconds_total"),
            "schedule": schedule,
            "throughput": rates,
        },
        "planned_epoch": {
            "path": str(out / "planned_epoch.json"),
            "seconds": planned.get("seconds"),
            "plan": planned.get("plan"),
            "length_trend": planned.get("length_trend"),
        },
        "dev_evals": {
            "path": str(out / "logs" / "dev_evals.jsonl"),
            "n_evals": len(trajectory),
            "best_step_by_selection_rule": best["step"],
            "best_metrics_by_selection_rule": best["dev_metrics_at_selection"],
            "selection_rule": best["selection_rule"],
            "additional_brier_first_rule": best.get("additional_analysis"),
            "trajectory": trajectory,
            "sample_sha256": best["dev_eval"].get("sample_sha256"),
            "sample_by_template": best["dev_eval"].get("by_template"),
        },
        "checkpoint": {
            "path": str(best_dir),
            "step": best["step"],
            "calibration_embedded": final["calibration_embedded"],
            "best_json": str(out / "best.json"),
        },
        "temperature": final["temperature"],
        "dev_final": {
            "path": str(out / "dev_final.json"),
            "n_items": len(final_items),
            "calibrated_overall": final["calibrated"]["overall"],
            "uncalibrated_overall": final["uncalibrated"]["overall"],
            "calibrated_per_qtype": final["calibrated"]["per_qtype"],
        },
        "exit": f"stopped at step {steps[-1]['step']}: {args.finalise_reason}",
    }
    write_json(out / "run.json", run)
    write_json(
        out / "finalisation.json",
        {
            "task": "S9",
            "kind": "finalisation_of_stopped_run",
            "at_utc": utcnow(),
            "command": " ".join(sys.argv),
            "reason": args.finalise_reason,
            "step_log_rows": len(steps),
            "last_step": steps[-1]["step"],
            "dev_evals": len(trajectory),
            "checkpoint_step": best["step"],
            "temperature_applied": final["applied"],
        },
    )
    print(
        f"[S9] finalised a stopped run at step {steps[-1]['step']}: best step {best['step']}, "
        f"temperature {json.dumps(final['applied'])}, wrote {out / 'run.json'}",
        flush=True,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out: Path = Path(args.out)
    log_path = Path(args.log) if args.log else out / "logs" / "train_steps.jsonl"
    evals_path = Path(args.dev_evals) if args.dev_evals else out / "logs" / "dev_evals.jsonl"
    best_dir = Path(args.best) if args.best else out / "best"
    pidfile = Path(args.pidfile) if args.pidfile else out / "train.pid"
    config = build_config(args)
    process_started = time.perf_counter()

    out.mkdir(parents=True, exist_ok=True)
    pidfile.parent.mkdir(parents=True, exist_ok=True)
    pidfile.write_text(
        json.dumps({"pid": os.getpid(), "started_at_utc": utcnow(), "argv": sys.argv}) + "\n",
        encoding="utf-8",
    )
    print(f"[S9] pid {os.getpid()} -> {pidfile}", flush=True)
    print(f"[S9] command: {' '.join(sys.argv)}", flush=True)
    print(f"[S9] config: {json.dumps(config.to_dict())}", flush=True)

    # ---- data -------------------------------------------------------------
    t0 = time.perf_counter()
    train_items = read_items(config.train_path, limit=args.train_limit, stride=args.train_stride)
    check_split(train_items, "train", path=config.train_path)
    dev_items_all = read_items(config.dev_path)
    check_split(dev_items_all, "dev", path=config.dev_path)
    read_seconds = time.perf_counter() - t0
    print(f"[S9] read {len(train_items)} train + {len(dev_items_all)} dev items "
          f"in {read_seconds:.1f}s", flush=True)

    eval_items = stratified_sample_by(
        dev_items_all, key="template_id", per_group=args.eval_per_template, seed=config.seed
    )
    if len(eval_items) < args.min_eval_items:
        raise ValueError(
            f"the periodic dev sample has {len(eval_items)} items, below the required "
            f"{args.min_eval_items}: raise --eval-per-template (templates: "
            f"{len({i.template_id for i in dev_items_all})})"
        )
    eval_sample = {
        "path": str(out / "dev_eval_ids.json"),
        "n_items": len(eval_items),
        "per_template": args.eval_per_template,
        "min_eval_items": args.min_eval_items,
        "seed": config.seed,
        "sha256": sample_digest(eval_items),
        "by_qtype": {
            q: sum(1 for i in eval_items if str(i.qtype) == q)
            for q in sorted({str(i.qtype) for i in eval_items})
        },
        "by_template": {
            t: sum(1 for i in eval_items if i.template_id == t)
            for t in sorted({i.template_id for i in eval_items})
        },
    }
    write_json(
        out / "dev_eval_ids.json",
        {**eval_sample, "item_ids": sorted(str(i.item_id) for i in eval_items)},
    )
    final_items = (
        dev_items_all
        if args.dev_final_per_qtype is None
        else stratified_sample_by(
            dev_items_all, key="qtype", per_group=args.dev_final_per_qtype, seed=config.seed
        )
    )
    print(f"[S9] periodic dev eval: {len(eval_items)} items over "
          f"{len(eval_sample['by_template'])} templates "
          f"({json.dumps(eval_sample['by_qtype'])}) sha256={eval_sample['sha256'][:16]}", flush=True)
    if args.finalise_from is not None:
        return finalise_stopped_run(args, out)
    if args.dry_run:
        print("[S9] dry run: config, data and the eval sample are fine; not training")
        pidfile.unlink(missing_ok=True)
        return 0

    # ---- model ------------------------------------------------------------
    seed_everything(config.seed)
    model = MedDecideModel(
        config.base_model,
        lora=config.lora,
        head_settings=config.head,
        dtype=config.dtype,
        device=config.device,
        variant=config.variant,
        revision=config.revision,
        max_prompt_tokens=config.max_prompt_tokens,
    )
    info = provenance(config, model)
    trainer = Trainer(model, config, log_path=log_path)
    print(f"[S9] model ready: {json.dumps(model.describe())[:400]}", flush=True)

    if args.padding_check:
        probe_model = model
        if args.padding_check_from is not None:
            probe_model = MedDecideModel.load(
                args.padding_check_from,
                device=config.device,
                dtype=args.padding_check_dtype,
            )
        elif args.padding_check_dtype is not None:
            probe_model = MedDecideModel(
                config.base_model,
                lora=config.lora,
                head_settings=config.head,
                dtype=args.padding_check_dtype,
                device=config.device,
                variant=config.variant,
                revision=config.revision,
                max_prompt_tokens=config.max_prompt_tokens,
            )
        probe = padding_check(
            probe_model,
            eval_items,
            max_prompt_tokens=config.max_prompt_tokens,
        )
        probe["model"] = probe_model.describe()
        probe["checkpoint"] = (
            str(args.padding_check_from) if args.padding_check_from is not None else None
        )
        probe["command"] = " ".join(sys.argv)
        probe_path = (
            Path(args.padding_check_out)
            if args.padding_check_out is not None
            else out / "padding_check.json"
        )
        write_json(probe_path, probe)
        print(
            f"[S9] padding check: {probe['verdict']} — alone vs left-padded batch "
            f"max|dlogit| = {probe['alone_vs_left_padded_batch']['max_abs_delta']:.3e}, "
            f"uniform-batch control = {probe['alone_vs_uniform_batch_control']['max_abs_delta']:.3e}"
            f" (tolerance {probe['tolerance']})",
            flush=True,
        )
        pidfile.unlink(missing_ok=True)
        return 0 if probe["verdict"] == "PASS" else 5
    write_json(
        out / "run_started.json",
        {
            "task": "S9",
            "kind": "training_run_started",
            "model_id": args.model_id,
            "command": " ".join(sys.argv),
            "started_at_utc": utcnow(),
            "pid": os.getpid(),
            "provenance": info,
            "config": config.to_dict(),
            "data": {
                "train_path": config.train_path,
                "dev_path": config.dev_path,
                "train_items": len(train_items),
                "train_limit": args.train_limit,
                "train_stride": args.train_stride,
                "dev_items": len(dev_items_all),
                "read_seconds": read_seconds,
            },
            "diag_note": args.diag_note,
            "eval_sample": eval_sample,
            "selection_rule": SELECTION_RULE,
            "log_path": str(log_path),
            "dev_evals_path": str(evals_path),
            "pidfile": str(pidfile),
        },
    )

    # ---- planned epoch: batch plan, length-trend check, LR-schedule total -----
    t_plan = time.perf_counter()
    # the plan must use the *same* item order the trainer will use (a per-epoch shuffle): with
    # raw file order the chunks would be source/template blocks, whose very different lengths
    # would leave a spurious batch-index trend that training never sees
    plan_order = shuffled_order(len(train_items), seed=config.seed, epoch=0)
    detail = plan_epoch_detail(
        model,
        train_items,
        order=plan_order,
        batch_size=config.batch_size,
        max_batch_tokens=config.max_batch_tokens,
        max_prompt_tokens=config.max_prompt_tokens,
        augment_options=config.shuffle_options,
        epoch=0,
        seed=config.seed,
        chunk_factor=config.batch_chunk_factor,
    )
    plan_seconds = time.perf_counter() - t_plan
    trend = length_trend(detail.batch_tokens, detail.batch_mean_chars)
    planned = {
        "task": "S9",
        "kind": "planned_epoch",
        "planned_at_utc": utcnow(),
        "seconds": plan_seconds,
        "epoch": 0,
        "chunk_factor": config.batch_chunk_factor,
        "chunk_size_items": max(
            config.batch_size, config.batch_chunk_factor * config.batch_size
        ),
        "plan": detail.plan.to_dict(),
        "length_trend": trend,
        "batch_tokens": detail.batch_tokens,
        "batch_mean_chars": detail.batch_mean_chars,
        "batch_items": detail.batch_items,
    }
    write_json(out / "planned_epoch.json", planned)
    print(f"[S9] planned epoch: {detail.plan.n_batches} batches, "
          f"{detail.plan.real_tokens} tokens, {plan_seconds:.1f}s to plan; "
          f"spearman(batch index, batch tokens) = {trend['rho_tokens']:.4f}, "
          f"first-20 vs last-20 of the first {trend['head_batches']} batches = "
          f"{trend['head_first20_mean_tokens']:.0f} -> {trend['head_last20_mean_tokens']:.0f} tokens",
          flush=True)
    if trend["rho_tokens"] is not None and abs(trend["rho_tokens"]) >= 0.1:
        raise RuntimeError(
            f"batch length still trends with batch index (rho={trend['rho_tokens']:.3f}); "
            f"refusing to train under an accidental length curriculum"
        )

    # ---- training with dev evals every `eval_every` steps ------------------
    selector = DevSelector(
        trainer,
        eval_items,
        eval_every=config.eval_every,
        evals_path=evals_path,
        best_dir=best_dir,
        best_json_path=out / "best.json",
        eval_sample=eval_sample,
        model_id=args.model_id,
        save_every_eval=args.save_every_eval,
        checkpoints_dir=out / "checkpoints",
    )
    if args.diag_note:
        print(f"[S9] diag note: {args.diag_note}", flush=True)

    pre_train_seconds = time.perf_counter() - process_started
    remaining = (
        None
        if args.max_seconds is None
        else max(1.0, float(args.max_seconds) - pre_train_seconds)
    )
    print(f"[S9] training starts: pre-train {pre_train_seconds:.1f}s, step budget "
          f"{'unbounded' if remaining is None else f'{remaining:.0f}s'}", flush=True)
    train_started = time.perf_counter()
    result = trainer.train(
        train_items,
        steps=args.steps,
        eval_items=None,  # selection is done in the hook, by Brier
        checkpoint_dir=None,
        max_seconds=remaining,
        on_step=selector,
        schedule_steps=detail.plan.n_batches,
    )
    train_seconds = time.perf_counter() - train_started
    rates = throughput_of(list(result.history))
    items_seen = sum(r.n_items for r in result.history)
    print(f"[S9] training done: steps={result.steps} epochs={result.epochs_completed} "
          f"stopped_early={result.stopped_early!r} wall={train_seconds:.0f}s "
          f"items={items_seen} items/s={rates.get('items_per_s')}", flush=True)

    # ---- final: best checkpoint, temperature fit on dev, per-template report --
    if selector.best_step is None:
        # no dev eval ever ran (budget shorter than one eval interval): evaluate once and save
        print("[S9] no dev eval ran during training; scoring the final weights once", flush=True)
        selector.evaluate(result.steps)
    if selector.best_step is None or selector.best_metrics is None:
        print("BLOCKED — training finished without a usable dev evaluation", file=sys.stderr)
        return 4

    # fit fresh from the *saved* checkpoint: the in-memory weights are the last step's, not the
    # selected step's (the whole point of a best checkpoint), and `model.json` must carry the
    # temperatures so `run_student.py` applies them without a second fit.
    best_model = MedDecideModel.load(best_dir, device=config.device)
    final = calibrate_and_report(
        best_model,
        final_items,
        config=config,
        out=out,
        best_dir=best_dir,
        best_step=selector.best_step,
        model_id=args.model_id,
        dev_items_available=len(dev_items_all),
        limited_sample=args.dev_final_per_qtype is not None,
    )
    temperature_payload = final["temperature"]
    applied = final["applied"]
    calibrated_report = final["calibrated"]
    uncalibrated_report = final["uncalibrated"]
    calibration_embedded = final["calibration_embedded"]

    macro_best = selector.brier_best()
    run = {
        "task": "S9-diag" if args.diag_note else "S9",
        "kind": "training_run",
        "model_id": args.model_id,
        "diag_note": args.diag_note,
        "command": " ".join(sys.argv),
        "generated_at_utc": utcnow(),
        "provenance": info,
        "config": config.to_dict(),
        "data": {
            "train_path": config.train_path,
            "dev_path": config.dev_path,
            "train_items": len(train_items),
            "train_limit": args.train_limit,
            "train_stride": args.train_stride,
            "dev_items": len(dev_items_all),
            "read_seconds": read_seconds,
            "plan_seconds": plan_seconds,
            "items_seen": items_seen,
            "fraction_of_train_seen": items_seen / len(train_items),
            "tokens_seen": rates.get("real_tokens"),
            "padded_tokens_seen": rates.get("padded_tokens"),
        },
        "training": {
            "steps": result.steps,
            "epochs_completed": result.epochs_completed,
            "stopped_early": result.stopped_early,
            "pass_completed": result.stopped_early is None,
            "train_seconds": train_seconds,
            "pre_train_seconds": pre_train_seconds,
            "budget_seconds": args.max_seconds,
            "train_budget_seconds": remaining,
            "schedule": result.schedule,
            "throughput": rates,
        },
        "planned_epoch": {
            "path": str(out / "planned_epoch.json"),
            "seconds": plan_seconds,
            "plan": detail.plan.to_dict(),
            "length_trend": trend,
        },
        "dev_evals": {
            "path": str(evals_path),
            "n_evals": len(selector.trajectory()),
            "n_errors": selector.n_errors,
            "eval_every": config.eval_every,
            "n_items_per_eval": len(eval_items),
            "sample_sha256": eval_sample["sha256"],
            "sample_by_template": eval_sample["by_template"],
            "trajectory": selector.trajectory(),
            "best_step_by_selection_rule": selector.best_step,
            "best_metrics_by_selection_rule": selector.best_metrics,
            "selection_rule": SELECTION_RULE,
            "additional_brier_first_rule": macro_best,
            "checkpoints_per_eval": (
                selector.eval_checkpoints if selector.save_every_eval else []
            ),
        },
        "checkpoint": {
            "path": str(best_dir),
            "step": selector.best_step,
            "saved_at_utc": selector.best_saved_at_utc,
            "save_error": selector.save_error,
            "calibration_embedded": calibration_embedded,
            "best_json": str(out / "best.json"),
        },
        "temperature": temperature_payload,
        "dev_final": {
            "path": str(out / "dev_final.json"),
            "n_items": len(final_items),
            "calibrated_overall": calibrated_report["overall"],
            "uncalibrated_overall": uncalibrated_report["overall"],
            "calibrated_per_qtype": calibrated_report["per_qtype"],
        },
        "wall_clock_s": time.perf_counter() - process_started,
        "exit": (
            "completed one pass"
            if result.stopped_early is None
            else f"stopped early: {result.stopped_early}"
        ),
    }
    write_json(out / "run.json", run)
    print(f"[S9] best step {selector.best_step}: "
          f"{json.dumps(selector.best_metrics)[:300]}", flush=True)
    print(f"[S9] temperature fit: {json.dumps(applied)}", flush=True)
    print(f"[S9] wrote {out / 'run.json'}; total wall {run['wall_clock_s']:.0f}s", flush=True)
    pidfile.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
