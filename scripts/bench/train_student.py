#!/usr/bin/env python
"""S9: train MedDecide-0.8B (instruct base) — one pass over the S6 mix.

A thin CLI over :class:`meddecide.train.Trainer`. It adds exactly four things the library does
not do on its own, all of them about *this* run's bookkeeping:

1. the S9 recipe from ``configs/student_v0.yaml`` held fixed (LoRA r=16, LoRA lr 2e-4, batch 8,
   8k prompt cap, one epoch) with a wall-clock budget covering data loading and dev evals;
2. **dev evaluation every ``--eval-every`` steps** on a fixed, reproducible stratified sample of
   ``data/train/student_v0/dev.jsonl`` (the sample's item ids and a digest are written beside the
   run, so the selection can be re-derived);
3. **best-checkpoint selection by dev Brier** (tie-break: higher dev macro accuracy, then the
   earlier step). The library's built-in rule selects on macro accuracy with Brier as the
   tie-break; S9's brief fixes Brier as the primary key, so the selection lives here
   (:class:`DevSelector`) and both orderings are reported. A checkpoint is written every time the
   rule improves;
4. after training, the **per-qtype temperature fit on dev** (never on test), written to
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

from meddecide.eval.metrics import accuracy, brier_score, ece_from_confidence, macro_accuracy
from meddecide.model.meddecide_model import MedDecideModel, ScoredItems
from meddecide.train.config import DEFAULT_CONFIG_PATH, StudentConfig, load_config
from meddecide.train.data import read_items, stratified_sample
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
# selection rule, quoted verbatim into every artifact that reports a best step
SELECTION_RULE = (
    "lowest dev Brier on the fixed dev evaluation sample; ties (|dBrier| <= 1e-12) broken by "
    "higher dev macro accuracy, then by the earlier step"
)
# how many dev items the *periodic* evaluation scores, per qtype (the full dev split is used for
# the final fit and the per-template report)
DEFAULT_EVAL_PER_QTYPE = 256


# --------------------------------------------------------------------------- pure helpers
def selection_key(metrics: dict[str, Any]) -> tuple[float, float]:
    """Sort key whose minimum is the selected checkpoint: ``(Brier, -macro accuracy)``."""
    return (float(metrics["brier"]), -float(metrics["macro_accuracy"]))


def is_better(candidate: dict[str, Any], best: dict[str, Any] | None, *, tol: float = 1e-12) -> bool:
    """Whether ``candidate`` beats ``best`` under :data:`SELECTION_RULE` (strict improvement)."""
    if best is None:
        return True
    cand_brier, cand_macro = selection_key(candidate)
    best_brier, best_macro = selection_key(best)
    if cand_brier < best_brier - tol:
        return True
    return abs(cand_brier - best_brier) <= tol and cand_macro < best_macro


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


def throughput_of(history: Sequence[Any]) -> dict[str, Any]:
    """Items/s and tokens/s over the optimiser steps only (dev evals are not training time)."""
    if not history:
        return {"steps": 0}
    step_seconds = float(sum(r.elapsed_s for r in history))
    items = sum(r.n_items for r in history)
    real = sum(r.real_tokens for r in history)
    padded = sum(r.padded_tokens for r in history)
    elapsed = np.asarray([r.elapsed_s for r in history], dtype=np.float64)
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
        "gpu_peak_memory_gb": max((r.gpu_peak_gb for r in history), default=0.0),
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
    """The training hook: dev-evaluate every ``eval_every`` steps, keep the best by Brier.

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
        self.print = print_fn
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
    def macro_best(self) -> dict[str, Any] | None:
        """Additional analysis: the step the library's macro-accuracy rule would pick."""
        rows = [r for r in self.evals if r.get("event") == "dev_eval"]
        if not rows:
            return None
        best = max(rows, key=lambda r: (r["metrics"]["macro_accuracy"], -r["metrics"]["brier"]))
        return {
            "note": (
                "additional — the library's built-in rule selects on macro accuracy with Brier as "
                "the tie-break; reported, never used for S9's selection"
            ),
            "macro_accuracy_best_step": int(best["step"]),
            "macro_accuracy_best_value": float(best["metrics"]["macro_accuracy"]),
            "macro_accuracy_best_brier": float(best["metrics"]["brier"]),
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
                "per_qtype": self.eval_sample.get("per_qtype"),
                "item_ids_path": self.eval_sample.get("path"),
                "sample_sha256": self.eval_sample.get("sha256"),
                "eval_every": self.eval_every,
            },
            "n_evals_so_far": len([r for r in self.evals if r.get("event") == "dev_eval"]),
            "written_at_utc": utcnow(),
            "save_error": self.save_error,
            "additional_analysis": self.macro_best(),
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
    parser.add_argument("--eval-per-qtype", type=int, default=DEFAULT_EVAL_PER_QTYPE,
                        help="dev items per qtype in the periodic evaluation")
    parser.add_argument("--dev-final-per-qtype", type=int, default=None,
                        help="cap per qtype for the final dev fit/report (default: every dev item)")
    parser.add_argument("--temperature-items-per-qtype", type=int, default=None,
                        help="cap per qtype for the temperature fit (default: every dev item)")
    parser.add_argument("--seed", type=int, default=None, help="default: the config's seed")
    parser.add_argument("--device", default=None, help="default: the config's device")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="default: the config's batch_size (8)")
    parser.add_argument("--train-limit", type=int, default=None,
                        help="debug: read only the first N training items (recorded in run.json)")
    parser.add_argument("--log", type=Path, default=None,
                        help="per-step JSONL (default <out>/logs/train_steps.jsonl)")
    parser.add_argument("--dev-evals", type=Path, default=None,
                        help="per-eval JSONL (default <out>/logs/dev_evals.jsonl)")
    parser.add_argument("--best", type=Path, default=None,
                        help="best-checkpoint directory (default <out>/best)")
    parser.add_argument("--pidfile", type=Path, default=None,
                        help="default <out>/train.pid; removed on a clean exit")
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
        "eval_items": args.eval_per_qtype,
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
    return replace(config, **overrides)


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
    train_items = read_items(config.train_path, limit=args.train_limit)
    check_split(train_items, "train", path=config.train_path)
    dev_items_all = read_items(config.dev_path)
    check_split(dev_items_all, "dev", path=config.dev_path)
    read_seconds = time.perf_counter() - t0
    print(f"[S9] read {len(train_items)} train + {len(dev_items_all)} dev items "
          f"in {read_seconds:.1f}s", flush=True)

    eval_items = stratified_sample(
        dev_items_all, per_qtype=args.eval_per_qtype, seed=config.seed
    )
    eval_sample = {
        "path": str(out / "dev_eval_ids.json"),
        "n_items": len(eval_items),
        "per_qtype": args.eval_per_qtype,
        "seed": config.seed,
        "sha256": sample_digest(eval_items),
        "by_qtype": {
            q: sum(1 for i in eval_items if str(i.qtype) == q)
            for q in sorted({str(i.qtype) for i in eval_items})
        },
    }
    write_json(
        out / "dev_eval_ids.json",
        {**eval_sample, "item_ids": sorted(str(i.item_id) for i in eval_items)},
    )
    final_items = (
        dev_items_all
        if args.dev_final_per_qtype is None
        else stratified_sample(dev_items_all, per_qtype=args.dev_final_per_qtype, seed=config.seed)
    )
    print(f"[S9] periodic dev eval: {len(eval_items)} items ({json.dumps(eval_sample['by_qtype'])})"
          f" sha256={eval_sample['sha256'][:16]}", flush=True)
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
                "dev_items": len(dev_items_all),
                "read_seconds": read_seconds,
            },
            "eval_sample": eval_sample,
            "selection_rule": SELECTION_RULE,
            "log_path": str(log_path),
            "dev_evals_path": str(evals_path),
            "pidfile": str(pidfile),
        },
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
    )

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
    t0 = time.perf_counter()
    scored_final = best_model.score_items(
        final_items,
        batch_size=config.batch_size,
        max_batch_tokens=config.max_batch_tokens,
        max_prompt_tokens=config.max_prompt_tokens,
    )
    final_scored_seconds = time.perf_counter() - t0
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
        "checkpoint_step": selector.best_step,
        "model_id": args.model_id,
        "n_dev_items_scored": len(final_items),
        "dev_path": config.dev_path,
        "dev_split_sha256": sample_digest(final_items),
        "limited_sample": args.dev_final_per_qtype is not None,
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
        "checkpoint_step": selector.best_step,
        "dev_path": config.dev_path,
        "n_items": len(final_items),
        "n_items_available": len(dev_items_all),
        "sample_sha256": sample_digest(final_items),
        "scored_seconds": final_scored_seconds,
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
    calibration_embedded = reopened.get("calibration") or {}

    macro_best = selector.macro_best()
    run = {
        "task": "S9",
        "kind": "training_run",
        "model_id": args.model_id,
        "command": " ".join(sys.argv),
        "generated_at_utc": utcnow(),
        "provenance": info,
        "config": config.to_dict(),
        "data": {
            "train_path": config.train_path,
            "dev_path": config.dev_path,
            "train_items": len(train_items),
            "train_limit": args.train_limit,
            "dev_items": len(dev_items_all),
            "read_seconds": read_seconds,
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
            "throughput": rates,
        },
        "dev_evals": {
            "path": str(evals_path),
            "n_evals": len(selector.trajectory()),
            "n_errors": selector.n_errors,
            "eval_every": config.eval_every,
            "n_items_per_eval": len(eval_items),
            "sample_sha256": eval_sample["sha256"],
            "trajectory": selector.trajectory(),
            "best_step_by_selection_rule": selector.best_step,
            "best_metrics_by_selection_rule": selector.best_metrics,
            "selection_rule": SELECTION_RULE,
            "additional_macro_accuracy_best": macro_best,
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
