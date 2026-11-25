"""200-step smoke run of the training path (task S7).

Runs the real training loop — same encoding, same loss, same optimiser, same batching — on a
small deterministic slice of ``data/train/student_v0/train.jsonl``, records the loss at steps
1/50/100/150/200, throughput, GPU peak memory and the config, and writes everything to
``outputs/student_v0/S7/smoke.json``.

Usage::

    uv run python -m meddecide.train.smoke --steps 200 --slice 2048 --stride 100 \\
        --out outputs/student_v0/S7/smoke.json

The slice is drawn with a fixed stride over the file, so it spreads across the training mix
instead of taking one source's block, and the same command reproduces the same items.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from meddecide.model.meddecide_model import MedDecideModel
from meddecide.train.config import DEFAULT_CONFIG_PATH, StudentConfig, load_config
from meddecide.train.data import iter_batches, read_items, stratified_sample
from meddecide.train.trainer import Trainer, provenance

REPORT_STEPS = (1, 50, 100, 150, 200)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="S7 smoke run of the MedDecide training path")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--reference-items", type=int, default=64,
                        help="fixed set the learning curve is measured on")
    parser.add_argument("--slice", type=int, default=2048, help="training items to sample")
    parser.add_argument("--stride", type=int, default=100, help="sample every Nth JSONL line")
    parser.add_argument("--dev-per-qtype", type=int, default=256,
                        help="dev items per qtype (stratified; the whole dev file is read)")
    parser.add_argument("--temperature-items-per-qtype", type=int, default=128)
    parser.add_argument("--out", default="outputs/student_v0/S7/smoke.json")
    parser.add_argument("--checkpoint", default="outputs/student_v0/S7/checkpoint")
    parser.add_argument("--log", default="outputs/student_v0/S7/logs/smoke_train.jsonl")
    parser.add_argument("--train-items", type=int, default=None,
                        help="training-set size for the extrapolation (default: S6 manifest)")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--no-pass", action="store_true",
                        help="skip the whole-slice throughput pass")
    return parser.parse_args(argv)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def slice_summary(items: list[Any]) -> dict[str, Any]:
    return {
        "n_items": len(items),
        "qtype": dict(sorted(Counter(str(i.qtype) for i in items).items())),
        "source": dict(sorted(Counter(i.source for i in items).items())),
        "template_id": dict(sorted(Counter(i.template_id for i in items).items())),
        "n_options": dict(sorted(Counter(i.n_options for i in items).items())),
        "mean_state_chars": float(np.mean([len(i.state) for i in items])),
    }


def reference_items(items: list[Any], *, n: int = 64) -> list[Any]:
    """``n`` items spread evenly across the slice's length order.

    The batch plan is length-sorted, so a handful of the shortest items would not represent it;
    spreading the reference set over the whole length range does.
    """
    order = sorted(
        range(len(items)), key=lambda i: (len(items[i].state) + len(items[i].question))
    )
    picks = [order[round(k * (len(order) - 1) / max(1, n - 1))] for k in range(n)]
    return [items[i] for i in sorted(set(picks))]


def slice_stats(
    model: Any,
    items: list[Any],
    *,
    batch_size: int,
    max_batch_tokens: int,
    max_prompt_tokens: int,
) -> dict[str, Any]:
    """Prompt-token statistics of the **whole** slice and how many batches it makes.

    Encodes every item once (no model forward). Without this, a throughput number measured on
    the items a 200-step run happened to reach would describe only the short end of a
    length-sorted plan — which is exactly the mistake the first smoke run made.
    """
    counts: list[int] = []
    batches = 0
    for _, batch in iter_batches(
        model,
        items,
        batch_size=batch_size,
        max_batch_tokens=max_batch_tokens,
        max_prompt_tokens=max_prompt_tokens,
    ):
        batches += 1
        counts.extend(entry.n_tokens for entry in batch)
    arr = np.asarray(counts, dtype=np.int64)
    return {
        "n_items": int(arr.size),
        "n_batches": int(batches),
        "total_tokens": int(arr.sum()),
        "mean_tokens": float(arr.mean()),
        "p50_tokens": float(np.percentile(arr, 50)),
        "p95_tokens": float(np.percentile(arr, 95)),
        "max_tokens": int(arr.max()),
        "share_capped": float((arr >= max_prompt_tokens).mean()),
    }


def throughput(result: Any) -> dict[str, Any]:
    records = result.history
    elapsed = [r.elapsed_s for r in records]
    real = sum(r.real_tokens for r in records)
    padded = sum(r.padded_tokens for r in records)
    items = sum(r.n_items for r in records)
    step_time = float(sum(elapsed))
    return {
        "optimiser_step_seconds_total": step_time,
        "optimiser_step_seconds_mean": float(np.mean(elapsed)),
        "optimiser_step_seconds_p50": float(np.percentile(elapsed, 50)),
        "optimiser_step_seconds_p95": float(np.percentile(elapsed, 95)),
        "n_items": items,
        "real_tokens": real,
        "padded_tokens": padded,
        "items_per_s": items / step_time,
        "tokens_per_s": real / step_time,
        "padded_tokens_per_s": padded / step_time,
        "wall_clock_s": result.wall_clock_s,
        "wall_clock_items_per_s": items / result.wall_clock_s,
        "wall_clock_tokens_per_s": real / result.wall_clock_s,
        "gpu_peak_memory_gb": max((r.gpu_peak_gb for r in records), default=0.0),
    }


def extrapolate(
    result: Any,
    *,
    train_items: int,
    mean_tokens_full_slice: float,
    n_batches_slice: int,
    slice_items: int,
    tokens_per_s: float,
    items_per_s: float,
) -> dict[str, Any]:
    """What one pass over the training set costs, three ways.

    The raw items/s is measured on the items actually processed, which is the **short** end of
    the length-sorted batch plan (1600 of 2048 items) — so it is an optimistic number and is
    labelled as such. The other two estimates use the whole slice's token mean. The fitted
    model regresses measured step time on real tokens per step (the run varied 847 -> 4363
    tokens/step, which is exactly the range a fit needs) so that per-step overhead is counted.
    """
    step_tokens = np.asarray([r.real_tokens for r in result.history], dtype=np.float64)
    step_seconds = np.asarray([r.elapsed_s for r in result.history], dtype=np.float64)
    slope, intercept = np.polyfit(step_tokens, step_seconds, 1)
    batches_per_pass = n_batches_slice * train_items / max(1, slice_items)
    total_pass_tokens = train_items * mean_tokens_full_slice
    fitted_seconds = intercept * batches_per_pass + slope * total_pass_tokens
    naive_seconds = train_items / items_per_s
    token_seconds = total_pass_tokens / tokens_per_s
    return {
        "train_items": train_items,
        "mean_prompt_tokens_per_item_processed": float(step_tokens.sum() / max(1, sum(
            r.n_items for r in result.history))),
        "mean_prompt_tokens_per_item_full_slice": mean_tokens_full_slice,
        "total_pass_tokens": total_pass_tokens,
        "batches_per_pass_estimate": batches_per_pass,
        "step_time_model": {
            "seconds_per_step_intercept": float(intercept),
            "seconds_per_token_slope": float(slope),
            "r_squared": float(np.corrcoef(step_tokens, step_seconds)[0, 1] ** 2),
        },
        "minutes_per_pass_from_items_per_s_optimistic": naive_seconds / 60.0,
        "minutes_per_pass_from_tokens_per_s_full_slice": token_seconds / 60.0,
        "minutes_per_pass_from_fitted_step_model": fitted_seconds / 60.0,
        "hours_per_pass_from_fitted_step_model": fitted_seconds / 3600.0,
    }


def measure_full_pass(trainer: Any, train_items: list[Any], *, n_batches: int) -> dict[str, Any]:
    """Time one training pass over the **whole** slice (all batches, so the long tail counts).

    The 200-step run only reaches the short end of a length-sorted plan, so it cannot say what a
    pass costs; this runs the real training step over every batch the slice makes and times it.
    It happens after the checkpoint and the calibration are written, so the model it leaves
    behind is not the artifact.
    """
    import time as _time

    before = len(trainer.history)
    started = _time.perf_counter()
    trainer.train(train_items, steps=n_batches)
    wall = _time.perf_counter() - started
    records = trainer.history[before:]
    items = sum(r.n_items for r in records)
    tokens = sum(r.real_tokens for r in records)
    step_seconds = float(sum(r.elapsed_s for r in records))
    return {
        "n_steps": len(records),
        "n_items": items,
        "real_tokens": tokens,
        "padded_tokens": sum(r.padded_tokens for r in records),
        "step_seconds_total": step_seconds,
        "wall_clock_s": wall,
        "items_per_s": items / wall,
        "tokens_per_s": tokens / wall,
        "step_seconds_mean": step_seconds / max(1, len(records)),
        "gpu_peak_memory_gb": max((r.gpu_peak_gb for r in records), default=0.0),
    }


def train_set_size(train_path: str, *, fallback: int = 212_481) -> int:
    """Item count of the full training mix, read from the S6 manifest beside the file."""
    manifest = Path(train_path).with_name("manifest.json")
    if manifest.exists():
        try:
            totals = json.loads(manifest.read_text(encoding="utf-8")).get("totals", {})
            if "train" in totals:
                return int(totals["train"])
        except (json.JSONDecodeError, TypeError, ValueError):  # pragma: no cover - bad manifest
            pass
    return fallback


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config: StudentConfig = load_config(args.config)
    seed_everything(config.seed)

    t0 = time.perf_counter()
    train_items = read_items(config.train_path, limit=args.slice, stride=args.stride)
    # the whole dev file, then a per-qtype stratified sample: `score` items are 33 of 16,454, so
    # a plain head-of-file or stride slice can contain none of them and the per-qtype fit for
    # `score` would silently be the identity (R5: a denominator of zero is not a fit)
    dev_all = read_items(config.dev_path)
    dev_items = stratified_sample(dev_all, per_qtype=args.dev_per_qtype, seed=config.seed)
    read_s = time.perf_counter() - t0
    print(f"[smoke] read {len(train_items)} train + {len(dev_items)} dev items in {read_s:.1f}s")

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

    trainer = Trainer(model, config, log_path=args.log)
    ref_items = reference_items(train_items, n=args.reference_items)
    fixed_losses: dict[str, Any] = {}

    def hook(step: int) -> None:
        # the raw per-step loss is not comparable across steps (length-sorted batches), so the
        # learning curve is measured on a fixed reference set as well
        if step in REPORT_STEPS:
            fixed_losses[str(step)] = trainer.evaluate_loss(ref_items)

    result = trainer.train(
        train_items,
        steps=args.steps,
        eval_items=dev_items,
        checkpoint_dir=None if args.no_save else args.checkpoint,
        on_step=hook,
    )
    rates = throughput(result)
    n_train = args.train_items or train_set_size(config.train_path)
    stats = slice_stats(
        model,
        train_items,
        batch_size=config.batch_size,
        max_batch_tokens=config.max_batch_tokens,
        max_prompt_tokens=config.max_prompt_tokens,
    )
    extrapolation = extrapolate(
        result,
        train_items=n_train,
        mean_tokens_full_slice=stats["mean_tokens"],
        n_batches_slice=int(stats["n_batches"]),
        slice_items=stats["n_items"],
        tokens_per_s=rates["tokens_per_s"],
        items_per_s=rates["items_per_s"],
    )

    calibration: dict[str, Any] = {}
    calibration_started = time.perf_counter()
    if config.fit_temperature:
        fits = trainer.fit_calibration(
            dev_items,
            limit_per_qtype=args.temperature_items_per_qtype,
            per_qtype=config.temperature_per_qtype,
        )
        calibration = {k: v.to_dict() for k, v in fits.items()}
    calibration_s = time.perf_counter() - calibration_started

    checkpoint = None
    if not args.no_save:
        checkpoint = str(trainer.save_checkpoint(args.checkpoint))

    full_pass = None
    if not args.no_pass:
        full_pass = measure_full_pass(
            trainer, train_items, n_batches=int(stats["n_batches"])
        )
        full_pass["minutes_per_pass_over_the_full_training_set"] = (
            full_pass["wall_clock_s"] * n_train / max(1, stats["n_items"]) / 60.0
        )

    losses = {str(step): result.loss_at(step) for step in REPORT_STEPS if step <= args.steps}
    report = {
        "task": "S7",
        "kind": "smoke_run",
        "command": " ".join(sys.argv),
        "provenance": info,
        "config": config.to_dict(),
        "slice": {
            "train": slice_summary(train_items),
            "dev": slice_summary(dev_items),
            "dev_available": slice_summary(dev_all),
            "stride": args.stride,
            "dev_per_qtype": args.dev_per_qtype,
            "read_seconds": read_s,
        },
        "steps": args.steps,
        "losses_at_steps": losses,
        "loss_curve_decreased": (
            None
            if len([v for v in losses.values() if v is not None]) < 2
            else losses[str(max(int(k) for k in losses))] < losses[str(min(int(k) for k in losses))]
        ),
        # the per-step loss is measured on whichever items the step happened to hold, and the
        # batch plan is length-sorted, so it also moves with item difficulty. The same five
        # steps measured on a fixed reference set is the learning curve proper.
        "losses_at_steps_fixed_reference_set": fixed_losses,
        "loss_curve_decreased_fixed_reference_set": (
            None
            if len(fixed_losses) < 2
            else fixed_losses[str(max(int(k) for k in fixed_losses))]["loss"]
            < fixed_losses[str(min(int(k) for k in fixed_losses))]["loss"]
        ),
        "fixed_reference_set": {
            "n_items": len(ref_items),
            "mean_state_chars": float(np.mean([len(i.state) for i in ref_items])),
            "description": "64 items spread evenly over the slice's length order",
        },
        "slice_tokens": stats,
        "throughput": rates,
        "extrapolation": extrapolation,
        "calibration": {
            "seconds": calibration_s,
            "items_per_qtype_cap": args.temperature_items_per_qtype,
            "fits": calibration,
        },
        "full_slice_pass": full_pass,
        "checkpoint": checkpoint,
        "wall_clock_s": result.wall_clock_s + calibration_s,
        "stopped_early": result.stopped_early,
        "best_dev": result.best_dev,
        "best_step": result.best_step,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in (
        "steps", "losses_at_steps", "loss_curve_decreased",
        "losses_at_steps_fixed_reference_set", "loss_curve_decreased_fixed_reference_set",
        "slice_tokens", "throughput", "extrapolation", "calibration", "full_slice_pass",
    )}, indent=2))
    print(f"[smoke] wrote {out}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
