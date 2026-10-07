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
from meddecide.train.data import read_items, stratified_sample
from meddecide.train.trainer import Trainer, provenance

REPORT_STEPS = (1, 50, 100, 150, 200)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="S7 smoke run of the MedDecide training path")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--steps", type=int, default=200)
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


def extrapolate(items_per_s: float, tokens_per_s: float, mean_tokens: float,
                train_items: int) -> dict[str, Any]:
    """What one pass over the training set costs at the measured rates."""
    minutes_items = train_items / items_per_s / 60.0
    minutes_tokens = train_items * mean_tokens / tokens_per_s / 60.0
    return {
        "train_items": train_items,
        "mean_prompt_tokens_per_item": mean_tokens,
        "minutes_per_pass_from_items_per_s": minutes_items,
        "minutes_per_pass_from_tokens_per_s": minutes_tokens,
        "hours_per_pass_from_items_per_s": minutes_items / 60.0,
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
    result = trainer.train(
        train_items,
        steps=args.steps,
        eval_items=dev_items,
        checkpoint_dir=None if args.no_save else args.checkpoint,
    )
    rates = throughput(result)
    mean_tokens = rates["real_tokens"] / max(1, rates["n_items"])
    n_train = args.train_items or train_set_size(config.train_path)
    extrapolation = extrapolate(
        rates["items_per_s"], rates["tokens_per_s"], mean_tokens, n_train
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
        "throughput": rates,
        "extrapolation": extrapolation,
        "calibration": {
            "seconds": calibration_s,
            "items_per_qtype_cap": args.temperature_items_per_qtype,
            "fits": calibration,
        },
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
        "steps", "losses_at_steps", "loss_curve_decreased", "throughput", "extrapolation",
    )}, indent=2))
    print(f"[smoke] wrote {out}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
