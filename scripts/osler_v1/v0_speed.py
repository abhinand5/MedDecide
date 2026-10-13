"""V0 step 4: time the batching configurations for osler_v1 (ADVISORY V0 step 4; D26).

Each configuration runs 150 optimiser steps of Qwen3.5-4B with the option-code head on the mix v2 train rows, with the
same seed, base, learning rates and data order as arm L; only the batch limits and gradient checkpointing change. One
process per configuration, so peak memory is not carried over from another run. A run that runs out of memory is recorded
and not chosen. ``--aggregate`` applies the pre-registered rule in ``meddecide.train.speed`` and writes speed.json.

Outputs (gitignored, aggregates only are copied to STATE): outputs/osler_v1/V0/speed/<name>.json and
outputs/osler_v1/V0/speed.json.

Usage (one GPU job at a time; the chain script runs them in order):
    source scripts/pod_env.sh && uv run --frozen python scripts/osler_v1/v0_speed.py --config a
    source scripts/pod_env.sh && uv run --frozen python scripts/osler_v1/v0_speed.py --aggregate
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

from meddecide.model.meddecide_model import MedDecideModel
from meddecide.train.config import load_config
from meddecide.train.mix_items import mix_row_to_item
from meddecide.train.speed import MEMORY_CAP_GB, choose_config, summarise
from meddecide.train.trainer import Trainer
from meddecide.utils.hashing import file_sha256
from meddecide.utils.io import read_json, write_json
from meddecide.utils.provenance import git_commit, utcnow

REPO = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO / "configs/osler_v0/arm_L.yaml"
OUT = REPO / "outputs" / "osler_v1" / "V0"
SPEED_DIR = OUT / "speed"
STEPS = 150

# the configurations of ADVISORY V0 step 4; "d" runs only if "b" runs out of memory (the chain script checks)
CONFIGS: dict[str, dict[str, Any]] = {
    "a": {"batch_size": 8, "max_batch_tokens": 8192, "gradient_checkpointing": True},  # the osler_v0 recipe
    "b": {"batch_size": 32, "max_batch_tokens": 32768, "gradient_checkpointing": False},
    "c": {"batch_size": 64, "max_batch_tokens": 65536, "gradient_checkpointing": False},
    "d": {"batch_size": 32, "max_batch_tokens": 32768, "gradient_checkpointing": True},
    # added after a-d were measured (STATE Deviations): a without gradient checkpointing. Checkpointing recomputes the
    # forward pass and changes no number, so this is a pure speed-memory setting; the same rule chooses among all five.
    "e": {"batch_size": 8, "max_batch_tokens": 8192, "gradient_checkpointing": False},
}


def run_config(name: str) -> dict[str, Any]:
    base = load_config(CONFIG_PATH)
    cfg = dataclasses.replace(base, **CONFIGS[name], output_dir=str(SPEED_DIR / name))
    train_path = REPO / cfg.train_path
    manifest = REPO / Path(cfg.train_path).parent / "manifest.json"
    provenance = {
        "task": "V0",
        "name": name,
        "started_utc": utcnow(),
        "git_commit": git_commit(REPO),
        "command": f"scripts/osler_v1/v0_speed.py --config {name}",
        "config_file": str(CONFIG_PATH.relative_to(REPO)),
        "overrides": CONFIGS[name],
        "steps_target": STEPS,
        "seed": cfg.seed,
        "base_model": cfg.base_model,
        "revision": cfg.revision,
        "readout": cfg.readout,
        "lr": cfg.lr,
        "lora_lr": cfg.lora_lr,
        "train_path": cfg.train_path,
        "train_bytes": train_path.stat().st_size,
        "mix_manifest_sha256": file_sha256(manifest) if manifest.exists() else None,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "memory_cap_gb": MEMORY_CAP_GB,
    }
    print(f"config {name}: {CONFIGS[name]}", flush=True)
    t_load = time.perf_counter()
    items = []
    with train_path.open(encoding="utf-8") as fh:
        for line in fh:
            items.append(mix_row_to_item(json.loads(line)))
    print(f"config {name}: read {len(items)} train items in {time.perf_counter() - t_load:.1f} s", flush=True)
    provenance["train_items"] = len(items)

    model = MedDecideModel(
        cfg.base_model,
        revision=cfg.revision,
        lora=cfg.lora,
        head_settings=cfg.head,
        dtype=cfg.dtype,
        device=cfg.device,
        variant=cfg.variant,
        readout=cfg.readout,
        bidirectional_full_attention=cfg.bidirectional_full_attention,
        max_prompt_tokens=cfg.max_prompt_tokens,
    )
    if cfg.gradient_checkpointing:
        model.enable_gradient_checkpointing()
    trainer = Trainer(model, cfg)
    records: list[dict[str, Any]] = []
    origin = time.perf_counter()

    def on_step(step: int) -> None:
        rec = trainer.history[-1]
        records.append({
            "step": step,
            "n_items": rec.n_items,
            "real_tokens": rec.real_tokens,
            "padded_tokens": rec.padded_tokens,
            "loss": rec.loss,
            "grad_norm": rec.grad_norm,
            "elapsed_s": rec.elapsed_s,
            "t_end_s": time.perf_counter() - origin,
            # the trainer resets the peak at the start of each step, so this is that step's peak reserved memory
            "peak_reserved_gb": torch.cuda.max_memory_reserved() / 1e9,
        })
        if step % 25 == 0:
            print(f"config {name}: step {step} loss {rec.loss:.4f} items/step {rec.n_items} "
                  f"peak reserved {records[-1]['peak_reserved_gb']:.1f} GB", flush=True)

    result: dict[str, Any] = {"provenance": provenance}
    try:
        outcome = trainer.train(items, steps=STEPS, on_step=on_step)
        result["stopped_early"] = outcome.stopped_early
        result["steps_run"] = outcome.steps
        if outcome.steps == STEPS:
            result["summary"] = summarise(records)
        else:
            result["summary"] = {"n_steps": outcome.steps, "peak_reserved_gb": max(
                (r["peak_reserved_gb"] for r in records), default=None)}
    except torch.cuda.OutOfMemoryError as exc:
        result["oom"] = True
        result["oom_at_step"] = len(records) + 1
        result["oom_message"] = str(exc).splitlines()[0][:200]
        result["summary"] = {"n_steps": len(records), "peak_reserved_gb": max(
            (r["peak_reserved_gb"] for r in records), default=None)}
        print(f"config {name}: out of memory at step {len(records) + 1}", flush=True)
    result["provenance"]["finished_utc"] = utcnow()
    result["provenance"]["cuda_max_memory_reserved_gb"] = torch.cuda.max_memory_reserved() / 1e9
    # every step record is stored, so the summary can be re-derived from the file (R6); the file is small
    result["records"] = records
    result["records_n"] = len(records)
    SPEED_DIR.mkdir(parents=True, exist_ok=True)
    write_json(SPEED_DIR / f"{name}.json", result)
    return result


def aggregate() -> dict[str, Any]:
    summaries: dict[str, dict[str, Any]] = {}
    for name in sorted(CONFIGS):
        path = SPEED_DIR / f"{name}.json"
        if not path.exists():
            continue
        res = read_json(path)
        summary = dict(res["summary"])
        if res.get("oom"):
            summary = {"oom": True, "oom_at_step": res["oom_at_step"], "n_steps": summary.get("n_steps"),
                       "peak_reserved_gb": summary.get("peak_reserved_gb")}
        summaries[name] = summary
    chosen, reason = choose_config(summaries, target_steps=STEPS)
    payload = {
        "task": "V0",
        "written_utc": utcnow(),
        "git_commit": git_commit(REPO),
        "rule": ("highest steady-state wall-clock items/s (steps after the 10-step warm-up) among runs that finish "
                 f"{STEPS} steps with peak reserved GPU memory at most {MEMORY_CAP_GB:g} GB; a run that runs out of "
                 "memory is not eligible"),
        "configurations": CONFIGS,
        "summaries": summaries,
        "chosen": chosen,
        "reason": reason,
        "learning_rates_unchanged": True,
    }
    write_json(OUT / "speed.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--config", choices=sorted(CONFIGS))
    group.add_argument("--aggregate", action="store_true")
    args = parser.parse_args()
    if args.aggregate:
        payload = aggregate()
        print(f"chosen: {payload['chosen']}; {payload['reason']}", flush=True)
        return
    result = run_config(args.config)
    s = result["summary"]
    if result.get("oom"):
        print(f"config {args.config}: OOM at step {result['oom_at_step']}", flush=True)
        return
    print(f"config {args.config}: steps {s['n_steps']}, items/s {s['items_per_s']:.3f}, "
          f"tokens/s {s['real_tokens_per_s']:.1f}, peak reserved {s['peak_reserved_gb']:.1f} GB, "
          f"loss(last 50) {s['loss_last_window_mean']:.4f}", flush=True)
    sys.exit(0)


if __name__ == "__main__":
    main()
