"""osler_v2 W6: one block run (ADVISORY W6; attention-only LoRA D33; D40 speed recipe; D39 base).

A block is a training file (C0, C1 or C2) on one backbone (qwen = Qwen3.5-4B, pplx = pplx-decider-flash-v1.2). The recipe is
configs/osler_v2/block_arm.yaml. The trainer runs for the step count of the block's replay file (13,992 steps, written by
w6_replay.py for that block and backbone; the block must match the replay's file hash), evaluates at every quarter of the
steps on the whole fast dev v2, and keeps the best checkpoint on the pooled template macro (the trainer's selection).

Writes outputs/osler_v2/W6/<block>_<backbone>/ (provenance.json, arm_result.json, logs/, dev_predictions.jsonl,
checkpoints) and block_result.json (aggregates and the validation of every logged step against the replay).

Usage (launch detached; check the GPU is free first):
    source scripts/pod_env.sh && setsid nohup uv run --frozen python scripts/osler_v2/w6_block_run.py --block C0 --backbone qwen \
        > outputs/osler_v2/W6/logs/C0_qwen.log 2>&1 < /dev/null &
    smoke (8 steps, 200 dev items, every step logged):  ... --smoke 8
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

from meddecide.model.pplx import ATTENTION_TARGETS
from meddecide.train.arm_run import run_arm
from meddecide.train.blocks import EVAL_QUARTERS
from meddecide.train.config import load_config
from meddecide.train.mix_items import mix_row_to_item
from meddecide.train.osler_arm import stratified_subset
from meddecide.utils.hashing import file_sha256
from meddecide.utils.io import read_json, write_json
from meddecide.utils.provenance import git_commit, utcnow

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "configs" / "osler_v2" / "block_arm.yaml"
TRAIN_DIR = REPO / "data" / "train" / "osler_v2"
DEV = REPO / "data" / "bench" / "osler_v2" / "fastdev_v2.jsonl"
OUT = REPO / "outputs" / "osler_v2" / "W6"
REPLAY_DIR = OUT / "replay"
SMOKE_DEV_ITEMS = 200


def _v8():
    spec = importlib.util.spec_from_file_location("v8_block_run", REPO / "scripts" / "osler_v1" / "v8_block_run.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_V8 = _v8()
validate_logged_steps = _V8.validate_logged_steps


def read_mix(path: Path) -> list[Any]:
    with path.open(encoding="utf-8") as fh:
        return [mix_row_to_item(json.loads(line)) for line in fh if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--block", required=True, choices=["C0", "C1", "C2"])
    parser.add_argument("--backbone", required=True, choices=["qwen", "pplx"])
    parser.add_argument("--smoke", type=int, default=0, help="run this many steps on a dev subset (a pipeline check)")
    args = parser.parse_args()
    started = time.time()
    name = f"{args.block}_{args.backbone}"
    replay = read_json(REPLAY_DIR / f"{args.block}_{args.backbone}.json")
    replay_steps = int(replay["steps"])
    if args.smoke:
        if args.smoke % EVAL_QUARTERS or args.smoke > replay_steps:
            raise SystemExit("--smoke must be a multiple of four no larger than the replay's step count")
        steps, out_dir = args.smoke, OUT / "smoke" / name
    else:
        steps, out_dir = replay_steps, OUT / name
    base = load_config(CONFIG)
    train_path = TRAIN_DIR / f"block_{args.block}.jsonl"
    train_sha = file_sha256(train_path)
    if replay.get("train_file_sha256") != train_sha:
        raise SystemExit(f"{name}: the replay was made from another file (replay the block again)")
    config = dataclasses.replace(
        base,
        train_path=str(train_path.relative_to(REPO)),
        dev_path=str(DEV.relative_to(REPO)),
        output_dir=str(out_dir.relative_to(REPO)),
        eval_every=steps // EVAL_QUARTERS,
        backbone=args.backbone,
        lora=dataclasses.replace(base.lora, target_modules=ATTENTION_TARGETS),
        log_every=1 if args.smoke else base.log_every,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{name}: loading {train_path.name}", flush=True)
    train_items = read_mix(train_path)
    dev_items = read_mix(DEV)
    if args.smoke:  # a template-stratified subset, so the temperature fits see every question type
        dev_items = stratified_subset(dev_items, key="template_id", size=SMOKE_DEV_ITEMS, seed=0)
    print(f"{name}: {len(train_items)} train items, {len(dev_items)} dev items, {steps} steps, "
          f"eval every {config.eval_every}", flush=True)
    write_json(out_dir / "provenance.json", {
        "built_at_utc": utcnow(), "git_commit": git_commit(REPO), "block": args.block, "backbone": args.backbone,
        "command": f"scripts/osler_v2/w6_block_run.py --block {args.block} --backbone {args.backbone}"
                   + (f" --smoke {args.smoke}" if args.smoke else ""),
        "config_file": str(CONFIG.relative_to(REPO)), "train_file": str(train_path.relative_to(REPO)),
        "train_sha256": train_sha, "dev_file": str(DEV.relative_to(REPO)), "dev_sha256": file_sha256(DEV),
        "dev_items_used": len(dev_items), "train_items": len(train_items), "steps": steps,
        "replay_steps": replay_steps, "base_model": config.base_model if args.backbone == "qwen" else "pplx (base_model ignored)",
        "revision": config.revision, "lora_target_modules": list(config.lora.target_modules),
        "gradient_checkpointing": config.gradient_checkpointing, "checkpoint_above_tokens": config.checkpoint_above_tokens,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
        "torch": torch.__version__, "python": sys.version.split()[0],
    })
    summary = run_arm(config, train_items=train_items, dev_items=dev_items, out_dir=out_dir, steps=steps,
                      eval_subset_size=len(dev_items))
    expected_evals = [steps * q // EVAL_QUARTERS for q in range(1, EVAL_QUARTERS + 1)]
    eval_steps = [e["step"] for e in summary["evaluations"]]
    validation = {
        "steps_taken": summary["steps"], "steps_as_planned": summary["steps"] == steps,
        "stopped_early": summary["stopped_early"],
        "evaluation_steps": eval_steps, "evaluation_steps_as_planned": eval_steps == expected_evals,
        "eval_subset_items": summary["eval_subset_items"], "dev_items": len(dev_items),
        "eval_subset_is_full_dev": summary["eval_subset_items"] == len(dev_items),
        "logged_steps": validate_logged_steps(out_dir / "logs" / "train.jsonl", replay, steps),
    }
    realised = replay["sizes"][:steps]
    block_result = {
        "block": args.block, "backbone": args.backbone, "smoke_steps": args.smoke or None, "steps": steps,
        "train_items": len(train_items), "realised_items": sum(realised),
        "real_tokens": sum(replay["real_tokens"][:steps]), "padded_tokens": sum(replay["padded_tokens"][:steps]),
        "selected": summary["selected"], "best_step": summary["best_step"], "evaluations": summary["evaluations"],
        "full_dev_metrics": summary["full_dev_metrics"], "temperature_fits_keys": sorted(summary["temperature_fits"]),
        "validation": validation, "train_wall_clock_s": summary["train_wall_clock_s"],
        "total_wall_clock_s": round(time.time() - started, 1), "config": config.to_dict(),
    }
    write_json(out_dir / "block_result.json", block_result)
    print(f"{name}: steps {summary['steps']}, realised items {sum(realised)}, selected {summary['selected']}, "
          f"validation {validation['logged_steps']['mismatches']} log mismatches, "
          f"{'all evaluations on the planned steps' if validation['evaluation_steps_as_planned'] else 'EVALUATION STEPS OFF PLAN'}",
          flush=True)


if __name__ == "__main__":
    main()
