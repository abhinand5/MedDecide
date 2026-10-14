"""V8: one matched short block run (ADVISORY V8; one GPU job at a time).

Qwen3.5-4B with the recipe of configs/osler_v1/block_arm.yaml (the osler_v0 arm L constants, with V0's batching), trained on
the block's file for the step count of the replay (the same for every block, from v8_replay_steps.py). The fast dev set is
the evaluation set: the trainer evaluates at 25, 50, 75 and 100 % of the steps, and the selected checkpoint is the best on
the pooled template macro accuracy (the trainer's selection; ties go to the lower Brier). The arm runs through
meddecide.train.arm_run.run_arm, the same code as the osler_v0 arms.

B0b is B0 with the LoRA restricted to the attention projections (ADVISORY V8, the alternative run after a failed B0
knowledge check). It reads B0's file and B0's replay.

Writes outputs/osler_v1/V8/<block>/ (provenance.json, arm_result.json, logs/, dev_predictions.jsonl, checkpoints/best) and
outputs/osler_v1/V8/<block>/block_result.json (the aggregates and the validation of the logged steps against the replay).

Usage (launch detached; check the GPU is free first):
    source scripts/pod_env.sh && setsid nohup uv run --frozen python scripts/osler_v1/v8_block_run.py --block B0 \
        > outputs/osler_v1/V8/logs/B0.log 2>&1 < /dev/null &
    smoke test (8 steps, 200 dev items, every step logged):  ... --block B0 --smoke 8
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

from meddecide.bench.schema import Item
from meddecide.train.arm_run import run_arm
from meddecide.train.blocks import EVAL_QUARTERS
from meddecide.train.config import load_config
from meddecide.train.mix_items import mix_row_to_item
from meddecide.utils.hashing import file_sha256
from meddecide.utils.io import read_json, write_json
from meddecide.utils.provenance import git_commit, utcnow

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "configs" / "osler_v1" / "block_arm.yaml"
DEV = REPO / "data" / "bench" / "osler_mr_v0" / "fastdev_mix.jsonl"
OUT = REPO / "outputs" / "osler_v1" / "V8"
V7_MANIFEST = REPO / "outputs" / "osler_v1" / "V7" / "blocks.json"
BLOCK_FILE ={"B0": "B0", "B0b": "B0", "B1": "B1", "B2": "B2", "B3": "B3"}  # the training file of each block
REPLAY_OF = {"B0b": "B0"}  # B0b replays B0's batches: the same file, the same batcher
ATTENTION_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj")
SMOKE_DEV_ITEMS = 200


def read_mix(path: Path) -> list[Item]:
    with path.open(encoding="utf-8") as fh:
        return [mix_row_to_item(json.loads(line)) for line in fh if line.strip()]


def check_block_manifest(block: str, train_path: Path) -> dict[str, Any]:
    """The block's V7 record must exist, its leakage check must be clean, the file must be the one recorded, and B2 and B3
    may train only on final teacher prose. A failed check stops the run before the GPU is used."""
    info = read_json(V7_MANIFEST).get("blocks", {}).get(block)
    if info is None:
        raise SystemExit(f"{block}: no V7 record; build the block with v7_blocks.py first")
    if not info.get("leakage_clean"):
        raise SystemExit(f"{block}: the V7 leakage check is not clean ({info.get('leakage_hits')}, "
                         f"{info.get('qualified_record_hits')})")
    if file_sha256(train_path) != info.get("sha256_file"):
        raise SystemExit(f"{block}: the block file does not match its V7 record; rebuild the block")
    if block in ("B2", "B3") and info.get("teacher_final_flag") is not True:
        raise SystemExit(f"{block}: the teacher prose is not final for this block (build it with --teacher-final)")
    return info


def validate_logged_steps(train_log: Path, replay: dict[str, Any], steps: int) -> dict[str, Any]:
    """The logged step records against the replay: every logged batch size and token count must agree exactly."""
    logged: dict[int, dict[str, Any]] = {}
    for line in train_log.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("event") == "step":
            logged[int(row["step"])] = row
    checked = [s for s in sorted(logged) if s <= steps]
    mismatches = []
    for s in checked:
        expect = (replay["sizes"][s - 1], replay["real_tokens"][s - 1], replay["padded_tokens"][s - 1])
        got = tuple(logged[s].get(key) for key in ("n_items", "real_tokens", "padded_tokens"))
        if expect != got:  # a missing field (None) is a mismatch, reported rather than raised
            mismatches.append({"step": s, "replay": expect, "logged": got})
    return {"logged_steps": len(logged), "logged_steps_checked": len(checked), "mismatches": len(mismatches),
            "first_mismatches": mismatches[:5]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--block", required=True, choices=sorted(BLOCK_FILE))
    parser.add_argument("--smoke", type=int, default=0, help="run this many steps on a dev subset (a pipeline check)")
    args = parser.parse_args()
    started = time.time()
    name = args.block
    replay_name = REPLAY_OF.get(name, name)
    replay = read_json(OUT / "replay" / f"{replay_name}.json")
    replay_steps = int(replay["steps"])
    if args.smoke:
        if args.smoke % EVAL_QUARTERS or args.smoke > replay_steps:
            raise SystemExit("--smoke must be a multiple of four no larger than the replay's step count")
        steps = args.smoke
        out_dir = OUT / "smoke" / name
    else:
        steps = replay_steps
        out_dir = OUT / name
    base = load_config(CONFIG)
    train_path = REPO / "data" / "train" / "osler_v1" / f"block_{BLOCK_FILE[name]}.jsonl"
    lora = base.lora
    if name == "B0b":
        lora = dataclasses.replace(base.lora, target_modules=ATTENTION_TARGETS)
    config = dataclasses.replace(
        base,
        train_path=str(train_path.relative_to(REPO)),
        output_dir=str(out_dir.relative_to(REPO)),
        eval_every=steps // EVAL_QUARTERS,
        lora=lora,
        log_every=1 if args.smoke else base.log_every,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    block_record = check_block_manifest(BLOCK_FILE[name], train_path)
    if replay.get("train_file_sha256") != block_record["sha256_file"]:
        raise SystemExit(f"{name}: the replay was made from another file (replay the block again)")
    print(f"{name}: loading {train_path.name}", flush=True)
    train_items = read_mix(train_path)
    dev_items = read_mix(DEV)
    if args.smoke:
        dev_items = dev_items[:SMOKE_DEV_ITEMS]
    print(f"{name}: {len(train_items)} train items, {len(dev_items)} dev items, {steps} steps, "
          f"eval every {config.eval_every}", flush=True)
    write_json(out_dir / "provenance.json", {
        "built_at_utc": utcnow(), "git_commit": git_commit(REPO), "block": name,
        "command": f"scripts/osler_v1/v8_block_run.py --block {name}" + (f" --smoke {args.smoke}" if args.smoke else ""),
        "config_file": str(CONFIG.relative_to(REPO)), "train_file": str(train_path.relative_to(REPO)),
        "train_sha256": file_sha256(train_path), "dev_file": str(DEV.relative_to(REPO)),
        "dev_sha256": file_sha256(DEV), "dev_items_used": len(dev_items), "train_items": len(train_items),
        "steps": steps, "replay_steps": replay_steps, "base_model": config.base_model, "revision": config.revision,
        "lora_target_modules": list(config.lora.target_modules),
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
        "block": name, "smoke_steps": args.smoke or None, "steps": steps, "replay_of": replay_name,
        "train_items": len(train_items),
        "realised_items": sum(realised), "real_tokens": sum(replay["real_tokens"][:steps]),
        "padded_tokens": sum(replay["padded_tokens"][:steps]),
        "selected": summary["selected"], "best_step": summary["best_step"],
        "evaluations": summary["evaluations"], "full_dev_metrics": summary["full_dev_metrics"],
        "temperature_fits_keys": sorted(summary["temperature_fits"]),
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
