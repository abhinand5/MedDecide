"""V9: the final Osler-4B run on the kept mix (osler_v1 ADVISORY V9; Osler-9B deferred to the next loop by the operator, V9-D1).

The kept mix is the last kept block of the V8 keep rules (STATE §3, CLAIMS). One pass over it, capped at 250,000 realised
items: the replay at that cap (eight evaluations, so the step count is a multiple of eight) sets the step count. The recipe
is configs/osler_v1/block_arm.yaml, the same as the block runs. Eight evaluations on the fast dev set; the selected checkpoint
is the best on the pooled template macro accuracy (the trainer's selection); the temperatures are fitted on the dev split and
saved with the selected checkpoint (run_arm). The checks, the arm and the logged-step validation are the V8 block driver's,
imported from scripts/osler_v1/v8_block_run.py, which is not changed by this script.

Writes outputs/osler_v1/V9/osler_4b/ (provenance.json, arm_result.json, block_result.json, logs/, dev_predictions.jsonl,
checkpoints/best). The replay for the kept block is written first by:
    source scripts/pod_env.sh && uv run --frozen python scripts/osler_v1/v8_replay_steps.py --blocks <BLOCK> \
        --target-items 250000 --multiple 8 --out outputs/osler_v1/V9

Usage (one GPU job at a time; launch only after the keep decisions name the kept block):
    source scripts/pod_env.sh && setsid nohup uv run --frozen python scripts/osler_v1/v9_final.py --block <BLOCK> \
        > outputs/osler_v1/V9/logs/osler_4b.log 2>&1 < /dev/null &
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import sys
import time
from pathlib import Path

import torch

from meddecide.train.arm_run import run_arm
from meddecide.train.config import load_config
from meddecide.utils.hashing import file_sha256
from meddecide.utils.io import read_json, write_json
from meddecide.utils.provenance import git_commit, utcnow

REPO = Path(__file__).resolve().parents[2]
V8_SCRIPT = REPO / "scripts" / "osler_v1" / "v8_block_run.py"
OUT = REPO / "outputs" / "osler_v1" / "V9"
CONFIG = REPO / "configs" / "osler_v1" / "block_arm.yaml"
TRAIN_DIR = REPO / "data" / "train" / "osler_v1"
EVALUATIONS = 8  # ADVISORY V9: eight evaluations per run on the fast dev set
TARGET_ITEMS = 250_000  # ADVISORY V9: one pass capped at 250,000 realised items


def load_v8():
    """The V8 block driver's checks, reader and validation, imported from its file (the file itself is not changed)."""
    spec = importlib.util.spec_from_file_location("v8_block_run", V8_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("v8_block_run", module)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--block", required=True, choices=["B1", "B2", "B3"],
                        help="the kept block (V8 keep rules); B0 and B0b are not training candidates for the final run")
    args = parser.parse_args()
    v8 = load_v8()
    started = time.time()
    block = args.block
    replay = read_json(OUT / "replay" / f"{block}.json")
    steps = int(replay["steps"])
    if steps % EVALUATIONS:
        raise SystemExit(f"the replay of {block} does not give {EVALUATIONS} evaluations on whole steps")
    if replay["realised_items"] < TARGET_ITEMS * 0.99:
        raise SystemExit(f"the replay of {block} holds {replay['realised_items']} items, not about {TARGET_ITEMS}")
    base = load_config(CONFIG)
    train_path = TRAIN_DIR / f"block_{block}.jsonl"
    out_dir = OUT / "osler_4b"
    config = dataclasses.replace(
        base,
        train_path=str(train_path.relative_to(REPO)),
        output_dir=str(out_dir.relative_to(REPO)),
        eval_every=steps // EVALUATIONS,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    block_record = v8.check_block_manifest(block, train_path)
    if replay.get("train_file_sha256") != block_record["sha256_file"]:
        raise SystemExit(f"{block}: the final replay was made from another file (replay it again)")
    print(f"osler_4b on {block}: loading {train_path.name} ({replay['train_items_in_block']} rows)", flush=True)
    train_items = v8.read_mix(train_path)
    dev_items = v8.read_mix(v8.DEV)
    print(f"osler_4b on {block}: {len(train_items)} train items, {len(dev_items)} dev items, {steps} steps, "
          f"eval every {config.eval_every}", flush=True)
    write_json(out_dir / "provenance.json", {
        "built_at_utc": utcnow(), "git_commit": git_commit(REPO), "block": block, "model": "osler_4b",
        "deferred": "Osler-9B is deferred to the next loop by the operator (V9-D1)",
        "command": f"scripts/osler_v1/v9_final.py --block {block}",
        "config_file": str(CONFIG.relative_to(REPO)), "train_file": str(train_path.relative_to(REPO)),
        "train_sha256": block_record["sha256_file"], "dev_file": str(v8.DEV.relative_to(REPO)),
        "dev_sha256": file_sha256(v8.DEV), "dev_items_used": len(dev_items), "train_items": len(train_items),
        "steps": steps, "evaluations": EVALUATIONS, "target_realised_items": TARGET_ITEMS,
        "base_model": config.base_model, "revision": config.revision,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
        "torch": torch.__version__, "python": sys.version.split()[0],
    })
    summary = run_arm(config, train_items=train_items, dev_items=dev_items, out_dir=out_dir, steps=steps,
                      eval_subset_size=len(dev_items))
    expected = [steps * q // EVALUATIONS for q in range(1, EVALUATIONS + 1)]
    eval_steps = [e["step"] for e in summary["evaluations"]]
    validation = {
        "steps_taken": summary["steps"], "steps_as_planned": summary["steps"] == steps,
        "stopped_early": summary["stopped_early"],
        "evaluation_steps": eval_steps, "evaluation_steps_as_planned": eval_steps == expected,
        "eval_subset_items": summary["eval_subset_items"], "dev_items": len(dev_items),
        "eval_subset_is_full_dev": summary["eval_subset_items"] == len(dev_items),
        "logged_steps": v8.validate_logged_steps(out_dir / "logs" / "train.jsonl", replay, steps),
    }
    realised = replay["sizes"][:steps]
    block_result = {
        "block": block, "model": "osler_4b", "steps": steps, "evaluations": EVALUATIONS,
        "train_items": len(train_items), "realised_items": sum(realised),
        "real_tokens": sum(replay["real_tokens"][:steps]), "padded_tokens": sum(replay["padded_tokens"][:steps]),
        "selected": summary["selected"], "best_step": summary["best_step"],
        "evaluations_detail": summary["evaluations"], "full_dev_metrics": summary["full_dev_metrics"],
        "temperature_fits_keys": sorted(summary["temperature_fits"]),
        "validation": validation, "train_wall_clock_s": summary["train_wall_clock_s"],
        "total_wall_clock_s": round(time.time() - started, 1), "config": config.to_dict(),
    }
    write_json(out_dir / "block_result.json", block_result)
    print(f"osler_4b on {block}: steps {summary['steps']}, realised items {sum(realised)}, selected {summary['selected']}, "
          f"validation {validation['logged_steps']['mismatches']} log mismatches, "
          f"{'evaluations on the planned steps' if validation['evaluation_steps_as_planned'] else 'EVALUATION STEPS OFF PLAN'}",
          flush=True)


if __name__ == "__main__":
    main()
