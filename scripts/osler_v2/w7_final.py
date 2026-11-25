"""osler_v2 W7: the final Osler-4B on the kept mix (ADVISORY W7; D39: Osler-9B deferred).

One pass over the kept block (C0, C1 or C2; chosen by the keep rules in blocks.md), attention-only LoRA (D33), the D40 speed
recipe, the chosen base (pplx-decider-flash-v1.2). The step count is the smallest multiple of eight whose batches hold at least
250,000 realised items (the replay below); eight evaluations on the whole fast dev v2; the selected checkpoint is the best on the
pooled template macro (the trainer's selection); the temperatures are fitted on the dev split and saved with the checkpoint.

The replay is written first, by the trainer's own batcher with the backbone's encoder (w6_replay's encoders; no weights):
    uv run --frozen python scripts/osler_v2/w7_final.py --block C2 --replay-only
Then the final run (detached; one GPU job at a time):
    setsid nohup uv run --frozen --with torchvision==0.29.* --with pillow python scripts/osler_v2/w7_final.py --block C2 \
        > outputs/osler_v2/W7/logs/osler_4b.log 2>&1 < /dev/null &

Writes outputs/osler_v2/W7/replay/<block>_pplx_250k.json and outputs/osler_v2/W7/osler_4b/ (provenance, block_result, checkpoints,
dev_predictions.jsonl).
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
import time
from pathlib import Path
from typing import Any

import torch

from meddecide.model.pplx import ATTENTION_TARGETS
from meddecide.train.arm_run import run_arm
from meddecide.train.blocks import steps_for_items
from meddecide.train.config import load_config
from meddecide.train.data import iter_batches, shuffled_order
from meddecide.utils.hashing import file_sha256
from meddecide.utils.io import read_json, write_json
from meddecide.utils.provenance import git_commit, utcnow

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "osler_v2"))
import w6_block_run as w6  # noqa: E402  (the block runner's reading, checks and validation helpers)
import w6_replay as rp  # noqa: E402  (the replay encoders and batch record)

CONFIG = REPO / "configs" / "osler_v2" / "block_arm.yaml"
TRAIN_DIR = REPO / "data" / "train" / "osler_v2"
DEV = REPO / "data" / "bench" / "osler_v2" / "fastdev_v2.jsonl"
OUT = REPO / "outputs" / "osler_v2" / "W7"
EVALUATIONS = 8  # ADVISORY W7 / osler_v1 V9: eight evaluations on the fast dev set
TARGET_ITEMS = 250_000  # ADVISORY W7: one pass capped at 250,000 realised items
MULTIPLE = 8


def replay_to_target(block: str, backbone: str, config: Any) -> dict[str, Any]:
    """The step count and per-step records of the block until the realised items reach the target (multiple of eight)."""
    path = TRAIN_DIR / f"block_{block}.jsonl"
    items = rp.read_block(path)
    encoder = rp.encoder_for(backbone, config)
    order = shuffled_order(len(items), seed=config.seed, epoch=0)
    batches = iter_batches(encoder, items, batch_size=config.batch_size, max_batch_tokens=config.max_batch_tokens,
                           max_prompt_tokens=config.max_prompt_tokens, order=order, augment_options=config.shuffle_options,
                           epoch=0, seed=config.seed)
    records: list[tuple[int, int, int]] = []

    def stream():
        for _, batch in batches:
            record = rp.batch_record(batch)
            records.append(record)
            yield record[0]

    steps = steps_for_items(stream(), TARGET_ITEMS, multiple=MULTIPLE)
    batches.close()
    if len(records) != steps:
        raise RuntimeError(f"replayed {len(records)} batches for {steps} steps")
    sizes = [r[0] for r in records]
    replay = {"block": block, "backbone": backbone, "steps": steps, "target_items": TARGET_ITEMS,
              "train_file": str(path.relative_to(REPO)), "train_file_sha256": file_sha256(path),
              "train_items_in_block": len(items), "sizes": sizes, "real_tokens": [r[1] for r in records],
              "padded_tokens": [r[2] for r in records], "realised_items": sum(sizes),
              "written_utc": utcnow(), "git_commit": git_commit(REPO)}
    write_json(OUT / "replay" / f"{block}_{backbone}_250k.json", replay)
    return replay


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--block", required=True, choices=["C0", "C1", "C2"])
    ap.add_argument("--backbone", default="pplx", choices=["qwen", "pplx"])
    ap.add_argument("--replay-only", action="store_true")
    args = ap.parse_args()
    base = load_config(CONFIG)
    replay_path = OUT / "replay" / f"{args.block}_{args.backbone}_250k.json"
    if replay_path.exists():
        replay = read_json(replay_path)
    else:
        replay = replay_to_target(args.block, args.backbone, base)
    if args.replay_only:
        print(f"replay {args.block}/{args.backbone}: steps {replay['steps']}, realised items {replay['realised_items']}",
              flush=True)
        return
    started = time.time()
    steps = int(replay["steps"])
    if steps % EVALUATIONS:
        raise SystemExit("the replay does not give eight evaluations on whole steps")
    train_path = TRAIN_DIR / f"block_{args.block}.jsonl"
    if replay.get("train_file_sha256") != file_sha256(train_path):
        raise SystemExit("the final replay was made from another file (replay it again)")
    out_dir = OUT / "osler_4b"
    config = dataclasses.replace(
        base, train_path=str(train_path.relative_to(REPO)), dev_path=str(DEV.relative_to(REPO)),
        output_dir=str(out_dir.relative_to(REPO)), eval_every=steps // EVALUATIONS, backbone=args.backbone,
        lora=dataclasses.replace(base.lora, target_modules=ATTENTION_TARGETS))
    out_dir.mkdir(parents=True, exist_ok=True)
    train_items = w6.read_mix(train_path)
    dev_items = w6.read_mix(DEV)
    write_json(out_dir / "provenance.json", {
        "built_at_utc": utcnow(), "git_commit": git_commit(REPO), "block": args.block, "backbone": args.backbone,
        "command": f"scripts/osler_v2/w7_final.py --block {args.block} --backbone {args.backbone}",
        "train_sha256": replay["train_file_sha256"], "dev_sha256": file_sha256(DEV), "steps": steps,
        "evaluations": EVALUATIONS, "target_realised_items": TARGET_ITEMS, "realised_items": replay["realised_items"],
        "lora_target_modules": list(config.lora.target_modules), "decision": "D39 base; D40 recipe; D33 adapter",
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none", "torch": torch.__version__,
    })
    summary = run_arm(config, train_items=train_items, dev_items=dev_items, out_dir=out_dir, steps=steps,
                      eval_subset_size=len(dev_items))
    expected = [steps * q // EVALUATIONS for q in range(1, EVALUATIONS + 1)]
    eval_steps = [e["step"] for e in summary["evaluations"]]
    validation = {
        "steps_taken": summary["steps"], "steps_as_planned": summary["steps"] == steps,
        "stopped_early": summary["stopped_early"], "evaluation_steps": eval_steps,
        "evaluation_steps_as_planned": eval_steps == expected,
        "eval_subset_is_full_dev": summary["eval_subset_items"] == len(dev_items),
        "logged_steps": w6.validate_logged_steps(out_dir / "logs" / "train.jsonl", replay, steps),
    }
    realised = replay["sizes"][:steps]
    result = {
        "block": args.block, "backbone": args.backbone, "model": "osler_4b", "steps": steps, "evaluations": EVALUATIONS,
        "realised_items": sum(realised), "selected": summary["selected"], "best_step": summary["best_step"],
        "evaluations_detail": summary["evaluations"], "full_dev_metrics": summary["full_dev_metrics"],
        "temperature_fits_keys": sorted(summary["temperature_fits"]), "validation": validation,
        "train_wall_clock_s": summary["train_wall_clock_s"], "total_wall_clock_s": round(time.time() - started, 1),
    }
    write_json(out_dir / "block_result.json", result)
    print(f"osler_4b: steps {steps}, realised items {sum(realised)}, selected {summary['selected']}, "
          f"validation {validation['logged_steps']['mismatches']} log mismatches", flush=True)


if __name__ == "__main__":
    main()
