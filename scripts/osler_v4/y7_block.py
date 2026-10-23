"""Y7 (osler_v4 ADVISORY Y7): one Stage B block (B1 to B4), trained and scored on the chosen Stage A recipe (A0: KL 0, 13,992 steps).

The recipe is configs/osler_v3/block_arm.yaml with kl_to_base 0 and no distillation (STATE §7: the chosen recipe is A0, the control).
Each block is one training run on the block's file (scripts/osler_v4/y7_build_block.py) and its replay (y7_replay.py). The run is
evaluated on fast dev v4 at each quarter of the steps; the selected checkpoint is scored on fast dev v4 and CG v2 by `score`.

Subcommands (one GPU job at a time; detached; check the card is free first):
  train  --arm B1|B2|B3|B4 --train F --replay R        a new run; writes outputs/osler_v4/Y7/<arm>/
  score  --arm B1|B2|B3|B4                              the arm's best checkpoint on fast dev v4 and CG v2 (predictions and aggregates)

Usage:
    source scripts/pod_env.sh && setsid nohup uv run --frozen python scripts/osler_v4/y7_block.py train --arm B1 \
        --train data/train/osler_v4/block_B1.jsonl --replay outputs/osler_v4/Y7/replays/B1_pplx.json \
        > outputs/osler_v4/Y7/logs/B1_train.log 2>&1 < /dev/null &
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

from meddecide.bench.schema import Item
from meddecide.model.pplx import ATTENTION_TARGETS, load_any
from meddecide.train.arm_run import run_arm
from meddecide.train.blocks import EVAL_QUARTERS
from meddecide.train.config import load_config
from meddecide.train.mix_items import mix_row_to_item
from meddecide.train.osler_arm import dev_prediction_rows
from meddecide.train.trainer import Trainer
from meddecide.utils.hashing import file_sha256
from meddecide.utils.io import read_json, write_json, write_jsonl
from meddecide.utils.provenance import git_commit, utcnow

REPO = Path(__file__).resolve().parents[2]
RECIPE = REPO / "configs" / "osler_v3" / "block_arm.yaml"
DEV = REPO / "data" / "bench" / "osler_v4" / "fastdev_v4.jsonl"
CG = REPO / "data" / "bench" / "osler_v4" / "cg_v2.jsonl"
OUT = REPO / "outputs" / "osler_v4" / "Y7"
ARMS = ("B1", "B2", "B3", "B4")


def _v8():
    spec = importlib.util.spec_from_file_location("v8_block_run", REPO / "scripts" / "osler_v1" / "v8_block_run.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


validate_logged_steps = _v8().validate_logged_steps


def read_mix(path: Path) -> list[Item]:
    with path.open(encoding="utf-8") as fh:
        return [mix_row_to_item(json.loads(line)) for line in fh if line.strip()]


def cmd_train(args: argparse.Namespace) -> None:
    started = time.time()
    replay = read_json(args.replay)
    steps = int(replay["steps"])
    if steps % EVAL_QUARTERS:
        raise SystemExit("the step count must be a multiple of four (four evaluations)")
    train_sha = file_sha256(args.train)
    if replay.get("train_file_sha256") != train_sha:
        raise SystemExit(f"{args.arm}: the replay was made from another file (replay it again)")
    base = load_config(RECIPE)
    out_dir = OUT / args.arm
    config = dataclasses.replace(
        base,
        train_path=str(args.train),
        dev_path=str(DEV.relative_to(REPO)),
        output_dir=str(out_dir.relative_to(REPO)),
        eval_every=steps // EVAL_QUARTERS,
        backbone="pplx",
        lora=dataclasses.replace(base.lora, target_modules=ATTENTION_TARGETS),
        kl_to_base=0.0,
        kl_exclude_template_prefixes=(),
        distill_weight=0.0,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    train_items = read_mix(args.train)
    dev_items = read_mix(DEV)
    write_json(out_dir / "provenance.json", {
        "built_at_utc": utcnow(), "git_commit": git_commit(REPO), "arm": args.arm, "command": " ".join(sys.argv),
        "config_file": str(RECIPE.relative_to(REPO)), "train_file": str(args.train), "train_sha256": train_sha,
        "replay_file": str(args.replay), "dev_file": str(DEV.relative_to(REPO)), "dev_sha256": file_sha256(DEV),
        "dev_items": len(dev_items), "train_items": len(train_items), "steps": steps, "kl_to_base": 0.0,
        "distill_weight": 0.0, "base_model": "pplx-decider-flash-v1.2 (D39)", "recipe": "A0 (the chosen Stage A recipe)",
    })
    summary = run_arm(config, train_items=train_items, dev_items=dev_items, out_dir=out_dir, steps=steps,
                      eval_subset_size=len(dev_items))
    expected = [steps * q // EVAL_QUARTERS for q in range(1, EVAL_QUARTERS + 1)]
    eval_steps = [e["step"] for e in summary["evaluations"]]
    sizes = replay["sizes"][:steps]
    validation = {
        "steps_taken": summary["steps"], "steps_as_planned": summary["steps"] == steps,
        "stopped_early": summary["stopped_early"], "evaluation_steps": eval_steps,
        "evaluation_steps_as_planned": eval_steps == expected,
        "logged_steps": validate_logged_steps(out_dir / "logs" / "train.jsonl", replay, steps),
    }
    write_json(out_dir / "block_result.json", {
        "arm": args.arm, "kl_to_base": 0.0, "distill_weight": 0.0, "steps": steps,
        "realised_items": sum(sizes), "selected": summary["selected"], "best_step": summary["best_step"],
        "evaluations": summary["evaluations"], "full_dev_metrics": summary["full_dev_metrics"],
        "validation": validation, "train_wall_clock_s": summary["train_wall_clock_s"],
        "total_wall_clock_s": round(time.time() - started, 1),
    })
    print(f"{args.arm}: steps {summary['steps']}, realised items {sum(sizes)}, selected {summary['selected']}, "
          f"validation {validation['logged_steps']['mismatches']} log mismatches, "
          f"{'evaluations on the planned steps' if validation['evaluation_steps_as_planned'] else 'EVALUATION STEPS OFF PLAN'}",
          flush=True)


def cmd_score(args: argparse.Namespace) -> None:
    started = time.time()
    checkpoint = OUT / args.arm / "checkpoints" / "best"
    if not checkpoint.exists():
        raise SystemExit(f"{args.arm}: no best checkpoint at {checkpoint}")
    config = dataclasses.replace(load_config(RECIPE), kl_to_base=0.0, distill_weight=0.0)
    model = load_any(checkpoint, device=config.device, dtype=config.dtype)
    trainer = Trainer(model, config)
    out = OUT / args.arm
    out.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {"arm": args.arm, "checkpoint": str(checkpoint.relative_to(REPO)), "sets": {}}
    for name, path in (("fastdev_v4", DEV), ("cg_v2", CG)):
        items = read_mix(path)
        scored = trainer.score(items)
        write_jsonl(out / f"{name}_predictions.jsonl", dev_prediction_rows(scored))
        report["sets"][name] = {"items": len(items),
                                "metrics": {k: v for k, v in scored.metrics().items() if isinstance(v, (int, float))}}
    report.update({"scored_utc": utcnow(), "git_commit": git_commit(REPO), "wall_clock_s": round(time.time() - started, 1)})
    write_json(out / "score_result.json", report)
    print(f"{args.arm}: fast dev v4 {report['sets']['fastdev_v4']['metrics'].get('accuracy')}; "
          f"CG v2 {report['sets']['cg_v2']['metrics'].get('accuracy')}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train")
    train.add_argument("--arm", required=True, choices=ARMS)
    train.add_argument("--train", required=True, type=Path)
    train.add_argument("--replay", required=True, type=Path)
    score = sub.add_parser("score")
    score.add_argument("--arm", required=True, choices=ARMS)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    {"train": cmd_train, "score": cmd_score}[args.command](args)


if __name__ == "__main__":
    main()
