"""Y6 (osler_v4 ADVISORY Y6; D48): Stage A on fast dev v4. The recipe is configs/osler_v3/block_arm.yaml (the osler_v3 recipe:
13,992 steps, seed 0). A0, A1 and A3 are osler_v3's checkpoints, re-scored and never retrained; A2 is not re-scored; A4 and A5
are new runs.

Subcommands (one GPU job at a time; detached; check the card is free first):
  train   --arm A4|A5 --train F --replay R --kl-to-base X [--distill-weight Y]   a new run, evaluated on fast dev v4
  score   --arm A0|A1|A3|A4|A5   the arm's best checkpoint on fast dev v4 and on CG v2 (predictions and aggregates)
  choose  (CPU) the pre-registered rule (meddecide.train.stage_a_choice) with the paired bootstrap of
          meddecide.train.blocks.keep_statistics; writes outputs/osler_v4/Y6/choice.json

Usage:
    source scripts/pod_env.sh && setsid nohup uv run --frozen python scripts/osler_v4/y6_stage_a.py train --arm A4 \
        --train data/train/osler_v3/block_A3.jsonl --replay outputs/osler_v3/replays/A3_pplx.json --kl-to-base 0.1 \
        > outputs/osler_v4/Y6/logs/A4.log 2>&1 < /dev/null &
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
from meddecide.eval.readout_gate import gate_by_template
from meddecide.model.pplx import ATTENTION_TARGETS, load_any
from meddecide.train.arm_run import run_arm
from meddecide.train.blocks import EVAL_QUARTERS, keep_statistics, pair_by_item
from meddecide.train.config import load_config
from meddecide.train.mix_items import mix_row_to_item
from meddecide.train.osler_arm import dev_prediction_rows
from meddecide.train.osler_v3 import DOMAIN_PREFIXES
from meddecide.train.stage_a_choice import (
    REFERENCE,
    ArmResult,
    choose,
    common_templates,
    passing_templates,
    qualifies,
    restrict_pairs,
)
from meddecide.train.trainer import Trainer
from meddecide.utils.hashing import file_sha256
from meddecide.utils.io import iter_jsonl, read_json, write_json, write_jsonl
from meddecide.utils.provenance import git_commit, utcnow

REPO = Path(__file__).resolve().parents[2]
RECIPE = REPO / "configs" / "osler_v3" / "block_arm.yaml"
DEV = REPO / "data" / "bench" / "osler_v4" / "fastdev_v4.jsonl"
CG = REPO / "data" / "bench" / "osler_v4" / "cg_v2.jsonl"
OSLER_V3 = REPO / "outputs" / "osler_v3" / "stage_a"
OUT = REPO / "outputs" / "osler_v4" / "Y6"
RESCORED = ("A0", "A1", "A3")
SCORED_ARMS = ("A0", "A1", "A3", "A4", "A5")


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


def arm_checkpoint(arm: str) -> Path:
    return (OSLER_V3 / arm if arm in RESCORED else OUT / arm) / "checkpoints" / "best"


def cmd_train(args: argparse.Namespace) -> None:
    started = time.time()
    replay = read_json(args.replay)
    steps = int(args.steps or replay["steps"])
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
        kl_to_base=args.kl_to_base,
        kl_exclude_template_prefixes=DOMAIN_PREFIXES if args.kl_to_base > 0 else (),
        distill_weight=args.distill_weight,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    train_items = read_mix(args.train)
    dev_items = read_mix(DEV)
    write_json(out_dir / "provenance.json", {
        "built_at_utc": utcnow(), "git_commit": git_commit(REPO), "arm": args.arm, "command": " ".join(sys.argv),
        "config_file": str(RECIPE.relative_to(REPO)), "train_file": str(args.train), "train_sha256": train_sha,
        "replay_file": str(args.replay), "dev_file": str(DEV.relative_to(REPO)), "dev_sha256": file_sha256(DEV),
        "dev_items": len(dev_items), "train_items": len(train_items), "steps": steps, "kl_to_base": args.kl_to_base,
        "distill_weight": args.distill_weight, "base_model": "pplx-decider-flash-v1.2 (D39)",
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
        "arm": args.arm, "kl_to_base": args.kl_to_base, "distill_weight": args.distill_weight, "steps": steps,
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
    checkpoint = arm_checkpoint(args.arm)
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


def _predictions(arm: str, name: str) -> list[dict[str, Any]]:
    path = OUT / arm / f"{name}_predictions.jsonl"
    if not path.exists():
        raise SystemExit(f"{arm}: {path} is missing (score the arm first)")
    return list(iter_jsonl(path))


def cmd_choose(args: argparse.Namespace) -> None:
    """The pre-registered rule with the D12 restriction (STATE §7, Y6): a cell that fails the readout gate is not compared.

    Each run's MR and knowledge lower bounds use the templates that pass for that run and for A0 (the same paired items). The
    qualifying runs are then ranked on the CG v2 templates that pass for all of them and for A0. When every cell passes, these are
    the full sets and the rule is exactly as written.
    """
    arms = [arm for arm in SCORED_ARMS if (OUT / arm / "cg_v2_predictions.jsonl").exists()]
    if REFERENCE not in arms:
        raise SystemExit(f"{REFERENCE} has no predictions: score it first")
    dev = {arm: _predictions(arm, "fastdev_v4") for arm in arms}
    cg = {arm: _predictions(arm, "cg_v2") for arm in arms}
    reference_dev = dev[REFERENCE]
    lowers: dict[str, tuple[float, float]] = {}
    detail: dict[str, Any] = {}
    for arm in arms:
        failing = {"fastdev_v4": sorted(t for t, g in gate_by_template(dev[arm]).items() if not g.passes),
                   "cg_v2": sorted(t for t, g in gate_by_template(cg[arm]).items() if not g.passes)}
        detail[arm] = {"failing_cells": failing}
        if arm == REFERENCE:
            continue
        templates = passing_templates(dev[arm]) & passing_templates(reference_dev)
        pairs = restrict_pairs(pair_by_item(dev[arm], reference_dev), templates)
        stats = keep_statistics(pairs)
        lowers[arm] = (stats["mr"].lo, stats["knowledge"].lo)
        detail[arm].update({"fastdev_v4_templates_compared": len(templates), "fastdev_v4_items_compared": len(pairs),
                            "mr_macro_difference": stats["mr"].as_dict(),
                            "knowledge_micro_difference": stats["knowledge"].as_dict()})
    qualifying = [REFERENCE] + [arm for arm in lowers if qualifies(*lowers[arm])]
    cg_common = common_templates([passing_templates(cg[arm]) for arm in qualifying])
    results: list[ArmResult] = []
    for arm in arms:
        rows = [r for r in cg[arm] if str(r["template_id"]) in cg_common]
        cg_accuracy = sum(bool(r["correct"]) for r in rows) / len(rows)
        if arm == REFERENCE:
            results.append(ArmResult(REFERENCE, None, None, cg_accuracy))
        else:
            results.append(ArmResult(arm, lowers[arm][0], lowers[arm][1], cg_accuracy))
        detail.setdefault(arm, {})
        detail[arm].update({"cg_v2_templates_compared": len(cg_common), "cg_v2_items_compared": len(rows),
                            "cg_v2_accuracy_on_compared": cg_accuracy})
    decision = choose(results)
    write_json(OUT / "choice.json", {"decision": decision, "cg_v2_templates_compared": sorted(cg_common), "arms": detail,
                                     "rule": {"mr_and_knowledge_lower_bound_limit": -0.01, "tie_margin": 0.003,
                                              "resamples": 1000, "seed": 0,
                                              "restriction": "D12: templates that pass the readout gate for every compared run"},
                                     "generated_utc": utcnow()})
    print(f"chosen: {decision['chosen']} ({decision['reason']})", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train")
    train.add_argument("--arm", required=True, choices=("A4", "A5"))
    train.add_argument("--train", required=True, type=Path)
    train.add_argument("--replay", required=True, type=Path)
    train.add_argument("--kl-to-base", required=True, type=float)
    train.add_argument("--distill-weight", type=float, default=0.0)
    train.add_argument("--steps", type=int, default=None)
    score = sub.add_parser("score")
    score.add_argument("--arm", required=True, choices=SCORED_ARMS)
    sub.add_parser("choose")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    {"train": cmd_train, "score": cmd_score, "choose": cmd_choose}[args.command](args)


if __name__ == "__main__":
    main()
