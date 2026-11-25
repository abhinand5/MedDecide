"""osler_v3 block run (ADVISORY X7 Stage A, X8 Stage B, X9 final): one attention-only pplx run with an explicit KL-to-base setting.

The recipe is configs/osler_v3/block_arm.yaml (osler_v2's recipe plus the KL keys). The run takes its training file and its replay
(the batches the trainer will draw, from scripts/osler_v3/x7_replay.py or osler_v2's replay of the same file), trains for the
replay's step count, evaluates on the dev set at each quarter of the steps, keeps the best checkpoint on the pooled template macro
(the trainer's selection) and fits temperatures on the dev set. The logged step sizes are validated against the replay.

Writes outputs/osler_v3/<stage>/<arm>/ (provenance.json, arm_result.json, logs/, dev_predictions.jsonl, checkpoints/best) and
block_result.json (aggregates and validation). Scores on the clinical-general partition and the mean KL are written by
scripts/osler_v3/x7_score.py afterwards, for the selected checkpoint.

Usage (detached; check the GPU is free first):
    source scripts/pod_env.sh && setsid nohup uv run --frozen python scripts/osler_v3/x7_block_run.py --stage stage_a --arm A1 \
        --train data/train/osler_v3/block_C1.jsonl --replay outputs/osler_v2/W6/replay/C1_pplx.json --kl-to-base 0.1 \
        > outputs/osler_v3/X7/logs/A1.log 2>&1 < /dev/null &
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
from meddecide.train.osler_v3 import DOMAIN_PREFIXES
from meddecide.utils.hashing import file_sha256
from meddecide.utils.io import read_json, write_json
from meddecide.utils.provenance import git_commit, utcnow

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "configs" / "osler_v3" / "block_arm.yaml"
FASTDEV_V3 = REPO / "data" / "bench" / "osler_v3" / "fastdev_v3.jsonl"
FASTDEV_V2 = REPO / "data" / "bench" / "osler_v2" / "fastdev_v2.jsonl"
OUT_ROOT = REPO / "outputs" / "osler_v3"


def _v8():
    spec = importlib.util.spec_from_file_location("v8_block_run", REPO / "scripts" / "osler_v1" / "v8_block_run.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


validate_logged_steps = _v8().validate_logged_steps


def read_mix(path: Path) -> list[Any]:
    from meddecide.train.mix_items import mix_row_to_item

    with path.open(encoding="utf-8") as fh:
        return [mix_row_to_item(json.loads(line)) for line in fh if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stage", required=True, help="stage_a, stage_b or final")
    ap.add_argument("--arm", required=True, help="A0..A3, B1..B3 or the final name")
    ap.add_argument("--train", required=True, type=Path, help="the block's training file (JSONL)")
    ap.add_argument("--replay", required=True, type=Path, help="the replay JSON of that file (steps, sizes, file hash)")
    ap.add_argument("--kl-to-base", required=True, type=float, help="0 for the control")
    ap.add_argument("--dev", type=Path, default=None, help="dev file (default: fast dev v3 when built, else fast dev v2)")
    ap.add_argument("--steps", type=int, default=None, help="step count (default: the replay's)")
    args = ap.parse_args()
    started = time.time()
    dev_path = args.dev or (FASTDEV_V3 if FASTDEV_V3.exists() else FASTDEV_V2)
    replay = read_json(args.replay)
    steps = int(args.steps or replay["steps"])
    if steps % EVAL_QUARTERS:
        raise SystemExit("the step count must be a multiple of four (four evaluations)")
    train_sha = file_sha256(args.train)
    if replay.get("train_file_sha256") != train_sha:
        raise SystemExit(f"{args.arm}: the replay was made from another file (replay it again)")
    base = load_config(CONFIG)
    out_dir = OUT_ROOT / args.stage / args.arm
    config = dataclasses.replace(
        base,
        train_path=str(args.train.resolve().relative_to(REPO)) if args.train.is_absolute() else str(args.train),
        dev_path=str(dev_path.resolve().relative_to(REPO)),
        output_dir=str(out_dir.relative_to(REPO)),
        eval_every=steps // EVAL_QUARTERS,
        backbone="pplx",
        lora=dataclasses.replace(base.lora, target_modules=ATTENTION_TARGETS),
        kl_to_base=args.kl_to_base,
        kl_exclude_template_prefixes=DOMAIN_PREFIXES if args.kl_to_base > 0 else (),
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    train_items = read_mix(args.train)
    dev_items = read_mix(dev_path)
    write_json(out_dir / "provenance.json", {
        "built_at_utc": utcnow(), "git_commit": git_commit(REPO), "stage": args.stage, "arm": args.arm,
        "command": " ".join(sys.argv), "config_file": str(CONFIG.relative_to(REPO)),
        "train_file": str(args.train), "train_sha256": train_sha, "replay_file": str(args.replay),
        "dev_file": str(dev_path), "dev_sha256": file_sha256(dev_path), "dev_items": len(dev_items),
        "train_items": len(train_items), "steps": steps, "kl_to_base": args.kl_to_base,
        "kl_exclude_template_prefixes": list(config.kl_exclude_template_prefixes),
        "lora_target_modules": list(config.lora.target_modules), "base_model": "pplx-decider-flash-v1.2 (D39)",
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none", "torch": torch.__version__,
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
    result = {
        "stage": args.stage, "arm": args.arm, "kl_to_base": args.kl_to_base, "steps": steps,
        "realised_items": sum(sizes), "selected": summary["selected"], "best_step": summary["best_step"],
        "evaluations": summary["evaluations"], "full_dev_metrics": summary["full_dev_metrics"],
        "temperature_fits_keys": sorted(summary["temperature_fits"]), "validation": validation,
        "train_wall_clock_s": summary["train_wall_clock_s"], "total_wall_clock_s": round(time.time() - started, 1),
    }
    write_json(out_dir / "block_result.json", result)
    print(f"{args.arm}: steps {summary['steps']}, realised items {sum(sizes)}, selected {summary['selected']}, "
          f"validation {validation['logged_steps']['mismatches']} log mismatches, "
          f"{'evaluations on the planned steps' if validation['evaluation_steps_as_planned'] else 'EVALUATION STEPS OFF PLAN'}",
          flush=True)


if __name__ == "__main__":
    main()
