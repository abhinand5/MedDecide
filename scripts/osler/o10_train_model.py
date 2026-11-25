"""O10: train Osler-9B or Osler-0.8B with the head O9 chose (GPU; one GPU job at a time; ADVISORY O10).

Runs the O6-O8 recipe (configs/osler_v0/arm_L.yaml) on the base named in the config. Before the model loads, it checks that
the base and its revision are the pinned ones, that the config's recipe is arm L's apart from the base, output directory
and head (src/meddecide/train/osler_configs.py), and that the head is the one in outputs/osler_v0/O9/head_choice.json.
A failed check stops the run with the reason. The O6 driver (scripts/osler/o6_train_arm.py) is left as it was, so the
running O6-O8 chain keeps its code.

Usage (launch detached; check the GPU is free first):
    source scripts/pod_env.sh && mkdir -p outputs/osler_v0/O10/logs && setsid nohup uv run --frozen python \
        scripts/osler/o10_train_model.py --config configs/osler_v0/osler_9b.yaml \
        > outputs/osler_v0/O10/logs/osler_9b.log 2>&1 < /dev/null &
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

from meddecide.train.arm_run import run_arm
from meddecide.train.config import load_config
from meddecide.train.mix_items import mix_row_to_item
from meddecide.train.osler_configs import check_head_matches, check_pinned_base, recipe_differences
from meddecide.utils.hashing import file_sha256
from meddecide.utils.io import write_json
from meddecide.utils.provenance import git_commit, utcnow

REPO = Path(__file__).resolve().parents[2]
REFERENCE_CONFIG = REPO / "configs/osler_v0/arm_L.yaml"
HEAD_CHOICE = REPO / "outputs/osler_v0/O9/head_choice.json"


def read_mix(path: Path) -> list:
    items = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            items.append(mix_row_to_item(json.loads(line)))
    return items


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path, help="a config under configs/osler_v0/")
    parser.add_argument("--steps", type=int, default=None, help="override the step count (the budget is the default)")
    parser.add_argument("--head-choice", type=Path, default=HEAD_CHOICE)
    args = parser.parse_args()
    started = time.time()
    config_path = args.config if args.config.is_absolute() else REPO / args.config
    config = load_config(config_path)
    reference = load_config(REFERENCE_CONFIG)
    check_pinned_base(config)
    differences = recipe_differences(config, reference)
    if differences:
        raise SystemExit(f"{config_path.name}: the recipe differs from arm L in {differences}; "
                         "record the deviation first")
    head_record = json.loads(args.head_choice.read_text(encoding="utf-8"))
    chosen = str(head_record["chosen"])
    check_head_matches(config, chosen)
    out_dir = REPO / config.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = REPO / config.train_path
    dev_path = REPO / config.dev_path
    print(f"O10 {config_path.name}: head {chosen} (checked against {args.head_choice.name}); "
          f"loading {train_path.name} and {dev_path.name}", flush=True)
    train_items = read_mix(train_path)
    dev_items = read_mix(dev_path)
    print(f"O10 {config_path.name}: {len(train_items)} train items, {len(dev_items)} dev items", flush=True)
    provenance = {
        "built_at_utc": utcnow(),
        "git_commit": git_commit(REPO),
        "command": f"scripts/osler/o10_train_model.py --config {args.config}"
        + (f" --steps {args.steps}" if args.steps else ""),
        "config_file": str(config_path.relative_to(REPO)),
        "config_sha256": file_sha256(config_path),
        "reference_config": str(REFERENCE_CONFIG.relative_to(REPO)),
        "reference_config_sha256": file_sha256(REFERENCE_CONFIG),
        "head_choice": chosen,
        "head_choice_sha256": file_sha256(args.head_choice),
        "train_sha256": file_sha256(train_path),
        "dev_sha256": file_sha256(dev_path),
        "train_items": len(train_items),
        "dev_items": len(dev_items),
        "base_model": config.base_model,
        "revision": config.revision,
        "lora_r": config.lora.r,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
        "torch": torch.__version__,
        "python": sys.version.split()[0],
    }
    write_json(out_dir / "provenance.json", provenance)
    summary = run_arm(config, train_items=train_items, dev_items=dev_items, out_dir=out_dir, steps=args.steps)
    summary["wall_clock_s"] = round(time.time() - started, 1)
    summary["head_choice"] = chosen
    write_json(out_dir / "arm_result.json", summary)
    print(f"O10 {config_path.name}: steps {summary['steps']}, stopped_early {summary['stopped_early']}, "
          f"selected {summary['selected']}, full-dev macro {summary['full_dev_metrics'].get('macro_accuracy')}",
          flush=True)


if __name__ == "__main__":
    main()
