"""O6-O8: train one matched Osler-4B arm on training mix v2 (GPU; one GPU job at a time).

Reads configs/osler_v0/arm_<arm>.yaml, loads the mix's train and dev rows through the training reader (mix_items), runs
``meddecide.train.arm_run.run_arm``, and writes the result and a provenance record under the arm's output directory.

Usage (launch detached; check the GPU is free first):
    source scripts/pod_env.sh && setsid nohup uv run --frozen python scripts/osler/o6_train_arm.py --arm L \
        > outputs/osler_v0/O6/logs/arm_L.log 2>&1 < /dev/null &
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
from meddecide.utils.hashing import file_sha256
from meddecide.utils.io import write_json
from meddecide.utils.provenance import git_commit, utcnow

REPO = Path(__file__).resolve().parents[2]


def read_mix(path: Path) -> list:
    items = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            items.append(mix_row_to_item(json.loads(line)))
    return items


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True, choices=["L", "P", "N"])
    parser.add_argument("--steps", type=int, default=None, help="override the step count (the budget is the default)")
    args = parser.parse_args()
    started = time.time()
    config_path = REPO / f"configs/osler_v0/arm_{args.arm}.yaml"
    config = load_config(config_path)
    out_dir = REPO / config.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = REPO / config.train_path
    dev_path = REPO / config.dev_path
    print(f"arm {args.arm}: loading {train_path.name} and {dev_path.name}", flush=True)
    train_items = read_mix(train_path)
    dev_items = read_mix(dev_path)
    print(f"arm {args.arm}: {len(train_items)} train items, {len(dev_items)} dev items", flush=True)
    provenance = {
        "built_at_utc": utcnow(),
        "git_commit": git_commit(REPO),
        "command": f"scripts/osler/o6_train_arm.py --arm {args.arm}" + (f" --steps {args.steps}" if args.steps else ""),
        "config_file": str(config_path.relative_to(REPO)),
        "config_sha256": file_sha256(config_path),
        "train_sha256": file_sha256(train_path),
        "dev_sha256": file_sha256(dev_path),
        "train_items": len(train_items),
        "dev_items": len(dev_items),
        "base_model": config.base_model,
        "revision": config.revision,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
        "torch": torch.__version__,
        "python": sys.version.split()[0],
    }
    write_json(out_dir / "provenance.json", provenance)
    summary = run_arm(config, train_items=train_items, dev_items=dev_items, out_dir=out_dir, steps=args.steps)
    summary["wall_clock_s"] = round(time.time() - started, 1)
    write_json(out_dir / "arm_result.json", summary)
    print(f"arm {args.arm}: steps {summary['steps']}, stopped_early {summary['stopped_early']}, "
          f"selected {summary['selected']}, full-dev macro {summary['full_dev_metrics'].get('macro_accuracy')}",
          flush=True)


if __name__ == "__main__":
    main()
