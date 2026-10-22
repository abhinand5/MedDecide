"""Y4 baseline (osler_v4 ADVISORY Y4 §4): a trainer-based osler checkpoint scored on our items. Used for osler_v2's Osler-4B
(outputs/osler_v2/W7/osler_4b/checkpoints/best), which the ADVISORY names as a baseline.

The checkpoint loads with meddecide.model.pplx.load_any, and the calibrated temperature is the one it saved (model.json). Item ids
are the trainer's (mix_row_to_item), so these predictions pair by item id with the other arms (blocks.pair_by_item).

Writes outputs/osler_v4/Y4/baselines/<name>/predictions.jsonl (meddecide.train.osler_arm.dev_prediction_rows) and manifest.json.

Usage (one GPU job at a time; detached):
    source scripts/pod_env.sh && setsid nohup uv run --frozen python scripts/osler_v4/y4_checkpoint_score.py \
        --checkpoint outputs/osler_v2/W7/osler_4b/checkpoints/best --items data/bench/osler_v4/fastdev_v4.jsonl \
        --name fastdev_v4_osler_v2_4b > outputs/osler_v4/Y4/logs/osler_v2_fastdev.log 2>&1 < /dev/null &
"""

from __future__ import annotations

import argparse
import dataclasses
import time
from pathlib import Path

from meddecide.model.pplx import load_any
from meddecide.train.config import load_config
from meddecide.train.mix_items import mix_row_to_item
from meddecide.train.osler_arm import dev_prediction_rows
from meddecide.train.trainer import Trainer
from meddecide.utils.io import iter_jsonl, write_json, write_jsonl
from meddecide.utils.provenance import git_commit, gpu_name, utcnow

REPO = Path(__file__).resolve().parents[2]
RECIPE = REPO / "configs" / "osler_v3" / "block_arm.yaml"
OUT = REPO / "outputs" / "osler_v4" / "Y4" / "baselines"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--items", required=True, type=Path)
    parser.add_argument("--name", required=True)
    parser.add_argument("--limit", type=int, default=0, help="score only the first N items (smoke runs)")
    args = parser.parse_args()
    started = time.time()
    config = dataclasses.replace(load_config(RECIPE), kl_to_base=0.0, distill_weight=0.0)
    model = load_any(args.checkpoint, device=config.device, dtype=config.dtype)
    rows = list(iter_jsonl(args.items))
    if args.limit:
        rows = rows[: args.limit]
    items = [mix_row_to_item(row) for row in rows]  # a row the schema rejects stops the run (loud)
    scored = Trainer(model, config).score(items)
    out = OUT / args.name
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "predictions.jsonl", dev_prediction_rows(scored))
    write_json(out / "manifest.json", {
        "model": "osler_v2 Osler-4B (trainer-based; pplx-decider-flash-v1.2 base, D39)",
        "checkpoint": str(args.checkpoint), "items_file": str(args.items), "items": len(items),
        "metrics": {k: v for k, v in scored.metrics().items() if isinstance(v, (int, float))},
        "gpu": gpu_name(), "git_commit": git_commit(REPO), "wall_clock_s": round(time.time() - started, 1),
        "generated_utc": utcnow(),
    })
    print(f"[{utcnow()}] {args.name}: scored {len(items)} items", flush=True)


if __name__ == "__main__":
    main()
