#!/usr/bin/env python
"""osler_v7 — score a saved checkpoint on any JSONL item file and write prediction rows (GPU, one job at a time).

Used for the gate inputs (T10) and for any additional evaluation: the item file is read with `mix_row_to_item`, so the
prediction rows carry the same recomputed item ids the other runs use (`meddecide.train.blocks.pair_by_item` pairs
them). Probabilities are kept because the D12 readout gate needs the label mass; the output stays under the gitignored
`outputs/`.

Run (GPU; detached)::

    source scripts/pod_env.sh && setsid nohup uv run --frozen python scripts/osler_v7/score_items.py \
        --checkpoint outputs/osler_v7/T6/arm/checkpoints/step_013992 \
        --items data/bench/osler_v7/confirm/general.jsonl --name t6_general --out outputs/osler_v7/T10 \
        > outputs/osler_v7/T10/logs/score_t6_general.log 2>&1 < /dev/null &
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from meddecide.model.pplx import PplxDecisionModel  # noqa: E402
from meddecide.train.mix_items import mix_row_to_item  # noqa: E402
from meddecide.train.osler_arm import dev_prediction_rows  # noqa: E402
from meddecide.utils.io import write_json, write_jsonl  # noqa: E402
from meddecide.utils.provenance import git_commit, utcnow  # noqa: E402


def read_items(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--items", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-batch-tokens", type=int, default=32768)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    args.checkpoint = args.checkpoint if args.checkpoint.is_absolute() else ROOT / args.checkpoint
    args.items = args.items if args.items.is_absolute() else ROOT / args.items
    args.out = args.out if args.out.is_absolute() else ROOT / args.out
    rows = read_items(args.items)
    if args.limit:
        rows = rows[: args.limit]
    items = [mix_row_to_item(row) for row in rows]
    if len({item.item_id for item in items}) != len(items):
        raise SystemExit("the item file yields duplicate item ids after conversion: stop")
    items.sort(key=lambda item: len(item.state) + len(item.question))
    started = time.time()
    model = PplxDecisionModel.load(args.checkpoint, device=args.device, dtype=args.dtype)
    model.eval_mode()
    scored = model.score_items(items, batch_size=args.batch_size, max_batch_tokens=args.max_batch_tokens)
    predictions = dev_prediction_rows(scored)
    out_dir = args.out / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "predictions.jsonl", predictions)
    write_json(out_dir / "manifest.json", {
        "generated_utc": utcnow(),
        "git_commit": git_commit(ROOT),
        "command": "uv run --frozen python scripts/osler_v7/score_items.py",
        "checkpoint": str(args.checkpoint.relative_to(ROOT)),
        "items": str(args.items.relative_to(ROOT)),
        "rows": len(predictions),
        "items_read": len(rows),
        "qtypes": {qtype: sum(1 for row in predictions if str(row["qtype"]) == qtype)
                   for qtype in sorted({str(row["qtype"]) for row in predictions})},
        "device": args.device,
        "dtype": args.dtype,
        "torch": torch.__version__,
        "batch_size": args.batch_size,
        "max_batch_tokens": args.max_batch_tokens,
        "wall_seconds": round(time.time() - started, 1),
    })
    print(json.dumps({"name": args.name, "rows": len(predictions), "wall_s": round(time.time() - started, 1)},
                     indent=1), flush=True)


if __name__ == "__main__":
    main()
