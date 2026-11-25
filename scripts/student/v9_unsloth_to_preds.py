#!/usr/bin/env python
"""Convert Unsloth raw answers (v8_finalize.py / v8_unsloth_predict.py) to run_student-format predictions.

Each raw line carries Unsloth's probabilities for one item; they are ordered by the item's option keys
(``meddecide.eval.unsloth_predictions``) and written with the same fields run_student writes, so g1.py
reads the file unchanged. Usage:
  uv run python scripts/student/v9_unsloth_to_preds.py --raw <raw.jsonl> --items <items.jsonl> \
      --model-id meddecide-v1-arm-c --split test --out <preds.jsonl>
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from meddecide.eval.unsloth_predictions import option_probabilities, prediction_row  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--items", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    items = {}
    for line in args.items.open(encoding="utf-8"):
        row = json.loads(line)
        items[row["item_id"]] = row
    run_id = f"{args.model_id}:{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with args.raw.open(encoding="utf-8") as src, args.out.open("w", encoding="utf-8") as dst:
        for line in src:
            raw = json.loads(line)
            item = items[raw["item_id"]]
            keys = [str(o["key"]) for o in item["options"]]
            probs = option_probabilities(item["qtype"], raw["probabilities"], keys)
            row = prediction_row(item, probs, model_id=args.model_id, run_id=run_id, split=args.split,
                                 latency_s=raw.get("latency_s"))
            dst.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1
    print(json.dumps({"rows": written, "out": str(args.out)}))


if __name__ == "__main__":
    main()
