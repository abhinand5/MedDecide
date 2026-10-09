#!/usr/bin/env python
"""Convert F7 JEV-9B rows (``preds_jev9b.jsonl``) into the harness prediction format for G1.

Every F7 row is converted or named in the summary; nothing is dropped silently. Only items found
in the given benchmark directories are converted, and the summary counts any row without an item.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from meddecide.bench.schema import Item
from meddecide.eval.jev_harness import harness_row
from meddecide.utils.io import read_jsonl

MODEL_ID = "autotrust/JEV-9B"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--f7", type=Path, required=True, help="preds_jev9b.jsonl from F7")
    parser.add_argument("--bench-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--run-id", default="jev9b_v02_test")
    parser.add_argument("--split", default="test")
    args = parser.parse_args()

    items: dict[str, Item] = {}
    for directory in args.bench_dirs:
        for path in sorted(directory.glob("*.jsonl")):
            loaded, validation = read_jsonl(path, Item)
            validation.check_closes()
            for item in loaded:
                if str(item.split) == args.split:
                    items[item.item_id] = item

    f7_rows = [json.loads(line) for line in args.f7.read_text(encoding="utf-8").splitlines() if line]
    converted: list[dict] = []
    unmatched: Counter[str] = Counter()
    seen: set[str] = set()
    for row in f7_rows:
        item = items.get(row["item_id"])
        if item is None:
            unmatched["item not in the given benchmark directories at this split"] += 1
            continue
        if row["item_id"] in seen:
            raise SystemExit(f"duplicate F7 row for item {row['item_id']}")
        seen.add(row["item_id"])
        converted.append(harness_row(row, item, run_id=args.run_id, model_id=MODEL_ID))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for out in converted:
            fh.write(json.dumps(out, ensure_ascii=False) + "\n")
    per_template = Counter(r["template_id"] for r in converted)
    summary = {
        "f7_rows": len(f7_rows),
        "converted": len(converted),
        "unmatched": dict(unmatched),
        "split_items_in_dirs": len(items),
        "per_template": dict(sorted(per_template.items())),
        "out": str(args.out),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
