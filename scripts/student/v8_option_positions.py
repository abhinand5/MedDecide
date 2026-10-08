#!/usr/bin/env python
"""Where the gold option sits in the file's option order, per qtype and option count (V8 check, CPU).

Arms A and B permute option order for every item they see (``shuffle_options``, one permutation per
(seed, epoch, item)). Arm C's Unsloth rows keep the file's option order. This counts the gold position in
that file order, so the effect of the difference can be judged: if the gold position is uniform, the file
order is already a random permutation, and a run that sees each item once (about a quarter of the mix)
gains nothing from re-permuting. Reads the train file only; no dev or test split. Writes
``outputs/student_v1/V8/option_positions.json`` (aggregate counts, no item text).

Run: ``uv run python scripts/student/v8_option_positions.py``
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from meddecide.train.batch_order import permute_options  # noqa: E402

TRAIN = ROOT / "data" / "train" / "student_v1" / "train.jsonl"
OUT = ROOT / "outputs" / "student_v1" / "V8" / "option_positions.json"
QTYPES = ("choice", "noul")
ORDER_SEED = 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=TRAIN)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--augment", action="store_true",
                        help="count after the arms A/B option-order augmentation (seed 0, epoch 0)")
    args = parser.parse_args()
    out = args.out or (OUT.with_name("option_positions_augmented.json") if args.augment else OUT)
    with args.train.open(encoding="utf-8") as handle:
        rows_in = [json.loads(line) for line in handle]
    changed = None
    if args.augment:
        rows_in, changed = permute_options(rows_in, seed=ORDER_SEED, epoch=0)
    counts: dict[tuple[str, int], Counter] = defaultdict(Counter)
    skipped: Counter = Counter()
    rows = len(rows_in)
    for row in rows_in:
        if row["qtype"] not in QTYPES:
            skipped[f"qtype:{row['qtype']}"] += 1
            continue
        keys = [o["key"] for o in row["options"]]
        if str(row["gold"]) not in keys:
            skipped["gold_not_in_options"] += 1
            continue
        counts[(row["qtype"], len(keys))][keys.index(str(row["gold"]))] += 1
    groups = []
    for (qtype, n), position in sorted(counts.items()):
        total = sum(position.values())
        expected = 1 / n
        se = math.sqrt(expected * (1 - expected) / total)
        proportions = [position[k] / total for k in range(n)]
        chi2 = sum((position[k] - total * expected) ** 2 / (total * expected) for k in range(n))
        groups.append({
            "qtype": qtype, "options": n, "items": total,
            "gold_position_counts": [position[k] for k in range(n)],
            "gold_position_proportions": proportions,
            "expected_per_position": expected,
            "max_abs_z_vs_uniform": max(abs(p - expected) / se for p in proportions),
            "chi2": chi2, "df": n - 1,
        })
    result = {"augmented": args.augment, "augmented_items_changed": changed, "rows_read": rows,
              "items_counted": sum(g["items"] for g in groups), "skipped": dict(skipped), "groups": groups}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for g in groups:
        print(f"{g['qtype']:>7} n={g['options']:<2} items={g['items']:>7} "
              f"max|z|={g['max_abs_z_vs_uniform']:.2f} chi2={g['chi2']:.1f} df={g['df']}")
    print(json.dumps({"rows_read": rows, "items_counted": result["items_counted"], "skipped": result["skipped"],
                      "augmented_items_changed": changed}))


if __name__ == "__main__":
    main()
