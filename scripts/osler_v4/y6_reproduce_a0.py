"""Y6 (osler_v4 ADVISORY Y6): the A0 re-score against osler_v3's recorded A0 predictions (CPU).

A0 is osler_v3's checkpoint, scored again by scripts/osler_v4/y6_stage_a.py score. This script checks the new scores against the
predictions osler_v3 wrote for the same checkpoint, on the items the two sets share: the decisions (the predicted key), the
correct flags, the accuracy on the shared items, and the probability differences (the largest, and the share above 0.01 and
0.05 per question type). Item text is never printed; only counts and differences.

Usage (CPU; the same check applies to each re-scored arm whose osler_v3 record exists, A0, A1 or A3):
    uv run --frozen python scripts/osler_v4/y6_reproduce_a0.py [--arm A0|A1|A3]
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from meddecide.utils.io import iter_jsonl

PAIRS = {"cg": ("cg_predictions.jsonl", "cg_v2_predictions.jsonl"), "dev": ("dev_predictions.jsonl", "fastdev_v4_predictions.jsonl")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--arm", default="A0", choices=("A0", "A1", "A3"))
    args = parser.parse_args()
    new_dir = Path("outputs/osler_v4/Y6") / args.arm
    old_dir = Path("outputs/osler_v3/stage_a") / args.arm
    print(f"{args.arm}: osler_v3 record {old_dir} against the re-score {new_dir}")
    for name, (old_file, new_file) in PAIRS.items():
        old = {str(r["item_id"]): r for r in iter_jsonl(old_dir / old_file)}
        new = {str(r["item_id"]): r for r in iter_jsonl(new_dir / new_file)}
        shared = [k for k in old if k in new]
        same_key = sum(old[k]["predicted_key"] == new[k]["predicted_key"] for k in shared)
        same_flag = sum(bool(old[k]["correct"]) == bool(new[k]["correct"]) for k in shared)
        acc_old = sum(bool(old[k]["correct"]) for k in shared) / len(shared)
        acc_new = sum(bool(new[k]["correct"]) for k in shared) / len(shared)
        diffs: dict[str, list[float]] = defaultdict(list)
        for k in shared:
            d = max(abs(a - b) for a, b in zip(old[k]["probs"], new[k]["probs"], strict=True))
            diffs[str(old[k]["qtype"])].append(d)
        largest = max(max(v) for v in diffs.values())
        print(f"{name}: osler_v3 items {len(old)}; re-scored items {len(new)}; shared {len(shared)}; "
              f"same predicted key {same_key}; same correct flag {same_flag}; accuracy on shared items: osler_v3 {acc_old:.4f}, "
              f"re-score {acc_new:.4f}; largest probability difference {largest:.3f}")
        for qtype, ds in sorted(diffs.items()):
            n = len(ds)
            print(f"   {qtype}: items {n}; share with difference above 0.01 {sum(d > 0.01 for d in ds) / n:.4f}; "
                  f"above 0.05 {sum(d > 0.05 for d in ds)} items; largest {max(ds):.3f}")


if __name__ == "__main__":
    main()
