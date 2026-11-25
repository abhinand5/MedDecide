#!/usr/bin/env python
"""Write the v0.2 test items that run_student.py scores, as one JSONL, for the Unsloth evaluator (V9).

Uses run_student's own loader and screen filter, so the item set is the one every other arm was scored on
(tier 1 + fresh, screen-kept templates). The count is asserted against the stored predictions file of
student_v0's run. Output: outputs/student_v1/V9/test_items.jsonl (gitignored; items are benchmark data).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "bench"))

import run_student as rs  # noqa: E402

OUT = ROOT / "outputs" / "student_v1" / "V9" / "test_items.jsonl"
REFERENCE = ROOT / "outputs" / "student_v0" / "S9_run3" / "preds_meddecide-0p8b-lora-pointer__test.jsonl"


def main() -> None:
    keep = rs.load_keep_set(rs.DEFAULT_SCREEN)
    bench = rs.load_benchmark(rs.DEFAULT_TIER1, rs.DEFAULT_FRESH, keep, split="test")
    items = bench["tier1"] + bench["fresh"]
    reference_ids = {json.loads(line)["item_id"] for line in REFERENCE.open(encoding="utf-8")}
    ids = {item.item_id for item in items}
    if ids != reference_ids:
        raise SystemExit(f"item set differs from the reference predictions: {len(ids)} vs {len(reference_ids)}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as handle:
        for item in sorted(items, key=lambda i: i.item_id):
            handle.write(item.model_dump_json() + "\n")
    print(json.dumps({"items": len(items), "tier1": len(bench["tier1"]), "fresh": len(bench["fresh"]),
                      "output": str(OUT.relative_to(ROOT))}))


if __name__ == "__main__":
    main()
