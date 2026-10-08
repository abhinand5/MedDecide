#!/usr/bin/env python
"""Fresh-process re-derivation of the V2 padding-effect point estimates (R6).

Plain Python: no numpy and no ``meddecide`` import, so it cannot share a bug with
``v2_effect.py``. Re-reads the three prediction files and checks every point estimate in
``effect.json``. Run as ``python -I scripts/student/v2_rederive_effect.py``. Exits 1 on any mismatch.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EFFECT = ROOT / "outputs" / "student_v1" / "V2" / "effect.json"
FILES = {
    "old": ROOT / "outputs" / "student_v0" / "S9_run3" / "preds_meddecide-0p8b-lora-pointer__test.jsonl",
    "fixed16": ROOT / "outputs" / "student_v1" / "V2" / "fixed_b16" / "preds_meddecide-0p8b-lora-pointer__test.jsonl",
    "fixed1": ROOT / "outputs" / "student_v1" / "V2" / "fixed_b1" / "preds_meddecide-0p8b-lora-pointer__test.jsonl",
    "unpadded1": ROOT / "outputs" / "student_v1" / "V2" / "unpadded_b1" / "preds_meddecide-0p8b-lora-pointer__test.jsonl",
}
COMPARISONS = [("unpadded1", "old"), ("fixed16", "old"), ("unpadded1", "fixed16"), ("fixed1", "unpadded1")]
TOLERANCE = 1e-9


def load(path: Path) -> dict[str, dict]:
    rows = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                rows[row["item_id"]] = row
    return rows


def per_item(rows: dict[str, dict]) -> dict[str, tuple[int, float, str]]:
    """item_id -> (correct, brier, template) using only the stored fields and plain arithmetic."""
    out = {}
    for item_id, row in rows.items():
        keys = row["option_keys"]
        gold = keys.index(row["gold_key"])
        brier = sum((p - (1.0 if k == gold else 0.0)) ** 2 for k, p in enumerate(row["option_probs"]))
        out[item_id] = (1 if row["correct"] else 0, brier, row["template_id"])
    return out


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def main() -> int:
    effect = json.loads(EFFECT.read_text(encoding="utf-8"))
    rows = {name: load(path) for name, path in FILES.items()}
    items = {name: per_item(r) for name, r in rows.items()}
    failures = []
    lines = []

    def check(label: str, got: float, expected: float) -> None:
        ok = math.isfinite(got) and abs(got - expected) <= TOLERANCE
        lines.append(f"{'MATCH   ' if ok else 'MISMATCH'} {label}: fresh={got!r} effect.json={expected!r}")
        if not ok:
            failures.append(label)

    for name, path in FILES.items():
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != effect["inputs"][name]["sha256"]:
            failures.append(f"sha256 {name}")
        lines.append(f"input {name}: rows={len(rows[name])} sha256={digest[:16]}…")
        scoring = effect["scorings"][name]
        check(f"{name} n", float(len(rows[name])), float(scoring["n"]))
        check(f"{name} micro_accuracy", mean([c for c, _, _ in items[name].values()]), scoring["micro_accuracy"])
        check(f"{name} micro_brier", mean([b for _, b, _ in items[name].values()]), scoring["micro_brier"])
        by_t: dict[str, list[int]] = defaultdict(list)
        for c, _, t in items[name].values():
            by_t[t].append(c)
        check(f"{name} macro_accuracy", mean([mean(v) for v in by_t.values()]), scoring["macro_accuracy"])

    for a, b in COMPARISONS:
        key = f"{a}-{b}"
        comp = effect["comparisons"][key]
        ia, ib = items[a], items[b]
        if ia.keys() != ib.keys():
            failures.append(f"item ids differ for {key}")
        ids = sorted(ia)
        check(f"{key} micro_accuracy_diff", mean([ia[i][0] - ib[i][0] for i in ids]),
              comp["micro_accuracy_diff"]["point"])
        check(f"{key} micro_brier_diff", mean([ia[i][1] - ib[i][1] for i in ids]),
              comp["micro_brier_diff"]["point"])
        groups: dict[str, list[str]] = defaultdict(list)
        for i in ids:
            groups[ia[i][2]].append(i)
        macro = mean([mean([ia[i][0] - ib[i][0] for i in members]) for members in groups.values()])
        check(f"{key} macro_accuracy_diff", macro, comp["macro_accuracy_diff"]["point"])
        flips = sum(1 for i in ids if rows[a][i]["argmax_key"] != rows[b][i]["argmax_key"])
        check(f"{key} argmax_changed_total", float(flips), float(comp["argmax_changed_total"]))
        for template, stats in comp["per_template"].items():
            members = groups[template]
            check(f"{key} {template} accuracy_diff",
                  mean([ia[i][0] - ib[i][0] for i in members]), stats["accuracy_diff"]["point"])

    lines.append(f"checked {len(lines)} lines; failures: {len(failures)}")
    print("\n".join(lines))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
