#!/usr/bin/env python
"""Expectation baselines on the v0.2 test items (V9 self-audit, R6): chance and the per-template majority.

Uniform-guess expected accuracy is the mean of 1/(number of options) over the items. The per-template majority
baseline predicts the most common gold answer of each template (the test golds are used only for this reference
number, not for any fit). Writes outputs/student_v1/V9/expectation_baselines.json.

Run: ``uv run python scripts/student/v9_baselines.py``
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ITEMS = ROOT / "outputs" / "student_v1" / "V9" / "test_items.jsonl"
OUT = ROOT / "outputs" / "student_v1" / "V9" / "expectation_baselines.json"


def main() -> None:
    rows = [json.loads(line) for line in ITEMS.open(encoding="utf-8")]
    n = len(rows)
    uniform = sum(1.0 / len(r["options"]) for r in rows) / n
    by_template: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_template[r["template_id"]].append(r)
    majority_correct = 0
    per_template = {}
    for template, items in sorted(by_template.items()):
        best = Counter(str(r["gold"]) for r in items).most_common(1)[0][1]
        majority_correct += best
        per_template[template] = {"items": len(items), "majority_accuracy": best / len(items)}
    result = {"kind": "v9_expectation_baselines", "items": n,
              "uniform_guess_expected_accuracy": uniform,
              "per_template_majority_accuracy_micro": majority_correct / n,
              "per_template": per_template}
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"items": n, "uniform": round(uniform, 4),
                      "majority_micro": round(majority_correct / n, 4)}))


if __name__ == "__main__":
    main()
