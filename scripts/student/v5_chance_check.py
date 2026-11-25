#!/usr/bin/env python
"""V5 chance check, computed in the main environment with the repository's metric code.

Reads ``outputs/student_v1/V5/chance_predictions.json`` (the untrained Unsloth head's answers on
seen-template v0.2 dev items, written by ``scripts/student/v5_unsloth_validate.py``) and reports
accuracy, macro accuracy (``meddecide.eval.metrics``), the expected chance accuracy (the mean of
1/k over the items' option counts) and the majority baseline. Writes
``outputs/student_v1/V5/chance_check.json``.

Score answers: the Unsloth decision API returns the level number; the MedDecide gold for score items is
the 1-based level, so an answer equal to the gold level counts as correct. Score items are reported
separately because the untrained head's score convention is not documented in the fetched docs.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from meddecide.eval.metrics import accuracy, macro_accuracy  # noqa: E402

PRED = ROOT / "outputs" / "student_v1" / "V5" / "chance_predictions.json"
OUT = ROOT / "outputs" / "student_v1" / "V5" / "chance_check.json"


def predicted_key(row: dict) -> str | None:
    raw = row.get("answer_raw")
    if row["qtype"] == "noul":
        return "yes" if raw in (True, "true", "True", 1) else "no"
    if row["qtype"] == "choice":
        return str(raw) if raw is not None else None
    if row["qtype"] == "score":
        return str(raw) if raw is not None else None
    return None


def main() -> None:
    rows = [r for r in json.loads(PRED.read_text(encoding="utf-8")) if not r.get("refused")]
    usable = [r for r in rows if predicted_key(r) is not None]
    result = {"kind": "v5_chance_check", "predictions": str(PRED.relative_to(ROOT)),
              "items_read": len(rows), "items_scored": len(usable), "by_qtype": {}}
    for qtype in sorted({r["qtype"] for r in usable}):
        group = [r for r in usable if r["qtype"] == qtype]
        pred = [predicted_key(r) for r in group]
        gold = [str(r["gold"]) for r in group]
        acc = accuracy(pred, gold)
        chance = sum(1.0 / len(r["option_keys"]) for r in group) / len(group)
        result["by_qtype"][qtype] = {
            "n": len(group),
            "accuracy": acc.micro,
            "macro_accuracy": macro_accuracy(pred, gold),
            "expected_chance": chance,
            "majority_baseline": acc.majority_baseline,
            "predicted_distribution": dict(Counter(pred)),
        }
    pred_all = [predicted_key(r) for r in usable]
    gold_all = [str(r["gold"]) for r in usable]
    overall = accuracy(pred_all, gold_all)
    result["overall"] = {"n": len(usable), "accuracy": overall.micro,
                         "macro_accuracy": macro_accuracy(pred_all, gold_all),
                         "expected_chance": sum(1.0 / len(r["option_keys"]) for r in usable) / len(usable),
                         "majority_baseline": overall.majority_baseline}
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["overall"], indent=2))


if __name__ == "__main__":
    main()
