#!/usr/bin/env python
"""Fresh-process re-derivation of the G1 point estimates (R6 check, stdlib only).

Reads the raw prediction files and item files directly and prints the seen / held-out macro
accuracy and mean multi-class Brier for MedDecide, zero-shot and JEV-9B, on the item sets that
``scripts/bench/g1.py`` uses. It does not import ``meddecide`` or ``g1``, so it cannot share a bug
with them. Aggregates only; no item text is printed.
"""

from __future__ import annotations

import glob
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HELDOUT = {"ct_phase_choice_v1", "fda_boxed_warning_noul_v1", "pubmed_humans_noul_v1", "ct_arm_role_noul_v1"}
FRESH = {
    "ct_arm_role_noul_v1", "ct_claim_set_choice_v1", "ct_healthy_volunteers_noul_v1", "ct_phase_choice_v1",
    "ct_randomised_noul_v1", "fda_boxed_warning_noul_v1", "fda_class_choice_v2", "fda_route_claim_noul_v1",
    "pubmed_humans_noul_v1", "pubmed_mesh_major_choice_v2", "pubmed_observational_noul_v1",
}
MEDDECIDE = "outputs/student_v0/S9_run3/preds_meddecide-0p8b-lora-pointer__test.jsonl"
ZEROSHOT = "outputs/student_v0/S11/preds_qwen3p5-0p8b.jsonl"
JEV9B = "outputs/student_v1/V1/preds/preds_jev9b__test.jsonl"


def load_rows(relative: str) -> dict[str, dict]:
    rows = {}
    for line in (ROOT / relative).read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        rows[row["item_id"]] = row
    return rows


def multiclass_brier(row: dict) -> float:
    gold = row["option_keys"].index(row["gold_key"])
    return sum((p - (1.0 if k == gold else 0.0)) ** 2 for k, p in enumerate(row["option_probs"]))


def summarise(rows: dict[str, dict], ids: list[str]) -> dict:
    per_template: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0])
    for item_id in ids:
        row = rows[item_id]
        bucket = per_template[row["template_id"]]
        bucket[0] += 1
        bucket[1] += bool(row["correct"])
        bucket[2] += multiclass_brier(row)
    macro = sum(b[1] / b[0] for b in per_template.values()) / len(per_template)
    brier = sum(b[2] for b in per_template.values()) / sum(b[0] for b in per_template.values())
    return {"items": len(ids), "templates": len(per_template),
            "macro_accuracy": round(macro, 4), "mean_brier": round(brier, 4)}


def main() -> None:
    fresh_test: dict[str, str] = {}
    for path in sorted(glob.glob(str(ROOT / "data/bench/v0.2/fresh/*.jsonl"))):
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            item = json.loads(line)
            if item["split"] == "test" and item["template_id"] in FRESH:
                fresh_test[item["item_id"]] = item["template_id"]
    seen = [i for i, t in fresh_test.items() if t not in HELDOUT]
    held = [i for i, t in fresh_test.items() if t in HELDOUT]
    print(f"fresh test items {len(fresh_test)}  seen {len(seen)}  held-out {len(held)}")

    md, zs, jev = load_rows(MEDDECIDE), load_rows(ZEROSHOT), load_rows(JEV9B)
    for label, ids in (("seen", seen), ("heldout", held)):
        print(label, "meddecide", summarise(md, ids))
        print(label, "zeroshot ", summarise(zs, ids))
        common = [i for i in ids if i in jev]
        print(label, "jev9b intersection size", len(common), "missing", len(ids) - len(common))
        print(label, "meddecide on intersection", summarise(md, common))
        print(label, "zeroshot on intersection ", summarise(zs, common))
        print(label, "jev9b on intersection    ", summarise(jev, common))

    dropped = [i for i in seen if i not in jev]
    kept = [i for i in seen if i in jev]

    def accuracy(rows: dict[str, dict], ids: list[str]) -> float:
        return round(sum(bool(rows[i]["correct"]) for i in ids) / len(ids), 4)

    print("additional: meddecide item accuracy on jev-dropped seen items", accuracy(md, dropped), len(dropped))
    print("additional: meddecide item accuracy on jev-kept seen items   ", accuracy(md, kept), len(kept))
    print("additional: jev9b item accuracy on jev-kept seen items       ", accuracy(jev, kept), len(kept))


if __name__ == "__main__":
    main()
