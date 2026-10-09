#!/usr/bin/env python
"""Regression gate: the pointer path after the letter-readout change reproduces V3's predictions.

Scores 300 seen dev items with the step-1,000 checkpoint in bf16, one item per batch, no padding (the V3
path), and compares the probabilities with ``outputs/student_v1/V3/preds/step_1000.jsonl``. Writes
``outputs/student_v1/V6/regression_check.json``. Passes if the largest difference is below 1e-5.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from meddecide.model.meddecide_model import MedDecideModel  # noqa: E402
from meddecide.train.data import read_items  # noqa: E402

CKPT = ROOT / "outputs" / "student_v0" / "S9_run3" / "checkpoints" / "step_1000"
REF = ROOT / "outputs" / "student_v1" / "V3" / "preds" / "step_1000.jsonl"
BENCH = ROOT / "data" / "bench" / "v0.2"
OUT = ROOT / "outputs" / "student_v1" / "V6" / "regression_check.json"


def main() -> None:
    ref = {}
    for line in REF.open(encoding="utf-8"):
        row = json.loads(line)
        if row["group"] == "seen":
            ref[row["item_id"]] = np.asarray(row["option_probs"])
    ids = sorted(ref)[:300]
    want = set(ids)
    found = {}
    for folder in ("tier1", "fresh"):
        for path in sorted((BENCH / folder).glob("*.jsonl")):
            for item in read_items(path):
                if item.item_id in want:
                    found[item.item_id] = item
    items = [found[i] for i in ids]
    model = MedDecideModel.load(CKPT, device="cuda:0", dtype="bfloat16")
    scored = model.score_items(items, batch_size=1, round_to_chunk=False)
    diffs = []
    for item, probs in zip(items, scored.probs, strict=True):
        diffs.append(float(np.max(np.abs(np.asarray(probs) - ref[item.item_id]))))
    worst = max(diffs)
    result = {
        "kind": "pointer_regression_check",
        "checkpoint": str(CKPT.relative_to(ROOT)),
        "reference": str(REF.relative_to(ROOT)),
        "items": len(items),
        "max_abs_prob_diff": worst,
        "mean_abs_prob_diff": float(np.mean(diffs)),
        "gate": "PASS" if worst < 1e-5 else "FAIL",
        "gate_rule": "max |Δ probability| < 1e-5 against the V3 predictions (same path, same checkpoint)",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("items", "max_abs_prob_diff", "gate")}))


if __name__ == "__main__":
    main()
