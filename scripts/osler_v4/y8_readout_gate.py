"""Y8 calibrated re-score checks (osler_v4 ADVISORY Y8; D12, D21; labelled additional; CPU).

For each set (fast dev v4 and CG v2), the calibrated re-score of the selected checkpoint (outputs/osler_v4/Y8/calibrated/<set>/
predictions.jsonl, written by scripts/osler_v4/y4_checkpoint_score.py with --arm-result) is gated template by template with
meddecide.eval.readout_gate.gate_by_template. Its gate is compared with the gate of the run's end-of-run predictions
(dev_predictions.jsonl and cg_v2_predictions.jsonl, at temperature 1), and the count of argmax changes is written. Calibration
keeps every argmax, so the two gates must agree and the count must be 0. A failing cell is reported and is not a result (D12).
Writes outputs/osler_v4/Y8/readout_gate_calibrated.json.

Usage (CPU):
    CUDA_VISIBLE_DEVICES= uv run --frozen python scripts/osler_v4/y8_readout_gate.py
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from meddecide.eval.readout_gate import gate_by_template
from meddecide.train.blocks import KNOWLEDGE_SOURCES, MR_SOURCES
from meddecide.utils.io import iter_jsonl, write_json
from meddecide.utils.provenance import utcnow

OUT = Path("outputs/osler_v4/Y8")
SETS = {
    "fastdev_v4": ("dev_predictions.jsonl", "calibrated/fastdev_v4/predictions.jsonl"),
    "cg_v2": ("cg_v2_predictions.jsonl", "calibrated/cg_v2/predictions.jsonl"),
}


def partition(source: str) -> str:
    if source in MR_SOURCES:
        return "mr"
    return "knowledge" if source in KNOWLEDGE_SOURCES else "general"


def main() -> None:
    report: dict[str, object] = {"generated_utc": utcnow(), "sets": {}}
    sets: dict[str, object] = {}
    for name, (end_of_run, calibrated) in SETS.items():
        rows = list(iter_jsonl(OUT / end_of_run))
        cal_rows = list(iter_jsonl(OUT / calibrated))
        if len(rows) != len(cal_rows):
            raise SystemExit(f"{name}: {len(rows)} end-of-run rows but {len(cal_rows)} calibrated rows")
        changed = 0
        for row, cal in zip(rows, cal_rows, strict=True):
            if row["item_id"] != cal["item_id"]:
                raise SystemExit(f"{name}: the calibrated re-score is not in the same item order")
            changed += row["predicted_index"] != cal["predicted_index"]
        gates_run = gate_by_template(rows)
        gates_cal = gate_by_template(cal_rows)
        disagree = sorted(t for t in gates_run if gates_run[t].status != gates_cal[t].status)
        parts: dict[str, str] = {}
        for row in cal_rows:
            parts.setdefault(str(row["template_id"]), partition(str(row["source"])))
        cells = {t: {"partition": parts[t], "items": g.items, "accuracy": round(g.accuracy, 4), "status": g.status}
                 for t, g in gates_cal.items()}
        failing = sorted(t for t, g in gates_cal.items() if not g.passes)
        sets[name] = {"items": len(cal_rows), "templates": len(gates_cal), "failing": failing,
                      "argmax_changed_by_calibration": changed, "gate_disagreements": disagree,
                      "partitions": dict(sorted(Counter(parts.values()).items())), "cells": cells}
        print(f"{name}: items {len(cal_rows)}; templates {len(gates_cal)}; failing cells {len(failing)}"
              + (f" ({', '.join(failing)})" if failing else "") + f"; argmax changed {changed}; gate disagreements {len(disagree)}",
              flush=True)
    report["sets"] = sets
    write_json(OUT / "readout_gate_calibrated.json", report)


if __name__ == "__main__":
    main()
