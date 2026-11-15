#!/usr/bin/env python
"""osler_v7 T11 — Gate v7 for the 63,304-step 9B final against the d3-flash zero-shot base (additional to T10).

The T10 gate measured the selected 13,992-step arm; this applies the same §7 checks to the final checkpoint using the
predictions written by `t10_score_d3.py --name 9b_final_*` and the already-scored `9b_base_*` predictions. Partial
coverage on any check keeps the verdict NOT MEASURED, as in T10.

Run (CPU): ``uv run --frozen python scripts/osler_v7/t11_gate.py``
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from meddecide.train.blocks import macro_difference, pair_by_item  # noqa: E402
from meddecide.utils.io import write_json  # noqa: E402
from meddecide.utils.provenance import git_commit, utcnow  # noqa: E402

T10 = ROOT / "outputs" / "osler_v7" / "T10"
OUT = ROOT / "outputs" / "osler_v7" / "T11"
THRESHOLDS = {"G-a": -0.01, "G-b": 0.0, "G-c": -0.02}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def compare(candidate: Path, base: Path) -> dict[str, Any]:
    if not candidate.is_file() or not base.is_file():
        return {"status": "NOT MEASURED", "candidate": str(candidate), "base": str(base)}
    cand_rows, base_rows = read_jsonl(candidate), read_jsonl(base)
    cand_ids = {row["item_id"] for row in cand_rows}
    base_ids = {row["item_id"] for row in base_rows}
    status = "MEASURED" if cand_ids == base_ids else "PARTIAL"
    if status == "PARTIAL":
        cand_rows = [row for row in cand_rows if row["item_id"] in base_ids]
        base_rows = [row for row in base_rows if row["item_id"] in cand_ids]
    diff = macro_difference(pair_by_item(cand_rows, base_rows)).as_dict()
    return {"status": status, "items": diff["n"], "point": round(diff["point"], 8), "ci_lo": round(diff["ci_lo"], 8),
            "ci_hi": round(diff["ci_hi"], 8),
            "coverage": {"candidate_rows": len(cand_ids), "base_rows": len(base_ids),
                         "only_candidate": len(cand_ids - base_ids), "only_base": len(base_ids - cand_ids)}}


def main() -> None:
    checks: dict[str, Any] = {
        "G-a": compare(T10 / "9b_final_general" / "predictions.jsonl", T10 / "9b_base_general" / "predictions.jsonl"),
    }
    mr = compare(T10 / "9b_final_mr_verdict" / "predictions.jsonl", T10 / "9b_base_mr_verdict" / "predictions.jsonl")
    mimic = compare(T10 / "9b_final_mimic_suite" / "predictions.jsonl", T10 / "9b_base_mimic_suite" / "predictions.jsonl")
    if mr["status"] == "MEASURED" and mimic["status"] == "MEASURED":
        pairs = pair_by_item(read_jsonl(T10 / "9b_final_mr_verdict" / "predictions.jsonl"),
                             read_jsonl(T10 / "9b_base_mr_verdict" / "predictions.jsonl")) + \
            pair_by_item(read_jsonl(T10 / "9b_final_mimic_suite" / "predictions.jsonl"),
                         read_jsonl(T10 / "9b_base_mimic_suite" / "predictions.jsonl"))
        diff = macro_difference(pairs).as_dict()
        checks["G-b"] = {"status": "MEASURED", "items": diff["n"], "point": round(diff["point"], 8),
                         "ci_lo": round(diff["ci_lo"], 8), "ci_hi": round(diff["ci_hi"], 8)}
    else:
        checks["G-b"] = {"status": "NOT MEASURED"}
    checks["G-c"] = compare(T10 / "9b_final_knowledge" / "predictions.jsonl", T10 / "9b_base_knowledge" / "predictions.jsonl")
    for name, check in checks.items():
        if check["status"] in ("MEASURED", "PARTIAL"):
            check["threshold_lower_bound"] = THRESHOLDS[name]
            check["passes"] = bool(check["ci_lo"] > THRESHOLDS[name])
    verdict = "NOT MEASURED"
    if all(check["status"] == "MEASURED" for check in checks.values()):
        verdict = "PASS" if all(check["passes"] for check in checks.values()) else "FAIL"
    write_json(OUT / "gate_final.json", {
        "generated_utc": utcnow(), "git_commit": git_commit(ROOT),
        "command": "uv run --frozen python scripts/osler_v7/t11_gate.py",
        "model": "Osler-9B final at 63,304 steps (T11) vs the d3-flash zero-shot base",
        "checks": checks, "verdict": verdict,
    })
    print(json.dumps({"verdict": verdict}, indent=1))
    print({name: (check.get("ci_lo"), check.get("passes")) for name, check in checks.items()})


if __name__ == "__main__":
    main()
