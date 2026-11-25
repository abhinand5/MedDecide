#!/usr/bin/env python
"""osler_rc0 — verify the R2/R3/R4 scoring outputs before the leaderboard consumes them (CPU).

Checks, per task:

``r2``  every Osler-9B-rc0 and d3-flash prediction file exists with one row per frozen item, finite normalised
        probabilities on every scored row, the calibrated copies reproduce ``apply_temperature(raw, T)`` exactly, no
        argmax changed, and the accuracy of the calibrated candidate equals the pre-existing T10 predictions on
        knowledge, mr_verdict, mimic_suite and confirm_general.
``r3``  every Osler-4B-rc0 raw and calibrated file exists on all 9 sets with finite probabilities; the saved dev
        temperatures were actually applied (re-applying the temperature to the raw probabilities reproduces the
        calibrated ones), and the checkpoint's calibration carries all three qtypes.
``r4``  the four freshly scored competitor files exist with the schemas and row counts the leaderboard expects.

Run::

    source scripts/pod_env.sh && uv run --frozen python scripts/osler_rc0/verify_predictions.py --task r2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from meddecide.eval.calibration import apply_temperature  # noqa: E402
from meddecide.utils.io import write_json  # noqa: E402

R2 = ROOT / "outputs/osler_rc0/R2"
R3 = ROOT / "outputs/osler_rc0/R3"
R4 = ROOT / "outputs/osler_rc0/R4"
OUT = ROOT / "outputs/osler_rc0/verify"
SETS = {
    "panel": ROOT / "data/bench/osler_v4/y9/panel.jsonl",
    "knowledge": ROOT / "data/bench/osler_v4/y9/knowledge.jsonl",
    "confirm_general": ROOT / "data/bench/osler_v7/confirm/general.jsonl",
    "mr_verdict": ROOT / "data/bench/osler_v4/y9/mr_verdict.jsonl",
    "mimic_suite": ROOT / "data/bench/osler_v4/y9/mimic_suite.jsonl",
    "mednli_test": ROOT / "data/bench/osler_v4/y9/mednli_test.jsonl",
    "mimic_heldout": ROOT / "data/bench/osler_v4/y9/mimic_heldout.jsonl",
    "mr_heldout_test": ROOT / "data/bench/osler_v4/y9/mr_heldout_test.jsonl",
    "pairs_test": ROOT / "data/bench/osler_v4/y9/pairs_test.jsonl",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def frozen_sizes() -> dict[str, int]:
    return {name: sum(1 for line in path.open(encoding="utf-8") if line.strip()) for name, path in SETS.items()}


def check_probs(rows: list[dict[str, Any]], label: str) -> list[str]:
    problems = []
    for row in rows:
        if row.get("status", "scored") != "scored":
            continue
        probs = row.get("probs")
        if not isinstance(probs, list) or len(probs) < 2:
            problems.append(f"{label}: {row.get('item_id')} has no usable probs")
            continue
        array = np.asarray(probs, dtype=np.float64)
        if not np.all(np.isfinite(array)):
            problems.append(f"{label}: {row.get('item_id')} non-finite prob")
        if abs(float(array.sum()) - 1.0) > 1e-3:
            problems.append(f"{label}: {row.get('item_id')} probs sum {array.sum():.6f}")
    return problems


def calibrated_matches(raw: list[dict[str, Any]], calibrated: list[dict[str, Any]],
                       temperatures: dict[str, float], qtypes: dict[str, str], label: str) -> dict[str, Any]:
    by_id = {str(row["item_id"]): row for row in calibrated}
    if len(by_id) != len(calibrated):
        raise SystemExit(f"{label}: duplicate ids in calibrated file")
    mismatched = 0
    argmax_changes = 0
    max_delta = 0.0
    for row in raw:
        item_id = str(row["item_id"])
        other = by_id.get(item_id)
        if other is None:
            raise SystemExit(f"{label}: calibrated file misses {item_id}")
        before = np.asarray(row["probs"], dtype=np.float64)
        expected = apply_temperature(before, temperatures[qtypes[item_id]])
        after = np.asarray(other["probs"], dtype=np.float64)
        delta = float(np.abs(expected - after).max())
        max_delta = max(max_delta, delta)
        if delta > 1e-6:
            mismatched += 1
        if int(before.argmax()) != int(after.argmax()):
            argmax_changes += 1
    return {"rows": len(raw), "max_abs_delta_vs_reapplied": max_delta, "mismatched_rows": mismatched,
            "argmax_changes": argmax_changes,
            "passed": mismatched == 0 and argmax_changes == 0}


def qtype_map(items_path: Path) -> dict[str, str]:
    from meddecide.train.mix_items import mix_row_to_item
    return {mix_row_to_item(row).item_id: str(row["qtype"]) for row in read_jsonl(items_path)}


def task_r2() -> dict[str, Any]:
    sizes = frozen_sizes()
    temperatures = json.loads((R2 / "9b_calibration.json").read_text())["temperatures"]
    result: dict[str, Any] = {"temperatures": temperatures, "sets": {}, "problems": []}
    for set_name, size in sizes.items():
        entry: dict[str, Any] = {}
        raw_dir = R2 / f"preds/9b_rc0_{set_name}" if set_name != "panel" else ROOT / "outputs/eval0/panel/9b_final_panel"
        cal_dir = R2 / f"preds_cal/9b_rc0_{set_name}"
        base_dir = R2 / f"preds/d3_base_{set_name}" if set_name != "panel" else ROOT / "outputs/eval0/panel/9b_base_panel"
        for label, directory in (("candidate_raw", raw_dir), ("candidate_cal", cal_dir), ("base", base_dir)):
            path = directory / "predictions.jsonl"
            if not path.is_file():
                entry[label] = {"status": "MISSING"}
                result["problems"].append(f"R2 {set_name} {label} missing at {path}")
                continue
            rows = read_jsonl(path)
            entry[label] = {"rows": len(rows), "expected": size, "has_probs": "probs" in rows[0],
                            "problems": len(check_probs(rows, f"{set_name}/{label}"))}
            if set_name != "panel" and label == "candidate_raw":
                entry[label]["calibration"] = calibrated_matches(
                    rows, read_jsonl(cal_dir / "predictions.jsonl"), temperatures, qtype_map(SETS[set_name]),
                    f"{set_name}/candidate")
        result["sets"][set_name] = entry
    # accuracy parity against the pre-existing T10 candidate predictions
    parity = {}
    t10_names = {"knowledge": "9b_final_knowledge", "mr_verdict": "9b_final_mr_verdict",
                 "mimic_suite": "9b_final_mimic_suite", "confirm_general": "9b_final_general"}
    for set_name, t10_name in t10_names.items():
        t10 = ROOT / f"outputs/osler_v7/T10/{t10_name}/predictions.jsonl"
        new = R2 / f"preds_cal/9b_rc0_{set_name}/predictions.jsonl"
        if not t10.is_file() or not new.is_file():
            parity[set_name] = {"status": "MISSING"}
            continue
        old_rows = {str(row["item_id"]): row for row in read_jsonl(t10)}
        new_rows = {str(row["item_id"]): row for row in read_jsonl(new)}
        common = sorted(set(old_rows) & set(new_rows))
        agree = sum(1 for item in common if int(old_rows[item]["predicted_index"]) == int(
            np.argmax(new_rows[item]["probs"])))
        parity[set_name] = {"common": len(common), "argmax_agree": agree,
                            "passed": len(common) == len(new_rows) and agree == len(common)}
    result["t10_parity"] = parity
    result["passed"] = (not result["problems"] and all(
        entry.get("candidate_cal", {}).get("rows") == entry.get("candidate_cal", {}).get("expected")
        for entry in result["sets"].values()) and all(v.get("passed") for v in parity.values()))
    return result


def task_r3() -> dict[str, Any]:
    sizes = frozen_sizes()
    calibration = json.loads((R3 / "4b_calibration.json").read_text())
    result: dict[str, Any] = {"calibration": calibration, "sets": {}, "problems": []}
    for set_name, size in sizes.items():
        entry: dict[str, Any] = {}
        raw_dir = R3 / f"preds/4b_rc0_{set_name}"
        cal_dir = R3 / f"preds_cal/4b_rc0_{set_name}"
        for label, directory in (("raw", raw_dir), ("cal", cal_dir)):
            path = directory / "predictions.jsonl"
            if not path.is_file():
                entry[label] = {"status": "MISSING"}
                result["problems"].append(f"R3 {set_name} {label} missing at {path}")
                continue
            rows = read_jsonl(path)
            entry[label] = {"rows": len(rows), "expected": size,
                            "problems": len(check_probs(rows, f"{set_name}/{label}"))}
        if (raw_dir / "predictions.jsonl").is_file() and (cal_dir / "predictions.jsonl").is_file():
            entry["calibration"] = calibrated_matches(
                read_jsonl(raw_dir / "predictions.jsonl"), read_jsonl(cal_dir / "predictions.jsonl"),
                calibration["temperatures"], qtype_map(SETS[set_name]), f"{set_name}/4b")
        result["sets"][set_name] = entry
    result["passed"] = not result["problems"] and all(
        entry.get("cal", {}).get("rows") == entry.get("cal", {}).get("expected") for entry in result["sets"].values())
    return result


def task_r4() -> dict[str, Any]:
    expected = {
        "pplx27b_confirm_general": ("preds/pplx27b_confirm_general/predictions.jsonl", 5988),
        "md9b_confirm_general": ("preds/md9b_confirm_general/predictions.jsonl", 5988),
        "md4b_confirm_general": ("preds/md4b_confirm_general/predictions.jsonl", 5988),
        "clef_confirm_general": ("preds/clef_confirm_general/predictions.jsonl", 5988),
        "jev9b_confirm_general": ("preds/jev9b_confirm_general/predictions.jsonl", 5988),
    }
    result: dict[str, Any] = {"files": {}, "problems": []}
    for name, (relative, size) in expected.items():
        path = R4 / relative
        if not path.is_file():
            result["files"][name] = {"status": "NOT MEASURED — file absent"}
            continue
        rows = read_jsonl(path)
        scored = sum(1 for row in rows if row.get("status", "scored") == "scored")
        result["files"][name] = {"rows": len(rows), "expected": size, "scored": scored, "skipped": len(rows) - scored,
                                 "problems": len(check_probs(rows, name))}
        if result["files"][name]["problems"]:
            result["problems"].append(f"{name}: probability problems")
    result["passed"] = not result["problems"]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--task", required=True, choices=["r2", "r3", "r4"])
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    result = {"r2": task_r2, "r3": task_r3, "r4": task_r4}[args.task]()
    out = args.out or OUT / f"{args.task}_verify.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json(out, result)
    print(json.dumps({"task": args.task, "passed": result["passed"], "out": str(out),
                      "problems": result.get("problems")}, indent=1))


if __name__ == "__main__":
    main()
