#!/usr/bin/env python
"""osler_rc0 R9 — five-claim fresh-process spot-check.

Each claim is re-derived from the raw artifacts (prediction files, eval outputs, membership scores) in this process —
not copied from any run summary — and compared with the number the loop reports. Any mismatch is a failure.

Run::

    source scripts/pod_env.sh && uv run --frozen python scripts/osler_rc0/r9_spotcheck.py
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

from meddecide.eval.audit_y10 import membership_auc  # noqa: E402
from meddecide.eval.calibration import apply_temperature  # noqa: E402
from meddecide.eval.composite import composite  # noqa: E402
from meddecide.eval.rc0_board import (  # noqa: E402
    cell_metrics,
    exact_dominates,
    load_cell,
    load_frozen,
    paired_difference,
)
from meddecide.utils.provenance import git_commit, utcnow  # noqa: E402

OUT = ROOT / "outputs/osler_rc0/R9"
R1 = ROOT / "outputs/osler_rc0/R1/arm"
R2 = ROOT / "outputs/osler_rc0/R2"
R3 = ROOT / "outputs/osler_rc0/R3"
R6 = ROOT / "outputs/osler_rc0/R6"
R7 = ROOT / "outputs/osler_rc0/R7"
SET_FILES = {
    "panel": ROOT / "data/bench/osler_v4/y9/panel.jsonl",
    "knowledge": ROOT / "data/bench/osler_v4/y9/knowledge.jsonl",
    "confirm_general": ROOT / "data/bench/osler_v7/confirm/general.jsonl",
    "mr_verdict": ROOT / "data/bench/osler_v4/y9/mr_verdict.jsonl",
    "mimic_suite": ROOT / "data/bench/osler_v4/y9/mimic_suite.jsonl",
}
HEADLINE = list(SET_FILES)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def claim1() -> dict[str, Any]:
    arm = read_json(R1 / "arm_result.json")
    step = int(arm["steps"])
    evaluations = arm["evaluations"]
    final = evaluations[str(step)]
    fast = read_jsonl(R1 / f"evals/step_{step:06d}/fast_dev.jsonl")
    cg = read_jsonl(R1 / f"evals/step_{step:06d}/cg_v2.jsonl")
    parts = composite(fast, cg)
    passes = [final["positional_original"]["passes"]]
    passes += [entry["passes"] for entry in final["positional_permuted"].values()]
    reported = float(final["composite"]["S"])
    return {
        "claim": "Osler-4B-rc0 step-63,304 dev S and all-orders positional health",
        "steps": step, "stopped_early": arm["stopped_early"],
        "S_reported": reported, "S_recomputed": parts.s,
        "S_match": abs(reported - parts.s) < 1e-12,
        "parts_recomputed": parts.to_dict(),
        "positional_all_orders": bool(all(passes)), "positional_orders": len(passes),
        "temperature_fits_present": sorted(arm.get("temperature_fits", {})),
        # the trainer reports a completion-reason string, not a boolean: a run that reached its requested step budget
        # with no tripwire did not stop early in substance (recorded in the R1 self-audit)
        "stopped_early_is_completion_reason": "step budget" in str(arm["stopped_early"]).lower(),
        "passed": step == 63304 and "step budget" in str(arm["stopped_early"]).lower()
        and abs(reported - parts.s) < 1e-12 and all(passes)
        and sorted(arm.get("temperature_fits", {})) == ["choice", "noul", "score"],
    }


def claim2() -> dict[str, Any]:
    frozen = load_frozen("panel", SET_FILES["panel"])
    cell = load_cell("Osler-9B-rc0", frozen, R2 / "preds_cal/9b_rc0_panel/predictions.jsonl")
    metrics = cell_metrics(frozen, cell)
    board = read_json(R7 / "rc0_board.json")["models"]["Osler-9B-rc0"]["panel"]
    return {
        "claim": "Osler-9B-rc0 dev-calibrated panel accuracy, Brier, ECE",
        "accuracy_recomputed": metrics["accuracy"], "accuracy_board": board["accuracy"],
        "brier_recomputed": metrics["brier"], "brier_board": board["brier"],
        "ece_recomputed": metrics["ece"], "ece_board": board["ece"],
        "coverage": metrics["coverage"],
        "match": all(abs(metrics[key] - board[key]) < 1e-9 for key in ("accuracy", "brier", "ece")),
        "passed": abs(metrics["accuracy"] - board["accuracy"]) < 1e-9
        and abs(metrics["brier"] - board["brier"]) < 1e-9 and abs(metrics["ece"] - board["ece"]) < 1e-9,
    }


def claim3() -> dict[str, Any]:
    out = {}
    pairs = {
        "Osler-9B-rc0": (R2 / "preds_cal/9b_rc0_confirm_general/predictions.jsonl",
                         R2 / "preds/d3_base_confirm_general/predictions.jsonl"),
        "Osler-4B-rc0": (R3 / "preds_cal/4b_rc0_confirm_general/predictions.jsonl",
                         ROOT / "outputs/osler_v7/T10/baselines/4b_base_general/predictions.jsonl"),
    }
    frozen = load_frozen("confirm_general", SET_FILES["confirm_general"])
    board = read_json(R7 / "rc0_board.json")["paired"]
    for model, (cand_path, base_path) in pairs.items():
        a = load_cell(model, frozen, cand_path)
        b = load_cell(f"{model} base", frozen, base_path)
        diff = paired_difference(frozen, a, b)
        reported = board[model]["d3-flash" if "9B" in model else "pplx-flash"]["confirm_general"]
        out[model] = {"diff_recomputed": diff["accuracy_diff"], "diff_reported": reported["accuracy_diff"],
                      "ci_recomputed": diff["accuracy_diff_ci95"], "ci_reported": reported["accuracy_diff_ci95"],
                      "n": diff["n"],
                      "match": abs(diff["accuracy_diff"] - reported["accuracy_diff"]) < 1e-9
                      and diff["accuracy_diff_ci95"] == reported["accuracy_diff_ci95"]}
    return {"claim": "confirm_general paired deltas vs own bases", "models": out,
            "passed": all(entry["match"] for entry in out.values())}


def claim4() -> dict[str, Any]:
    items = read_jsonl(R6 / "visited_membership.jsonl")
    temperatures = {
        "Osler-9B-rc0": read_json(R2 / "9b_calibration.json")["temperatures"],
        "Osler-4B-rc0": read_json(R3 / "4b_calibration.json")["temperatures"],
    }
    sources = {
        "Osler-9B-rc0": (R6 / "scores/9b_rc0/predictions.jsonl", True),
        "Osler-4B-rc0": (R6 / "scores/4b_rc0/predictions.jsonl", True),
    }
    reported = read_json(R6 / "membership_auc.json")["models"]
    out = {}
    for model, (path, calibrated) in sources.items():
        by_id = {str(row["item_id"]): row for row in read_jsonl(path)}
        losses = {}
        for row in items:
            prediction = by_id[str(row["r6_score_id"])]
            probs = prediction["probs"]
            if calibrated:
                probs = apply_temperature([probs], temperatures[model][str(row["qtype"])])[0].tolist()
            losses[str(row["r6_score_id"])] = -float(np.log(max(float(probs[int(prediction["gold_index"])]), 1e-300)))
        members = [losses[str(row["r6_score_id"])] for row in items if row["r6_label"] == "member"]
        nonmembers = [losses[str(row["r6_score_id"])] for row in items if row["r6_label"] == "nonmember"]
        auc = membership_auc(members, nonmembers)
        out[model] = {"auc_recomputed": round(auc, 6), "auc_reported": reported[model]["auc"],
                      "match": abs(auc - reported[model]["auc"]) < 1e-6}
    return {"claim": "visited-member MIMIC membership AUC for 9B and 4B", "models": out,
            "passed": all(entry["match"] for entry in out.values())}


def claim5() -> dict[str, Any]:
    board = read_json(R7 / "rc0_board.json")
    metrics = board["models"]
    params = {model: board["params_b"][model] for model in metrics}
    le10 = [model for model in metrics if params[model] <= 10.0]
    points = {}
    for model in le10:
        values = [metrics[model].get(set_name, {}).get("accuracy") for set_name in HEADLINE]
        if all(isinstance(value, float) for value in values):
            points[model] = dict(zip(HEADLINE, values, strict=True))
    recomputed = []
    for a in points:
        for b in points:
            if a != b and exact_dominates(points[a], points[b], HEADLINE):
                recomputed.append({"dominator": a, "dominated": b})
    reported = board["pareto"]["quality_le10b"]["dominance_pairs"]
    reported_pairs = [{"dominator": entry["dominator"], "dominated": entry["dominated"]} for entry in reported]
    key = lambda entry: (entry["dominator"], entry["dominated"])  # noqa: E731
    return {
        "claim": "formal <=10B Pareto dominance output from rc0_board.json",
        "comparable_models": sorted(points), "recomputed_pairs": sorted(recomputed, key=key),
        "reported_pairs": sorted(reported_pairs, key=key),
        "match": sorted(recomputed, key=key) == sorted(reported_pairs, key=key),
        "passed": sorted(recomputed, key=key) == sorted(reported_pairs, key=key),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, default=OUT / "spotcheck.json")
    args = parser.parse_args()
    claims = {"generated_utc": utcnow(), "git_commit": git_commit(ROOT)}
    for number, function in ((1, claim1), (2, claim2), (3, claim3), (4, claim4), (5, claim5)):
        claims[f"claim_{number}"] = function()
        print(f"claim {number}: {'PASS' if claims[f'claim_{number}']['passed'] else 'FAIL'}")
    claims["all_passed"] = all(claims[f"claim_{number}"]["passed"] for number in range(1, 6))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(claims, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"all_passed": claims["all_passed"], "out": str(args.out)}, indent=1))


if __name__ == "__main__":
    main()
