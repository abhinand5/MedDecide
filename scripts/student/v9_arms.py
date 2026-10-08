#!/usr/bin/env python
"""Arm-versus-arm paired comparisons (V9): paired bootstrap on the same test items, seen and held-out.

Inputs are run_student-format prediction files, one per arm (any number of arms). For each pair (A, B) the
script reports, on the seen templates and on the four held-out templates (D14), the paired differences of:

  * micro accuracy (item-weighted),
  * macro accuracy (mean over templates of per-template accuracy, the G1 definition),
  * mean Brier (uncalibrated probabilities as stored, multi-class),

each with a 95 % percentile bootstrap interval. The macro interval resamples items within each template
(``stratified_macro_difference``); micro and Brier intervals resample items (``paired_difference``). Only items
present in both files are compared; the counts are reported. Pure metrics live in ``meddecide.eval.paired``.

Usage: ``uv run python scripts/student/v9_arms.py --arm A=<preds> --arm B=<preds> --out <json>``
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from meddecide.eval.paired import (  # noqa: E402
    item_brier,
    paired_difference,
    stratified_macro_difference,
)

HELD_OUT = (
    "ct_phase_choice_v1",
    "fda_boxed_warning_noul_v1",
    "pubmed_humans_noul_v1",
    "ct_arm_role_noul_v1",
)


def load(path: Path) -> dict[str, dict]:
    rows = {}
    for line in path.open(encoding="utf-8"):
        row = json.loads(line)
        if row["item_id"] in rows:
            raise ValueError(f"{path}: duplicate item {row['item_id']}")
        keys = row["option_keys"]
        gold = keys.index(row["gold_key"])
        rows[row["item_id"]] = {
            "template": row["template_id"],
            "correct": 1.0 if row["correct"] else 0.0,
            "brier": item_brier(row["option_probs"], gold),
        }
    return rows


def compare(a: dict[str, dict], b: dict[str, dict], ids: list[str]) -> dict:
    groups_a, groups_b = defaultdict(list), defaultdict(list)
    brier_a, brier_b, corr_a, corr_b = [], [], [], []
    for item_id in ids:
        ra, rb = a[item_id], b[item_id]
        groups_a[ra["template"]].append(ra["correct"])
        groups_b[ra["template"]].append(rb["correct"])
        corr_a.append(ra["correct"])
        corr_b.append(rb["correct"])
        brier_a.append(ra["brier"])
        brier_b.append(rb["brier"])
    micro = paired_difference(corr_a, corr_b, n_resamples=2000)
    brier = paired_difference(brier_a, brier_b, n_resamples=2000)
    macro = stratified_macro_difference(groups_a, groups_b, n_resamples=2000)
    return {
        "n": len(ids),
        "templates": len(groups_a),
        "micro_accuracy_diff": micro.as_dict(),
        "macro_accuracy_diff": macro.as_dict(),
        "mean_brier_diff": brier.as_dict(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", action="append", required=True, help="label=predictions.jsonl (repeatable)")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    arms = {}
    for spec in args.arm:
        label, _, path = spec.partition("=")
        arms[label] = load(Path(path))
    report = {"kind": "arm_pairs", "held_out_templates": list(HELD_OUT), "arms": {}, "pairs": {}}
    for label, rows in arms.items():
        report["arms"][label] = {"n": len(rows)}
    for left, right in combinations(sorted(arms), 2):
        common = sorted(set(arms[left]) & set(arms[right]))
        seen = [i for i in common if arms[left][i]["template"] not in HELD_OUT]
        held = [i for i in common if arms[left][i]["template"] in HELD_OUT]
        report["pairs"][f"{left}-{right}"] = {
            "common_items": len(common),
            "seen": compare(arms[left], arms[right], seen),
            "held_out": compare(arms[left], arms[right], held),
            "all": compare(arms[left], arms[right], common),
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, default=float) + "\n", encoding="utf-8")
    for key, pair in report["pairs"].items():
        held = pair["held_out"]["macro_accuracy_diff"]
        print(f"{key}: held-out macro diff {held['point'] * 100:+.2f} pts "
              f"[{held['ci_lo'] * 100:+.2f}, {held['ci_hi'] * 100:+.2f}] (n={pair['held_out']['n']})")


if __name__ == "__main__":
    main()
