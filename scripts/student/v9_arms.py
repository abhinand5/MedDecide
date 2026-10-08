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

Verdict scope (the G1 scope, ADVISORY V9): the fresh tier only, and every template whose cell G1 excluded under
D12 for either arm of the pair is dropped (``--exclude LABEL=g1.json[:model]``). A set with no items is reported
as NOT MEASURED. The all-tier comparison without the D12 exclusions is reported as an additional analysis,
labelled as such.

Usage: ``uv run python scripts/student/v9_arms.py --arm A=<preds> --arm B=<preds> --items <test items> \
  --exclude B=outputs/student_v1/V9/g1_arm_b.json:arm_b --out <json>``
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


def load_tiers(path: Path) -> dict[str, str]:
    tiers: dict[str, str] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            tiers[row["item_id"]] = row["tier"]
    return tiers


def excluded_templates(path: Path, model: str) -> set[str]:
    """Fresh-tier templates whose (model) cell G1 excluded under D12 (the verdict scope of G1)."""
    report = json.loads(path.read_text(encoding="utf-8"))
    return {c["template_id"] for c in report["excluded_cells"] if c["model"] == model and c["tier"] == "fresh"}


def compare(a: dict[str, dict], b: dict[str, dict], ids: list[str]) -> dict:
    if not ids:
        return {"n": 0, "templates": 0, "status": "NOT MEASURED — no common items in this set"}
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


def describe(block: dict) -> str:
    if "status" in block:
        return block["status"]
    held = block["macro_accuracy_diff"]
    return (f"macro diff {held['point'] * 100:+.2f} pts [{held['ci_lo'] * 100:+.2f}, {held['ci_hi'] * 100:+.2f}] "
            f"(n={block['n']}, templates={block['templates']})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arm", action="append", required=True, help="label=predictions.jsonl (repeatable)")
    parser.add_argument("--items", type=Path, required=True, help="the v0.2 test items (MedDecide JSONL): the tier")
    parser.add_argument("--tier", default="fresh", help="the verdict tier (the G1 scope is fresh)")
    parser.add_argument("--exclude", action="append", default=[],
                        help="LABEL=g1.json[:model]: the cells G1 excluded under D12 for that arm (repeatable)")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    arms = {}
    for spec in args.arm:
        label, _, path = spec.partition("=")
        arms[label] = load(Path(path))
    tiers = load_tiers(args.items)
    excluded: dict[str, set[str]] = {}
    for spec in args.exclude:
        label, _, rest = spec.partition("=")
        path, _, model = rest.partition(":")
        excluded[label] = excluded_templates(Path(path), model or label)
    report = {"kind": "arm_pairs", "held_out_templates": list(HELD_OUT), "verdict_tier": args.tier,
              "d12_excluded_templates": {k: sorted(v) for k, v in excluded.items()}, "arms": {}, "pairs": {}}
    for label, rows in arms.items():
        report["arms"][label] = {"n": len(rows)}
    for left, right in combinations(sorted(arms), 2):
        a, b = arms[left], arms[right]
        common = sorted(set(a) & set(b))
        drop = excluded.get(left, set()) | excluded.get(right, set())
        verdict = [i for i in common if tiers.get(i) == args.tier and a[i]["template"] not in drop]
        v_seen = [i for i in verdict if a[i]["template"] not in HELD_OUT]
        v_held = [i for i in verdict if a[i]["template"] in HELD_OUT]
        a_seen = [i for i in common if a[i]["template"] not in HELD_OUT]
        a_held = [i for i in common if a[i]["template"] in HELD_OUT]
        report["pairs"][f"{left}-{right}"] = {
            "common_items": len(common),
            "verdict": {
                "tier": args.tier, "excluded_templates": sorted(drop), "items": len(verdict),
                "seen": compare(a, b, v_seen),
                "held_out": compare(a, b, v_held),
            },
            "additional_all_tiers_no_d12": {
                "label": "additional, not a verdict: all tiers, no D12 exclusions",
                "seen": compare(a, b, a_seen),
                "held_out": compare(a, b, a_held),
                "all": compare(a, b, common),
            },
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, default=float) + "\n", encoding="utf-8")
    for key, pair in report["pairs"].items():
        print(f"{key} verdict ({pair['verdict']['tier']}, {pair['verdict']['items']} items, "
              f"excluded {pair['verdict']['excluded_templates']}):")
        print(f"  seen:     {describe(pair['verdict']['seen'])}")
        print(f"  held-out: {describe(pair['verdict']['held_out'])}")
        additional = pair["additional_all_tiers_no_d12"]
        print(f"{key} additional (all tiers, no D12): seen {describe(additional['seen'])}; "
              f"held-out {describe(additional['held_out'])}")


if __name__ == "__main__":
    main()
