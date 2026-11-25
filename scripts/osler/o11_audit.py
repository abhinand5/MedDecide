"""O11 self-audit (ADVISORY R6; CPU; reads the scored O11 outputs, runs no model).

For one Osler model, re-derives the O11 numbers from the raw prediction files in a fresh process and checks them:
    - denominators: every in-scope item once in each file (both_orders.jsonl and single_order.jsonl), no extra or missing ids;
    - status: every row scored;
    - alignment: probabilities match the row's option keys and sum to one (float64, tolerance 1e-6);
    - orders: choice rows were averaged over two orders (both-order and single-order vectors differ), and noul and score rows
      have identical vectors in both files (one order only);
    - the answer: accuracy and Brier by benchmark and question type, the most frequent predicted option's share (a constant-answer
      check) and the majority-of-gold baseline.
Writes outputs/osler_v0/O11/<name>/audit.json (gitignored); the committed record is the CLAIMS rows.

Usage:
    uv run --frozen python scripts/osler/o11_audit.py --name osler_4b_L
"""

from __future__ import annotations

import argparse
import collections
from pathlib import Path
from typing import Any

import numpy as np

from meddecide.eval.o2_io import Scope, iter_items
from meddecide.eval.paired import item_brier
from meddecide.utils.io import iter_jsonl, write_json

REPO = Path(__file__).resolve().parents[2]
ITEMS = REPO / "data/bench/v0.3_ext/o2_items.jsonl"
OUT = REPO / "outputs/osler_v0/O11"
SCOPE = Scope("common_set", ("v0.2", "ext_panel", "robustness"))
SUM_TOLERANCE = 1e-6


def predicted_key(row: dict[str, Any]) -> str:
    return row["option_keys"][int(np.argmax(row["probs"]))]


def is_correct(row: dict[str, Any]) -> bool:
    return predicted_key(row) == row["gold"]


def check_rows(rows: list[dict[str, Any]], expected: set[str], *, label: str) -> dict[str, Any]:
    ids = [r["item_id"] for r in rows]
    counter = collections.Counter(ids)
    missing = sorted(expected - set(ids))
    extra = sorted(set(ids) - expected)
    duplicated = sum(1 for v in counter.values() if v > 1)
    not_scored = sum(1 for r in rows if r["status"] != "scored")
    misaligned = sum(1 for r in rows if r["probs"] is None or len(r["probs"]) != len(r["option_keys"]))
    sum_violations = sum(1 for r in rows if r["probs"] is not None
                         and abs(float(np.sum(r["probs"])) - 1.0) > SUM_TOLERANCE)
    gold_not_offered = sum(1 for r in rows if r["gold"] not in r["option_keys"])
    if missing or extra or duplicated or not_scored or misaligned or sum_violations or gold_not_offered:
        raise ValueError(f"{label}: missing {len(missing)}, extra {len(extra)}, duplicated {duplicated}, not scored "
                         f"{not_scored}, misaligned {misaligned}, probability sums off {sum_violations}, gold not offered "
                         f"{gold_not_offered}")
    return {"rows": len(rows), "expected": len(expected), "unique_ids": len(counter), "not_scored": not_scored,
            "misaligned": misaligned, "sum_violations": sum_violations, "gold_not_offered": gold_not_offered}


def accuracy_table(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, float]]:
    groups: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for r in rows:
        groups[str(r[key])].append(r)
    return {g: {"n": len(v), "accuracy": float(np.mean([is_correct(r) for r in v])),
                "brier": float(np.mean([item_brier(r["probs"], r["option_keys"].index(r["gold"])) for r in v]))}
            for g, v in sorted(groups.items())}


def template_checks(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Per template: the item count, the accuracy, and the share of the most frequent predicted option and of the most frequent
    gold option. Inside one template the options are the same, so a share near one would mean a constant answer."""
    groups: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for r in rows:
        groups[str(r["template_id"])].append(r)
    out: dict[str, dict[str, Any]] = {}
    for template, members in sorted(groups.items()):
        predicted = collections.Counter(predicted_key(r) for r in members).most_common(1)[0]
        golds = collections.Counter(r["gold"] for r in members).most_common(1)[0]
        out[template] = {"n": len(members), "accuracy": float(np.mean([is_correct(r) for r in members])),
                         "top_predicted_key": predicted[0], "top_predicted_share": predicted[1] / len(members),
                         "top_gold_key": golds[0], "top_gold_share": golds[1] / len(members)}
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, help="the O11 output directory name, e.g. osler_4b_L")
    parser.add_argument("--out-root", type=Path, default=OUT,
                        help="the directory that holds the output directories (default outputs/osler_v0/O11)")
    args = parser.parse_args()
    run_dir = args.out_root / args.name
    expected = {row["item_id"] for row in iter_items(ITEMS, SCOPE)}
    both = list(iter_jsonl(run_dir / "both_orders.jsonl"))
    single = list(iter_jsonl(run_dir / "single_order.jsonl"))
    counts_both = check_rows(both, expected, label="both_orders")
    counts_single = check_rows(single, expected, label="single_order")

    single_by_id = {r["item_id"]: r for r in single}
    choice_pairs = [(r, single_by_id[r["item_id"]]) for r in both if r["qtype"] == "choice"]
    averaged = [b for b, s in choice_pairs if not np.allclose(b["probs"], s["probs"], atol=1e-12)]
    orders_ok = all(b["extras"].get("orders") == 2 for b, _ in choice_pairs)
    fixed = [(b, s) for b in both if b["qtype"] != "choice" for s in [single_by_id[b["item_id"]]]]
    identical_fixed = sum(1 for b, s in fixed if np.allclose(b["probs"], s["probs"], atol=1e-12))
    orders_single = all(s["extras"].get("orders") == 1 for s in single)

    per_template = template_checks(both)
    worst = max(per_template, key=lambda t: per_template[t]["top_predicted_share"])
    audit = {
        "name": args.name,
        "expected_items": len(expected),
        "both_orders": counts_both,
        "single_order": counts_single,
        "choice_rows": len(choice_pairs),
        "choice_rows_averaged": len(averaged),
        "choice_rows_two_orders_recorded": orders_ok,
        "fixed_order_rows": len(fixed),
        "fixed_order_rows_identical": identical_fixed,
        "single_rows_one_order_recorded": orders_single,
        "headline_both_orders": {"accuracy": float(np.mean([is_correct(r) for r in both])),
                                 "brier": float(np.mean([item_brier(r["probs"], r["option_keys"].index(r["gold"]))
                                                         for r in both]))},
        "headline_single_order": {"accuracy": float(np.mean([is_correct(r) for r in single]))},
        "by_benchmark_both_orders": accuracy_table(both, "benchmark"),
        "by_qtype_both_orders": accuracy_table(both, "qtype"),
        "by_template_both_orders": per_template,
        "constant_answer_check": {"templates": len(per_template), "worst_template": worst,
                                  "worst_top_predicted_share": per_template[worst]["top_predicted_share"]},
    }
    write_json(run_dir / "audit.json", audit)
    print(f"{args.name}: {counts_both['rows']} rows in both_orders, {counts_single['rows']} in single_order; "
          f"choice rows averaged {len(averaged)} of {len(choice_pairs)}; fixed-order rows identical {identical_fixed} of {len(fixed)}")
    print(f"accuracy both orders {audit['headline_both_orders']['accuracy']:.4f}, single order "
          f"{audit['headline_single_order']['accuracy']:.4f}, Brier {audit['headline_both_orders']['brier']:.4f}; "
          f"worst per-template top predicted share {per_template[worst]['top_predicted_share']:.3f} ({worst}, "
          f"{len(per_template)} templates)")


if __name__ == "__main__":
    main()
