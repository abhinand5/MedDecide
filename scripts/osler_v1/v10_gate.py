"""V10: Gate O2 (PROGRAM D31) for Osler-4B. CPU only; reads prediction files and runs no model.

The Osler-9B verdict is NOT MEASURED: it is deferred to the next loop by the operator (V9-D1).

Primary checks (they decide the verdict):
  (i)   MR audit suite, scope mr_test (the MR test suite; MIMIC is absent, so these are the generated parts), against MedDecider-4B
        and zero-shot Qwen3.5-4B: accuracy lower bound > 0 and Brier upper bound < 0 (``Comparison.passes``);
  (ii)  external clinical panel, against both baselines: accuracy lower bound > -0.02 (non-inferiority);
  (iii) knowledge guard (D24), Osler-4B against zero-shot Qwen3.5-4B on MedQA and MedMCQA test: lower bound > -0.02.
Baselines: the MR suite from the V6 baseline runs (outputs/osler_v1/V6/baselines); the panel and the knowledge items from the O2
common-set runs (outputs/osler_v0/O2), the protocol the O1 and O2 verdicts used.

Reported beside the verdict, never in it (labelled additional): the held-out MR file (its test rows), the panel's superiority,
the single-order rows, and the d1-3B row. MDACE and the MIMIC part of the MR suite are NOT MEASURED (the data is absent).

Outputs: outputs/osler_v1/V10/gate_o2.json (every number, with its item counts) and loops/osler_v1/gate_o2.md (the verdicts).

Usage:
    uv run --frozen python scripts/osler_v1/v10_gate.py
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from meddecide.eval.gate import (
    KNOWLEDGE_GUARD_FLOOR,
    Comparison,
    accuracy_difference,
    compare_rows,
    knowledge_guard,
)
from meddecide.eval.gate_o2 import PANEL_NON_INFERIORITY_FLOOR, o2_verdict
from meddecide.eval.paired import Difference
from meddecide.utils.io import iter_jsonl, write_json
from meddecide.utils.provenance import git_commit, utcnow

REPO = Path(__file__).resolve().parents[2]
ITEMS = REPO / "data/bench/osler_v1/v10_items.jsonl"
V10 = REPO / "outputs/osler_v1/V10"
REPORT = REPO / "loops/osler_v1/gate_o2.md"
V6_BASELINES = REPO / "outputs/osler_v1/V6/baselines"
O2_BASELINES = REPO / "outputs/osler_v0/O2"
MR_BASELINES = {"MedDecider-4B": V6_BASELINES / "meddecider-4b", "zero-shot Qwen3.5-4B": V6_BASELINES / "qwen35-4b"}
PANEL_BASELINES = {"MedDecider-4B": O2_BASELINES / "meddecider-4b", "zero-shot Qwen3.5-4B": O2_BASELINES / "qwen35-4b"}
GUARD_NAME = "zero-shot Qwen3.5-4B"
GUARD_DIR = O2_BASELINES / "qwen35-4b"
ADDITIONAL_D1 = ("d1-3B", V6_BASELINES / "d1-3b")
N_RESAMPLES = 1000
SEED = 0


def members_of(rows: Sequence[Mapping[str, Any]], scope: str, *, test_only: bool = False) -> list[dict[str, Any]]:
    keep = [r for r in rows if r["v10_scope"] == scope and (not test_only or r["split"] == "test")]
    return [{"item_id": r["item_id"], "template_id": r["template_id"]} for r in keep]


def by_item(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    return {str(row["item_id"]): row for row in iter_jsonl(path)}


def compared(scope: str, name: str, members: list[dict], osler: Mapping, base: Mapping) -> tuple[dict[str, Any], Comparison | None]:
    """One comparison: its record, and the comparison itself (None when it could not be measured, which is never a failure)."""
    try:
        paired = compare_rows(scope, name, members, osler, base, n_resamples=N_RESAMPLES, seed=SEED)
    except ValueError as exc:
        return {"scope": scope, "baseline": name, "not_measured": str(exc)}, None
    if paired.n_osler_missing:
        reason = f"{paired.n_osler_missing} scope items have no Osler row"
        return {"scope": scope, "baseline": name, "not_measured": reason}, None
    comp = paired.comparison
    record = {"scope": scope, "baseline": name, "n_scope": paired.n_scope, "n_paired": paired.n_paired,
              "n_baseline_skipped": paired.n_baseline_skipped, "n_baseline_missing": paired.n_baseline_missing,
              "n_osler_missing": paired.n_osler_missing, "accuracy": comp.accuracy.__dict__, "brier": comp.brier.__dict__,
              "passes_mr_rule": comp.passes, "accuracy_lower_above_panel_floor": comp.accuracy.lo > PANEL_NON_INFERIORITY_FLOOR}
    return record, comp


def table_row(record: dict[str, Any], check: str, rule: str | None) -> str:
    """One table row. ``rule`` names the pass rule shown in the pass column: "mr" (D31 (i)), "panel" (D31 (ii)) or None (no rule)."""
    if "not_measured" in record:
        cells = [check, record["baseline"], "NOT MEASURED", "", "", "—", record["not_measured"]]
    else:
        a, b = record["accuracy"], record["brier"]
        passes = {"mr": record["passes_mr_rule"], "panel": record["accuracy_lower_above_panel_floor"]}.get(rule or "")
        cells = [check, record["baseline"], str(record["n_paired"]),
                 f"{a['point']:+.4f} [{a['lo']:+.4f}, {a['hi']:+.4f}]", f"{b['point']:+.4f} [{b['lo']:+.4f}, {b['hi']:+.4f}]",
                 "—" if passes is None else ("yes" if passes else "no"),
                 f"{record['n_baseline_skipped']} / {record['n_baseline_missing']}, {record['n_osler_missing']}"]
    return "| " + " | ".join(cells) + " |"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--v10-root", type=Path, default=V10, help=argparse.SUPPRESS)  # pipeline checks only
    parser.add_argument("--items", type=Path, default=ITEMS, help=argparse.SUPPRESS)  # pipeline checks only
    parser.add_argument("--report", type=Path, default=REPORT, help=argparse.SUPPRESS)  # pipeline checks only
    args = parser.parse_args()
    v10_root = args.v10_root

    rows = list(iter_jsonl(args.items))
    osler_run = json.loads((v10_root / "osler_4b" / "run.json").read_text(encoding="utf-8"))
    if not osler_run["checkpoint_calibration_matches_dev_fits"]:
        raise SystemExit("the Osler-4B calibration does not match its dev fits (ADVISORY V10)")
    osler_both = by_item(v10_root / "osler_4b" / "both_orders.jsonl")
    osler_single = by_item(v10_root / "osler_4b" / "single_order.jsonl")

    mr = members_of(rows, "mr_test")
    panel = members_of(rows, "external_panel")
    knowledge = members_of(rows, "knowledge")
    held_out_test = members_of(rows, "mr_heldout", test_only=True)
    scope_sizes = {"mr_test": len(mr), "external_panel": len(panel), "knowledge": len(knowledge),
                   "mr_heldout_test": len(held_out_test)}

    mr_records: dict[str, dict[str, Any]] = {}
    mr_comps: dict[str, Comparison | None] = {}
    for name, path in MR_BASELINES.items():
        mr_records[name], mr_comps[name] = compared("mr_test", name, mr, osler_both, by_item(path / "predictions.jsonl"))
    panel_records: dict[str, dict[str, Any]] = {}
    panel_comps: dict[str, Comparison | None] = {}
    for name, path in PANEL_BASELINES.items():
        panel_records[name], panel_comps[name] = compared("external_panel", name, panel, osler_both,
                                                          by_item(path / "predictions.jsonl"))

    guard_record: dict[str, Any] = {"baseline": GUARD_NAME}
    guard: Difference | None = None
    try:
        diff, n_paired = accuracy_difference(knowledge, osler_both, by_item(GUARD_DIR / "predictions.jsonl"),
                                             n_resamples=N_RESAMPLES, seed=SEED)
        if n_paired != len(knowledge):
            raise ValueError(f"{len(knowledge) - n_paired} knowledge items are not paired")
        guard = diff
        guard_record |= {"n_paired": n_paired, "accuracy": diff.__dict__, "passes": knowledge_guard(diff)}
    except ValueError as exc:
        guard_record |= {"not_measured": str(exc)}

    verdict, reason = o2_verdict(mr_comps, panel_comps, guard)

    # additional rows: never in the verdict
    held_out: dict[str, dict[str, Any]] = {}
    for name, path in MR_BASELINES.items():
        held_out[name], _ = compared("mr_heldout_test", name, held_out_test, osler_both, by_item(path / "predictions.jsonl"))
    d1_name, d1_dir = ADDITIONAL_D1
    d1_record, _ = compared("mr_test", d1_name, mr, osler_both, by_item(d1_dir / "predictions.jsonl"))
    single_mr: dict[str, dict[str, Any]] = {}
    single_panel: dict[str, dict[str, Any]] = {}
    for name, path in MR_BASELINES.items():
        single_mr[name], _ = compared("mr_test", name, mr, osler_single, by_item(path / "predictions.jsonl"))
    for name, path in PANEL_BASELINES.items():
        single_panel[name], _ = compared("external_panel", name, panel, osler_single, by_item(path / "predictions.jsonl"))

    results: dict[str, Any] = {
        "kind": "osler_v1_V10_gate_o2", "built_at_utc": utcnow(), "git_commit": git_commit(REPO),
        "command": "scripts/osler_v1/v10_gate.py", "scope_sizes": scope_sizes,
        "non_inferiority_floor": PANEL_NON_INFERIORITY_FLOOR, "knowledge_guard_floor": KNOWLEDGE_GUARD_FLOOR,
        "n_resamples": N_RESAMPLES, "seed": SEED,
        "osler_run": {k: osler_run.get(k) for k in ("checkpoint", "calibration", "checkpoint_calibration_matches_dev_fits",
                                                    "rows_in_items_file", "items_sha256")},
        "models": {
            "osler_4b": {"verdict": verdict, "verdict_reason": reason, "mr_suite": mr_records,
                         "external_panel": panel_records, "knowledge_guard": guard_record},
            "osler_9b": {"verdict": "NOT MEASURED", "verdict_reason": "deferred to the next loop by the operator (V9-D1)"},
        },
        "additional": {"held_out_test": held_out, "d1_3b_mr_test": d1_record, "single_order_mr_test": single_mr,
                       "single_order_external_panel": single_panel,
                       "mdace": {"not_measured": "the MDACE data is not on the volume"},
                       "mr_mimic_part": {"not_measured": "MIMIC is absent (V5c); the MR test suite is the generated part"}},
    }
    write_json(v10_root / "gate_o2.json", results)

    head = ["| check | baseline | items (paired) | accuracy diff [lower, upper] | Brier diff [lower, upper] | pass | skipped / missing (baseline), Osler missing |",
            "|---|---|---|---|---|---|---|"]
    lines = ["# osler_v1 — Gate O2 (PROGRAM D31), Osler-4B", "",
             "Osler rows: both-order averages for choice items, noul and score items scored once (the O2 convention). Accuracy is the",
             f"template macro, bootstrapped within templates ({N_RESAMPLES} resamples, seed {SEED}); Brier is the item mean. An interval",
             "is shown as point [lower, upper]. Items in scope: " + ", ".join(f"{k} {v}" for k, v in scope_sizes.items()) + ".", ""]
    lines += head
    lines += [table_row(mr_records[name], "(i) MR suite", "mr") for name in MR_BASELINES]
    lines += [table_row(panel_records[name], "(ii) external panel", "panel") for name in PANEL_BASELINES]
    if "not_measured" in guard_record:
        lines += ["", f"(iii) knowledge guard vs {GUARD_NAME}: NOT MEASURED ({guard_record['not_measured']})."]
    else:
        g = guard_record["accuracy"]
        lines += ["", f"(iii) knowledge guard vs {GUARD_NAME} (MedQA + MedMCQA test, micro accuracy): {guard_record['n_paired']} "
                      f"paired items, {g['point']:+.4f} [{g['lo']:+.4f}, {g['hi']:+.4f}]; guard "
                      f"{'passes' if guard_record['passes'] else 'fails'} (lower bound must exceed {KNOWLEDGE_GUARD_FLOOR:+.2f})."]
    lines += ["", f"**Osler-4B verdict: {verdict}.** {reason}.", "",
              "**Osler-9B verdict: NOT MEASURED — deferred to the next loop by the operator (V9-D1).**", "",
              "## Additional rows (labelled; never in the verdict)", "", *head,
              *[table_row(held_out[name], "held-out MR, test rows", "mr") for name in MR_BASELINES],
              table_row(d1_record, "d1-3B, MR test", "mr"), "",
              "Single-order rows (the original option order only; reported beside the gate, never in it):", "", *head,
              *[table_row(single_mr[name], "single order, MR test", "mr") for name in MR_BASELINES],
              *[table_row(single_panel[name], "single order, external panel", "panel") for name in PANEL_BASELINES], "",
              "- MDACE: NOT MEASURED — the dataset is not on the volume.",
              "- The MIMIC part of the MR suite: NOT MEASURED — MIMIC is absent (V5c). The MR test suite is the generated part."]
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Osler-4B: {verdict}. {reason}", flush=True)
    print(f"wrote {args.report} and {v10_root / 'gate_o2.json'}", flush=True)


if __name__ == "__main__":
    main()
