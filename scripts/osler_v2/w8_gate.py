"""osler_v2 W8: Gate O3 (PROGRAM D31 and D35) for Osler-4B. CPU only; reads prediction files and runs no model.

Primary checks (they decide the verdict; ADVISORY W8 step 2, D35, D39):
  (i)   MR suite v1 (D35): the mr_test rows whose template is in the verdict set (data/bench/osler_mr_v1/manifest.json),
        against MedDecider-4B and zero-shot pplx-decider-flash-v1.2 (D39 substitute for zero-shot Qwen3.5-4B): accuracy lower
        bound > 0 and Brier upper bound < 0;
  (ii)  external clinical panel, against both: accuracy lower bound > -0.02 (non-inferiority);
  (iii) knowledge guard (D24), Osler-4B against zero-shot pplx-decider-flash-v1.2 on the MedQA and MedMCQA test items: lower
        bound > -0.02.
The 9B verdict is NOT MEASURED (deferred by the operator, D39).

Reported beside the verdict, never in it (labelled additional): the reported-only MR templates (D35 (a) and (b)), the held-out
MR test rows, the MR suite without PMC, the MR suite vs zero-shot Qwen3.5-4B (not a baseline of the verdict under D39), the
osler_v1 Osler-4B on the same items, and the d1-3B row (not run in this loop).

Baselines: MedDecider-4B from osler_v1's V6 (MR) and osler_v0's O2 (panel); zero-shot pplx from this loop's W5 (MR, panel, knowledge).
The MedDecider-4B held-out readout-failed cells are excluded from the held-out additional row, as in osler_v1 V10.

Outputs: outputs/osler_v2/W8/gate_o3.json (every number with its item counts) and loops/osler_v2/gate_o3.md (verdicts).

Usage: uv run --frozen python scripts/osler_v2/w8_gate.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from meddecide.eval.gate import Comparison, accuracy_difference, compare_rows, knowledge_guard
from meddecide.eval.gate_o2 import (
    KNOWLEDGE_GUARD_FLOOR,
    PANEL_NON_INFERIORITY_FLOOR,
    o2_verdict,
    panel_superior,
)
from meddecide.utils.io import iter_jsonl, write_json
from meddecide.utils.provenance import git_commit, utcnow

REPO = Path(__file__).resolve().parents[2]
ITEMS = REPO / "data" / "bench" / "osler_v1" / "v10_items.jsonl"
MANIFEST = REPO / "data" / "bench" / "osler_mr_v1" / "manifest.json"
OSLER = REPO / "outputs" / "osler_v2" / "W8" / "osler_4b" / "both_orders.jsonl"
OSLER_V1 = REPO / "outputs" / "osler_v1" / "V10" / "osler_4b" / "both_orders.jsonl"
V6 = REPO / "outputs" / "osler_v1" / "V6" / "baselines"
V6_SUMMARY = REPO / "outputs" / "osler_v1" / "V6" / "baseline_summary.json"
O2 = REPO / "outputs" / "osler_v0" / "O2"
PPLX = REPO / "outputs" / "osler_v2" / "W5" / "pplx_test"
OUT = REPO / "outputs" / "osler_v2" / "W8" / "gate_o3.json"
PMC = "pmc_similar_patient_v1"
N_RESAMPLES = 1000
SEED = 0

BASELINES = {
    "MedDecider-4B": {"mr": V6 / "meddecider-4b", "panel": O2 / "meddecider-4b", "knowledge": None},
    "zero-shot pplx-decider-flash-v1.2": {"mr": PPLX / "mr" / "pplx-decider-flash-v1.2",
                                          "panel": PPLX / "panel" / "pplx-decider-flash-v1.2",
                                          "knowledge": PPLX / "knowledge" / "pplx-decider-flash-v1.2"},
}
ADDITIONAL = {
    "zero-shot Qwen3.5-4B": {"mr": V6 / "qwen35-4b", "panel": O2 / "qwen35-4b", "knowledge": O2 / "qwen35-4b"},
}


def by_item(path: Path) -> dict[str, dict[str, Any]]:
    return {str(row["item_id"]): row for row in iter_jsonl(path)}


def members(rows: list[dict[str, Any]], scope: str, *, templates: set[str] | None = None,
            test_only: bool = False) -> list[dict[str, Any]]:
    out = []
    for r in rows:
        if r["v10_scope"] != scope or (test_only and r["split"] != "test"):
            continue
        if templates is not None and r["template_id"] not in templates:
            continue
        out.append({"item_id": r["item_id"], "template_id": r["template_id"]})
    return out


def readout_failed(name: str) -> set[str]:
    summary = json.loads(V6_SUMMARY.read_text(encoding="utf-8"))
    return set(summary["models"]["meddecider-4b"]["sets"]["mr_heldout"]["readout_fail_templates"])


def compared(scope: str, name: str, rows: list[dict[str, Any]], osler: dict, base: dict) -> tuple[dict, Comparison | None]:
    try:
        paired = compare_rows(scope, name, rows, osler, base, n_resamples=N_RESAMPLES, seed=SEED)
    except ValueError as exc:
        return {"scope": scope, "baseline": name, "not_measured": str(exc)}, None
    if paired.n_osler_missing:
        return {"scope": scope, "baseline": name, "not_measured": f"{paired.n_osler_missing} items have no Osler row"}, None
    comp = paired.comparison
    return ({"scope": scope, "baseline": name, "n_scope": paired.n_scope, "n_paired": paired.n_paired,
             "accuracy": comp.accuracy.__dict__, "brier": comp.brier.__dict__, "passes_mr_rule": comp.passes,
             "accuracy_lower_above_panel_floor": comp.accuracy.lo > PANEL_NON_INFERIORITY_FLOOR,
             "accuracy_lower_above_zero": panel_superior(comp.accuracy)}, comp)


def main() -> None:
    rows = list(iter_jsonl(ITEMS))
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    verdict_t = set(manifest["verdict_templates"])
    reported_t = set(manifest["reported_only_templates"])
    osler = by_item(OSLER)

    mr = members(rows, "mr_test", templates=verdict_t)
    reported = members(rows, "mr_test", templates=reported_t)
    mr_without_pmc = [m for m in mr if m["template_id"] != PMC]
    panel = members(rows, "external_panel")
    knowledge = members(rows, "knowledge")
    held_out = members(rows, "mr_heldout", test_only=True)
    sizes = {"mr_verdict": len(mr), "mr_reported_only": len(reported), "external_panel": len(panel),
             "knowledge": len(knowledge), "mr_heldout_test": len(held_out)}

    mr_rec, mr_comp, panel_rec, panel_comp = {}, {}, {}, {}
    for name, paths in BASELINES.items():
        mr_rec[name], mr_comp[name] = compared("mr_verdict", name, mr, osler, by_item(paths["mr"] / "predictions.jsonl"))
        panel_rec[name], panel_comp[name] = compared("external_panel", name, panel, osler,
                                                     by_item(paths["panel"] / "predictions.jsonl"))
    guard_name = "zero-shot pplx-decider-flash-v1.2"
    guard_rec: dict[str, Any] = {"baseline": guard_name}
    guard = None
    try:
        diff, n_paired = accuracy_difference(knowledge, osler, by_item(BASELINES[guard_name]["knowledge"] / "predictions.jsonl"),
                                             n_resamples=N_RESAMPLES, seed=SEED)
        if n_paired != len(knowledge):
            raise ValueError(f"{len(knowledge) - n_paired} knowledge items are not paired")
        guard = diff
        guard_rec |= {"n_paired": n_paired, "accuracy": diff.__dict__, "passes": knowledge_guard(diff)}
    except ValueError as exc:
        guard_rec |= {"not_measured": str(exc)}

    verdict, reason = o2_verdict(mr_comp, panel_comp, guard)

    additional: dict[str, Any] = {}
    additional["reported_only_mr_templates"] = {
        name: compared("mr_reported_only", name, reported, osler, by_item(p["mr"] / "predictions.jsonl"))[0]
        for name, p in BASELINES.items()}
    failed = readout_failed("MedDecider-4B")
    held_pass = [m for m in held_out if m["template_id"] not in failed]
    additional["held_out_test"] = {
        "excluded_readout_fail_templates_for_MedDecider_4B": sorted(failed),
        **{name: compared("mr_heldout_test", name, held_pass, osler,
                          by_item(p["mr"] / "predictions.jsonl"))[0] for name, p in BASELINES.items()}}
    additional["mr_verdict_without_pmc"] = {
        name: compared("mr_verdict_without_pmc", name, mr_without_pmc, osler,
                       by_item(p["mr"] / "predictions.jsonl"))[0] for name, p in BASELINES.items()}
    additional["vs_zero_shot_Qwen_not_in_verdict"] = {
        name: compared("mr_verdict", name, mr, osler, by_item(p["mr"] / "predictions.jsonl"))[0]
        for name, p in ADDITIONAL.items()}
    if OSLER_V1.exists():
        v1 = by_item(OSLER_V1)
        additional["osler_v1_osler_4b_mr_verdict"] = {
            name: compared("mr_verdict", name, mr, v1, by_item(p["mr"] / "predictions.jsonl"))[0]
            for name, p in BASELINES.items()}

    record = {
        "kind": "osler_v2_W8_gate_o3", "written_utc": utcnow(), "git_commit": git_commit(REPO),
        "verdict": verdict, "reason": reason, "scope_sizes": sizes,
        "primary": {"mr_suite_v1": {"vs": dict(mr_rec)},
                    "external_panel": {"vs": dict(panel_rec)},
                    "knowledge_guard": guard_rec},
        "additional": additional, "non_inferiority_floor": PANEL_NON_INFERIORITY_FLOOR,
        "knowledge_guard_floor": KNOWLEDGE_GUARD_FLOOR, "n_resamples": N_RESAMPLES, "seed": SEED,
        "nine_b": "NOT MEASURED — deferred by the operator (D39)",
    }
    write_json(OUT, record)
    print(f"verdict: {verdict} — {reason}", flush=True)
    print(f"wrote {OUT.relative_to(REPO)}", flush=True)


if __name__ == "__main__":
    main()
