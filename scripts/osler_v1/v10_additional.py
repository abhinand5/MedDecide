"""V10 additional rows (labelled; never in the Gate O2 verdict, which v10_gate.py alone decides; CPU only).

Writes outputs/osler_v1/V10/additional.json (aggregates only) with:
  - the held-out MR test rows split into the held-out indicator (aki_creatinine_rise: the synthetic indicator template and the
    MR indicator template) and the held-out ICD family (heart_failure), against MedDecider-4B and zero-shot Qwen3.5-4B (V6);
  - the osler_v0 Osler-4B (arm L, the option-code head that osler_v0 kept) against Osler-4B (osler_v1) on the external panel and
    the knowledge items, the only V10 scopes the osler_v0 rows cover (outputs/osler_v0/O11);
  - a defect screen of the Osler-4B rows on the V10 items (below chance, constant answer, too good, per template).

The paired comparison is the gate's own function (meddecide.eval.gate.compare_rows): template macro accuracy and item-mean
Brier, bootstrapped within templates (1,000 resamples, seed 0).

Usage (CPU; no GPU):
    source scripts/pod_env.sh && uv run --frozen python -I scripts/osler_v1/v10_additional.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from meddecide.eval.gate import compare_rows
from meddecide.train.blocks import defect_screen
from meddecide.utils.io import iter_jsonl, write_json
from meddecide.utils.provenance import git_commit, utcnow

REPO = Path(__file__).resolve().parents[2]
ITEMS = REPO / "data/bench/osler_v1/v10_items.jsonl"
V10 = REPO / "outputs/osler_v1/V10"
V6_BASELINES = REPO / "outputs/osler_v1/V6/baselines"
O2_BASELINES = REPO / "outputs/osler_v0/O2"
OSLER_V0_L = REPO / "outputs/osler_v0/O11/osler_4b_L/both_orders.jsonl"
OUT = V10 / "additional.json"
V6_SUMMARY = V6_BASELINES.parent / "baseline_summary.json"
BASELINE_KEYS = {"MedDecider-4B": "meddecider-4b", "zero-shot Qwen3.5-4B": "qwen35-4b"}
GENERATED_SOURCES = (None, "synthea")  # V1 code-generated records (source None) and V5a Synthea pages
REAL_TEXT_SOURCES = ("pmc_patients", "codiesp", "mkurman", "mtsamples")  # V5b real clinical text
INDICATOR_TEMPLATES = ("synthea_indicator_evidence_v1", "mr_indicator_evidence_v1")
FAMILY_TEMPLATES = ("mr_icd_support_template_v2",)
N_RESAMPLES = 1000
SEED = 0


def readout_failed(name: str) -> set[str]:
    """The held-out templates whose readout fails the gate for one baseline (the V6 summary)."""
    summary = json.loads(V6_SUMMARY.read_text(encoding="utf-8"))
    return set(summary["models"][BASELINE_KEYS[name]]["sets"]["mr_heldout"]["readout_fail_templates"])


def by_item(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    return {str(row["item_id"]): row for row in iter_jsonl(path)}


def members(rows: Sequence[Mapping[str, Any]], scope: str, *, templates: tuple[str, ...] | None = None,
            test_only: bool = False) -> list[dict[str, Any]]:
    keep = [r for r in rows if r["v10_scope"] == scope and (not test_only or r["split"] == "test")
            and (templates is None or r["template_id"] in templates)]
    return [{"item_id": r["item_id"], "template_id": r["template_id"]} for r in keep]


def paired(scope: str, name: str, scope_rows: list[dict[str, Any]], osler: Mapping[str, Any],
           base: Mapping[str, Any]) -> dict[str, Any]:
    """One paired row: the gate's comparison on these items, or the reason it could not be measured."""
    try:
        result = compare_rows(scope, name, scope_rows, osler, base, n_resamples=N_RESAMPLES, seed=SEED)
    except ValueError as exc:
        return {"scope": scope, "baseline": name, "not_measured": str(exc)}
    if result.n_osler_missing:
        return {"scope": scope, "baseline": name, "not_measured": f"{result.n_osler_missing} scope items have no Osler row"}
    comp = result.comparison
    if comp is None:
        return {"scope": scope, "baseline": name, "not_measured": "no comparison could be formed"}
    return {"scope": scope, "baseline": name, "n_scope": result.n_scope, "n_paired": result.n_paired,
            "n_baseline_skipped": result.n_baseline_skipped, "n_baseline_missing": result.n_baseline_missing,
            "n_osler_missing": result.n_osler_missing,
            "accuracy": {"point": comp.accuracy.point, "lo": comp.accuracy.lo, "hi": comp.accuracy.hi},
            "brier": {"point": comp.brier.point, "lo": comp.brier.lo, "hi": comp.brier.hi}}


def brier_and_correct(row: Mapping[str, Any]) -> tuple[float, float]:
    """Correct (1.0 or 0.0) and the multi-class Brier score of one prediction row."""
    probs = [float(p) for p in row["probs"]]
    gold = list(row["option_keys"]).index(row["gold"])
    predicted = max(range(len(probs)), key=probs.__getitem__)
    brier = sum((p - (1.0 if k == gold else 0.0)) ** 2 for k, p in enumerate(probs))
    return (1.0 if predicted == gold else 0.0), brier


def per_template(scope_rows: Sequence[Mapping[str, Any]], osler: Mapping[str, Any],
                 base: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Per template on the paired items: Osler minus baseline accuracy and Brier (item means), worst Brier first."""
    sums: dict[str, list[float]] = defaultdict(lambda: [0.0] * 6)  # n, osler correct, base correct, osler brier, base brier
    for item in scope_rows:
        o, b = osler.get(item["item_id"]), base.get(item["item_id"])
        if o is None or b is None or b.get("status") != "scored":
            continue
        co, bo = brier_and_correct(o)
        cb, bb = brier_and_correct(b)
        slot = sums[item["template_id"]]
        slot[0] += 1
        slot[1] += co
        slot[2] += cb
        slot[3] += bo
        slot[4] += bb
    rows = []
    for template, (n, co, cb, bo, bb, _) in sums.items():
        rows.append({"template_id": template, "paired": int(n), "osler_accuracy": co / n, "baseline_accuracy": cb / n,
                     "accuracy_diff": (co - cb) / n, "brier_diff": (bo - bb) / n})
    return sorted(rows, key=lambda r: r["brier_diff"], reverse=True)


def screen_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Predictions in the form defect_screen reads: the argmax option against the gold key's index."""
    out = []
    for row in rows:
        probs = [float(p) for p in row["probs"]]
        keys = list(row["option_keys"])
        predicted = max(range(len(probs)), key=probs.__getitem__)
        gold = keys.index(row["gold"])
        out.append({"template_id": row["template_id"], "n_options": len(probs), "predicted_index": predicted,
                    "gold_index": gold, "correct": predicted == gold})
    return out


def main() -> None:
    rows = list(iter_jsonl(ITEMS))
    osler = by_item(V10 / "osler_4b" / "both_orders.jsonl")
    if len(osler) != len(rows):
        raise SystemExit(f"the Osler-4B rows ({len(osler)}) do not cover the V10 items ({len(rows)})")

    held_indicator = members(rows, "mr_heldout", templates=INDICATOR_TEMPLATES, test_only=True)
    held_family = members(rows, "mr_heldout", templates=FAMILY_TEMPLATES, test_only=True)
    baselines = {"MedDecider-4B": by_item(V6_BASELINES / "meddecider-4b" / "predictions.jsonl"),
                 "zero-shot Qwen3.5-4B": by_item(V6_BASELINES / "qwen35-4b" / "predictions.jsonl")}
    held_out: dict[str, Any] = {"indicator_test_items": len(held_indicator), "family_test_items": len(held_family),
                                "rows": []}
    for label, scope_rows, scope in (("indicator (aki_creatinine_rise)", held_indicator, "mr_heldout_indicator_test"),
                                     ("ICD family (heart_failure)", held_family, "mr_heldout_family_test")):
        for name, base in baselines.items():
            excluded = readout_failed(name)  # a failing readout cell reports no accuracy (V6; CLAIMS V089)
            kept = [m for m in scope_rows if m["template_id"] not in excluded]
            record = paired(scope, name, kept, osler, base)
            held_out["rows"].append({"check": label, "readout_excluded": sorted(excluded & {m["template_id"] for m in scope_rows}),
                                     **record})
    synthea_test = sum(1 for m in held_indicator if m["template_id"] == "synthea_indicator_evidence_v1")
    held_out["rows"].append({"check": f"held-out synthetic indicator (synthea_indicator_evidence_v1; {synthea_test} test items)",
                             "baseline": "MedDecider-4B",
                             "not_measured": "READOUT_FAIL for MedDecider-4B on mr_heldout (V6 readout gate; CLAIMS V089): "
                                             "no accuracy is reported"})

    panel = members(rows, "external_panel")
    knowledge = members(rows, "knowledge")
    v0 = by_item(OSLER_V0_L)
    v0_rows = {"items_in_file": len(v0), "panel_items": len(panel), "knowledge_items": len(knowledge)}
    osler_v0 = [{"check": "osler_v0 Osler-4B arm L (additional)", **paired(scope, "osler_v0 Osler-4B (arm L)", scope_rows,
                                                                       osler, v0)}
                for scope, scope_rows in (("external_panel", panel), ("knowledge", knowledge))]
    osler_v0_9b = "NOT MEASURED — no per-item O2 predictions for the osler_v0 Osler-9B on the V10 items were found under outputs/osler_v0"

    scored_scopes = ("mr_test", "external_panel", "knowledge")
    all_pred = [osler[str(r["item_id"])] for r in rows if r["v10_scope"] in scored_scopes]
    mr_pred = [osler[str(r["item_id"])] for r in rows if r["v10_scope"] == "mr_test"]
    screen_all = defect_screen(screen_rows(all_pred), losses=[])
    screen_mr = defect_screen(screen_rows(mr_pred), losses=[])
    too_good = sorted(set(screen_all["too_good_templates"]) | set(screen_mr["too_good_templates"]))
    template_accuracy = {}
    for name in too_good:
        flags = [bool(r["correct"]) for r in screen_rows([r for r in all_pred if r["template_id"] == name])]
        template_accuracy[name] = {"items": len(flags), "correct": sum(flags), "accuracy": sum(flags) / len(flags)}
    # ADVISORY §2, outcome 3: which templates the MR loss against MedDecider-4B sits on (per template, paired items)
    mr_vs_med = per_template(members(rows, "mr_test"), osler, baselines["MedDecider-4B"])
    macro_acc = sum(r["accuracy_diff"] for r in mr_vs_med) / len(mr_vs_med)
    # ADVISORY V10: the MR suite by part. Generated: the V1 code-generated records (source None) and the V5a Synthea pages;
    # real text: the V5b sources (PMC-Patients, CodiEsp, mkurman, MTSamples). MIMIC is absent.
    parts: dict[str, Any] = {"mimic": {"not_measured": "MIMIC is absent (V5c); no MIMIC rows exist in the V10 items"}}
    for label, sources in (("generated", GENERATED_SOURCES), ("real_text", REAL_TEXT_SOURCES)):
        part_members = [{"item_id": r["item_id"], "template_id": r["template_id"]} for r in rows
                        if r["v10_scope"] == "mr_test" and r.get("source") in sources]
        parts[label] = {"items": len(part_members),
                        "rows": [paired(f"mr_test_{label}", name, part_members, osler, base)
                                 for name, base in baselines.items()]}
    # post hoc (the templates were chosen after the screen flagged them): the MR suite without the flagged templates
    mr_members = members(rows, "mr_test")
    mr_unflagged = [m for m in mr_members if m["template_id"] not in too_good]
    unflagged = [paired("mr_test_without_flagged_templates", name, mr_unflagged, osler, base)
                 for name, base in baselines.items()]

    record = {"kind": "osler_v1_V10_additional", "built_at_utc": utcnow(), "git_commit": git_commit(REPO),
              "command": "scripts/osler_v1/v10_additional.py", "n_resamples": N_RESAMPLES, "seed": SEED,
              "held_out": held_out, "osler_v0_arm_L": {"coverage": v0_rows, "rows": osler_v0},
              "osler_v0_osler_9b": osler_v0_9b,
              "mr_parts": parts,
              "mr_per_template_vs_meddecider": {"templates": len(mr_vs_med), "accuracy_template_macro_diff": macro_acc,
                                                "rows": mr_vs_med},
              "mr_without_flagged_templates": {"post_hoc": True, "templates_removed": too_good,
                                               "items": len(mr_unflagged), "mr_test_items": len(mr_members),
                                               "rows": unflagged},
              "defect_screen": {"all_v10_scopes": screen_all, "mr_test": screen_mr,
                                "too_good_template_accuracy": template_accuracy}}
    write_json(OUT, record)
    print(json.dumps({"held_out_rows": [(r["check"], r.get("n_paired"), r.get("not_measured")) for r in held_out["rows"]],
                      "osler_v0_rows": [(r["check"], r.get("n_paired"), r.get("not_measured")) for r in osler_v0],
                      "screen_all_flags": screen_all["flags"], "screen_mr_flags": screen_mr["flags"],
                      "too_good_all": screen_all["too_good_templates"], "too_good_mr": screen_mr["too_good_templates"]},
                     indent=1, default=str))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
