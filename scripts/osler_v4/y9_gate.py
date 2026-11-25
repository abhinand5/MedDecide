"""Y9 (osler_v4 ADVISORY Y9; D49; STATE §7 Y9 pre-registration): Gate O4 and the size-class scoreboard from the scored predictions (CPU).

Reads the final model's and each baseline's predictions on the Y9 sets (data/bench/osler_v4/y9; the MIMIC suite and held-out runs of
the baselines are the Y4 runs). Each row is normalised against its item (meddecide.eval.gate_o4.normalise_row), paired with the final
model's row by item, and checked with the readout gate (D12) and the D44(a) pairing rule. A run that has not produced its predictions
is listed as NOT RUN, never skipped.

Writes outputs/osler_v4/Y9/gate.json (every figure), loops/osler_v4/gate_o4.md (the gate) and loops/osler_v4/results.md (the
scoreboard). Aggregates only: no item text, identifier or prediction is written to the committed files.

Usage (CPU; after the scoring chain):
    uv run --frozen python scripts/osler_v4/y9_gate.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from meddecide.eval.code_split import place_section, render_code_section
from meddecide.eval.gate_o4 import (
    MEDDECIDER_OPTION_CAP,
    CheckResult,
    check,
    comparison,
    gate,
    normalise_row,
    pair_rows,
    readout_failures,
)
from meddecide.eval.paired import item_brier
from meddecide.train.mix_items import mix_row_to_item
from meddecide.utils.io import iter_jsonl, write_json
from meddecide.utils.provenance import utcnow

REPO = Path(__file__).resolve().parents[2]
SETS_DIR = REPO / "data" / "bench" / "osler_v4" / "y9"
Y9_RUNS = REPO / "outputs" / "osler_v4" / "Y9" / "baselines"
Y4_RUNS = REPO / "outputs" / "osler_v4" / "Y4" / "baselines"
OUT_JSON = REPO / "outputs" / "osler_v4" / "Y9" / "gate.json"
GATE_MD = REPO / "loops" / "osler_v4" / "gate_o4.md"
RESULTS_MD = REPO / "loops" / "osler_v4" / "results.md"
CODE_SPLIT_JSON = REPO / "outputs" / "osler_v4" / "Y8" / "code_split.json"  # the Y8 steer's additional rows, kept in results.md

GATE_CHECKS = (("i", "mr_verdict", "mr_or_mimic"), ("ii", "mimic_suite", "mr_or_mimic"),
               ("iii", "panel", "accuracy"), ("iv", "knowledge", "accuracy"))
SETS = ("mr_verdict", "mr_reported", "mr_heldout_test", "mimic_suite", "mimic_heldout", "panel", "knowledge", "mednli_test",
        "pairs_test")
# model key -> (label, option cap of its interface, file prefix of its Y9 runs, Y4 run names for the MIMIC sets)
MODELS: dict[str, dict[str, Any]] = {
    "final": {"label": "Osler-4B (final, Y8)", "cap": None, "prefix": "final", "y4": {}},
    "pplx_flash": {"label": "zero-shot pplx-decider-flash-v1.2", "cap": None, "prefix": "pplx",
                   "y4": {"mimic_suite": "suite_flash", "mimic_heldout": "heldout_flash"}},
    "meddecider_4b": {"label": "MedDecider-4B", "cap": MEDDECIDER_OPTION_CAP, "prefix": "md4b",
                      "y4": {"mimic_suite": "suite_meddecider4b", "mimic_heldout": "heldout_meddecider4b"}},
    "osler_v2": {"label": "osler_v2 Osler-4B", "cap": None, "prefix": "osler_v2",
                 "y4": {"mimic_suite": "suite_osler_v2", "mimic_heldout": "heldout_osler_v2"}},
    "meddecider_9b": {"label": "MedDecider-9B", "cap": MEDDECIDER_OPTION_CAP, "prefix": "md9b",
                      "y4": {"mimic_suite": "suite_meddecider9b"}},
    "jev_9b": {"label": "JEV-9B", "cap": None, "prefix": "jev9b", "y4": {}, "ids": "raw"},
    "pplx_27b": {"label": "pplx-decider-v1.2-27b (ceiling, additional)", "cap": None, "prefix": "pplx27b", "y4": {}},
}
BASELINES_GATE = ("pplx_flash", "meddecider_4b")
Q6_NOTE = "STATE §6 question 6: the eight templates that pass for MedDecider-4B on the MIMIC audit suite"


def predictions_path(model: str, set_name: str) -> Path | None:
    """Where a model's predictions on a set are, or None when the set has no run (reported as NOT RUN)."""
    spec = MODELS[model]
    if set_name in spec["y4"]:
        return Y4_RUNS / spec["y4"][set_name] / "predictions.jsonl"
    if model == "jev_9b":
        return Y9_RUNS / f"jev9b_{set_name}" / "jev-9b" / "predictions.jsonl"
    return Y9_RUNS / f"{spec['prefix']}_{set_name}" / "predictions.jsonl"


def load_items(set_name: str) -> list[dict[str, Any]]:
    """The set's items, keyed by the recomputed item id (mix_row_to_item), which the trainer-based and pplx runners write; the file's
    own id is kept as ``raw_id``, which the JEV-9B runner writes."""
    items: list[dict[str, Any]] = []
    for row in iter_jsonl(SETS_DIR / f"{set_name}.jsonl"):
        items.append({**row, "raw_id": str(row["item_id"]), "item_id": mix_row_to_item(row).item_id})
    if len({i["item_id"] for i in items}) != len(items):
        raise ValueError(f"{set_name}: recomputed item ids are not unique")
    return items


def load_scored(model: str, set_name: str, items: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The model's normalised scored rows on a set, with the counts of skipped rows and of argmax disagreements; None if not run."""
    path = predictions_path(model, set_name)
    if path is None or not path.exists():
        return None
    key = "raw_id" if MODELS[model].get("ids", "item") == "raw" else "item_id"
    by_id = {str(i[key]): i for i in items}
    rows: list[dict[str, Any]] = []
    skipped = 0
    disagree = 0
    for raw in iter_jsonl(path):
        item = by_id.get(str(raw["item_id"]))
        if item is None:
            raise ValueError(f"{model} {set_name}: a prediction for an item outside the set")
        if raw.get("status", "scored") != "scored":
            skipped += 1
            continue
        row = normalise_row(raw, item)
        if "correct" in raw and bool(raw["correct"]) != row["correct"]:
            disagree += 1
        rows.append(row)
    return {"rows": rows, "path": str(path.relative_to(REPO)), "skipped_by_runner": skipped, "argmax_disagreements": disagree}


def scoreboard_line(rows: list[dict[str, Any]], items_by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """A model's own figures on the items it scored: template macro accuracy, Brier (item mean) and the readout status."""
    if not rows:  # every item unpaired or unscored for this model: reported, never a figure
        return {"scored_items": 0, "template_macro_accuracy": None, "brier_item_mean": None, "templates": 0,
                "readout_failures": {}, "status": "NOT MEASURED (no scored items)"}
    templates: dict[str, list[float]] = {}
    briers: list[float] = []
    for r in rows:
        templates.setdefault(r["template_id"], []).append(1.0 if r["correct"] else 0.0)
        item = items_by_id[r["item_id"]]
        gold = [str(o["key"]) for o in item["options"]].index(str(item["gold"]))
        briers.append(item_brier(r["probs"], gold))
    failing = readout_failures(rows)
    macro = sum(sum(v) / len(v) for v in templates.values()) / len(templates)
    return {"scored_items": len(rows), "template_macro_accuracy": round(macro, 6), "brier_item_mean": round(sum(briers) / len(briers), 6),
            "templates": len(templates), "readout_failures": failing,
            "status": "READOUT_FAIL (D12)" if failing else "PASS"}


def run_checks(model: str, base: str, final_by_set: dict[str, Any], base_by_set: dict[str, Any],
               items_by_set: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """The four Gate O4 checks of the final model against one baseline, with the pairing and readout facts of each."""
    results: dict[str, CheckResult] = {}
    facts: dict[str, Any] = {}
    for name, set_name, kind in GATE_CHECKS:
        f, b = final_by_set.get(set_name), base_by_set.get(set_name)
        if f is None or b is None:
            missing = "final model" if f is None else base
            results[name] = CheckResult(name, "NOT RUN", f"no predictions for {missing} on {set_name}")
            continue
        items = items_by_set[set_name]
        pairing = pair_rows(f["rows"], b["rows"], items, option_cap=MODELS[base]["cap"])
        f_fail = readout_failures(f["rows"])
        b_fail = readout_failures(b["rows"])
        restriction = None
        note = ""
        if base == "meddecider_4b" and set_name == "mimic_suite" and b_fail:
            restriction = frozenset(set(pairing.set_templates) - set(f_fail) - set(b_fail))
            note = Q6_NOTE
            if len(restriction) != 8:
                results[name] = CheckResult(name, "NOT MEASURED", f"the templates passing for both models number {len(restriction)}, "
                                            f"not the eight of the ruling (STATE §6 question 6)")
                continue
        results[name] = check(f"{name} {set_name}", pairing, kind=kind, model_failures=f_fail, base_failures=b_fail,
                              restriction=restriction, restriction_note=note)
        facts[name] = {"set": set_name, "items": len(items), "paired": len(pairing.pairs), "unpaired": pairing.unpaired,
                       "unpaired_share": round(pairing.unpaired_share, 6), "dropped_templates": sorted(pairing.dropped_templates),
                       "final_readout_failures": f_fail, "baseline_readout_failures": b_fail,
                       "restricted_to": sorted(restriction) if restriction is not None else None}
    verdict = gate(results) if all(r.verdict != "NOT RUN" for r in results.values()) else "NOT RUN"
    return {"verdict": verdict, "checks": {k: {"check": r.name, "verdict": r.verdict, "reason": r.reason,
                                              "accuracy": _diff(r.comparison.accuracy) if r.comparison else None,
                                              "brier": _diff(r.comparison.brier) if r.comparison else None,
                                              "items": r.comparison.items if r.comparison else None,
                                              "templates": list(r.comparison.templates) if r.comparison else None}
                                          for k, r in results.items()}, "facts": facts}


def _diff(d: Any) -> dict[str, float]:
    return {"point": round(d.point, 6), "lower": round(d.lo, 6), "upper": round(d.hi, 6)}


def additional(model: str, base: str, final_by_set: dict[str, Any], base_by_set: dict[str, Any],
               items_by_set: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """Labelled additional comparisons (no verdict): the other sets and the panel per dataset."""
    out: dict[str, Any] = {}
    for set_name in ("mr_reported", "mr_heldout_test", "mimic_heldout", "mednli_test", "pairs_test", "panel"):
        f, b = final_by_set.get(set_name), base_by_set.get(set_name)
        if f is None or b is None:
            out[set_name] = {"status": "NOT RUN"}
            continue
        items = items_by_set[set_name]
        pairing = pair_rows(f["rows"], b["rows"], items, option_cap=MODELS[base]["cap"])
        if not pairing.pairs:
            out[set_name] = {"status": "NOT MEASURED (no paired items)"}
            continue
        result = comparison(pairing)
        out[set_name] = {"items": result.items, "templates": list(result.templates), "accuracy": _diff(result.accuracy),
                         "brier": _diff(result.brier), "unpaired": pairing.unpaired,
                         "unpaired_share": round(pairing.unpaired_share, 6),
                         "readout_failures": {"final": readout_failures(f["rows"]), "baseline": readout_failures(b["rows"])}}
        if set_name == "panel":
            per = {}
            for template in sorted(result.templates):
                part = comparison(pairing, templates=frozenset({template}))
                per[template] = {"items": part.items, "accuracy": _diff(part.accuracy), "brier": _diff(part.brier)}
            out["panel_per_dataset"] = per
    # post hoc (decided after the verdict was read; osler_v1 flagged these two templates as too good, V180 to V183): the MR verdict
    # without them, labelled additional; the verdict above keeps all seven templates as pre-registered
    flagged = frozenset({"pmc_similar_patient_v1", "mr_icd_support_template_v2"})
    f, b = final_by_set.get("mr_verdict"), base_by_set.get("mr_verdict")
    if f is not None and b is not None:
        pairing = pair_rows(f["rows"], b["rows"], items_by_set["mr_verdict"], option_cap=MODELS[base]["cap"])
        kept = frozenset(comparison(pairing).templates) - flagged
        part = comparison(pairing, templates=kept)
        out["mr_verdict_without_osler_v1_flags"] = {"post_hoc": True, "items": part.items, "templates": sorted(kept),
                                                    "accuracy": _diff(part.accuracy), "brier": _diff(part.brier)}
    return out


def _fmt(d: dict[str, float] | None) -> str:
    if d is None:
        return "—"
    return f"{d['point']:+.4f} [{d['lower']:+.4f}; {d['upper']:+.4f}]"


def write_gate_markdown(report: dict[str, Any]) -> str:
    lines = ["# osler_v4 — Gate O4 (D49): the final Osler-4B against the two gate baselines", "",
             f"Generated {report['generated_utc']} by `scripts/osler_v4/y9_gate.py` from `outputs/osler_v4/Y9/gate.json` (aggregates only).",
             "Pre-registration: STATE §7, Y9 pre-registration (recorded before the final model was scored). Pairing per D44(a) as "
             "pre-registered; readout per D12; the MIMIC audit suite against MedDecider-4B on the eight passing templates "
             "(STATE §6 question 6). Accuracy is the template macro of the paired differences, final minus baseline; Brier is the item "
             "mean, final minus baseline (a negative value favours the final model). Bounds: paired bootstrap within templates, "
             "1,000 resamples, seed 0.", "",
             "## Verdict", "", "| baseline | Gate O4 | (i) MR verdict | (ii) MIMIC audit suite | (iii) external panel | (iv) knowledge |",
             "|---|---|---|---|---|---|"]
    for base in BASELINES_GATE:
        g = report["gate"][base]
        cells = []
        for name in ("i", "ii", "iii", "iv"):
            c = g["checks"][name]
            detail = f"{c['verdict']}" + (f" (accuracy {_fmt(c['accuracy'])}" + (f"; Brier {_fmt(c['brier'])})" if c["brier"] else ")")
                                          if c["accuracy"] else "")
            cells.append(detail)
        lines.append(f"| {MODELS[base]['label']} | **{g['verdict']}** | " + " | ".join(cells) + " |")
    lines += ["", "## Checks in detail", ""]
    for base in BASELINES_GATE:
        g = report["gate"][base]
        lines += [f"### Against {MODELS[base]['label']}", ""]
        for name in ("i", "ii", "iii", "iv"):
            c = g["checks"][name]
            f = g["facts"].get(name, {})
            lines.append(f"- ({name}) {c['check']}: **{c['verdict']}**. {c['reason']}.")
            if f:
                lines.append(f"  items {f['items']}; paired {f['paired']}; unpaired {f['unpaired'] or 'none'} "
                             f"(share {f['unpaired_share']:.2%}); dropped templates {f['dropped_templates'] or 'none'}; readout "
                             f"failures final {f['final_readout_failures'] or 'none'}, baseline {f['baseline_readout_failures'] or 'none'}"
                             + (f"; restricted to {len(f['restricted_to'])} templates" if f["restricted_to"] else "") + ".")
            if c["accuracy"]:
                lines.append(f"  accuracy {_fmt(c['accuracy'])}; Brier {_fmt(c['brier'])}; items {c['items']}; templates {c['templates']}.")
        lines.append("")
    lines += ["## Additional rows (labelled; no verdict)", ""]
    for base in BASELINES_GATE:
        lines.append(f"### Against {MODELS[base]['label']}")
        for set_name, entry in report["additional"][base].items():
            if set_name == "panel_per_dataset":
                continue
            if "accuracy" not in entry:
                lines.append(f"- {set_name}: {entry.get('status', 'NOT RUN')}")
                continue
            label = f"{set_name} (post hoc; additional)" if entry.get("post_hoc") else set_name
            lines.append(f"- {label}: accuracy {_fmt(entry['accuracy'])}; Brier {_fmt(entry['brier'])}; items {entry['items']}; "
                         f"unpaired {entry.get('unpaired') or 'none'}.")
        per = report["additional"][base].get("panel_per_dataset")
        if per:
            lines.append("- panel per dataset (accuracy, final minus baseline):")
            for template, row in per.items():
                lines.append(f"  - {template}: {_fmt(row['accuracy'])} on {row['items']} items")
        lines.append("")
    return "\n".join(lines) + "\n"


def write_results_markdown(report: dict[str, Any]) -> str:
    lines = ["# osler_v4 — size-class scoreboard (Y9; additional rows, labelled; no verdict)", "",
             f"Generated {report['generated_utc']} by `scripts/osler_v4/y9_gate.py` from `outputs/osler_v4/Y9/gate.json` (aggregates only). "
             "Each model's figures are its own, on the items it scored: the template macro accuracy, the item-mean Brier, and the readout "
             "status (D12). A model that cannot take an item (ten letters) has that item unscored here (D44(a)). Gate O4 is in gate_o4.md.",
             ""]
    for set_name in SETS:
        lines += [f"## {set_name}", "", "| model | scored items | template macro accuracy | Brier (item mean) | readout | skipped by runner |",
                  "|---|---|---|---|---|---|"]
        for model, spec in MODELS.items():
            e = report["scoreboard"].get(set_name, {}).get(model, {"status": "NOT RUN"})
            if "scored_items" not in e:
                lines.append(f"| {spec['label']} | NOT RUN | — | — | — | — |")
                continue
            macro = "—" if e["template_macro_accuracy"] is None else f"{e['template_macro_accuracy']:.4f}"
            brier = "—" if e["brier_item_mean"] is None else f"{e['brier_item_mean']:.4f}"
            lines.append(f"| {spec['label']} | {e['scored_items']} | {macro} | {brier} "
                         f"| {e['status']} | {e.get('skipped_by_runner', '—')} |")
        lines.append("")
    text = "\n".join(lines) + "\n"
    if CODE_SPLIT_JSON.exists():  # the additional rows are written by y8_code_split.py; a rewrite of this file keeps them
        text = place_section(text, render_code_section(json.loads(CODE_SPLIT_JSON.read_text(encoding="utf-8"))))
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--write-md", action="store_true", help="also write loops/osler_v4/gate_o4.md and results.md (after the runs)")
    args = parser.parse_args()
    items_by_set = {s: load_items(s) for s in SETS}
    scored: dict[str, dict[str, Any]] = {}
    for model in MODELS:
        scored[model] = {}
        for set_name in SETS:
            loaded = load_scored(model, set_name, items_by_set[set_name])
            if loaded is not None:
                scored[model][set_name] = loaded
    final = scored["final"]
    report: dict[str, Any] = {"generated_utc": utcnow(), "gate": {}, "additional": {}, "scoreboard": {}, "runs": {}}
    for model in MODELS:
        report["runs"][model] = {s: (scored[model][s]["path"] if s in scored[model] else "NOT RUN") for s in SETS}
    for base in BASELINES_GATE:
        report["gate"][base] = run_checks("final", base, final, scored[base], items_by_set)
        report["additional"][base] = additional("final", base, final, scored[base], items_by_set)
    for set_name in SETS:
        by_id = {str(i["item_id"]): i for i in items_by_set[set_name]}
        report["scoreboard"].setdefault(set_name, {})
        for model in MODELS:
            if set_name in scored[model]:
                entry = scoreboard_line(scored[model][set_name]["rows"], by_id)
                entry["skipped_by_runner"] = scored[model][set_name]["skipped_by_runner"]
                entry["argmax_disagreements"] = scored[model][set_name]["argmax_disagreements"]
                report["scoreboard"][set_name][model] = entry
            else:
                report["scoreboard"][set_name][model] = {"status": "NOT RUN"}
    write_json(OUT_JSON, report)
    if args.write_md:
        GATE_MD.write_text(write_gate_markdown(report), encoding="utf-8")
        RESULTS_MD.write_text(write_results_markdown(report), encoding="utf-8")
    print(f"gate: {({b: report['gate'][b]['verdict'] for b in BASELINES_GATE})}", flush=True)


if __name__ == "__main__":
    main()
