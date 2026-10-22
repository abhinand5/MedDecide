"""Y4 baselines report (osler_v4 ADVISORY Y4 §4): the accuracy of each baseline on the frozen MIMIC audit suite, the held-out items
(additional rows), the MIMIC partition of fast dev v4 and CG v2, with the readout checks of meddecide.eval.health.

Readout checks, per (set, model, template) cell: the constant-answer check (the modal answer on at least DEFAULT_MAX_MODAL_SHARE of
the items, on a template whose majority share is at most DEFAULT_DEGENERATE_MAJORITY_CEILING) and the below-chance check (the 95 %
CI upper bound of the accuracy is below the mean chance 1/n). The label-mass check is not applied to the authors' rows: their
decision models are trained readouts that normalise over the offered options, so label mass is a diagnostic there (D21). A cell
that fails reads READOUT_FAIL and its accuracy is not reported as a result (D12).

A set-level figure is a result only when every template cell in it passes (D12); otherwise the set's passing-cell figure is an
additional, labelled row. The set's macro is the template macro (the unweighted mean of the per-template accuracies, as in
meddecide.eval.metrics and D24). No readout check is pooled across templates, because the gold keys of different templates
are not the same classes.

Writes outputs/osler_v4/Y4/baselines_report.json (aggregates) and loops/osler_v4/baselines.md (tables; no item text).
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from meddecide.eval.health import DEFAULT_DEGENERATE_MAJORITY_CEILING, DEFAULT_MAX_MODAL_SHARE
from meddecide.eval.metrics import template_macro_accuracy
from meddecide.eval.readout_gate import cell_gate
from meddecide.utils.io import iter_jsonl, write_json

BASELINES = Path("outputs/osler_v4/Y4/baselines")
OUT = Path("outputs/osler_v4/Y4/baselines_report.json")
MD = Path("loops/osler_v4/baselines.md")
SETS = ("suite", "heldout", "fastdev_mimic", "cg_v2")
MODELS = {"flash": "pplx-decider-flash-v1.2 (zero-shot, authors' code)",
          "meddecider4b": "MedDecider-4B (authors' code)",
          "meddecider9b": "MedDecider-9B (authors' code; extra row)",
          "osler_v2": "osler_v2 Osler-4B (trainer-based)"}


def top_gap(probs: list[float]) -> float:
    """The gap between the two largest probabilities (zero when they tie)."""
    top = sorted(probs, reverse=True)
    return top[0] - top[1]


def cell_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """One template's cell: the readout gate of meddecide.eval.readout_gate, and diagnostics that are not gates."""
    gate = cell_gate([str(r["gold_key"]) for r in rows], [str(r["predicted_key"]) for r in rows],
                     [bool(r["correct"]) for r in rows], [1.0 / len(r["probs"]) for r in rows])
    # diagnostics only (not gates): a constant answer that is a tie artifact shows a zero median gap and many exact ties
    n = len(rows)
    gaps = sorted(top_gap(r["probs"]) for r in rows)
    positions = Counter(max(range(len(r["probs"])), key=lambda i: r["probs"][i]) for r in rows)
    return {"items": n, "micro_accuracy": round(gate.accuracy, 4), "chance": round(gate.chance, 4),
            "majority_share": round(gate.majority_share, 4), "modal_share": round(gate.modal_share, 4),
            "modal_key": gate.modal_key, "median_top_gap": round(gaps[n // 2], 6),
            "tied_top_share": round(sum(g < 1e-6 for g in gaps) / n, 4),
            "argmax_position_counts": {str(k): v for k, v in sorted(positions.items())}, "status": gate.status}


def set_report(rows: list[dict[str, Any]], cells: dict[str, dict[str, Any]], missing: list[str]) -> dict[str, Any]:
    """A set's figures: reportable only when every template cell passes (D12); the macro is the template macro (D24)."""
    n = len(rows)
    correct = [bool(r["correct"]) for r in rows]
    failed = [t for t, c in cells.items() if c["status"] != "PASS"]
    status = ("PASS" if not failed else
              f"READOUT_FAIL — {len(failed)} of {len(cells)} template cells fail the gate ({', '.join(failed)}); "
              "the set-level accuracy is not reported as a result (D12)")
    if missing:
        # a run that cannot score a template (more than ten options, C086) has a set-level figure over fewer templates
        status += f"; templates not scored by this run: {', '.join(missing)} (C086)"
    report: dict[str, Any] = {
        "items": n, "micro_accuracy": round(sum(correct) / n, 4),
        "template_macro_accuracy": round(template_macro_accuracy([float(c) for c in correct], [str(r["template_id"]) for r in rows]), 4),
        "chance": round(sum(1.0 / len(r["probs"]) for r in rows) / n, 4), "reportable": not failed, "status": status,
        "templates_missing": missing}
    if failed:
        passing = [r for r in rows if str(r["template_id"]) not in failed]
        report["passing_cells_additional"] = {
            "label": "additional, not a result: micro accuracy over the template cells that pass the gate",
            "templates": sorted(t for t in cells if t not in failed), "items": len(passing),
            "micro_accuracy": round(sum(bool(r["correct"]) for r in passing) / len(passing), 4) if passing else None}
    return report


def main() -> None:
    report: dict[str, Any] = {"checks": {"max_modal_share": DEFAULT_MAX_MODAL_SHARE,
                                         "degenerate_majority_ceiling": DEFAULT_DEGENERATE_MAJORITY_CEILING,
                                         "label_mass": "not applied (D21: trained readouts normalising over offered options)",
                                         "macro": "template macro: the unweighted mean of the per-template accuracies (D24)"},
                              "files": {}, "sets": {}}
    coverage: dict[str, set[str]] = defaultdict(set)  # every template that any run of the set scored
    for set_name in SETS:
        for model in MODELS:
            path = BASELINES / f"{set_name}_{model}" / "predictions.jsonl"
            if path.exists():
                coverage[set_name].update(str(r["template_id"]) for r in iter_jsonl(path))
    for set_name in SETS:
        for model in MODELS:
            path = BASELINES / f"{set_name}_{model}" / "predictions.jsonl"
            if not path.exists():
                report["files"][f"{set_name}_{model}"] = "not run"
                continue
            rows = [r for r in iter_jsonl(path) if r.get("status", "scored") == "scored"]
            report["files"][f"{set_name}_{model}"] = {"scored": len(rows)}
            by_template: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for r in rows:
                by_template[str(r["template_id"])].append(r)
            cells = {t: cell_report(v) for t, v in sorted(by_template.items())}
            missing = sorted(coverage[set_name] - set(by_template))
            overall = set_report(rows, cells, missing) if rows else None
            report["sets"][f"{set_name}_{model}"] = {"overall": overall, "templates": cells}
    write_json(OUT, report)
    lines = ["# osler_v4 Y4 — baselines (aggregates only)", "",
             "Readout checks per (set, model, template) follow D12 and D21 as written in the report script; label mass is not applied",
             "to the authors' rows. Cells that fail a check read READOUT_FAIL and are not reported as results; a set-level figure",
             "is reported only when every template cell in it passes. The macro is the template macro (D24).", ""]
    for key, value in report["sets"].items():
        if value["overall"] is None:
            continue
        o = value["overall"]
        figures = f"micro {o['micro_accuracy']}; template macro {o['template_macro_accuracy']}; " if o["reportable"] else ""
        lines += [f"## {key}", "", f"overall: {o['items']} items; {figures}chance {o['chance']}; status {o['status']}", ""]
        if "passing_cells_additional" in o:
            extra = o["passing_cells_additional"]
            lines += [f"{extra['label']}: {extra['items']} items; micro {extra['micro_accuracy']}", ""]
        lines += ["| template | items | micro | majority | modal share | status |", "|---|---|---|---|---|---|"]
        for t, c in value["templates"].items():
            lines.append(f"| {t} | {c['items']} | {c['micro_accuracy']} | {c['majority_share']} | {c['modal_share']} | {c['status']} |")
        lines.append("")
    MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print({k: (v["overall"]["micro_accuracy"] if v["overall"]["reportable"] else "not a result (READOUT_FAIL)")
           for k, v in report["sets"].items() if v["overall"]})


if __name__ == "__main__":
    main()
