"""V6 analysis: baseline numbers on the MR evaluation suite (osler_v1 ADVISORY V6, step 3; D12 for zero-shot cells).

Reads the O2 runners' prediction files (outputs/osler_v1/V6/baselines/<model>/predictions.jsonl) and, per model and set
(mr_test, mr_heldout), writes the set summary (coverage, accuracy with its bootstrap interval, template macro accuracy, Brier,
ECE) and the per-template readout-health verdict: D12 for the zero-shot models (label mass, greedy agreement, accuracy interval
above chance, constant answer) and D21 for the trained and competitor models (constant answer, accuracy interval). A cell that
fails is listed as READOUT_FAIL and its numbers are not reported as accuracies.

Writes outputs/osler_v1/V6/baseline_summary.json (aggregates only). The committed report, loops/osler_v1/mr_baselines.md, is
written from that file by hand (aggregates only).

Usage:
    source scripts/pod_env.sh && uv run --frozen python scripts/osler_v1/v6_analyse.py
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from meddecide.eval.o2_metrics import cell_health, score_row, summarise
from meddecide.utils.io import write_json
from meddecide.utils.provenance import git_commit, utcnow

REPO = Path(__file__).resolve().parents[2]
BASE = REPO / "outputs" / "osler_v1" / "V6" / "baselines"
OUT = REPO / "outputs" / "osler_v1" / "V6" / "baseline_summary.json"
MODELS = {  # slug -> (label, zero_shot, the O2 role)
    "qwen35-4b": ("zero-shot Qwen3.5-4B", True, "zero-shot base (Gate O2 baseline for Osler-4B)"),
    "qwen35-9b": ("zero-shot Qwen3.5-9B", True, "zero-shot base"),
    "meddecider-4b": ("MedDecider-4B", False, "competitor (Gate O2 baseline for Osler-4B)"),
    "meddecider-9b": ("MedDecider-9B", False, "competitor (Gate O2 baseline for Osler-9B)"),
    "jev-9b": ("JEV-9B", False, "competitor (Gate O2 baseline for Osler-9B)"),
}
SETS = ("mr_test", "mr_heldout")


def read_predictions(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def per_template(model_label: str, rows: list[dict[str, Any]], zero_shot: bool) -> dict[str, Any]:
    out: dict[str, Any] = {}
    by_template: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_template[r["template_id"]].append(r)
    for template, group in sorted(by_template.items()):
        scored = [score_row(r) for r in group if r["status"] == "scored"]
        golds = Counter(r["gold"] for r in group)
        health = cell_health(model_label, template, group, zero_shot=zero_shot)
        entry: dict[str, Any] = {
            "n_in_scope": len(group), "n_scored": len(scored),
            "skipped": dict(Counter(r["reason"] for r in group if r["status"] == "skipped")),
            "majority_gold_share": round(max(golds.values()) / len(group), 4),
            "readout_health": health["status"],
        }
        if scored and not health["status"].startswith("READOUT_FAIL"):
            entry["accuracy"] = round(sum(1 for s in scored if s.correct) / len(scored), 4)
            entry["brier"] = round(sum(s.brier for s in scored) / len(scored), 4)
        out[template] = entry
    return out


def main() -> None:
    summary: dict[str, Any] = {"written_utc": utcnow(), "git_commit": git_commit(REPO), "models": {}}
    for slug, (label, zero_shot, role) in MODELS.items():
        path = BASE / slug / "predictions.jsonl"
        if not path.exists():
            summary["models"][slug] = {"label": label, "role": role, "status": "NOT MEASURED — no predictions file"}
            continue
        rows = read_predictions(path)
        model: dict[str, Any] = {"label": label, "role": role, "zero_shot": zero_shot, "sets": {}}
        for set_name in SETS:
            set_rows = [r for r in rows if r.get("set_name") == set_name]
            if not set_rows:
                model["sets"][set_name] = {"status": "NOT MEASURED — no rows"}
                continue
            templates = per_template(label, set_rows, zero_shot)
            # D12 (zero-shot) and D21 (competitors): a cell that reads READOUT_FAIL reports no accuracy, so the set's accuracy
            # is taken over the other cells only, and the failing cells are named
            failed = sorted(t for t, e in templates.items() if e["readout_health"].startswith("READOUT_FAIL"))
            pass_rows = [r for r in set_rows if r["template_id"] not in failed]
            s = summarise(set_rows)  # coverage and skip counts over every row of the set (no accuracy is taken from it)
            p = summarise(pass_rows) if pass_rows else None
            model["sets"][set_name] = {
                "n_in_scope": s.n_in_scope, "n_scored": s.n_scored, "skipped_by_reason": s.skipped_by_reason,
                "coverage": round(s.coverage, 4) if s.coverage == s.coverage else None,
                "readout_fail_templates": failed,
                "items_in_pass_cells": len(pass_rows),
                "pass_cells_scored": p.n_scored if p else 0,
                "pass_cells_accuracy": round(p.accuracy, 4) if p and p.accuracy is not None else None,
                "pass_cells_accuracy_ci95": [round(x, 4) for x in p.accuracy_ci95] if p and p.accuracy_ci95 else None,
                "pass_cells_macro_accuracy": round(p.macro_accuracy, 4) if p and p.macro_accuracy is not None else None,
                "pass_cells_macro_brier": round(p.macro_brier, 4) if p and p.macro_brier is not None else None,
                "pass_cells_brier": round(p.brier, 4) if p and p.brier is not None else None,
                "pass_cells_ece": round(p.ece, 4) if p and p.ece is not None else None,
                "templates": templates,
            }
        summary["models"][slug] = model
    # additional, labelled: every model's pass-cell accuracy on the items that both MedDecider models scored (their common
    # support), per set; the same readout-health rule applies to each model's own failing cells
    summary["matched_to_meddecider_support"] = {}
    all_rows = {slug: read_predictions(BASE / slug / "predictions.jsonl")
                for slug in MODELS if (BASE / slug / "predictions.jsonl").exists()}
    for set_name in SETS:
        support: set[str] | None = None
        for slug in ("meddecider-4b", "meddecider-9b"):
            if slug not in all_rows:
                continue
            ids = {r["item_id"] for r in all_rows[slug] if r.get("set_name") == set_name and r["status"] == "scored"}
            support = ids if support is None else support & ids
        if support is None:
            continue
        # one common item set for every model: the templates that fail the gate for any model are excluded for all of them
        fail_all: set[str] = set()
        for slug in all_rows:
            fail_all |= set(summary["models"][slug].get("sets", {}).get(set_name, {}).get("readout_fail_templates", []))
        block: dict[str, Any] = {"support_items": len(support), "readout_fail_templates_excluded": sorted(fail_all),
                                 "models": {}}
        for slug, rows in all_rows.items():
            if "readout_fail_templates" not in summary["models"][slug].get("sets", {}).get(set_name, {}):
                continue
            matched = [r for r in rows if r.get("set_name") == set_name and r["item_id"] in support
                       and r["status"] == "scored" and r["template_id"] not in fail_all]
            if not matched:
                block["models"][slug] = {"items": 0, "accuracy": None}
                continue
            m = summarise(matched)
            block["models"][slug] = {"items": len(matched), "accuracy": round(m.accuracy, 4),
                                     "accuracy_ci95": [round(x, 4) for x in m.accuracy_ci95],
                                     "macro_accuracy": round(m.macro_accuracy, 4) if m.macro_accuracy is not None else None,
                                     "brier": round(m.brier, 4) if m.brier is not None else None}
        summary["matched_to_meddecider_support"][set_name] = block
    write_json(OUT, summary)
    for slug, model in summary["models"].items():
        for set_name, s in model.get("sets", {}).items():
            if "pass_cells_accuracy" in s:
                print(f"{slug:14s} {set_name:11s} scored {s['n_scored']}/{s['n_in_scope']} coverage {s['coverage']} "
                      f"pass-cell acc {s['pass_cells_accuracy']} macro {s['pass_cells_macro_accuracy']} "
                      f"brier {s['pass_cells_brier']} readout-fail {s['readout_fail_templates']}", flush=True)


if __name__ == "__main__":
    main()
