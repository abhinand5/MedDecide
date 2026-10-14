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
            s = summarise(set_rows)
            model["sets"][set_name] = {
                "n_in_scope": s.n_in_scope, "n_scored": s.n_scored, "skipped_by_reason": s.skipped_by_reason,
                "coverage": round(s.coverage, 4) if s.coverage == s.coverage else None,
                "accuracy": round(s.accuracy, 4) if s.accuracy is not None else None,
                "accuracy_ci95": [round(x, 4) for x in s.accuracy_ci95] if s.accuracy_ci95 else None,
                "macro_accuracy": round(s.macro_accuracy, 4) if s.macro_accuracy is not None else None,
                "macro_brier": round(s.macro_brier, 4) if s.macro_brier is not None else None,
                "brier": round(s.brier, 4) if s.brier is not None else None,
                "ece": round(s.ece, 4) if s.ece is not None else None,
                "templates": per_template(label, set_rows, zero_shot),
            }
        summary["models"][slug] = model
    write_json(OUT, summary)
    for slug, model in summary["models"].items():
        for set_name, s in model.get("sets", {}).items():
            if "accuracy" in s:
                print(f"{slug:14s} {set_name:11s} scored {s['n_scored']}/{s['n_in_scope']} acc {s['accuracy']} "
                      f"macro {s['macro_accuracy']} brier {s['brier']}", flush=True)


if __name__ == "__main__":
    main()
