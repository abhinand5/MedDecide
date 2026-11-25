"""osler_v5 Z10 report (CPU; ADVISORY §10 Z10; additional, labelled; no choice).

Reads the Z10 prediction files: outputs/osler_v5/Z10/zeroshot/<slug>/<set>.jsonl for the two zero-shot Qwen3.5 models, and
outputs/osler_v5/Z10/clef_flash/<set>.jsonl for Clef-Flash (only when its local-only check passed; otherwise its cells are
NOT MEASURED). The expected row counts are those of osler_v4's frozen Y9 sets; a file that is short is INCOMPLETE and is not reported.

Per model and set:
- the readout-health gate, per (set, template) cell: D12 (meddecide.eval.health through o2_metrics.cell_health) for the zero-shot
  Qwen models, and D21 (constant answer; accuracy CI not below chance) for Clef-Flash, whose readout normalises over the offered
  options. A failing cell is listed by name and status and left out of every reported accuracy; its item count is printed beside
  the numbers (as osler_v0's O2 scoreboard did).
- the template macro and the item micro of the reported cells (meddecide.eval.o2_metrics.summarise).
- the knowledge set: accuracy on the reported rows, and the paired difference against osler_v4's zero-shot pplx-decider-flash-v1.2
  (the authors' code; outputs/osler_v4/Y9/baselines/pplx_knowledge) on the items both sides scored (5,435 of 5,456 in osler_v4),
  joined on the id osler_v4 wrote (join_ids).
  The paired bootstrap is meddecide.train.blocks.micro_difference (within templates, 1,000 resamples, seed 0).

Writes outputs/osler_v5/Z10/report.json (figures and cell statuses) and loops/osler_v5/z10.md (aggregates only: no item, note or
prediction text). Claim rows are written by hand from these figures (CLAIMS.md).

Usage (CPU):
    uv run --frozen python scripts/osler_v5/z10_report.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from meddecide.eval.o2_metrics import ScoredItem, cell_health, score_row, summarise
from meddecide.train.blocks import micro_difference, pair_by_item
from meddecide.train.mix_items import mix_row_to_item
from meddecide.utils.io import iter_jsonl, write_json
from meddecide.utils.provenance import git_commit, utcnow

REPO = Path(__file__).resolve().parents[2]
Y9_DIR = REPO / "data" / "bench" / "osler_v4" / "y9"
Z10 = REPO / "outputs" / "osler_v5" / "Z10"
PPLX_KNOWLEDGE = REPO / "outputs" / "osler_v4" / "Y9" / "baselines" / "pplx_knowledge" / "predictions.jsonl"
CLEF_CHECK = Z10 / "clef_local_check.json"
REPORT = Z10 / "report.json"
MARKDOWN = REPO / "loops" / "osler_v5" / "z10.md"
SETS = ("mr_verdict", "mr_reported", "mimic_suite", "mimic_heldout", "mednli_test", "panel", "knowledge")
MODELS = (
    {"slug": "qwen35_4b", "label": "Qwen3.5-4B zero-shot", "zero_shot": True},
    {"slug": "qwen35_9b", "label": "Qwen3.5-9B zero-shot", "zero_shot": True},
    {"slug": "clef_flash", "label": "Clef-Flash (authors' code)", "zero_shot": False},
)


def load(path: Path) -> list[dict[str, Any]] | None:
    return list(iter_jsonl(path)) if path.exists() else None


def model_dir(model: dict[str, Any]) -> Path:
    if model["slug"] == "clef_flash":
        return Z10 / "clef_flash"
    return Z10 / "zeroshot" / model["slug"]


def expected_rows(set_name: str) -> int:
    with (Y9_DIR / f"{set_name}.jsonl").open(encoding="utf-8") as fh:
        return sum(1 for line in fh if line.strip())


def gate_cells(model: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """The readout-health verdict of every (set, template) cell in the set's rows."""
    cells: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        cells[(str(row["set_name"]), str(row["template_id"]))].append(row)
    out: dict[str, dict[str, Any]] = {}
    for (set_name, template), cell_rows in sorted(cells.items()):
        verdict = cell_health(model["label"], template, cell_rows, zero_shot=model["zero_shot"])
        out[f"{set_name}/{template}"] = {"status": verdict["status"], "n": len(cell_rows), "failures": verdict.get("failures", [])}
    return out


def reported_rows(rows: list[dict[str, Any]], gate: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    failing = {key for key, cell in gate.items() if cell["status"].startswith("READOUT_FAIL")}
    return [r for r in rows if f"{r['set_name']}/{r['template_id']}" not in failing]


def set_block(rows: list[dict[str, Any]], expected: int, gate: dict[str, dict[str, Any]]) -> dict[str, Any]:
    kept = reported_rows(rows, gate)
    summary = summarise(kept).as_dict()
    summary["readout_fail_items_excluded"] = len(rows) - len(kept)
    summary["rows_in_file"] = len(rows)
    summary["expected_rows"] = expected
    summary["failing_cells"] = {key: cell["status"] for key, cell in gate.items() if cell["status"].startswith("READOUT_FAIL")}
    return summary


def join_ids() -> dict[str, str]:
    """Each knowledge item's join id: the id mix_row_to_item gives it, which every osler_v4 baseline wrote (y4_pplx_flash.py).

    The Y9 file's own item_id is another namespace, and the pplx file has none of its ids (checked 2026-10-29). The derived ids are
    unique over the 5,456 items and are the ids in osler_v4's final knowledge predictions too.
    """
    return {str(row["item_id"]): mix_row_to_item(row).item_id for row in iter_jsonl(Y9_DIR / "knowledge.jsonl")}


def knowledge_pairs(kept: list[dict[str, Any]]) -> dict[str, Any]:
    """Model minus pplx on the items both scored, paired by join id; the 21 unpaired items are counted and named by count only."""
    join = join_ids()
    pplx = {str(r["item_id"]): r for r in iter_jsonl(PPLX_KNOWLEDGE)}
    model_scored: dict[str, ScoredItem] = {join[str(r["item_id"])]: score_row(r) for r in kept if r["status"] == "scored"}
    common = [item_id for item_id in model_scored if item_id in pplx]
    candidate = [{"item_id": i, "template_id": model_scored[i].template_id, "correct": model_scored[i].correct} for i in common]
    reference = [{"item_id": i, "template_id": str(pplx[i]["template_id"]), "correct": bool(pplx[i]["correct"])} for i in common]
    diff = micro_difference(pair_by_item(candidate, reference))
    n = len(common)
    return {
        "paired_items": n, "unpaired_items": len(model_scored) - n, "pplx_items": len(pplx),
        "model_accuracy_paired": sum(c["correct"] for c in candidate) / n if n else None,
        "pplx_accuracy_paired": sum(c["correct"] for c in reference) / n if n else None,
        "difference_point": float(diff.point), "difference_lo": float(diff.lo), "difference_hi": float(diff.hi),
    }


def evaluate(model: dict[str, Any], clef_ok: bool) -> dict[str, Any]:
    result: dict[str, Any] = {"label": model["label"], "zero_shot": model["zero_shot"], "sets": {}}
    if model["slug"] == "clef_flash" and not clef_ok:
        for set_name in SETS:
            result["sets"][set_name] = {"status": "NOT MEASURED — the Clef local-only check did not pass (ADVISORY §10 Z10.2)"}
        return result
    for set_name in SETS:
        path = model_dir(model) / f"{set_name}.jsonl"
        rows = load(path)
        expected = expected_rows(set_name)
        if rows is None:
            result["sets"][set_name] = {"status": f"NOT MEASURED — no prediction file ({path.relative_to(REPO)})"}
            continue
        if len(rows) != expected:
            result["sets"][set_name] = {"status": f"INCOMPLETE — {len(rows)} of {expected} rows; not reported"}
            continue
        gate = gate_cells(model, rows)
        result["sets"][set_name] = {"status": "MEASURED", **set_block(rows, expected, gate), "cells": gate}
        if set_name == "panel":
            kept = reported_rows(rows, gate)
            by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in kept:
                by_dataset[str(row["set_name"])].append(row)
            result["sets"][set_name]["per_dataset"] = {name: summarise(group).as_dict() for name, group in sorted(by_dataset.items())}
        if set_name == "knowledge":
            result["sets"][set_name]["paired_with_pplx"] = knowledge_pairs(reported_rows(rows, gate))
    return result


def fmt(value: Any) -> str:
    return "—" if value is None else f"{value:.4f}"


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# osler_v5 — Z10 comparator scoring (additional; choose nothing)",
        "",
        "Zero-shot Qwen3.5-4B and -9B (local weights) and Clef-Flash, scored on the frozen Y9 sets of osler_v4 with the current",
        "readout (Qwen: osler_v4's Y4 zero-shot readout; Clef: the authors' `systemone` through `envs/clef`). Dev is not used; these are",
        "the test-side sets named by ADVISORY §10 Z10. Accuracy is the reported figure only for cells that pass the readout-health gate.",
        "Template macro and item micro are both given. Counts are the reported items; excluded items are printed beside them.",
        "",
        f"Generated {report['generated_utc']} at git {report['git_commit'][:7]}. The raw predictions are in `outputs/osler_v5/Z10/`.",
        "",
        "## Clef-Flash data rule",
        "",
        f"{report['clef_check']}",
        "",
        "## Readout-health gate (D12 for zero-shot, D21 for Clef-Flash)",
        "",
        "| model | set | cells | cells failing | failing cells (status) |",
        "|---|---|---|---|---|",
    ]
    for model in report["models"]:
        for set_name in SETS:
            block = model["sets"][set_name]
            if block["status"] != "MEASURED":
                lines.append(f"| {model['label']} | {set_name} | — | — | {block['status']} |")
                continue
            failing = block["failing_cells"]
            text = "; ".join(f"{key}: {status.replace('READOUT_FAIL — ', '')}" for key, status in failing.items()) or "none"
            lines.append(f"| {model['label']} | {set_name} | {len(block['cells'])} | {len(failing)} | {text} |")
    lines += ["", "## Figures (reported cells only)", "",
              "| model | set | template macro | item micro | items reported (excluded) | expected |", "|---|---|---|---|---|---|"]
    for model in report["models"]:
        for set_name in SETS:
            block = model["sets"][set_name]
            if block["status"] != "MEASURED":
                lines.append(f"| {model['label']} | {set_name} | — | — | — | {block['status']} |")
                continue
            lines.append(f"| {model['label']} | {set_name} | {fmt(block['macro_accuracy'])} | {fmt(block['accuracy'])} | "
                         f"{block['n_scored']} ({block['readout_fail_items_excluded']}) | {block['expected_rows']} |")
    lines += ["", "## Panel by dataset (reported cells; template macro and item micro of each dataset)", "",
              "| model | dataset | template macro | item micro | items |", "|---|---|---|---|---|"]
    for model in report["models"]:
        block = model["sets"]["panel"]
        if block["status"] != "MEASURED":
            continue
        for name, row in block["per_dataset"].items():
            lines.append(f"| {model['label']} | {name} | {fmt(row['macro_accuracy'])} | {fmt(row['accuracy'])} | {row['n_scored']} |")
    lines += ["", "## Knowledge (MedQA and MedMCQA test) and the paired difference against zero-shot pplx", "",
              "| model | knowledge items (reported) | accuracy, all reported | paired items | model accuracy (paired) | pplx accuracy (paired) | model minus pplx [95 % interval] |",
              "|---|---|---|---|---|---|---|"]
    for model in report["models"]:
        block = model["sets"]["knowledge"]
        if block["status"] != "MEASURED":
            lines.append(f"| {model['label']} | — | — | — | — | — | {block['status']} |")
            continue
        p = block["paired_with_pplx"]
        lines.append(f"| {model['label']} | {block['n_scored']} | {fmt(block['accuracy'])} | {p['paired_items']} | "
                     f"{fmt(p['model_accuracy_paired'])} | {fmt(p['pplx_accuracy_paired'])} | "
                     f"{p['difference_point']:+.4f} [{p['difference_lo']:+.4f}, {p['difference_hi']:+.4f}] |")
    return "\n".join(lines) + "\n"


def main() -> None:
    clef_ok = False
    clef_text = "NOT MEASURED — no local-only check record (scripts/osler_v5/z10_clef_local_check.py has not run)."
    if CLEF_CHECK.exists():
        check = json.loads(CLEF_CHECK.read_text(encoding="utf-8"))
        clef_ok = bool(check.get("local_only") is True)
        hits = len(check.get("network_hits", []))
        clef_text = (f"Local-only check at {check.get('checked_utc')}: revision matches {check.get('revision_matches')}; "
                     f"network-client or URL lines in the authors' code {hits}; offline load and score "
                     f"{'ok' if check.get('offline_load_and_score', {}).get('ok') else 'failed'}; "
                     f"local_only {check.get('local_only')}.")
    models = [evaluate(model, clef_ok) for model in MODELS]
    report = {"generated_utc": utcnow(), "git_commit": git_commit(REPO), "clef_check": clef_text, "clef_local_only": clef_ok,
              "models": [{"slug": m["slug"], **res} for m, res in zip(MODELS, models, strict=True)]}
    write_json(REPORT, report)
    MARKDOWN.write_text(markdown(report), encoding="utf-8")
    print(f"wrote {REPORT.relative_to(REPO)} and {MARKDOWN.relative_to(REPO)}", flush=True)


if __name__ == "__main__":
    main()
