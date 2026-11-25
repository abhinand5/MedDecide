#!/usr/bin/env python
"""osler_rc0 R7 — unified 12-model leaderboard, per-category tables and formal Pareto frontier (CPU).

Every number is recomputed from the per-item prediction files through the frozen item universe and the
raw<->recomp id bijection (``meddecide.eval.rc0_board``). Nothing is copied from a run summary, no prediction row is
silently dropped (a row that does not map raises), and a model that does not cover a set is reported as NOT MEASURED
with the reason.

Run::

    source scripts/pod_env.sh && uv run --frozen python scripts/osler_rc0/rc0_leaderboard.py \
        --out outputs/osler_rc0/R7 --latency outputs/osler_rc0/R5/latency.json

Writes ``rc0_board.json`` under the gitignored ``--out`` and the committed reports
``loops/osler_rc0/{LEADERBOARD,PER_CATEGORY,PARETO}.md``.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from meddecide.eval.rc0_board import (  # noqa: E402
    Cell,
    FrozenSet,
    cell_metrics,
    exact_dominates,
    fmt,
    load_cell,
    load_frozen,
    mean_of,
    paired_difference,
    pareto_front,
    restricted_to_n_options,
)
from meddecide.utils.provenance import git_commit, utcnow  # noqa: E402

SETS: dict[str, Path] = {
    "panel": ROOT / "data/bench/osler_v4/y9/panel.jsonl",
    "knowledge": ROOT / "data/bench/osler_v4/y9/knowledge.jsonl",
    "confirm_general": ROOT / "data/bench/osler_v7/confirm/general.jsonl",
    "mr_verdict": ROOT / "data/bench/osler_v4/y9/mr_verdict.jsonl",
    "mimic_suite": ROOT / "data/bench/osler_v4/y9/mimic_suite.jsonl",
    "mednli_test": ROOT / "data/bench/osler_v4/y9/mednli_test.jsonl",
    "mimic_heldout": ROOT / "data/bench/osler_v4/y9/mimic_heldout.jsonl",
    "mr_heldout_test": ROOT / "data/bench/osler_v4/y9/mr_heldout_test.jsonl",
    "pairs_test": ROOT / "data/bench/osler_v4/y9/pairs_test.jsonl",
}
HEADLINE = ["panel", "knowledge", "confirm_general", "mr_verdict", "mimic_suite"]
ADDITIONAL = ["mednli_test", "mimic_heldout", "mr_heldout_test", "pairs_test"]
OOD = ["panel", "knowledge", "confirm_general"]
AUDIT = ["mr_verdict", "mimic_suite"]

R2 = ROOT / "outputs/osler_rc0/R2"
R3 = ROOT / "outputs/osler_rc0/R3"
R4 = ROOT / "outputs/osler_rc0/R4"
Y9 = ROOT / "outputs/osler_v4/Y9/baselines"
Y4 = ROOT / "outputs/osler_v4/Y4/baselines"
T10 = ROOT / "outputs/osler_v7/T10"
Z10 = ROOT / "outputs/osler_v5/Z10"
D1 = Path("/workspace/MedDecide-d1-2b/outputs/d1_bench")
CLEF = Z10 / "clef_flash"
QS9 = Z10 / "zeroshot/qwen35_9b"
QS4 = Z10 / "zeroshot/qwen35_4b"


def preds(directory: Path) -> Path:
    return directory / "predictions.jsonl"


MODELS: dict[str, dict[str, Any]] = {
    "Osler-9B-rc0": {"params_b": 8.39, "health_mode": "d21", "tier": "rc0", "sets": {
        s: preds(R2 / f"preds_cal/9b_rc0_{s}") for s in SETS}},
    "Osler-4B-rc0": {"params_b": 4.0, "health_mode": "d21", "tier": "rc0", "sets": {
        s: preds(R3 / f"preds_cal/4b_rc0_{s}") for s in SETS}},
    "d3-flash": {"params_b": 8.39, "health_mode": "d21", "tier": "base", "sets": {
        "panel": preds(ROOT / "outputs/eval0/panel/9b_base_panel"),
        # R2 re-scored the four prob-less T10 base files with full probabilities; use the complete ones
        **{s: preds(R2 / f"preds/d3_base_{s}") for s in
           ["knowledge", "confirm_general", "mr_verdict", "mimic_suite", *ADDITIONAL]}}},
    "pplx-flash": {"params_b": 4.0, "health_mode": "d21", "tier": "base", "sets": {
        **{s: preds(Y9 / f"pplx_{s}") for s in
           ["panel", "knowledge", "mr_verdict", "mednli_test", "mr_heldout_test", "pairs_test"]},
        "mimic_suite": preds(T10 / "baselines/4b_base_mimic_suite"),
        "confirm_general": preds(T10 / "baselines/4b_base_general")}},
    "MedDecider-9B": {"params_b": 9.0, "health_mode": "d21", "tier": "rival", "sets": {
        **{s: preds(Y9 / f"md9b_{s}") for s in
           ["panel", "knowledge", "mr_verdict", "mednli_test", "mimic_heldout", "mr_heldout_test", "pairs_test"]},
        "mimic_suite": preds(Y4 / "suite_meddecider9b"),
        "confirm_general": preds(R4 / "preds/md9b_confirm_general")}},
    "MedDecider-4B": {"params_b": 4.0, "health_mode": "d21", "tier": "rival", "sets": {
        **{s: preds(Y9 / f"md4b_{s}") for s in
           ["panel", "knowledge", "mr_verdict", "mednli_test", "mr_heldout_test", "pairs_test"]},
        "mimic_suite": preds(Y4 / "suite_meddecider4b"),
        "mimic_heldout": preds(Y4 / "heldout_meddecider4b"),
        "confirm_general": preds(R4 / "preds/md4b_confirm_general")}},
    "Clef-Flash": {"params_b": 9.0, "health_mode": "d21", "tier": "rival", "sets": {
        **{s: CLEF / f"{s}.jsonl" for s in
           ["panel", "knowledge", "mr_verdict", "mimic_suite", "mednli_test", "mimic_heldout", "mr_heldout_test"]},
        "confirm_general": preds(R4 / "preds/clef_confirm_general")}},
    "JEV-9B": {"params_b": 9.0, "health_mode": "d21", "tier": "rival", "sets": {
        **{s: Y9 / f"jev9b_{s}/jev-9b/predictions.jsonl" for s in SETS},
        "confirm_general": preds(R4 / "preds/jev9b_confirm_general")}},
    "d1-2B": {"params_b": 2.0, "health_mode": "d21", "tier": "rival", "sets": {
        s: preds(D1 / s) for s in ["panel", "knowledge", "mr_verdict", "mimic_suite", "mednli_test"]}},
    "Qwen3.5-9B": {"params_b": 9.0, "health_mode": "d12", "tier": "zeroshot", "sets": {
        s: QS9 / f"{s}.jsonl" for s in
        ["panel", "knowledge", "mr_verdict", "mimic_suite", "mednli_test", "mimic_heldout", "mr_heldout_test"]}},
    "Qwen3.5-4B": {"params_b": 4.0, "health_mode": "d12", "tier": "zeroshot", "sets": {
        s: QS4 / f"{s}.jsonl" for s in
        ["panel", "knowledge", "mr_verdict", "mimic_suite", "mednli_test", "mimic_heldout", "mr_heldout_test"]}},
    "pplx-27B": {"params_b": 27.0, "health_mode": "d21", "tier": "ceiling", "sets": {
        **{s: preds(Y9 / f"pplx27b_{s}") for s in SETS},
        "confirm_general": preds(R4 / "preds/pplx27b_confirm_general")}},
}
OWN_BASE = {"Osler-9B-rc0": "d3-flash", "Osler-4B-rc0": "pplx-flash"}
RIVAL_PAIRS = {
    "Osler-9B-rc0": ["d3-flash", "MedDecider-9B", "Clef-Flash", "JEV-9B", "pplx-27B"],
    "Osler-4B-rc0": ["pplx-flash", "MedDecider-4B", "Qwen3.5-4B", "Osler-9B-rc0"],
}


def accuracy_of(cell_metrics_row: dict[str, Any] | None) -> float | None:
    if not cell_metrics_row or cell_metrics_row.get("status") != "MEASURED":
        return None
    return cell_metrics_row.get("accuracy")


def md_table(headers: list[str], rows: list[list[Any]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        out.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(out)


def load_all(frozen: dict[str, FrozenSet]) -> tuple[dict[str, dict[str, Cell]], dict[str, dict[str, dict[str, Any]]]]:
    cells: dict[str, dict[str, Cell]] = {}
    metrics: dict[str, dict[str, dict[str, Any]]] = {}
    for model, spec in MODELS.items():
        cells[model] = {}
        metrics[model] = {}
        for set_name, path in spec["sets"].items():
            cell = load_cell(model, frozen[set_name], path)
            if cell is None:
                metrics[model][set_name] = {"status": "NOT MEASURED — prediction file absent"}
                continue
            cells[model][set_name] = cell
            metrics[model][set_name] = cell_metrics(frozen[set_name], cell, health_mode=spec["health_mode"])
            print(f"  {model}/{set_name}: {cell.scored}/{cell.n_universe} rows, "
                  f"acc {metrics[model][set_name].get('accuracy')}", flush=True)
    return cells, metrics


def five_set_macro(metrics: dict[str, dict[str, Any]]) -> float | None:
    values = [accuracy_of(metrics.get(s)) for s in HEADLINE]
    return mean_of(values)


def compute_pareto(frozen: dict[str, FrozenSet], metrics: dict[str, dict[str, Any]],
                   latency: dict[str, Any]) -> dict[str, Any]:
    def acc(model: str, set_name: str) -> float | None:
        return accuracy_of(metrics.get(model, {}).get(set_name))

    def point(model: str, sets: list[str]) -> dict[str, float] | None:
        values = [acc(model, s) for s in sets]
        if any(v is None for v in values):
            return None
        return dict(zip(sets, values, strict=True))

    le10 = [m for m, spec in MODELS.items() if spec["params_b"] <= 10]
    le4 = [m for m in le10 if MODELS[m]["params_b"] <= 4]
    out: dict[str, Any] = {"quality_le10b": {}, "quality_le4b": {}, "two_axis": {}, "efficiency": {}}

    for group_name, group in (("quality_le10b", le10), ("quality_le4b", le4)):
        points = {m: point(m, HEADLINE) for m in group}
        points = {m: p for m, p in points.items() if p is not None}
        dominance = []
        for a in points:
            for b in points:
                if a != b and exact_dominates(points[a], points[b], HEADLINE):
                    dominance.append({"dominator": a, "dominated": b,
                                      "deltas": {s: round(points[a][s] - points[b][s], 6) for s in HEADLINE}})
        undominated = [m for m in points if not any(d["dominated"] == m for d in dominance)]
        out[group_name] = {
            "models_with_all_five": sorted(points),
            "not_comparable": sorted(set(group) - set(points)),
            "dominance_pairs": dominance,
            "undominated": sorted(undominated),
            "points": points,
        }

    two = {}
    for model in MODELS:
        ood = mean_of([acc(model, s) for s in OOD])
        audit = mean_of([acc(model, s) for s in AUDIT])
        if ood is not None and audit is not None:
            two[model] = (ood, audit)
    out["two_axis"] = {
        "definition": "OOD General Avg = mean(panel, knowledge, confirm_general); Clinical Audit Avg = mean(mr_verdict, mimic_suite)",
        "points": {m: {"ood": round(v[0], 6), "audit": round(v[1], 6)} for m, v in two.items()},
        "front": pareto_front(two),
        "not_comparable": sorted(set(MODELS) - set(two)),
    }

    latency_models = latency.get("models", {}) if latency else {}
    efficiency = {}
    for model in MODELS:
        macro = mean_of([acc(model, s) for s in HEADLINE])
        if macro is None:
            continue
        entry = {"params_b": MODELS[model]["params_b"], "five_set_macro": round(macro, 6),
                 "ood": round(mean_of([acc(model, s) for s in OOD]), 6) if mean_of([acc(model, s) for s in OOD]) is not None else None,
                 "audit": round(mean_of([acc(model, s) for s in AUDIT]), 6) if mean_of([acc(model, s) for s in AUDIT]) is not None else None}
        measured = latency_models.get(model)
        if measured and measured.get("batch1", {}).get("p50_ms_per_item") is not None:
            entry["p50_ms_per_item"] = measured["batch1"]["p50_ms_per_item"]
            entry["peak_vram_gb"] = measured["batch1"].get("peak_vram_gb")
        efficiency[model] = entry
    out["efficiency"] = {"models": efficiency,
                         "definition": "params (B) and measured batch-1 p50 latency vs the three accuracy averages; "
                                       "pplx-27B included as the out-of-tier cost/accuracy reference"}
    return out


def write_leaderboard(path: Path, board: dict[str, Any]) -> None:
    metrics = board["models"]
    lines = ["# RC0 leaderboard — 12 models, 9 frozen evaluation sets", ""]
    lines += [f"Generated {board['generated_utc']} from `rc0_board.json` (git `{board['git_commit']}`) by "
              "`scripts/osler_rc0/rc0_leaderboard.py`. Every number is recomputed from per-item prediction files; "
              "accuracy is micro over the frozen set with a 95 % bootstrap CI (2,000 resamples, seed 0), macro is the "
              "unweighted mean over templates (CI resamples within template). Brier is the multi-class Brier "
              "(sum over offered options), ECE is 15-bin top-label. Coverage is scored / frozen universe; a set a model "
              "did not run is `NOT MEASURED — prediction file absent`.", ""]
    lines += ["## Headline sets", ""]
    for set_name in HEADLINE:
        rows = []
        ranked = sorted(MODELS, key=lambda m: (accuracy_of(metrics[m].get(set_name)) is None,
                                               -(accuracy_of(metrics[m].get(set_name)) or 0.0)))
        for model in ranked:
            m = metrics[model].get(set_name, {})
            if m.get("status") != "MEASURED":
                rows.append([model, "NOT MEASURED", "—", "—", "—", "—", "—"])
                continue
            ci = m["accuracy_ci95"]
            rows.append([model, fmt(m["accuracy"]), f"[{ci[0]:.4f}, {ci[1]:.4f}]", fmt(m["macro_accuracy"]),
                         fmt(m.get("brier")), fmt(m.get("ece")), f"{m['coverage']:.3f}"])
        lines += [f"### `{set_name}` ({board['sets'][set_name]:,} items)", "",
                  md_table(["model", "accuracy", "95% CI", "template macro", "Brier", "ECE", "coverage"], rows), ""]
    lines += ["## Additional / held-out sets", ""]
    for set_name in ADDITIONAL:
        rows = []
        for model in MODELS:
            m = metrics[model].get(set_name, {})
            if m.get("status") != "MEASURED":
                rows.append([model, "NOT MEASURED"])
                continue
            ci = m["accuracy_ci95"]
            rows.append([model, f"{m['accuracy']:.4f}", f"[{ci[0]:.4f}, {ci[1]:.4f}]", f"{m['coverage']:.3f}"])
        lines += [f"### `{set_name}` ({board['sets'][set_name]:,} items)", "",
                  md_table(["model", "accuracy", "95% CI", "coverage"], rows), ""]
    lines += ["## Readout health (D12/D21)", "",
              "D12 applies the label-mass gate to zero-shot letter readouts; for trained readouts that normalise over the "
              "offered options (D21) the label mass is a diagnostic and the constant-answer and below-chance checks are "
              "still applied. Cells that fail any applicable check read `READOUT_FAIL — <check>`.", ""]
    rows = []
    for model in MODELS:
        statuses = []
        for set_name in SETS:
            m = metrics[model].get(set_name, {})
            if m.get("status") != "MEASURED":
                continue
            status = m["readout_health"]["overall"]
            if status.startswith("READOUT_FAIL"):
                failing = [t["template_id"] for t in m["readout_health"]["templates"] if t["status"].startswith("READOUT_FAIL")]
                statuses.append(f"`{set_name}`: {status} ({', '.join(failing[:3])}{'…' if len(failing) > 3 else ''})")
        rows.append([model, "; ".join(statuses) if statuses else "—"])
    lines += [md_table(["model", "readout-health flags"], rows), ""]
    lines += ["## Paired differences (accuracy, 95 % paired bootstrap CI)", ""]
    for cand, pairs in board["paired"].items():
        lines += [f"### {cand}", ""]
        rows = []
        for base, per_set in pairs.items():
            for set_name, diff in per_set.items():
                if diff.get("n", 0) == 0 or diff.get("status") == "NOT MEASURED — no common items":
                    rows.append([base, set_name, "NOT MEASURED", "—", "—"])
                else:
                    ci = diff["accuracy_diff_ci95"]
                    rows.append([base, set_name, f"{diff['accuracy_diff']:+.4f}",
                                 f"[{ci[0]:+.4f}, {ci[1]:+.4f}]", f"{diff['n']:,}"])
        lines += [md_table(["vs", "set", "Δ accuracy", "95% CI", "paired n"], rows), ""]
    lines += ["## Coverage and NOT MEASURED", ""]
    rows = []
    for model in MODELS:
        missing = [s for s in SETS if metrics[model].get(s, {}).get("status") != "MEASURED"]
        rows.append([model, f"{len(SETS) - len(missing)}/9", ", ".join(missing) if missing else "—"])
    lines += [md_table(["model", "sets measured", "not measured"], rows), "",
              "Reasons for `NOT MEASURED`: competitor sets the model never ran (see §2 of the ADVISORY) and, where "
              "relevant, runner timeboxes. A missing cell is never imputed.", ""]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_per_category(path: Path, board: dict[str, Any], frozen: dict[str, FrozenSet],
                       cells: dict[str, dict[str, Cell]]) -> None:
    lines = ["# RC0 per-category tables — every source and template of the 9 frozen sets", "",
             f"Generated {board['generated_utc']} from `rc0_board.json` by `scripts/osler_rc0/rc0_leaderboard.py`. "
             "Each cell is micro accuracy on the items of that source/template that the model scored; `n` is the "
             "item count in the frozen set (all models are measured on the same frozen item list; coverage differences "
             "come from skipped items and are shown in LEADERBOARD.md).", ""]
    models = list(MODELS)
    for set_name in SETS:
        lines += [f"## `{set_name}`", ""]
        for dimension in ("source", "template_id"):
            groups: dict[str, list[int]] = defaultdict(list)
            for item in frozen[set_name].items:
                groups[getattr(item, dimension)].append(item.index)
            rows = []
            for group, positions in sorted(groups.items()):
                position_set = set(positions)
                row = [group, len(positions)]
                for model in models:
                    cell = cells[model].get(set_name)
                    if cell is None:
                        row.append("—")
                        continue
                    mask = [i for i, idx in enumerate(cell.index) if int(idx) in position_set]
                    row.append(f"{cell.correct[mask].mean():.4f}" if mask else "—")
                rows.append(row)
            lines += [f"### by {dimension}", "", md_table(["group", "n", *models], rows), ""]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_pareto(path: Path, board: dict[str, Any]) -> None:
    pareto = board["pareto"]
    lines = ["# RC0 Pareto frontier", "",
             f"Generated {board['generated_utc']} from `rc0_board.json` by `scripts/osler_rc0/rc0_leaderboard.py`. "
             "Both axes are accuracies (higher is better). A model is *comparable* only when it has a measured number "
             "on every axis; incomparable models are listed, never imputed.", ""]
    lines += ["## ≤10B quality frontier (5 headline sets)", ""]
    q = pareto["quality_le10b"]
    lines += [f"Comparable models: {', '.join(q['models_with_all_five']) or 'none'}.", "",
              f"Not comparable (missing a headline set): {', '.join(q['not_comparable']) or 'none'}.", ""]
    if q["dominance_pairs"]:
        lines += [md_table(["dominator", "dominated", *HEADLINE],
                           [[d["dominator"], d["dominated"], *[f"{d['deltas'][s]:+.4f}" for s in HEADLINE]]
                            for d in q["dominance_pairs"]]), ""]
    else:
        lines += ["No exact dominance pair among the comparable ≤10B models.", ""]
    lines += [f"Undominated on all five: {', '.join(q['undominated']) or 'none'}.", ""]
    lines += ["## 2-axis Pareto (OOD General Avg vs Clinical Audit Avg)", "",
              "OOD General Avg = mean(panel, knowledge, confirm_general); Clinical Audit Avg = mean(mr_verdict, mimic_suite).", ""]
    two = pareto["two_axis"]
    rows = [[m, f"{v['ood']:.4f}", f"{v['audit']:.4f}", "front" if m in two["front"] else ""]
            for m, v in sorted(two["points"].items(), key=lambda kv: -kv[1]["ood"])]
    lines += [md_table(["model", "OOD General Avg", "Clinical Audit Avg", "Pareto"], rows), "",
              f"Frontier: {', '.join(two['front'])}.", "",
              f"Not comparable: {', '.join(two['not_comparable']) or 'none'}.", ""]
    lines += ["## ≤4B tier frontier", ""]
    q4 = pareto["quality_le4b"]
    lines += [f"Comparable: {', '.join(q4['models_with_all_five']) or 'none'}; not comparable: "
              f"{', '.join(q4['not_comparable']) or 'none'}.", ""]
    if q4["dominance_pairs"]:
        lines += [md_table(["dominator", "dominated", *HEADLINE],
                           [[d["dominator"], d["dominated"], *[f"{d['deltas'][s]:+.4f}" for s in HEADLINE]]
                            for d in q4["dominance_pairs"]]), ""]
    else:
        lines += ["No exact dominance pair among the comparable ≤4B models.", ""]
    lines += [f"Undominated: {', '.join(q4['undominated']) or 'none'}.", ""]
    lines += ["## Efficiency frontier", "",
              pareto["efficiency"]["definition"], ""]
    rows = [[m, f"{v['params_b']:.2f}", fmt(v.get("p50_ms_per_item")), fmt(v.get("peak_vram_gb")),
             fmt(v["five_set_macro"]), fmt(v.get("ood")), fmt(v.get("audit"))]
            for m, v in sorted(pareto["efficiency"]["models"].items(), key=lambda kv: kv[1]["params_b"])]
    lines += [md_table(["model", "params (B)", "p50 ms/item (batch 1)", "peak VRAM (GB)", "5-set macro", "OOD avg",
                        "Audit avg"], rows), "",
              "Latency values come from `outputs/osler_rc0/R5/latency.json`; `NOT MEASURED` means the R5 timebox for "
              "that model was not met or its runner cannot batch.", ""]
    lines += ["## Paired comparison tables", ""]
    for cand, pairs in board["paired"].items():
        for base, per_set in pairs.items():
            rows = []
            for set_name in SETS:
                diff = per_set.get(set_name)
                if not diff or diff.get("n", 0) == 0:
                    rows.append([set_name, "NOT MEASURED", "—", "—"])
                    continue
                acc_ci = diff["accuracy_diff_ci95"]
                brier = diff.get("brier_diff")
                brier_ci = diff.get("brier_diff_ci95")
                rows.append([set_name, f"{diff['accuracy_diff']:+.4f}",
                             f"[{acc_ci[0]:+.4f}, {acc_ci[1]:+.4f}]",
                             f"{brier:+.4f} [{brier_ci[0]:+.4f}, {brier_ci[1]:+.4f}]" if isinstance(brier, float) else
                             str(brier)])
            lines += [f"### {cand} vs {base}", "",
                      md_table(["set", "Δ accuracy", "95% CI", "Δ Brier [95% CI] (lower better)"], rows), ""]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_pointer(path: Path, board: dict[str, Any]) -> None:
    """The repo-level pointer document: the RC0 headline table and links to the full reports."""
    metrics = board["models"]
    lines = ["# MedDecide leaderboard (RC0 snapshot)", "",
             f"Generated {board['generated_utc']} at git `{board['git_commit']}`. This is a pointer to the RC0 board: every "
             "number is recomputed from per-item prediction files by `scripts/osler_rc0/rc0_leaderboard.py` "
             "(12 models x 9 frozen evaluation sets, accuracy with 95% bootstrap CI, template macro, multi-class Brier, "
             "15-bin ECE, coverage).", "",
             "- Full board: [loops/osler_rc0/LEADERBOARD.md](loops/osler_rc0/LEADERBOARD.md)",
             "- Per-source and per-template tables: [loops/osler_rc0/PER_CATEGORY.md](loops/osler_rc0/PER_CATEGORY.md)",
             "- Pareto frontier and paired comparisons: [loops/osler_rc0/PARETO.md](loops/osler_rc0/PARETO.md)",
             "", "## Headline: accuracy on the five headline sets", "",
             "| model | panel | knowledge | confirm_general | mr_verdict | mimic_suite | 5-set macro |",
             "|---|---|---|---|---|---|---|"]
    rows = []
    for model in MODELS:
        values = []
        measured = []
        for set_name in HEADLINE:
            cell = metrics[model].get(set_name, {})
            if cell.get("status") != "MEASURED":
                values.append("NOT MEASURED")
            else:
                values.append(f"{cell['accuracy']:.4f}")
                measured.append(cell["accuracy"])
        macro = f"{sum(measured) / len(measured):.4f}" if len(measured) == len(HEADLINE) else "—"
        rows.append((model, values, macro))
    rows.sort(key=lambda row: (row[2] == "—", -float(row[2]) if row[2] != "—" else 0.0))
    lines += [f"| {model} | " + " | ".join(values) + f" | {macro} |" for model, values, macro in rows]
    pareto = board.get("pareto", {})
    lines += ["", "## Fronts", ""]
    two = pareto.get("two_axis", {})
    if two.get("front"):
        lines.append(f"- 2-axis Pareto front (OOD General Avg vs Clinical Audit Avg): {', '.join(two['front'])}.")
    q = pareto.get("quality_le10b", {})
    if q.get("undominated"):
        lines.append(f"- Undominated on all five headline sets (<=10B): {', '.join(q['undominated'])}.")
    lines += ["", "Coverage and NOT MEASURED reasons are in the full board; no cell is imputed. The prior loops' boards "
              "(`loops/osler_v*/FINDINGS.md`) remain the record for their own recipes.", ""]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/osler_rc0/R7")
    parser.add_argument("--latency", type=Path, default=ROOT / "outputs/osler_rc0/R5/latency.json")
    parser.add_argument("--reports", type=Path, default=ROOT / "loops/osler_rc0")
    parser.add_argument("--skip-reports", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    frozen = {name: load_frozen(name, path) for name, path in SETS.items()}
    cells, metrics = load_all(frozen)
    board: dict[str, Any] = {
        "generated_utc": utcnow(),
        "git_commit": git_commit(ROOT),
        "sets": {name: len(frozen[name]) for name in SETS},
        "models": metrics,
        "params_b": {m: spec["params_b"] for m, spec in MODELS.items()},
        "health_mode": {m: spec["health_mode"] for m, spec in MODELS.items()},
        "paired": {},
        "intersections": {},
    }
    for cand, bases in RIVAL_PAIRS.items():
        board["paired"][cand] = {}
        for base in bases:
            board["paired"][cand][base] = {}
            for set_name in SETS:
                a = cells[cand].get(set_name)
                b = cells[base].get(set_name)
                board["paired"][cand][base][set_name] = (
                    paired_difference(frozen[set_name], a, b) if a is not None and b is not None
                    else {"n": 0, "status": "NOT MEASURED — one cell absent"})
    for set_name in ("panel", "confirm_general"):
        restricted_cells = {}
        for name in MODELS:
            cell = cells[name].get(set_name)
            if cell is not None:
                restricted_cells[name] = restricted_to_n_options(cell, frozen[set_name], 10)
        board["intersections"][f"{set_name}_le10_options"] = {
            name: cell_metrics(frozen[set_name], cell) for name, cell in restricted_cells.items()}
    latency = json.loads(args.latency.read_text()) if args.latency.is_file() else {}
    board["latency"] = latency
    board["pareto"] = compute_pareto(frozen, metrics, latency)
    (args.out / "rc0_board.json").write_text(json.dumps(board, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.out / 'rc0_board.json'}")
    if not args.skip_reports:
        args.reports.mkdir(parents=True, exist_ok=True)
        write_leaderboard(args.reports / "LEADERBOARD.md", board)
        write_per_category(args.reports / "PER_CATEGORY.md", board, frozen, cells)
        write_pareto(args.reports / "PARETO.md", board)
        pointer = args.reports.parent.parent / "docs/leaderboard.md"
        write_pointer(pointer, board)
        print(f"wrote reports under {args.reports} and the pointer {pointer}")


if __name__ == "__main__":
    main()
