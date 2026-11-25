"""osler_v3 X10 report (operator ruling, 2026-10-20): the Stage A runs and block B1, calibrated, as ADDITIONAL rows. CPU only.

No Gate O3 verdict is computed or printed here: the operator ruled that no new final model is trained, so D35/D39/D44 are not
applied. Each arm is compared with the two baselines named by the ruling, zero-shot pplx-decider-flash-v1.2 (W5's files) and
osler_v2's Osler-4B (W8's both-orders file), on the same v10 items, paired by item id, with the gate helpers unchanged
(stratified macro accuracy, paired bootstrap, 1,000 resamples, seed 0; the same settings as osler_v2's W8 gate).

Scopes: the MR suite v1 verdict templates (mr_test), the external panel overall and per dataset (set_name), and the knowledge items.
The clinical-general (CG) accuracy and the mean KL to the base are read from each run's cg_result.json (the trade-off columns).

Writes outputs/osler_v3/X10/results.json (every number with its item counts) and loops/osler_v3/results.md (the tables).

Usage:  uv run --frozen python scripts/osler_v3/x10_report.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from meddecide.eval.gate import accuracy_difference, compare_rows
from meddecide.utils.io import iter_jsonl, write_json
from meddecide.utils.provenance import git_commit, utcnow

REPO = Path(__file__).resolve().parents[2]
ITEMS = REPO / "data" / "bench" / "osler_v1" / "v10_items.jsonl"
MANIFEST = REPO / "data" / "bench" / "osler_mr_v1" / "manifest.json"
X10 = REPO / "outputs" / "osler_v3" / "X10"
STAGE_A = REPO / "outputs" / "osler_v3" / "stage_a"
STAGE_B = REPO / "outputs" / "osler_v3" / "stage_b"
OSLER_V2 = REPO / "outputs" / "osler_v2" / "W8" / "osler_4b" / "both_orders.jsonl"
PPLX_NAME = "zero-shot pplx-decider-flash-v1.2"
PPLX = REPO / "outputs" / "osler_v2" / "W5" / "pplx_test"
OUT_JSON = X10 / "results.json"
OUT_MD = REPO / "loops" / "osler_v3" / "results.md"
N_RESAMPLES = 1000
SEED = 0

ARMS = {
    "A0 (control, kl 0)": ("A0", STAGE_A / "A0"),
    "A1 (KL-to-base 0.1)": ("A1", STAGE_A / "A1"),
    "A2 (KL-to-base 0.5)": ("A2", STAGE_A / "A2"),
    "A3 (broad replay, kl 0)": ("A3", STAGE_A / "A3"),
    "B1 (X1 real sources)": ("B1", STAGE_B / "B1"),
}


def by_item(path: Path) -> dict[str, dict[str, Any]]:
    return {str(row["item_id"]): row for row in iter_jsonl(path)}


def micro(scope_rows: list[dict[str, Any]], rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Micro accuracy over the scored items of a scope; the denominators are stated (R5)."""
    scored = [rows[str(r["item_id"])] for r in scope_rows if str(r["item_id"]) in rows
              and rows[str(r["item_id"])].get("status") == "scored"]
    correct = sum(max(range(len(r["probs"])), key=lambda i: r["probs"][i]) == r["option_keys"].index(r["gold"])
                  for r in scored)
    return {"n_scope": len(scope_rows), "n_scored": len(scored), "accuracy": correct / len(scored) if scored else None}


def paired(scope: str, scope_rows: list[dict[str, Any]], arm: dict[str, Any], base_name: str, base: dict[str, Any]) -> dict[str, Any]:
    try:
        result = compare_rows(scope, base_name, scope_rows, arm, base, n_resamples=N_RESAMPLES, seed=SEED)
    except ValueError as exc:
        return {"scope": scope, "baseline": base_name, "not_measured": str(exc)}
    comp = result.comparison
    return {"scope": scope, "baseline": base_name, "n_scope": result.n_scope, "n_paired": result.n_paired,
            "n_baseline_skipped": result.n_baseline_skipped, "n_baseline_missing": result.n_baseline_missing,
            "n_osler_missing": result.n_osler_missing, "accuracy": comp.accuracy.__dict__, "brier": comp.brier.__dict__}


def paired_knowledge(scope_rows: list[dict[str, Any]], arm: dict[str, Any], base_name: str, base: dict[str, Any]) -> dict[str, Any]:
    try:
        diff, n = accuracy_difference(scope_rows, arm, base, n_resamples=N_RESAMPLES, seed=SEED)
    except ValueError as exc:
        return {"scope": "knowledge", "baseline": base_name, "not_measured": str(exc)}
    return {"scope": "knowledge", "baseline": base_name, "n_scope": len(scope_rows), "n_paired": n,
            "accuracy": diff.__dict__}


def cg_summary(run_dir: Path) -> dict[str, Any]:
    result = json.loads((run_dir / "cg_result.json").read_text(encoding="utf-8"))
    return {"cg_items": result["cg_items"], "cg_accuracy_micro": result["cg_accuracy_micro"],
            "mean_kl_to_base_cg": result["mean_kl_to_base_cg"]}


def main() -> None:
    rows = list(iter_jsonl(ITEMS))
    verdict_templates = set(json.loads(MANIFEST.read_text(encoding="utf-8"))["verdict_templates"])
    mr = [r for r in rows if r["v10_scope"] == "mr_test" and r["template_id"] in verdict_templates]
    panel = [r for r in rows if r["v10_scope"] == "external_panel"]
    knowledge = [r for r in rows if r["v10_scope"] == "knowledge"]
    datasets = sorted({r["set_name"] for r in panel})
    sizes = {"mr_verdict": len(mr), "external_panel": len(panel), "knowledge": len(knowledge), "panel_datasets": {
        d: sum(r["set_name"] == d for r in panel) for d in datasets}}

    pplx = {scope: by_item(PPLX / folder / "pplx-decider-flash-v1.2" / "predictions.jsonl")
            for scope, folder in {"mr": "mr", "panel": "panel", "knowledge": "knowledge"}.items()}
    osler_v2 = by_item(OSLER_V2)
    references = {"mr_verdict": micro(mr, pplx["mr"]), "external_panel": micro(panel, pplx["panel"]),
                  "knowledge": micro(knowledge, pplx["knowledge"])}
    v2_scores = {"mr_verdict": micro(mr, osler_v2), "external_panel": micro(panel, osler_v2),
                 "knowledge": micro(knowledge, osler_v2)}

    results: dict[str, Any] = {"kind": "osler_v3_X10_results", "written_utc": utcnow(), "git_commit": git_commit(REPO),
                               "scope_sizes": sizes, "n_resamples": N_RESAMPLES, "seed": SEED,
                               "verdict": "NOT COMPUTED — no Gate O3 verdict (operator ruling, 2026-10-20)",
                               "micro_accuracy_zero_shot_pplx": references, "micro_accuracy_osler_v2_osler_4b": v2_scores,
                               "osler_v2_osler_4b_vs_zero_shot_pplx": {
                                   "mr_verdict": paired("mr_verdict", mr, osler_v2, PPLX_NAME, pplx["mr"]),
                                   "external_panel": paired("external_panel", panel, osler_v2, PPLX_NAME, pplx["panel"]),
                                   "knowledge": paired_knowledge(knowledge, osler_v2, PPLX_NAME, pplx["knowledge"])},
                               "arms": {}}
    for label, (name, stage_dir) in ARMS.items():
        arm = by_item(X10 / name / "both_orders.jsonl")
        results["arms"][label] = {
            "run": name, "cg": cg_summary(stage_dir),
            "micro_accuracy": {scope: micro(s, arm) for scope, s in
                               {"mr_verdict": mr, "external_panel": panel, "knowledge": knowledge}.items()},
            "vs_zero_shot_pplx": {
                "mr_verdict": paired("mr_verdict", mr, arm, PPLX_NAME, pplx["mr"]),
                "external_panel": paired("external_panel", panel, arm, PPLX_NAME, pplx["panel"]),
                "knowledge": paired_knowledge(knowledge, arm, PPLX_NAME, pplx["knowledge"]),
                "external_panel_by_dataset": {d: paired(f"external_panel:{d}", [r for r in panel if r["set_name"] == d],
                                                       arm, PPLX_NAME, pplx["panel"]) for d in datasets}},
            "vs_osler_v2_osler_4b": {
                "mr_verdict": paired("mr_verdict", mr, arm, "osler_v2 Osler-4B", osler_v2),
                "external_panel": paired("external_panel", panel, arm, "osler_v2 Osler-4B", osler_v2),
                "knowledge": paired_knowledge(knowledge, arm, "osler_v2 Osler-4B", osler_v2),
                "external_panel_by_dataset": {d: paired(f"external_panel:{d}", [r for r in panel if r["set_name"] == d],
                                                       arm, "osler_v2 Osler-4B", osler_v2) for d in datasets}},
        }
    write_json(OUT_JSON, results)
    OUT_MD.write_text(render(results, datasets), encoding="utf-8")
    print(f"wrote {OUT_JSON.relative_to(REPO)} and {OUT_MD.relative_to(REPO)}", flush=True)


def fmt(d: dict[str, Any] | None) -> str:
    if d is None or "not_measured" in d:
        return "NOT MEASURED"
    a = d["accuracy"]
    return f"{a['point']:+.4f} [{a['lo']:+.4f}, {a['hi']:+.4f}] (n {d['n_paired']})"


def render(results: dict[str, Any], datasets: list[str]) -> str:
    sizes = results["scope_sizes"]
    ref = results["micro_accuracy_zero_shot_pplx"]
    v2 = results["micro_accuracy_osler_v2_osler_4b"]
    lines = [
        "# osler_v3 — X10 results (additional rows; no Gate O3 verdict)",
        "",
        "Written by `scripts/osler_v3/x10_report.py` from `outputs/osler_v3/X10/results.json` (aggregates only). Operator ruling,",
        "2026-10-20: no new final model, so no verdict. Every difference is Arm minus baseline on the same paired items, with the",
        "paired 95 % bootstrap interval (stratified macro accuracy for MR and the panel; micro for knowledge; 1,000 resamples, seed 0).",
        "",
        f"Scopes (v10 items, `data/bench/osler_v1/v10_items.jsonl`): MR suite v1 verdict {sizes['mr_verdict']}; external panel "
        f"{sizes['external_panel']}; knowledge {sizes['knowledge']}.",
        "",
        "## 1. Baselines (micro accuracy on the scope's scored items)",
        "",
        "| model | MR verdict | external panel | knowledge |",
        "|---|---|---|---|",
        f"| zero-shot pplx-decider-flash-v1.2 | {ref['mr_verdict']['accuracy']:.4f} ({ref['mr_verdict']['n_scored']} of "
        f"{ref['mr_verdict']['n_scope']}) | {ref['external_panel']['accuracy']:.4f} ({ref['external_panel']['n_scored']} of "
        f"{ref['external_panel']['n_scope']}) | {ref['knowledge']['accuracy']:.4f} ({ref['knowledge']['n_scored']} of "
        f"{ref['knowledge']['n_scope']}) |",
        f"| osler_v2 Osler-4B | {v2['mr_verdict']['accuracy']:.4f} ({v2['mr_verdict']['n_scored']} of {v2['mr_verdict']['n_scope']}) "
        f"| {v2['external_panel']['accuracy']:.4f} ({v2['external_panel']['n_scored']} of {v2['external_panel']['n_scope']}) "
        f"| {v2['knowledge']['accuracy']:.4f} ({v2['knowledge']['n_scored']} of {v2['knowledge']['n_scope']}) |",
        "",
        "## 2. Stage A and B1 against both baselines",
        "",
        "| run | KL to base (CG) | CG accuracy | MR vs pplx | panel vs pplx | knowledge vs pplx | MR vs osler_v2 | panel vs osler_v2 | knowledge vs osler_v2 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for label, arm in results["arms"].items():
        cg = arm["cg"]
        p, v = arm["vs_zero_shot_pplx"], arm["vs_osler_v2_osler_4b"]
        lines.append(f"| {label} | {cg['mean_kl_to_base_cg']:.4f} | {cg['cg_accuracy_micro']:.4f} | {fmt(p['mr_verdict'])} | "
                     f"{fmt(p['external_panel'])} | {fmt(p['knowledge'])} | {fmt(v['mr_verdict'])} | "
                     f"{fmt(v['external_panel'])} | {fmt(v['knowledge'])} |")
    lines += ["", "osler_v2's Osler-4B against zero-shot pplx (reference row, same items):", "",
              f"- MR verdict: {fmt(results['osler_v2_osler_4b_vs_zero_shot_pplx']['mr_verdict'])}",
              f"- external panel: {fmt(results['osler_v2_osler_4b_vs_zero_shot_pplx']['external_panel'])}",
              f"- knowledge: {fmt(results['osler_v2_osler_4b_vs_zero_shot_pplx']['knowledge'])}", "",
              "## 3. External panel by dataset (Arm minus baseline)", ""]
    for base_key, title in (("vs_zero_shot_pplx", "zero-shot pplx-decider-flash-v1.2"), ("vs_osler_v2_osler_4b", "osler_v2 Osler-4B")):
        lines += [f"### against {title}", "", "| dataset (n panel items) | " + " | ".join(results["arms"]) + " |",
                  "|---|" + "---|" * len(results["arms"])]
        for d in datasets:
            cells = [fmt(arm[base_key]["external_panel_by_dataset"][d]) for arm in results["arms"].values()]
            n = results["scope_sizes"]["panel_datasets"][d]
            lines.append(f"| {d} ({n}) | " + " | ".join(cells) + " |")
        lines.append("")
    lines += ["Denominators are the paired counts in `outputs/osler_v3/X10/results.json` (n paired; baseline skips are counted there).",
              "The per-dataset rows are additional and carry no pass or fail.", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
