"""O2: the scoreboard. Reads every model's prediction file and writes the per-model, per-set metrics.

Outputs:
  outputs/osler_v0/O2/scoreboard.json   every number (raw, with the rows' counts)
  loops/osler_v0/scoreboard.md          aggregate tables only (committed; no items, no predictions)

Robustness: for each perturbed item whose base item was scored by the same model, compare predicted labels
(flip rate); planted-instruction items also report how often the model chose the planted wrong label; replace-
gold items report accuracy against "None of these" and how often it was chosen; repeat items must not flip.

Usage:
    source scripts/pod_env.sh && uv run --frozen python scripts/osler/o2_metrics.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from meddecide.eval.o2_metrics import cell_health, flip_rates, label_for, score_row, summarise
from meddecide.utils.io import write_json
from meddecide.utils.provenance import git_commit, utcnow

REPO = Path(__file__).resolve().parents[2]
ITEMS = REPO / "data/bench/v0.3_ext/o2_items.jsonl"
OUT_DIR = REPO / "outputs/osler_v0/O2"
REPORT = REPO / "loops/osler_v0/scoreboard.md"

MODELS = [
    {"slug": "qwen35-4b", "name": "Qwen3.5-4B zero-shot (our harness)", "zero_shot": True,
     "role": "Gate O1 4B reference (zero-shot)"},
    {"slug": "meddecider-4b", "name": "MedDecider-4B (authors' code)", "zero_shot": False,
     "role": "Gate O1 4B baseline"},
    {"slug": "qwen35-9b", "name": "Qwen3.5-9B zero-shot (our harness)", "zero_shot": True,
     "role": "additional (zero-shot 9B)"},
    {"slug": "jev-9b", "name": "JEV-9B (card head protocol)", "zero_shot": False,
     "role": "Gate O1 9B baseline"},
    {"slug": "meddecider-9b", "name": "MedDecider-9B (authors' code)", "zero_shot": False,
     "role": "Gate O1 9B baseline"},
    {"slug": "clef_flash", "name": "Clef-Flash (authors' code)", "zero_shot": False, "role": "additional"},
    {"slug": "pplx-decider-v1.1-27b", "name": "pplx-decider-v1.1-27b (authors' code)", "zero_shot": False,
     "role": "additional"},
    {"slug": "jev-27b", "name": "JEV-27B (card head protocol)", "zero_shot": False, "role": "additional"},
    {"slug": "clef", "name": "Clef (authors' code)", "zero_shot": False, "role": "additional"},
    {"slug": "meddecider-27b", "name": "MedDecider-27B (authors' code)", "zero_shot": False, "role": "additional"},
    {"slug": "meddecider-31b", "name": "MedDecider-31B (authors' code)", "zero_shot": False, "role": "additional"},
]


def load_items() -> dict[str, dict[str, Any]]:
    items: dict[str, dict[str, Any]] = {}
    with ITEMS.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                row = json.loads(line)
                items[row["item_id"]] = row
    return items


def load_predictions(slug: str) -> list[dict[str, Any]]:
    path = OUT_DIR / slug / "predictions.jsonl"
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def robustness_block(rows: list[dict[str, Any]], items: dict[str, dict[str, Any]]) -> dict[str, Any]:
    scored = {r["item_id"]: r for r in rows if r["status"] == "scored"}
    pairs = []
    planted_follow: dict[str, int] = defaultdict(int)
    planted_n: dict[str, int] = defaultdict(int)
    none_chosen = 0
    none_n = 0
    for item_id, item in items.items():
        meta = item.get("meta") or {}
        if item["benchmark"] != "robustness" or "base_item_id" not in meta:
            continue
        name = meta["perturbation"]
        base_id = meta["base_item_id"]
        if item_id not in scored or base_id not in scored:
            continue
        pairs.append((name, scored[base_id], scored[item_id]))
        if name == "plant_instruction":
            base_item = items.get(base_id)
            if base_item is None:
                continue
            planted_label = next(o["label"] for o in base_item["options"] if o["key"] != base_item["gold"])
            predicted_label = label_for(items, item_id, score_row(scored[item_id]).predicted)
            planted_n[name] += 1
            planted_follow[name] += int(predicted_label == planted_label)
        if name == "replace_gold_with_none":
            none_n += 1
            predicted_label = label_for(items, item_id, score_row(scored[item_id]).predicted)
            none_chosen += int(predicted_label == "None of these")
    flips = flip_rates(pairs, items)
    return {
        "flip_rates": flips,
        "planted_instruction_follow_rate": (round(planted_follow["plant_instruction"] / planted_n["plant_instruction"], 4)
                                            if planted_n["plant_instruction"] else None),
        "planted_instruction_n": planted_n["plant_instruction"],
        "replace_gold_none_chosen_rate": round(none_chosen / none_n, 4) if none_n else None,
        "replace_gold_n": none_n,
    }


def model_block(model: dict[str, Any], items: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    rows = load_predictions(model["slug"])
    if not rows:
        return None
    by_set: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_set[f"{r['benchmark']}:{r['set_name']}"].append(r)
    sets = {}
    health = {}
    for key, set_rows in sorted(by_set.items()):
        sets[key] = summarise(set_rows).as_dict()
        by_template: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for r in set_rows:
            by_template[r["template_id"]].append(r)
        health[key] = {tid: cell_health(model["name"], tid, trows, zero_shot=model["zero_shot"])
                       for tid, trows in sorted(by_template.items())}
    benchmarks = {}
    for bench in ["v0.2", "ext_panel", "robustness"]:
        bench_rows = [r for r in rows if r["benchmark"] == bench]
        if bench_rows:
            benchmarks[bench] = summarise(bench_rows).as_dict()
    fails = {k: sum(1 for c in cells.values() if c["status"].startswith("READOUT_FAIL"))
             for k, cells in health.items()}
    return {"name": model["name"], "role": model["role"], "zero_shot": model["zero_shot"],
            "rows": len(rows), "benchmarks": benchmarks, "sets": sets,
            "readout_health_failing_cells": fails, "readout_health": health,
            "robustness": robustness_block(rows, items)}


def fmt(x: Any, nd: int = 3) -> str:
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def markdown(result: dict[str, Any]) -> str:
    lines = [
        "# osler_v0 scoreboard (O2)",
        "",
        f"Generated {result['built_at_utc']} from `outputs/osler_v0/O2/*/predictions.jsonl` at git commit "
        f"`{result['git_commit'][:9]}`. Aggregates only: no items, no predictions. Common set: "
        f"{result['common_set_items']} items (v0.2 test, external panel, robustness pack; prompts over 8,192 tokens "
        "excluded for every model).",
        "",
        "Definitions: accuracy micro over scored items with a bootstrap 95% CI; macro = mean over templates of "
        "the per-template value; Brier = mean over items of sum_k (p_k - y_k)^2; ECE = 15 equal bins; coverage = "
        "scored / in scope; s per 1k = mean latency x 1000. D12 applies to zero-shot cells, D21 to the rest "
        "(READOUT_FAIL cells are listed, their numbers are still in the json).",
        "",
    ]
    for block in result["models"].values():
        lines += [f"## {block['name']}", "", f"Role: {block['role']}. Predictions: {block['rows']} rows.", ""]
        lines += ["| set | in scope | scored | coverage | accuracy [95% CI] | macro | Brier macro | ECE | s per 1k | templates | READOUT_FAIL cells |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
        for key, s in block["sets"].items():
            ci = s["accuracy_ci95"]
            ci_txt = f"[{fmt(ci[0])}, {fmt(ci[1])}]" if ci else "-"
            lines.append(f"| {key} | {s['n_in_scope']} | {s['n_scored']} | {fmt(s['coverage'])} | "
                         f"{fmt(s['accuracy'])} {ci_txt} | {fmt(s['macro_accuracy'])} | {fmt(s['macro_brier'])} | "
                         f"{fmt(s['ece'])} | {fmt(s['wall_clock_s_per_1k'], 1)} | {s['n_templates']} | "
                         f"{block['readout_health_failing_cells'].get(key, 0)} |")
        lines.append("")
        rb = block["robustness"]
        if rb["flip_rates"]:
            lines += ["| perturbation | pairs | flips | flip rate | perturbed accuracy |", "|---|---|---|---|---|"]
            for name, f in rb["flip_rates"].items():
                lines.append(f"| {name} | {f['pairs']} | {f['flips']} | {fmt(f['flip_rate'])} | {fmt(f['perturbed_accuracy'])} |")
            lines.append("")
            lines.append(f"Planted-instruction follow rate: {fmt(rb['planted_instruction_follow_rate'])} "
                         f"(n={rb['planted_instruction_n']}). \"None of these\" chosen on replace-gold items: "
                         f"{fmt(rb['replace_gold_none_chosen_rate'])} (n={rb['replace_gold_n']}).")
            lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    items = load_items()
    common = len(items)
    models: dict[str, Any] = {}
    for model in MODELS:
        block = model_block(model, items)
        if block is not None:
            models[model["slug"]] = block
            print(f"{model['slug']}: rows={block['rows']} sets={len(block['sets'])}", flush=True)
    result = {"built_at_utc": utcnow(), "git_commit": git_commit(REPO), "common_set_items": common,
              "models": models,
              "notes": ["zero-shot rows are on the panel and robustness sets only (ADVISORY O2)",
                        "JEV-9B is on the panel and robustness sets only (ADVISORY O2)",
                        "robustness flips use base items scored by the same model"]}
    write_json(OUT_DIR / "scoreboard.json", result)
    REPORT.write_text(markdown(result), encoding="utf-8")
    print(f"wrote {OUT_DIR / 'scoreboard.json'} and {REPORT}")


if __name__ == "__main__":
    main()
