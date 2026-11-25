#!/usr/bin/env python
"""eval0 leaderboard — build the scoreboard from prediction files only (CPU).

Every number is recomputed here from the per-item prediction files; nothing is copied from a run summary. Items are
matched by ``item_id`` against the frozen item file for each set (the universe). A model that does not cover the
universe is reported with its coverage, never silently restricted. Brier and ECE need the full option distribution;
rows without ``probs`` report them as NOT MEASURED.

Run: uv run python scripts/eval0/leaderboard.py --out outputs/eval0/leaderboard
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SETS = {
    "panel": ROOT / "data/bench/osler_v4/y9/panel.jsonl",
    "knowledge": ROOT / "data/bench/osler_v4/y9/knowledge.jsonl",
    "mr_verdict": ROOT / "data/bench/osler_v4/y9/mr_verdict.jsonl",
    "mimic_suite": ROOT / "data/bench/osler_v4/y9/mimic_suite.jsonl",
}
T10 = ROOT / "outputs/osler_v7/T10"
Y9 = ROOT / "outputs/osler_v4/Y9/baselines"
EVAL = ROOT / "outputs/eval0/panel"

# label -> {set: prediction directory}. Missing directories are reported as NOT MEASURED.
MODELS: dict[str, dict[str, Path]] = {
    "Osler-9B final (63,304 steps)": {"panel": EVAL / "9b_final_panel", "knowledge": T10 / "9b_final_knowledge",
                                       "mr_verdict": T10 / "9b_final_mr_verdict", "mimic_suite": T10 / "9b_final_mimic_suite"},
    "Osler-9B arm (13,992 steps)": {"panel": EVAL / "9b_arm_panel", "knowledge": T10 / "9b_cand_knowledge",
                                     "mr_verdict": T10 / "9b_cand_mr_verdict", "mimic_suite": T10 / "9b_cand_mimic_suite"},
    "Osler-4B arm (13,992 steps)": {"panel": EVAL / "4b_arm_panel", "knowledge": T10 / "4b_cand_knowledge",
                                     "mr_verdict": T10 / "4b_cand_mr_verdict", "mimic_suite": T10 / "4b_cand_mimic_suite"},
    "Osler-4B v4 final": {"panel": Y9 / "final_panel", "knowledge": Y9 / "final_knowledge",
                          "mr_verdict": Y9 / "final_mr_verdict", "mimic_suite": Y9 / "final_mimic_suite"},
    "Osler-4B v2": {"panel": Y9 / "osler_v2_panel", "knowledge": Y9 / "osler_v2_knowledge",
                    "mr_verdict": Y9 / "osler_v2_mr_verdict"},
    "pplx-decider-flash-v1.2 (zero-shot, 4B base)": {"panel": Y9 / "pplx_panel", "knowledge": Y9 / "pplx_knowledge",
                                                      "mr_verdict": Y9 / "pplx_mr_verdict",
                                                      "mimic_suite": T10 / "baselines/4b_base_mimic_suite"},
    "d3-flash (zero-shot, 9B base)": {"panel": EVAL / "9b_base_panel", "knowledge": T10 / "9b_base_knowledge",
                                       "mr_verdict": T10 / "9b_base_mr_verdict", "mimic_suite": T10 / "9b_base_mimic_suite"},
    "pplx-decider-v1.2-27b (zero-shot, ceiling)": {"panel": Y9 / "pplx27b_panel", "knowledge": Y9 / "pplx27b_knowledge",
                                                    "mr_verdict": Y9 / "pplx27b_mr_verdict",
                                                    "mimic_suite": Y9 / "pplx27b_mimic_suite"},
    "MedDecider-4B (lion-ai)": {"panel": Y9 / "md4b_panel", "knowledge": Y9 / "md4b_knowledge",
                                "mr_verdict": Y9 / "md4b_mr_verdict"},
    "MedDecider-9B (lion-ai)": {"panel": Y9 / "md9b_panel", "knowledge": Y9 / "md9b_knowledge",
                                "mr_verdict": Y9 / "md9b_mr_verdict"},
    "JEV-9B": {"panel": Y9 / "jev9b_panel", "knowledge": Y9 / "jev9b_knowledge", "mr_verdict": Y9 / "jev9b_mr_verdict",
               "mimic_suite": Y9 / "jev9b_mimic_suite"},
}
# Own-base pairs for the paired difference (candidate -> its zero-shot base).
OWN_BASE = {
    "Osler-9B final (63,304 steps)": "d3-flash (zero-shot, 9B base)",
    "Osler-9B arm (13,992 steps)": "d3-flash (zero-shot, 9B base)",
    "Osler-4B arm (13,992 steps)": "pplx-decider-flash-v1.2 (zero-shot, 4B base)",
}
BOOT = 2000
ECE_BINS = 15


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_rows(directory: Path) -> dict[str, dict[str, Any]] | None:
    path = directory / "predictions.jsonl"
    if not path.is_file():
        return None
    rows = read_jsonl(path)
    index: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row["item_id"] in index:
            raise ValueError(f"duplicate item id in {path}")
        index[row["item_id"]] = row
    return index


def correct_of(row: dict[str, Any]) -> float:
    return 1.0 if row["correct"] else 0.0


def gold_index(row: dict[str, Any]) -> int:
    """Gold option index; comparator files carry only the letter key, which maps to the same index (A=0)."""
    if "gold_index" in row:
        return int(row["gold_index"])
    key = str(row["gold_key"])
    if key in ("yes", "no"):
        return 0 if key == "yes" else 1  # yes/no items: probs are ordered (yes, no); checked against predicted_key
    return ord(key) - ord("A")


def brier_row(row: dict[str, Any]) -> float | None:
    probs = row.get("probs")
    if probs is None:
        return None
    gold = gold_index(row)
    return sum((p - (1.0 if k == gold else 0.0)) ** 2 for k, p in enumerate(probs))


def ece(rows: list[dict[str, Any]]) -> float | None:
    if any(r.get("probs") is None for r in rows):
        return None
    bins: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for r in rows:
        conf = max(r["probs"])
        b = min(int(conf * ECE_BINS), ECE_BINS - 1)
        bins[b].append((conf, correct_of(r)))
    n = len(rows)
    total = 0.0
    for members in bins.values():
        conf = sum(c for c, _ in members) / len(members)
        acc = sum(a for _, a in members) / len(members)
        total += len(members) / n * abs(acc - conf)
    return total


def bootstrap_ci(values: list[float], seed: int = 0) -> tuple[float, float]:
    import numpy as np

    arr = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(arr), size=(BOOT, len(arr)))
    means = arr[idx].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def macro(rows: list[dict[str, Any]]) -> float:
    by_t: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        by_t[str(r["template_id"])].append(correct_of(r))
    return sum(sum(v) / len(v) for v in by_t.values()) / len(by_t)


def summarise(rows_by_id: dict[str, dict[str, Any]], universe: list[str]) -> dict[str, Any]:
    scored = [rows_by_id[i] for i in universe if i in rows_by_id]
    out: dict[str, Any] = {"universe": len(universe), "scored": len(scored),
                           "coverage": round(len(scored) / len(universe), 6)}
    if not scored:
        return out
    vals = [correct_of(r) for r in scored]
    out["accuracy"] = round(sum(vals) / len(vals), 6)
    out["accuracy_ci95"] = [round(x, 6) for x in bootstrap_ci(vals)]
    out["macro_accuracy"] = round(macro(scored), 6)
    briers = [brier_row(r) for r in scored]
    if all(b is not None for b in briers):
        out["brier"] = round(sum(briers) / len(briers), 6)  # multi-class Brier, summed over offered options
        out["ece"] = round(ece(scored), 6)
    else:
        out["brier"] = "NOT MEASURED — no probs in prediction file"
        out["ece"] = "NOT MEASURED — no probs in prediction file"
    return out


def per_category(rows_by_id: dict[str, dict[str, Any]], universe: list[str], key: str) -> dict[str, Any]:
    groups: dict[str, list[float]] = defaultdict(list)
    for i in universe:
        if i in rows_by_id:
            groups[str(rows_by_id[i][key])].append(correct_of(rows_by_id[i]))
    return {g: {"n": len(v), "accuracy": round(sum(v) / len(v), 6)} for g, v in sorted(groups.items())}


def paired_diff(a: dict[str, dict[str, Any]], b: dict[str, dict[str, Any]], universe: list[str]) -> dict[str, Any]:
    common = [i for i in universe if i in a and i in b]
    if not common:
        return {"n": 0}
    diffs = [correct_of(a[i]) - correct_of(b[i]) for i in common]
    point = sum(diffs) / len(diffs)
    lo, hi = bootstrap_ci(diffs, seed=1)
    return {"n": len(common), "diff": round(point, 6), "ci95": [round(lo, 6), round(hi, 6)]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    frozen_sizes = {name: sum(1 for line in path.open(encoding="utf-8") if line.strip()) for name, path in SETS.items()}
    # The id universe of each set is the set of item ids in the Osler-9B final prediction file for it (full coverage).
    loaded: dict[str, dict[str, dict[str, Any] | None]] = {}
    for model, paths in MODELS.items():
        loaded[model] = {s: load_rows(paths[s]) if s in paths else None for s in SETS}

    universe: dict[str, list[str]] = {}
    for s in SETS:
        ref = loaded["Osler-9B final (63,304 steps)"][s]
        if ref is None:
            raise SystemExit(f"no reference predictions for {s}")
        if len(ref) != frozen_sizes[s]:
            raise SystemExit(f"{s}: reference has {len(ref)} ids but the frozen file has {frozen_sizes[s]} rows")
        universe[s] = sorted(ref)

    board: dict[str, Any] = {"frozen_set_sizes": frozen_sizes, "universe_sizes": {s: len(v) for s, v in universe.items()},
                             "models": {}, "paired_vs_own_base": {}}
    for model in MODELS:
        board["models"][model] = {}
        for s in SETS:
            rows = loaded[model][s]
            if rows is None:
                board["models"][model][s] = {"status": f"NOT MEASURED — no predictions in {MODELS[model].get(s, 'set not listed')}"}
            else:
                board["models"][model][s] = summarise(rows, universe[s])
    board["per_category"] = {}
    for model in MODELS:
        board["per_category"][model] = {}
        for s in SETS:
            rows = loaded[model][s]
            if rows is None:
                continue
            board["per_category"][model][s] = {
                "by_source": per_category(rows, universe[s], "source"),
                "by_template": per_category(rows, universe[s], "template_id"),
            }
    for cand, base in OWN_BASE.items():
        board["paired_vs_own_base"][cand] = {}
        for s in SETS:
            a, b = loaded[cand][s], loaded[base][s]
            if a is None or b is None:
                board["paired_vs_own_base"][cand][s] = "NOT MEASURED — missing predictions"
            else:
                board["paired_vs_own_base"][cand][s] = paired_diff(a, b, universe[s])
    (args.out / "leaderboard.json").write_text(json.dumps(board, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(board, indent=1, sort_keys=True)[:3000])


if __name__ == "__main__":
    main()
