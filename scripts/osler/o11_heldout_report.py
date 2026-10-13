"""O11: the Osler models on the held-out generators' test items, as their own table (CPU; reads the scored predictions).

heldout.md rule 3: held-out results are reported in their own row and never pooled with the seen generators. This script reads the
held-out predictions that o11_heldout_score.py wrote (outputs/osler_v0/O11/heldout/<name>/both_orders.jsonl, and single_order.jsonl
beside them) and writes:
    outputs/osler_v0/O11/heldout/report.json      the numbers the CLAIMS rows cite (gitignored)
    loops/osler_v0/heldout_generators.md          the table (committed; aggregates only)
The four D24 baselines were not scored on these items, so the comparisons against them are NOT MEASURED and are not computed here.
The paired differences between Osler models are additional, never in a verdict. Every model must have all 600 rows; a missing or
duplicated row raises (no silent drops).

Usage:
    uv run --frozen python scripts/osler/o11_heldout_report.py
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from meddecide.eval.paired import item_brier, stratified_macro_difference
from meddecide.gen.families import HELD_OUT
from meddecide.utils.io import iter_jsonl, write_json
from meddecide.utils.provenance import git_commit, utcnow

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "outputs/osler_v0/O11/heldout"
REPORT_MD = REPO / "loops/osler_v0/heldout_generators.md"
MODELS = ("osler_4b_L", "osler_4b_P", "osler_4b_N", "osler_9b", "osler_0p8b")
PAIRS = (("osler_4b_P", "osler_4b_L"), ("osler_4b_N", "osler_4b_L"), ("osler_9b", "osler_4b_L"))
EXPECTED_ROWS = 600
N_RESAMPLES = 2000
SEED = 0


def is_correct(row: Mapping[str, Any]) -> bool:
    """The answer is the option with the highest probability; the probabilities are aligned with the row's option keys."""
    return row["option_keys"][int(np.argmax(row["probs"]))] == row["gold"]


def item_table(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """One entry per item id: correctness, template, gold key and the item's Brier score. Refuses a missing or duplicated row."""
    table: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row["status"] != "scored":
            raise ValueError(f"{row['item_id']} is not scored ({row['status']})")
        if row["item_id"] in table:
            raise ValueError(f"{row['item_id']} is written twice")
        gold_index = row["option_keys"].index(row["gold"])
        table[row["item_id"]] = {"correct": float(is_correct(row)), "template": row["template_id"], "gold": row["gold"],
                                 "brier": item_brier(row["probs"], gold_index)}
    if len(table) != EXPECTED_ROWS:
        raise ValueError(f"{len(table)} held-out items scored; {EXPECTED_ROWS} expected")
    return table


def groups_by_template(table: Mapping[str, Mapping[str, Any]], ids: Sequence[str] | None = None) -> dict[str, list[float]]:
    """Per template, the 0/1 correctness in item-id order (the paired comparison needs the same order for both models)."""
    ids = sorted(table) if ids is None else ids
    groups: dict[str, list[float]] = {}
    for item_id in ids:
        groups.setdefault(table[item_id]["template"], []).append(table[item_id]["correct"])
    return groups


def macro_interval(groups: Mapping[str, Sequence[float]], *, n_resamples: int = N_RESAMPLES,
                   seed: int = SEED) -> tuple[float, float, float]:
    """Template macro accuracy with a seeded percentile bootstrap that resamples the items within each template."""
    rng = np.random.default_rng(seed)
    arrays = {t: np.asarray(v, dtype=np.float64) for t, v in sorted(groups.items())}
    point = float(np.mean([a.mean() for a in arrays.values()]))
    boot = np.mean([a[rng.integers(0, a.size, size=(n_resamples, a.size))].mean(axis=1) for a in arrays.values()], axis=0)
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return point, float(lo), float(hi)


def summarise(table: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    groups = groups_by_template(table)
    point, lo, hi = macro_interval(groups)
    majority = {}
    for template in sorted(groups):
        golds = [table[i]["gold"] for i in sorted(table) if table[i]["template"] == template]
        majority[template] = max(golds.count(g) for g in set(golds)) / len(golds)
    return {
        "n": len(table),
        "n_by_template": {t: len(v) for t, v in sorted(groups.items())},
        "accuracy_overall": float(np.mean([v["correct"] for v in table.values()])),
        "template_macro_accuracy": point,
        "template_macro_ci_95": [lo, hi],
        "accuracy_by_template": {t: float(np.mean(v)) for t, v in sorted(groups.items())},
        "brier_mean": float(np.mean([v["brier"] for v in table.values()])),
        "majority_by_template": majority,
        "majority_template_macro": float(np.mean(list(majority.values()))),
    }


def paired(table_a: Mapping[str, Mapping[str, Any]], table_b: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Template macro difference A - B over the same items (additional; resampled within template, seeded)."""
    if set(table_a) != set(table_b):
        raise ValueError("the two models do not cover the same items")
    ids = sorted(table_a)
    diff = stratified_macro_difference(groups_by_template(table_a, ids), groups_by_template(table_b, ids),
                                       n_resamples=N_RESAMPLES, seed=SEED)
    return {"n": diff.n, "point": diff.point, "ci_95": [diff.lo, diff.hi]}


def load(name: str, kind: str) -> list[dict[str, Any]]:
    path = OUT / name / f"{kind}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing; run o11_heldout_chain.sh first")
    return list(iter_jsonl(path))


def main() -> None:
    tables = {name: item_table(load(name, "both_orders")) for name in MODELS}
    singles = {name: item_table(load(name, "single_order")) for name in MODELS}
    report = {
        "kind": "osler_v0_O11_heldout_report",
        "built_at_utc": utcnow(),
        "git_commit": git_commit(REPO),
        "command": "scripts/osler/o11_heldout_report.py",
        "held_out_generators": sorted(HELD_OUT),
        "models": {name: {"both_orders": summarise(tables[name]), "single_order": summarise(singles[name])}
                   for name in MODELS},
        "paired_additional": {f"{a} - {b}": paired(tables[a], tables[b]) for a, b in PAIRS},
        "baselines": "NOT MEASURED: the four D24 baselines were not scored on the held-out generator items",
    }
    write_json(OUT / "report.json", report)
    write_report_md(report)
    print(f"report.json and {REPORT_MD.relative_to(REPO)} written")


def write_report_md(report: Mapping[str, Any]) -> None:
    lines = [
        "# osler_v0 — held-out generators (O11; additional, never in a verdict)",
        "",
        "The two locked held-out generators (`loops/osler_v0/heldout.md`): the 600 test items of `note_lab_range_v1` (choice, "
        "300 items) and `policy_triage_v1` (score, 300 items). Held-out results are reported here and never pooled with the seen "
        "generators (heldout.md rule 3). Choice items use both option orders averaged; score items have one order. Intervals: 95 % "
        "percentile bootstrap over the items, resampled within each generator (2,000 resamples, seed 0).",
        "",
        "**Baselines: NOT MEASURED.** The four D24 baselines (MedDecider-4B, zero-shot Qwen3.5-4B, MedDecider-9B, JEV-9B) were not "
        "scored on these items, so the held-out verdicts are not computed (deviation 58).",
        "",
        "| model | template macro accuracy (95 % CI) | note_lab_range_v1 | policy_triage_v1 | Brier (mean) | "
        "majority baseline (template macro) | single-order template macro |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, m in report["models"].items():
        both = m["both_orders"]
        acc = both["accuracy_by_template"]
        lo, hi = both["template_macro_ci_95"]
        lines.append(
            f"| {name} | {both['template_macro_accuracy']:.4f} ({lo:.4f} to {hi:.4f}) | "
            f"{acc['note_lab_range_v1']:.4f} (n {both['n_by_template']['note_lab_range_v1']}) | "
            f"{acc['policy_triage_v1']:.4f} (n {both['n_by_template']['policy_triage_v1']}) | "
            f"{both['brier_mean']:.4f} | {both['majority_template_macro']:.4f} | "
            f"{m['single_order']['template_macro_accuracy']:.4f} |")
    lines += ["", "Paired differences (additional; template macro, items resampled within generator):", "",
              "| comparison | difference | 95 % CI | items |", "|---|---|---|---|"]
    for label, d in report["paired_additional"].items():
        lo, hi = d["ci_95"]
        lines.append(f"| {label} | {d['point']:+.4f} | {lo:+.4f} to {hi:+.4f} | {d['n']} |")
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
