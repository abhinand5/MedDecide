"""osler_v3 Stage A choice (ADVISORY X7; the rule was fixed before any Stage A run and is not changed here).

Rule (paired bootstrap within templates, 1,000 resamples, seed 0; blocks.macro_difference):
- a variant QUALIFIES if its MR partition minus A0's MR partition has a lower bound above -0.01 (A0 always qualifies);
- among the qualifying runs, the CHOICE is the one with the highest point estimate of (its CG accuracy minus zero-shot
  pplx-decider-flash-v1.2's CG accuracy): the closest to its base on the clinical-general proxy;
- a difference under 0.002 from A0 is a tie, and A0 wins it.
A run whose predictions are missing is NOT MEASURED and cannot qualify. A0 is the control (kl_to_base 0).

The MR partition is the fast dev v3 items of the MR sources (blocks.MR_SOURCES); knowledge is medqa and medmcqa; general is the rest.
CG accuracy is the item accuracy over the paired CG items (items the zero-shot pplx reference cannot score are excluded on both sides,
and counted). Writes outputs/osler_v3/X7/choice.json and loops/osler_v3/stage_a.md (aggregates only).

Usage:  source scripts/pod_env.sh && uv run --frozen python scripts/osler_v3/x7_choice.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from meddecide.eval.o2_metrics import score_row
from meddecide.train.blocks import (
    KNOWLEDGE_SOURCES,
    MR_SOURCES,
    macro_difference,
    micro_difference,
    pair_by_item,
)
from meddecide.train.osler_v3 import STAGE_A_TIE, choose_recipe
from meddecide.utils.io import write_json
from meddecide.utils.provenance import git_commit, utcnow

REPO = Path(__file__).resolve().parents[2]
STAGE_A = REPO / "outputs" / "osler_v3" / "stage_a"
ZERO_SHOT_CG = REPO / "outputs" / "osler_v3" / "X6" / "zeroshot_pplx_cg" / "predictions.jsonl"
OUT = REPO / "outputs" / "osler_v3" / "X7" / "choice.json"
DOC = REPO / "loops" / "osler_v3" / "stage_a.md"
ARMS = ("A0", "A1", "A2", "A3")
MR_MARGIN = -0.01


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def partition(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {"mr": [], "knowledge": [], "general": []}
    for r in rows:
        if r["source"] in MR_SOURCES:
            out["mr"].append(r)
            continue
        out["general"].append(r)
        if r["source"] in KNOWLEDGE_SOURCES:
            out["knowledge"].append(r)
    return out


def zero_shot_cg_correct() -> dict[str, bool]:
    """Item id -> correct for the zero-shot pplx scored items (its 'skipped' items are not in the map)."""
    if not ZERO_SHOT_CG.exists():
        return {}
    out: dict[str, bool] = {}
    for row in read_jsonl(ZERO_SHOT_CG):
        if row.get("status") == "scored":
            out[row["item_id"]] = bool(score_row(row).correct)
    return out


def main() -> int:
    arms: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        pred = STAGE_A / arm / "dev_predictions.jsonl"
        cg = STAGE_A / arm / "cg_result.json"
        if not (pred.exists() and cg.exists()):
            arms[arm] = {"status": "NOT MEASURED — no finished run and CG score"}
            continue
        arms[arm] = {"status": "measured", "rows": read_jsonl(pred), "cg": json.loads(cg.read_text(encoding="utf-8"))}
    if arms["A0"].get("status") != "measured":
        raise SystemExit("A0 (the control) has no finished run: the choice cannot be made")
    a0_rows = arms["A0"]["rows"]
    a0_parts = partition(a0_rows)
    zs = zero_shot_cg_correct()
    report: dict[str, Any] = {"built_utc": utcnow(), "git_commit": git_commit(REPO), "rule": {
        "mr_margin": MR_MARGIN, "tie": STAGE_A_TIE, "resamples": 1000, "seed": 0, "cg_metric": "item accuracy on paired CG items"},
        "arms": {}}
    for arm in ARMS:
        info = arms[arm]
        if info["status"] != "measured":
            report["arms"][arm] = info
            continue
        parts = partition(info["rows"])
        entry: dict[str, Any] = {"status": "measured", "kl_to_base_cg": info["cg"].get("mean_kl_to_base_cg")}
        if arm == "A0":
            entry["mr_macro_vs_A0"] = entry["knowledge_micro_vs_A0"] = entry["general_micro_vs_A0"] = None
        else:
            entry["mr_macro_vs_A0"] = macro_difference(pair_by_item(parts["mr"], a0_parts["mr"])).as_dict()
            entry["knowledge_micro_vs_A0"] = micro_difference(pair_by_item(parts["knowledge"], a0_parts["knowledge"])).as_dict()
            entry["general_micro_vs_A0"] = micro_difference(pair_by_item(parts["general"], a0_parts["general"])).as_dict()
        entry["mr_accuracy"] = sum(bool(r["correct"]) for r in parts["mr"]) / max(1, len(parts["mr"]))
        entry["knowledge_accuracy"] = sum(bool(r["correct"]) for r in parts["knowledge"]) / max(1, len(parts["knowledge"]))
        entry["general_accuracy"] = sum(bool(r["correct"]) for r in parts["general"]) / max(1, len(parts["general"]))
        cg_rows = read_jsonl(STAGE_A / arm / "cg_predictions.jsonl")
        paired = [r for r in cg_rows if r["item_id"] in zs]
        arm_correct = {r["item_id"]: bool(r["correct"]) for r in cg_rows}
        entry["cg_items_paired_with_zero_shot_pplx"] = len(paired)
        entry["cg_items_unpaired"] = len(cg_rows) - len(paired)
        if paired:
            entry["cg_accuracy_arm"] = sum(arm_correct[r["item_id"]] for r in paired) / len(paired)
            entry["cg_accuracy_zero_shot_pplx"] = sum(zs[r["item_id"]] for r in paired) / len(paired)
            entry["cg_score"] = entry["cg_accuracy_arm"] - entry["cg_accuracy_zero_shot_pplx"]
        else:
            entry["cg_score"] = None
        report["arms"][arm] = entry
    qualifying = ["A0"]
    for arm in ("A1", "A2", "A3"):
        entry = report["arms"].get(arm, {})
        if entry.get("status") != "measured" or entry.get("mr_macro_vs_A0") is None:
            continue
        if entry["mr_macro_vs_A0"]["ci_lo"] > MR_MARGIN:
            qualifying.append(arm)
        entry["qualifies"] = entry["mr_macro_vs_A0"]["ci_lo"] > MR_MARGIN
    report["a0_cg_score"] = report["arms"]["A0"].get("cg_score")
    scores = {a: report["arms"][a].get("cg_score") for a in ARMS}
    best, tie = choose_recipe(scores, qualifying)
    report["tie_rule_applied"] = tie
    report["qualifying"] = qualifying
    report["chosen"] = best
    write_json(OUT, report)
    lines = ["# osler_v3 — Stage A: the recipe choice (ADVISORY X7; aggregates only)", "",
             f"Written {utcnow()} by `scripts/osler_v3/x7_choice.py`; the rule is in its docstring and was fixed before any Stage A run.",
             "", f"**Chosen recipe: {best}** (qualifying: {', '.join(qualifying)}; tie rule applied: {report['tie_rule_applied']}).", "",
             "| arm | KL to base (CG) | MR macro vs A0 (point, 95 % CI) | qualifies | CG acc | CG - zero-shot pplx | knowledge acc | general acc |",
             "|---|---|---|---|---|---|---|---|"]
    for arm in ARMS:
        e = report["arms"][arm]
        if e.get("status") != "measured":
            lines.append(f"| {arm} | NOT MEASURED | NOT MEASURED | — | — | — | — | — |")
            continue
        mr = e.get("mr_macro_vs_A0")
        mr_txt = "control" if mr is None else f"{mr['point']:+.4f} [{mr['ci_lo']:+.4f}, {mr['ci_hi']:+.4f}]"
        lines.append(f"| {arm} | {e['kl_to_base_cg']:.4f} | {mr_txt} | {'control' if arm == 'A0' else e.get('qualifies')} | "
                     f"{e.get('cg_accuracy_arm', float('nan')):.4f} | {e['cg_score'] if e['cg_score'] is None else round(e['cg_score'], 4)} | "
                     f"{e['knowledge_accuracy']:.4f} | {e['general_accuracy']:.4f} |")
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"chosen {best}; qualifying {qualifying}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
