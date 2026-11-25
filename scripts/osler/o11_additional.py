"""O11 additional comparisons (ADVISORY O11, the 'Additional' list; never in a verdict; CPU, reads the calibrated O11 outputs).

Arm-vs-arm at 4B: the paired difference of two-order accuracy (template macro, bootstrapped within templates) and of the mean
Brier, for osler_4b_N and osler_4b_P against osler_4b_L, on the whole O2 common set and on the external panel. The knowledge guard
of each Osler model against the zero-shot Qwen3.5-4B base (MedQA and MedMCQA, micro accuracy, the gate's statistic), so that the
arms can be compared on tier-1 knowledge as well. The knowledge guard of osler_0p8b has no zero-shot 0.8B base on the common set,
so it is NOT MEASURED. Writes outputs/osler_v0/O11/calibrated/additional.json (gitignored); the committed record is the CLAIMS rows.

Usage:
    uv run --frozen python scripts/osler/o11_additional.py
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from meddecide.eval.gate import accuracy_difference, compare_rows
from meddecide.eval.o2_io import Scope, iter_items
from meddecide.utils.io import iter_jsonl, write_json
from meddecide.utils.provenance import git_commit, utcnow

REPO = Path(__file__).resolve().parents[2]
ITEMS = REPO / "data/bench/v0.3_ext/o2_items.jsonl"
CAL = REPO / "outputs/osler_v0/O11/calibrated"
O2 = REPO / "outputs/osler_v0/O2"
OUT = CAL / "additional.json"
SCOPE = Scope("common_set", ("v0.2", "ext_panel", "robustness"))
KNOWLEDGE_SETS = frozenset({"medqa", "medmcqa"})
ARMS = ("osler_4b_N", "osler_4b_P")
REFERENCE_ARM = "osler_4b_L"
ZERO_SHOT_4B = "qwen35-4b"
N_RESAMPLES = 1000
SEED = 0


def knowledge_items(items: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """The knowledge guard's items: v0.2 established items from MedQA and MedMCQA (the gate's knowledge scope)."""
    return [{"item_id": r["item_id"], "template_id": r["template_id"]} for r in items
            if r["benchmark"] == "v0.2" and r["meta"]["tier"] == "established" and r["set_name"] in KNOWLEDGE_SETS]


def by_item(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    return {row["item_id"]: row for row in iter_jsonl(path)}


def arm_difference(scope_name: str, scope_rows: Sequence[Mapping[str, Any]], arm: str,
                   reference: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    comp = compare_rows(scope_name, REFERENCE_ARM, scope_rows, by_item(CAL / arm / "both_orders.jsonl"), reference,
                        n_resamples=N_RESAMPLES, seed=SEED)
    a, b = comp.comparison.accuracy, comp.comparison.brier
    return {"scope": scope_name, "arm": arm, "reference": REFERENCE_ARM, "n_paired": comp.n_paired,
            "accuracy_diff": {"point": a.point, "lo": a.lo, "hi": a.hi},
            "brier_diff": {"point": b.point, "lo": b.lo, "hi": b.hi}}


def guard(arm: str, knowledge: Sequence[Mapping[str, Any]], zero_shot: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    osler = by_item(CAL / arm / "both_orders.jsonl")
    if not osler:
        return {"arm": arm, "not_measured": f"no calibrated O11 outputs for {arm} yet"}
    if not zero_shot:
        return {"arm": arm, "not_measured": "no zero-shot base predictions"}
    diff, n = accuracy_difference(knowledge, osler, zero_shot, n_resamples=N_RESAMPLES, seed=SEED)
    return {"arm": arm, "reference": ZERO_SHOT_4B, "n_paired": n,
            "accuracy_diff": {"point": diff.point, "lo": diff.lo, "hi": diff.hi}}


def main() -> None:
    items = list(iter_items(ITEMS, SCOPE))
    scope_all = [{"item_id": r["item_id"], "template_id": r["template_id"]} for r in items]
    scope_panel = [s for s, r in zip(scope_all, items, strict=True) if r["benchmark"] == "ext_panel"]
    reference = by_item(CAL / REFERENCE_ARM / "both_orders.jsonl")
    knowledge = knowledge_items(items)
    zero_shot = by_item(O2 / ZERO_SHOT_4B / "predictions.jsonl")
    record = {
        "kind": "osler_v0_O11_additional",
        "built_at_utc": utcnow(),
        "git_commit": git_commit(REPO),
        "command": "scripts/osler/o11_additional.py",
        "resamples": N_RESAMPLES,
        "seed": SEED,
        "arm_vs_arm": [arm_difference(scope, rows, arm, reference)
                       for scope, rows in (("common_set", scope_all), ("external_panel", scope_panel))
                       for arm in ARMS],
        # the 0.8B's guard needs its own zero-shot base (zero-shot Qwen3.5-0.8B), which the O2 common set does not have
        "knowledge_guard": [guard(arm, knowledge, zero_shot) for arm in (*ARMS, REFERENCE_ARM)]
        + [guard("osler_0p8b", knowledge, {})],
        "knowledge_items": len(knowledge),
    }
    write_json(OUT, record)
    for row in record["arm_vs_arm"]:
        print(row["scope"], row["arm"], "minus", row["reference"], row["n_paired"],
              round(row["accuracy_diff"]["point"], 4), round(row["accuracy_diff"]["lo"], 4), round(row["accuracy_diff"]["hi"], 4),
              "| brier", round(row["brier_diff"]["point"], 4), round(row["brier_diff"]["lo"], 4), round(row["brier_diff"]["hi"], 4))
    for row in record["knowledge_guard"]:
        print("knowledge guard", row["arm"], row.get("accuracy_diff") or row.get("not_measured"))


if __name__ == "__main__":
    main()
