"""osler_v5 Z3 checkpoint selection (ADVISORY Z3): among the evaluations that pass positional health in all three orders, the one with the
highest composite S. An exact tie goes to the later step. If no evaluation passes, the outcome is "no evaluation passes" and nothing is
selected (ADVISORY Z3, outcome 4). Pure function over the per-evaluation records written by scripts/osler_v5/run_arm.py."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def select_checkpoint(records: Mapping[int, Mapping[str, Any]]) -> dict[str, Any]:
    """The selection with the evaluated S of every evaluation, so the choice can be read from the report."""
    evaluated = {int(step): {"S": float(r["composite"]["S"]), "passes": bool(r["positional_passes_all_orders"])}
                 for step, r in records.items()}
    eligible = {step: v for step, v in evaluated.items() if v["passes"]}
    report: dict[str, Any] = {"evaluations": {str(s): v for s, v in sorted(evaluated.items())},
                              "eligible_steps": sorted(eligible)}
    if not eligible:
        report.update({"outcome": "no evaluation passes positional health (ADVISORY Z3, outcome 4): nothing is selected",
                       "selected": None})
        return report
    best = max(eligible, key=lambda step: (eligible[step]["S"], step))
    report.update({"outcome": "selected", "selected": best, "selected_S": eligible[best]["S"]})
    return report
