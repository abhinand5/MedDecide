"""Checkpoint selection by S for the osler_v6 final (ADVISORY §2.7).

The final runs ten evaluations, each with a positional-health check. The checkpoint is chosen by S (the composite mean of the
MR and MIMIC template-macro accuracies and the CG v2 item accuracy) **among the evaluations that pass positional health in
all three orders**; an evaluation that fails positional health is not eligible even if its S is higher. If no evaluation
passes, the final is BLOCKED and reported — no checkpoint is chosen by falling back to an earlier rule.

This module holds the pure rule; `scripts/osler_v6/z6_select.py` reads the final's evaluations, copies (and calibrates) the
chosen checkpoint and writes the record.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Selection:
    """The chosen evaluation step, its S and the eligibility table (every evaluation, in step order)."""

    chosen_step: int | None
    chosen_s: float | None
    outcome: str
    eligible: tuple[int, ...]
    table: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {"chosen_step": self.chosen_step, "chosen_s": self.chosen_s, "outcome": self.outcome,
                "eligible_steps": list(self.eligible), "evaluations": list(self.table)}


def select_by_s(evaluations: Mapping[str, Any]) -> Selection:
    """Apply ADVISORY §2.7 to an `evaluations.json`: highest S among the positional-health passes; none passing = BLOCKED."""
    if not evaluations:
        raise ValueError("no evaluations to select from")
    table = []
    eligible: list[tuple[int, float]] = []
    for key in sorted(evaluations, key=lambda k: int(k)):
        step = int(key)
        record = evaluations[key]
        parts = record["composite"]
        s = float(parts["S"])
        passes = bool(record["positional_passes_all_orders"])
        table.append({"step": step, "S": s, "mr": float(parts["mr"]), "mimic": float(parts["mimic"]),
                      "cg_micro": float(parts["cg_micro"]), "positional_passes_all_orders": passes,
                      "knowledge_dev_accuracy": float(record.get("knowledge_dev_accuracy", float("nan")))})
        if passes:
            eligible.append((step, s))
    if not eligible:
        return Selection(None, None, "BLOCKED — no evaluation passes positional health in all three orders",
                         (), tuple(table))
    best_s = max(s for _, s in eligible)
    best_steps = sorted(step for step, s in eligible if s == best_s)
    if len(best_steps) > 1:
        raise ValueError(f"S is tied between the evaluations {best_steps}: the pre-registration names no tie-break")
    return Selection(best_steps[0], best_s,
                     f"chosen step {best_steps[0]}: the highest S ({best_s:.6f}) among the {len(eligible)} positional-health passes",
                     tuple(step for step, _ in eligible), tuple(table))
