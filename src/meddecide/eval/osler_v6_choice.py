"""The osler_v6 choice rule (ADVISORY §2.6, pre-registered before any arm ran).

A G1 recipe is a qualifier when, at 13,992 steps:

* it passes positional health in all three orders (the original option order and two seeded permutations);
* the lower bound of the paired (G1 - G0) knowledge-dev-v2 item-micro difference is above -0.01;
* the lower bounds of the paired MR and MIMIC template-macro differences against G0 are above -0.03;
* its CG v2 micro point estimate is at least G0's.

G0 (the F2 seed-0 reference) is not a candidate: its bounds against itself are zero, so it never qualifies. The choice is
the qualifier with the highest knowledge-dev-v2 accuracy; if none qualifies, the loop's outcome is (4) — report and stop,
no final is trained. Seed 0 is G1's candidate; a seed-1 run is a stability check that is reported beside it and never
replaces it, so the caller passes only seed-0 arms here.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

KNOWLEDGE_LOWER_LIMIT = -0.01
MR_LOWER_LIMIT = -0.03
MIMIC_LOWER_LIMIT = -0.03


@dataclass(frozen=True)
class ArmChoice:
    """One G1 arm's choice-rule inputs at 13,992 steps (all measured on dev only)."""

    name: str
    knowledge_dev: float
    knowledge_lower: float
    mr_lower: float
    mimic_lower: float
    cg_micro: float
    positional_all_orders: bool


@dataclass(frozen=True)
class ControlChoice:
    """The G0 reference's CG v2 micro accuracy (and its knowledge dev accuracy, for the ranking table)."""

    name: str = "G0"
    cg_micro: float = 0.0
    knowledge_dev: float = 0.0


@dataclass
class ChoiceReport:
    """The rule's verdict: per-arm failures, the qualifying set and the chosen arm (None = outcome 4)."""

    chosen: str | None
    outcome: str
    rows: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"chosen": self.chosen, "outcome": self.outcome, "rows": self.rows}


def failed_checks(arm: ArmChoice, control: ControlChoice) -> list[str]:
    """The pre-registered checks the arm fails (an empty list means it qualifies)."""
    failed = []
    if not arm.positional_all_orders:
        failed.append("positional health (all three orders)")
    if not arm.knowledge_lower > KNOWLEDGE_LOWER_LIMIT:
        failed.append(f"knowledge lower bound {arm.knowledge_lower:+.4f} <= {KNOWLEDGE_LOWER_LIMIT:+.2f}")
    if not arm.mr_lower > MR_LOWER_LIMIT:
        failed.append(f"MR lower bound {arm.mr_lower:+.4f} <= {MR_LOWER_LIMIT:+.2f}")
    if not arm.mimic_lower > MIMIC_LOWER_LIMIT:
        failed.append(f"MIMIC lower bound {arm.mimic_lower:+.4f} <= {MIMIC_LOWER_LIMIT:+.2f}")
    if not arm.cg_micro >= control.cg_micro:
        failed.append(f"CG v2 micro {arm.cg_micro:.4f} < control {control.cg_micro:.4f}")
    return failed


def choose_recipe(arms: Sequence[ArmChoice], control: ControlChoice) -> ChoiceReport:
    """Apply the rule to the seed-0 candidates; the highest knowledge-dev-v2 qualifier wins, else outcome (4)."""
    if not arms:
        raise ValueError("choose_recipe needs at least one G1 arm")
    rows: dict[str, dict[str, Any]] = {}
    qualifying: list[ArmChoice] = []
    for arm in arms:
        failed = failed_checks(arm, control)
        rows[arm.name] = {
            "knowledge_dev": arm.knowledge_dev,
            "knowledge_lower": arm.knowledge_lower,
            "mr_lower": arm.mr_lower,
            "mimic_lower": arm.mimic_lower,
            "cg_micro": arm.cg_micro,
            "positional_all_orders": arm.positional_all_orders,
            "failed": failed,
            "qualifies": not failed,
        }
        if not failed:
            qualifying.append(arm)
    if not qualifying:
        return ChoiceReport(chosen=None, outcome="(4) no arm qualified: report and stop; no final is trained", rows=rows)
    best = max(qualifying, key=lambda arm: arm.knowledge_dev)
    tied = sorted(arm.name for arm in qualifying if arm.knowledge_dev == best.knowledge_dev)
    if len(tied) > 1:
        raise ValueError(f"the knowledge-dev-v2 maximum is tied between {tied}: the rule names no tie-break")
    return ChoiceReport(chosen=best.name, outcome=f"chosen {best.name}: the highest knowledge dev v2 among the qualifiers",
                        rows=rows)
