"""osler_v5 Z2 choice rule for the corrected arms (ADVISORY §8.4, D56c; pre-registered). Pure functions over each corrected arm's final
evaluation (13,992 steps). Where they differ, this supersedes ADVISORY §4 Z2 for F0, F2 and F2s1. The R-arm rule in ``arm_choice`` is the
superseded design and is kept as written.

An arm qualifies if it passes positional health in all three orders, its knowledge-dev lower bound against F0 is above -0.01, and its MR and
MIMIC lower bounds against F0 are above -0.03 (paired bootstrap, 1,000 resamples, seed 0). The control's own bounds are zero. The choice is
the qualifier with the highest CG v2 micro. A qualifier within 0.010 of that top value that is the fuller fix wins the tie; two arms of the
same fix level in one tie are refused, because the rule does not order them. If no arm qualifies, nothing is chosen (ADVISORY §4 Z2, outcome 4).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

KNOWLEDGE_LOWER_LIMIT = -0.01  # D56c: the knowledge-dev lower bound against F0 must be above this
MR_MIMIC_LOWER_LIMIT = -0.03  # D56c: the MR and MIMIC lower bounds against F0 must be above this
TIE_BAND = 0.010  # D56c: qualifiers within this of the top CG v2 micro are tied


@dataclass(frozen=True)
class ArmResult:
    name: str
    fix_level: int  # 0 for osler_v4's recipe (the control), 1 for per-item weighting with augmentation; the tie's "fuller fix"
    positional_all_orders: bool
    knowledge_lower: float  # lower bound of the paired knowledge-dev micro difference against F0
    mr_lower: float  # lower bound of the paired MR template-macro difference against F0
    mimic_lower: float  # lower bound of the paired MIMIC template-macro difference against F0
    cg_micro: float


def failed_checks(arm: ArmResult) -> list[str]:
    """The D56c checks this arm fails, by name; an empty list means it qualifies."""
    failed = []
    if not arm.positional_all_orders:
        failed.append("positional health in all orders")
    if not arm.knowledge_lower > KNOWLEDGE_LOWER_LIMIT:
        failed.append(f"knowledge lower bound {arm.knowledge_lower:+.4f} is not above {KNOWLEDGE_LOWER_LIMIT:+.2f}")
    if not arm.mr_lower > MR_MIMIC_LOWER_LIMIT:
        failed.append(f"MR lower bound {arm.mr_lower:+.4f} is not above {MR_MIMIC_LOWER_LIMIT:+.2f}")
    if not arm.mimic_lower > MR_MIMIC_LOWER_LIMIT:
        failed.append(f"MIMIC lower bound {arm.mimic_lower:+.4f} is not above {MR_MIMIC_LOWER_LIMIT:+.2f}")
    return failed


def choose_arm(arms: Sequence[ArmResult]) -> dict[str, Any]:
    """The D56c choice over ``arms``; ``chosen`` is None when no arm qualifies."""
    names = [arm.name for arm in arms]
    if len(set(names)) != len(names):
        raise ValueError("an arm appears twice")
    rows: dict[str, Any] = {}
    for arm in arms:
        failed = failed_checks(arm)
        rows[arm.name] = {"qualifies": not failed, "failed": failed, "fix_level": arm.fix_level,
                          "positional_all_orders": arm.positional_all_orders, "knowledge_lower": arm.knowledge_lower,
                          "mr_lower": arm.mr_lower, "mimic_lower": arm.mimic_lower, "cg_micro": arm.cg_micro}
    qualifiers = [arm for arm in arms if not failed_checks(arm)]
    if not qualifiers:
        return {"qualifiers": [], "chosen": None, "tied": [], "rows": rows,
                "outcome": "(4): no arm qualifies; choose nothing and report (ADVISORY §4 Z2)"}
    top = max(arm.cg_micro for arm in qualifiers)
    tied = [arm for arm in qualifiers if top - arm.cg_micro <= TIE_BAND]
    fullest = max(arm.fix_level for arm in tied)
    winners = [arm for arm in tied if arm.fix_level == fullest]
    if len(winners) > 1:
        raise ValueError(f"a tie within {TIE_BAND} of the top CG v2 micro between arms of one fix level "
                         f"({[arm.name for arm in winners]}); the rule does not order them, so the operator decides")
    return {"qualifiers": [arm.name for arm in qualifiers], "chosen": winners[0].name,
            "tied": [arm.name for arm in tied], "rows": rows,
            "outcome": "chosen: the highest CG v2 micro among the qualifiers, with the tie band applied (ADVISORY §8.4)"}
