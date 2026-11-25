"""osler_v5 Z2 choice rule (ADVISORY Z2; pre-registered before any arm runs). Pure functions over per-arm statistics at step 13,992.

An arm qualifies if (a) its positional health passes in all three orders (the original and the two seeded permutations, D52b), (b) its
knowledge-dev paired lower bound against R0 is above -0.01, and (c) its MR and MIMIC paired lower bounds against R0 are above -0.03.
The control R0 is its own reference, so its differences are zero and the bounds do not apply to it.

Among the qualifying arms, the highest CG v2 micro is chosen. Arms within max(0.005, |R0 - R0'|) of that value are tied, and the
fuller fix (the higher-numbered arm in ``order``) wins the tie. If no arm qualifies, the outcome is "no arm qualifies" and nothing is
chosen (ADVISORY Z2, outcome 4).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

KNOWLEDGE_MARGIN = -0.01
PARTITION_MARGIN = -0.03
TIE_FLOOR = 0.005


@dataclass(frozen=True)
class ArmStats:
    """One arm's values at step 13,992. The lower bounds are against the control R0 (None for the control itself)."""

    name: str
    cg_micro: float
    positional_passes: bool
    knowledge_lower: float | None
    mr_lower: float | None
    mimic_lower: float | None

    def qualifies(self) -> bool:
        return (
            self.positional_passes
            and (self.knowledge_lower is None or self.knowledge_lower > KNOWLEDGE_MARGIN)
            and (self.mr_lower is None or self.mr_lower > PARTITION_MARGIN)
            and (self.mimic_lower is None or self.mimic_lower > PARTITION_MARGIN)
        )


def choose(arms: Sequence[ArmStats], *, control: str, repeat: str, order: Sequence[str]) -> dict[str, Any]:
    """The choice under the pre-registered rule, with every quantity it used. ``repeat`` is the control's seed repeat (R0')."""
    by_name = {arm.name: arm for arm in arms}
    if control not in by_name or repeat not in by_name:
        raise ValueError("the control and its seed repeat must be among the arms")
    if len({arm.name for arm in arms}) != len(arms):
        raise ValueError("an arm is named twice")
    unknown = [arm.name for arm in arms if arm.name not in order]
    if unknown:
        raise ValueError(f"arms missing from the tie order: {unknown}")
    qualifying = [arm for arm in arms if arm.qualifies()]
    band = max(TIE_FLOOR, abs(by_name[control].cg_micro - by_name[repeat].cg_micro))
    report: dict[str, Any] = {
        "rule": {"knowledge_lower_above": KNOWLEDGE_MARGIN, "partition_lower_above": PARTITION_MARGIN, "tie_band": band,
                 "tie_order": list(order)},
        "arms": {arm.name: {"qualifies": arm.qualifies(), "cg_micro": arm.cg_micro, "positional_passes": arm.positional_passes,
                            "knowledge_lower": arm.knowledge_lower, "mr_lower": arm.mr_lower, "mimic_lower": arm.mimic_lower}
                 for arm in arms},
    }
    if not qualifying:
        report.update({"outcome": "no arm qualifies (ADVISORY Z2 outcome 4): nothing is chosen", "chosen": None, "tied": []})
        return report
    best = max(arm.cg_micro for arm in qualifying)
    tied = [arm for arm in qualifying if best - arm.cg_micro <= band]
    chosen = max(tied, key=lambda arm: list(order).index(arm.name))
    report.update({"outcome": "chosen", "chosen": chosen.name, "best_cg_micro": best,
                   "tied": [arm.name for arm in tied]})
    return report
