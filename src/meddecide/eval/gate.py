"""Gate O1 decision rule (PROGRAM D24), as the pre-registered verdict logic over paired differences.

Each comparison is Osler minus one baseline, with an item-paired bootstrap interval (1,000 resamples, items within datasets
or templates). A comparison passes when the macro-accuracy interval's lower bound is above 0 and the mean-Brier interval's
upper bound is below 0 (lower Brier is better). The gate passes only when every required comparison passes, on the fresh
seen-template set and on the external clinical panel, against both baselines. A knowledge guard also applies: on MedQA plus
MedMCQA, the accuracy interval's lower bound of Osler minus the zero-shot base of the same size must be above -0.02.

The thresholds and the set of comparisons are fixed by D24 before any result. This module implements them and changes
nothing about them.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from meddecide.eval.paired import Difference

KNOWLEDGE_GUARD_FLOOR = -0.02


@dataclass(frozen=True)
class Comparison:
    """Osler minus one baseline on one scope (a set of items), with both paired intervals."""

    scope: str
    baseline: str
    accuracy: Difference  # macro accuracy, Osler minus baseline
    brier: Difference  # mean Brier, Osler minus baseline (lower is better)

    @property
    def passes(self) -> bool:
        return self.accuracy.lo > 0 and self.brier.hi < 0


def gate_verdict(comparisons: Sequence[Comparison]) -> bool:
    """PASS only when at least one comparison is required and every one passes."""
    return len(comparisons) > 0 and all(c.passes for c in comparisons)


def knowledge_guard(difference: Difference) -> bool:
    """The guard passes when the accuracy interval's lower bound (Osler minus zero-shot base) is above -0.02."""
    return difference.lo > KNOWLEDGE_GUARD_FLOOR
