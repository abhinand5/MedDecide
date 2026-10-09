"""The pre-registered head-choice rule at 4B (ADVISORY O9): dev only, never test and never held-out.

Start with arm L (the option-code head). Switch to a challenger (P or N) only if its dev macro accuracy exceeds L's by at
least 1.0 point, with a paired bootstrap lower bound above 0 on dev. If both qualify, take the larger gain. The bootstrap is
the one D24 uses for Gate O1 (1,000 resamples, items resampled within templates), so the two rules share one statistic.

Pure functions: arrays in, a decision out. The script that reads the arms' dev predictions and writes the report lives in
scripts/osler/o9_head_choice.py.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from meddecide.eval.paired import Difference, paired_macro_accuracy_difference

MARGIN = 0.010  # one point of macro accuracy, as the ADVISORY states it
N_RESAMPLES = 1000  # the D24 bootstrap size
BASELINE = "L"


@dataclass(frozen=True)
class DevScores:
    """One arm's per-item dev scores, aligned to the same items as every other arm."""

    name: str
    correct: np.ndarray  # 1 where the argmax is the gold option, else 0
    gold: np.ndarray  # gold labels (the classes of the macro)
    templates: np.ndarray  # stratum for the bootstrap


@dataclass(frozen=True)
class Challenge:
    name: str
    difference: Difference  # macro accuracy, challenger minus L, with its paired bootstrap interval
    qualifies: bool


@dataclass(frozen=True)
class HeadChoice:
    chosen: str
    challenges: tuple[Challenge, ...]
    reason: str


def challenge(arm: DevScores, baseline: DevScores, *, n_resamples: int = N_RESAMPLES, seed: int = 0) -> Challenge:
    """Compare one challenger with L on dev (macro accuracy, paired, resampled within templates)."""
    if arm.gold.shape != baseline.gold.shape or not np.array_equal(arm.gold, baseline.gold):
        raise ValueError(f"{arm.name} and {baseline.name} must be scored on the same items in the same order")
    difference = paired_macro_accuracy_difference(arm.correct, baseline.correct, arm.gold, arm.templates,
                                                  n_resamples=n_resamples, seed=seed)
    qualifies = difference.point >= MARGIN and difference.lo > 0
    return Challenge(name=arm.name, difference=difference, qualifies=qualifies)


def choose_head(baseline: DevScores, challengers: Sequence[DevScores], *, n_resamples: int = N_RESAMPLES,
                seed: int = 0) -> HeadChoice:
    """Apply the pre-registered rule. ``baseline`` is arm L; ``challengers`` are the arms P and N."""
    if baseline.name != BASELINE:
        raise ValueError(f"the baseline must be arm {BASELINE}, got {baseline.name}")
    if not challengers:
        raise ValueError("choose_head needs at least one challenger")
    challenges = tuple(challenge(arm, baseline, n_resamples=n_resamples, seed=seed) for arm in challengers)
    qualifying = [c for c in challenges if c.qualifies]
    if not qualifying:
        chosen = BASELINE
        reason = (f"no challenger exceeds {BASELINE} by at least {MARGIN:.3f} with a paired lower bound above 0 on dev; "
                  f"keep {BASELINE}")
    else:
        best = max(qualifying, key=lambda c: c.difference.point)
        chosen = best.name
        reason = (f"{best.name} exceeds {BASELINE} by {best.difference.point:.4f} with lower bound "
                  f"{best.difference.lo:.4f} on dev")
        if len(qualifying) > 1:
            reason += "; both challengers qualify, so the larger gain decides"
    return HeadChoice(chosen=chosen, challenges=challenges, reason=reason)
