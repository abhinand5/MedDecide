"""The O9 head-choice rule and the Gate O1 decision logic (PROGRAM D24; ADVISORY O9 and O11), on synthetic scores."""

from __future__ import annotations

import numpy as np
import pytest

from meddecide.eval.gate import Comparison, gate_verdict, knowledge_guard
from meddecide.eval.head_choice import DevScores, challenge, choose_head
from meddecide.eval.paired import Difference


def _diff(point: float, lo: float, hi: float, n: int = 100) -> Difference:
    return Difference(n=n, point=point, lo=lo, hi=hi)


def _scores(name: str, p_correct: float, n: int = 400, seed: int = 0) -> DevScores:
    rng = np.random.default_rng(seed)
    gold = rng.choice(np.array(["A", "B", "C", "D"]), size=n)
    templates = np.array([f"t{i % 8}" for i in range(n)])
    correct = (rng.random(n) < p_correct).astype(float)
    return DevScores(name=name, correct=correct, gold=gold, templates=templates)


# --- Gate O1 (D24) -----------------------------------------------------------------------------------------------------

def test_a_comparison_passes_only_with_both_bounds_on_the_right_side() -> None:
    good = Comparison("fresh seen templates", "MedDecider-4B", accuracy=_diff(0.03, 0.01, 0.05),
                      brier=_diff(-0.02, -0.04, -0.005))
    assert good.passes
    accuracy_lower_at_zero = Comparison("panel", "zero-shot", accuracy=_diff(0.01, 0.0, 0.02),
                                        brier=_diff(-0.02, -0.04, -0.005))
    assert not accuracy_lower_at_zero.passes  # lower bound must be strictly above 0
    brier_upper_at_zero = Comparison("panel", "zero-shot", accuracy=_diff(0.03, 0.01, 0.05),
                                     brier=_diff(-0.02, -0.04, 0.0))
    assert not brier_upper_at_zero.passes  # upper bound must be strictly below 0


def test_the_gate_needs_every_comparison_to_pass() -> None:
    passing = Comparison("fresh seen templates", "MedDecider-4B", accuracy=_diff(0.03, 0.01, 0.05),
                         brier=_diff(-0.02, -0.04, -0.005))
    failing = Comparison("external panel", "zero-shot", accuracy=_diff(-0.01, -0.03, 0.01),
                         brier=_diff(0.01, -0.01, 0.03))
    assert gate_verdict([passing, passing])
    assert not gate_verdict([passing, failing])


def test_an_empty_comparison_list_cannot_pass_the_gate() -> None:
    assert not gate_verdict([])


def test_knowledge_guard_threshold_is_minus_two_points() -> None:
    assert knowledge_guard(_diff(0.0, -0.0199, 0.02))
    assert not knowledge_guard(_diff(0.0, -0.02, 0.02))  # exactly -0.02 fails: strictly above


# --- O9 head choice ----------------------------------------------------------------------------------------------------

def test_identical_arms_do_not_qualify() -> None:
    base = _scores("L", 0.6, seed=1)
    same = DevScores(name="P", correct=base.correct.copy(), gold=base.gold, templates=base.templates)
    result = challenge(same, base, n_resamples=200, seed=2)
    assert result.difference.point == 0.0
    assert not result.qualifies


def test_a_clearly_better_challenger_is_chosen() -> None:
    base = _scores("L", 0.5, seed=3)
    better = _scores("P", 0.9, seed=4)
    better = DevScores(name="P", correct=better.correct, gold=base.gold, templates=base.templates)
    choice = choose_head(base, [better], n_resamples=300, seed=5)
    assert choice.chosen == "P"
    assert choice.challenges[0].qualifies


def test_no_challenger_qualifies_keeps_l() -> None:
    base = _scores("L", 0.6, seed=6)
    same_like = DevScores(name="N", correct=base.correct.copy(), gold=base.gold, templates=base.templates)
    choice = choose_head(base, [same_like], n_resamples=100, seed=7)
    assert choice.chosen == "L"
    assert "keep L" in choice.reason


def test_the_baseline_must_be_arm_l_and_items_must_align() -> None:
    base = _scores("P", 0.6, seed=8)
    with pytest.raises(ValueError, match="baseline must be arm L"):
        choose_head(base, [base], n_resamples=10)
    arm = _scores("N", 0.6, seed=9)
    other = _scores("L", 0.6, seed=10)  # different gold labels
    with pytest.raises(ValueError, match="same items"):
        challenge(arm, other, n_resamples=10)


def test_head_choice_uses_the_template_macro() -> None:
    # two templates of unequal size: the template macro weights them equally, the item mean does not
    templates = np.array(["big"] * 300 + ["small"] * 20)
    gold = np.array(["A"] * 320)
    base = DevScores(name="L", correct=np.r_[np.ones(300), np.zeros(20)], gold=gold, templates=templates)
    arm = DevScores(name="P", correct=np.r_[np.ones(300), np.ones(20)], gold=gold, templates=templates)
    result = challenge(arm, base, n_resamples=200, seed=0)
    assert result.difference.point == pytest.approx(0.5)  # (0 + 1) / 2 templates, not 20 / 320 items
