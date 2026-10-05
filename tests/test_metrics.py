"""Hand-computed cases for the metric primitives (T6 will reuse these)."""

from __future__ import annotations

import numpy as np
import pytest

from meddecide.eval.metrics import (
    accuracy,
    bootstrap_ci,
    brier_score,
    ece,
    ece_detail,
    label_mass,
    macro_accuracy,
)


def test_accuracy_micro_and_majority_baseline() -> None:
    gold = ["A", "A", "A", "B"]
    pred = ["A", "B", "A", "B"]
    report = accuracy(pred, gold)
    assert report.n == 4
    assert report.n_correct == 3
    assert report.micro == pytest.approx(0.75)
    assert report.majority_label == "A"
    assert report.majority_baseline == pytest.approx(0.75)


def test_accuracy_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError, match="length mismatch"):
        accuracy(["A"], ["A", "B"])


def test_macro_accuracy_hand_computed() -> None:
    """Class A: 1 of 2 correct -> 0.5. Class B: 1 of 2 correct -> 0.5. Macro = 0.5.

    Corrected during T1: the first draft claimed class B was 2/2, which the arrays below
    contradict (gold B at positions 2 and 3; predictions A and B). The metric was right.
    """
    predicted = ["A", "B", "A", "B"]
    gold = ["A", "A", "B", "B"]
    assert macro_accuracy(predicted, gold) == pytest.approx(0.5)
    assert macro_accuracy(gold, gold) == pytest.approx(1.0)
    # A deliberately unbalanced case: macro ignores class frequency.
    # class A: 3 items, 3 correct -> 1.0 ; class B: 1 item, 0 correct -> 0.0
    assert macro_accuracy(["A", "A", "A", "A"], ["A", "A", "A", "B"]) == pytest.approx(0.5)
    # ... while micro on the same case is 3/4
    assert accuracy(["A", "A", "A", "A"], ["A", "A", "A", "B"]).micro == pytest.approx(0.75)


def test_brier_score_perfect_and_uniform() -> None:
    perfect = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert brier_score(perfect, [0, 1]) == pytest.approx(0.0)
    uniform = np.array([[0.5, 0.5], [0.5, 0.5]])
    # each row contributes (0.5-1)^2 + (0.5-0)^2 = 0.5
    assert brier_score(uniform, [0, 1]) == pytest.approx(0.5)


def test_brier_score_rejects_unnormalised_rows() -> None:
    with pytest.raises(ValueError, match="sum to 1"):
        brier_score(np.array([[0.7, 0.7]]), [0])


def test_ece_bins_are_right_closed_and_weighted() -> None:
    # 4 items: two at confidence 0.9 (one correct), two at confidence 0.5 (both correct)
    probs = np.array(
        [
            [0.9, 0.1],
            [0.9, 0.1],
            [0.5, 0.5],
            [0.5, 0.5],
        ]
    )
    gold = [0, 1, 0, 0]
    detail = ece_detail(probs, gold, n_bins=15)
    # bin 0.9: n=2, acc=0.5 -> gap 0.4 ; bin 0.5: n=2, acc=1.0 -> gap 0.5
    assert detail["ece"] == pytest.approx(0.5 * 0.4 + 0.5 * 0.5)
    assert sum(b["n"] for b in detail["bins"]) == 4
    bin_90 = next(b for b in detail["bins"] if b["n"] == 2 and b["confidence"] == pytest.approx(0.9))
    assert bin_90["accuracy"] == pytest.approx(0.5)
    assert bin_90["gap"] == pytest.approx(0.4)


def test_ece_perfect_calibration_is_zero_only_when_confidence_equals_accuracy() -> None:
    probs = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert ece(probs, [0, 1]) == pytest.approx(0.0)
    overconfident = np.array([[1.0, 0.0], [1.0, 0.0]])
    assert ece(overconfident, [0, 1]) == pytest.approx(0.5)


def test_bootstrap_ci_is_deterministic_and_brackets_the_point() -> None:
    values = [0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0]
    point, lo, hi = bootstrap_ci(values, seed=0, n_resamples=500)
    assert point == pytest.approx(np.mean(values))
    assert lo <= point <= hi
    assert (point, lo, hi) == bootstrap_ci(values, seed=0, n_resamples=500)


def test_label_mass_reports_token_mass_and_top_probability() -> None:
    probs = np.array([[0.6, 0.4]])
    masses = np.array([[0.06, 0.04]])
    out = label_mass(probs, masses)
    assert out["mean_option_token_mass"] == pytest.approx(0.1)
    assert out["mean_top_prob"] == pytest.approx(0.6)
