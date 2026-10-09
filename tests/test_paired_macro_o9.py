"""Paired macro-accuracy and stratified mean differences (ADVISORY O9 and O11; PROGRAM D24), on synthetic items."""

from __future__ import annotations

import numpy as np
import pytest

from meddecide.eval.metrics import macro_accuracy
from meddecide.eval.paired import (
    paired_macro_accuracy_difference,
    paired_mean_difference_stratified,
)


def _synthetic(n: int = 240, seed: int = 0) -> tuple[np.ndarray, list[str], list[str]]:
    rng = np.random.default_rng(seed)
    gold = [str(g) for g in rng.choice(["A", "B", "C", "yes", "no"], size=n, p=[0.3, 0.2, 0.2, 0.2, 0.1])]
    strata = [f"t{i % 4}" for i in range(n)]
    return rng, gold, strata


def test_point_estimate_equals_the_metric_difference() -> None:
    rng, gold, strata = _synthetic()
    correct_a = rng.random(len(gold)) < 0.7
    correct_b = rng.random(len(gold)) < 0.5
    pred_a = [g if c else "X" for g, c in zip(gold, correct_a, strict=True)]
    pred_b = [g if c else "X" for g, c in zip(gold, correct_b, strict=True)]
    expected = macro_accuracy(pred_a, gold) - macro_accuracy(pred_b, gold)
    result = paired_macro_accuracy_difference(correct_a.astype(float), correct_b.astype(float), gold, strata,
                                              n_resamples=50, seed=1)
    assert result.point == pytest.approx(expected)
    assert result.n == len(gold)


def test_identical_scorings_have_a_zero_interval() -> None:
    _, gold, strata = _synthetic()
    correct = np.arange(len(gold)) % 3 == 0
    result = paired_macro_accuracy_difference(correct.astype(float), correct.astype(float), gold, strata,
                                              n_resamples=200, seed=2)
    assert result.point == 0.0
    assert result.lo == 0.0 and result.hi == 0.0


def test_a_clearly_better_scoring_has_a_positive_lower_bound() -> None:
    rng, gold, strata = _synthetic(n=800, seed=3)
    correct_a = (rng.random(len(gold)) < 0.9).astype(float)
    correct_b = (rng.random(len(gold)) < 0.5).astype(float)
    result = paired_macro_accuracy_difference(correct_a, correct_b, gold, strata, n_resamples=300, seed=4)
    assert result.point > 0
    assert result.lo > 0
    assert result.lo <= result.point <= result.hi


def test_the_bootstrap_is_seeded() -> None:
    rng, gold, strata = _synthetic(n=300, seed=5)
    a = (rng.random(len(gold)) < 0.6).astype(float)
    b = (rng.random(len(gold)) < 0.55).astype(float)
    first = paired_macro_accuracy_difference(a, b, gold, strata, n_resamples=100, seed=7)
    second = paired_macro_accuracy_difference(a, b, gold, strata, n_resamples=100, seed=7)
    assert (first.lo, first.hi) == (second.lo, second.hi)


def test_macro_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="equal-length"):
        paired_macro_accuracy_difference([1.0, 0.0], [1.0], ["A", "B"], ["t", "t"], n_resamples=5)


def test_stratified_mean_of_a_constant_offset_is_exact() -> None:
    values_b = np.linspace(0.1, 0.9, 40)
    values_a = values_b - 0.05
    strata = [f"t{i % 3}" for i in range(40)]
    result = paired_mean_difference_stratified(values_a, values_b, strata, n_resamples=100, seed=0)
    assert result.point == pytest.approx(-0.05)
    assert result.lo == pytest.approx(-0.05) and result.hi == pytest.approx(-0.05)


def test_stratified_mean_needs_one_stratum_per_item() -> None:
    with pytest.raises(ValueError, match="one entry per item"):
        paired_mean_difference_stratified([0.1, 0.2], [0.2, 0.3], ["t"], n_resamples=5)
