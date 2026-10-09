"""Paired-comparison helpers: Brier per item, paired and stratified bootstrap differences."""

from __future__ import annotations

import math

import pytest

from meddecide.eval.paired import item_brier, paired_difference, stratified_macro_difference


def test_item_brier_matches_hand_computed_values() -> None:
    probs = [0.7, 0.2, 0.1]
    assert item_brier(probs, 0) == pytest.approx(0.3**2 + 0.2**2 + 0.1**2)
    assert item_brier(probs, 1) == pytest.approx(0.7**2 + 0.8**2 + 0.1**2)


def test_item_brier_accepts_rows_rounded_to_six_decimals() -> None:
    # stored predictions round each probability to 6 decimals, so the row may not sum to 1 exactly
    probs = [0.333333, 0.333333, 0.333334]
    assert item_brier(probs, 2) == pytest.approx(
        0.333333**2 + 0.333333**2 + (0.333334 - 1) ** 2
    )


def test_item_brier_rejects_out_of_range_gold() -> None:
    with pytest.raises(ValueError, match="outside"):
        item_brier([0.5, 0.5], 2)


def test_paired_difference_point_and_count() -> None:
    result = paired_difference([1, 0, 1, 0], [0, 0, 0, 0], n_resamples=500)
    assert result.n == 4
    assert result.point == pytest.approx(0.5)
    assert result.lo <= result.point <= result.hi


def test_paired_difference_of_identical_scorings_is_zero_with_zero_width() -> None:
    result = paired_difference([1.0, 0.0, 0.5], [1.0, 0.0, 0.5], n_resamples=200)
    assert (result.point, result.lo, result.hi) == (0.0, 0.0, 0.0)


def test_paired_difference_is_seeded() -> None:
    a, b = [1.0, 0.0, 1.0, 1.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0]
    assert paired_difference(a, b, seed=3) == paired_difference(a, b, seed=3)


def test_paired_difference_rejects_empty_and_mismatched_input() -> None:
    with pytest.raises(ValueError):
        paired_difference([], [])
    with pytest.raises(ValueError):
        paired_difference([1.0, 0.0], [1.0])


def test_stratified_macro_is_mean_of_group_means() -> None:
    a = {"t1": [1.0, 1.0, 1.0], "t2": [0.0, 1.0]}
    b = {"t1": [0.0, 0.0, 0.0], "t2": [0.0, 0.0]}
    result = stratified_macro_difference(a, b, n_resamples=300)
    # group t1 contributes +1.0, group t2 contributes +0.5; macro is their unweighted mean
    assert result.point == pytest.approx(0.75)
    assert result.n == 5
    assert result.lo <= result.point <= result.hi


def test_stratified_macro_rejects_different_groups() -> None:
    with pytest.raises(ValueError, match="same groups"):
        stratified_macro_difference({"a": [1.0]}, {"b": [1.0]})


def test_stratified_macro_interval_is_finite() -> None:
    a = {"x": [1.0, 0.0, 1.0], "y": [1.0, 1.0]}
    b = {"x": [0.0, 0.0, 1.0], "y": [0.0, 1.0]}
    result = stratified_macro_difference(a, b, n_resamples=400)
    assert all(math.isfinite(v) for v in (result.point, result.lo, result.hi))


def test_spearman_perfect_and_reversed_order() -> None:
    from meddecide.eval.paired import spearman

    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4], [40, 30, 20, 10]) == pytest.approx(-1.0)


def test_spearman_handles_ties_with_average_ranks() -> None:
    from meddecide.eval.paired import spearman

    # ranks of x are [1.5, 1.5, 3]; Pearson with [1, 2, 3] is 1.5 / sqrt(1.5 * 2)
    assert spearman([1, 1, 2], [1, 2, 3]) == pytest.approx(1.5 / math.sqrt(3.0))


def test_spearman_constant_side_is_nan() -> None:
    from meddecide.eval.paired import spearman

    assert math.isnan(spearman([1, 1, 1], [1, 2, 3]))


def test_bootstrap_spearman_point_matches_and_interval_brackets_it() -> None:
    from meddecide.eval.paired import bootstrap_spearman, spearman

    x = list(range(1, 23))
    y = [0.5 + 0.01 * v + (0.05 if v % 3 == 0 else -0.02) for v in x]
    point, lo, hi = bootstrap_spearman(x, y, n_resamples=300, seed=1)
    assert point == pytest.approx(spearman(x, y))
    assert lo <= point <= hi
    assert bootstrap_spearman(x, y, n_resamples=300, seed=1) == (point, lo, hi)
