"""osler_v4 Y10 (meddecide.eval.audit_y10): the membership AUC and the verbatim-run count, with hand-computed cases (CPU)."""

from __future__ import annotations

import pytest

from meddecide.eval.audit_y10 import (
    longest_aligned_run,
    membership_auc,
    membership_auc_interval,
    verbatim_match,
)


def test_members_with_lower_loss_give_an_auc_of_one() -> None:
    assert membership_auc([0.1, 0.2], [0.5, 0.9]) == pytest.approx(1.0)


def test_members_with_higher_loss_give_an_auc_of_zero() -> None:
    assert membership_auc([0.5, 0.9], [0.1, 0.2]) == pytest.approx(0.0)


def test_equal_losses_tie_and_give_one_half() -> None:
    assert membership_auc([0.5], [0.5]) == pytest.approx(0.5)


def test_a_mixed_case_matches_the_pairwise_count() -> None:
    # member scores (negative losses) -0.1 and -0.6 against non-member scores -0.3 and -0.9: pairs won 1, 1, 0, 1 of 4
    assert membership_auc([0.1, 0.6], [0.3, 0.9]) == pytest.approx(0.75)


def test_the_interval_brackets_the_point() -> None:
    members = [0.1 * k for k in range(1, 41)]
    nonmembers = [0.1 * k + 0.05 for k in range(1, 41)]
    result = membership_auc_interval(members, nonmembers)
    assert result.lo <= result.point <= result.hi
    assert result.n == 80


def test_longest_run_counts_consecutive_aligned_matches() -> None:
    assert longest_aligned_run([1, 2, 3, 9, 9], [1, 2, 3, 4, 5]) == 3
    assert longest_aligned_run([9] * 50, [9] * 50) == 50
    assert longest_aligned_run([1, 2, 9, 4, 5, 6], [1, 2, 3, 4, 5, 6]) == 3


def test_verbatim_needs_fifty_consecutive_matches() -> None:
    truth = list(range(64))
    assert verbatim_match(truth, truth) is True
    broken = list(truth)
    broken[25] = -1  # the longest run is then 25 (positions 0 to 24) and 38 (positions 26 to 63)
    assert longest_aligned_run(broken, truth) == 38
    assert verbatim_match(broken, truth) is False
