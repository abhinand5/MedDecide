"""Unit tests for the batched d3 scoring helpers (d3_distill step 0): batch planning and the probability checks."""

from __future__ import annotations

import pytest

from meddecide.distill.d3_batch import (
    PreparedQuestion,
    argmax_index,
    distribution_ok,
    length_batches,
    non_constant,
    padded_tokens,
)


def _rows(lengths: list[int]) -> list[PreparedQuestion]:
    return [PreparedQuestion(index=i, sequence=tuple(range(n)), count=4, state=10) for i, n in enumerate(lengths)]


def test_length_batches_covers_every_row_once_and_respects_the_row_cap() -> None:
    rows = _rows([10, 30, 20, 40, 5])
    batches = length_batches(rows, max_rows=2, max_tokens=1_000)
    assert sorted(r.index for batch in batches for r in batch) == [0, 1, 2, 3, 4]
    assert all(len(batch) <= 2 for batch in batches)
    # Longest first: the 40-token row opens the first batch.
    assert batches[0][0].index == 3


def test_length_batches_respects_the_padded_token_budget() -> None:
    rows = _rows([100, 100, 100, 10, 10, 10])
    batches = length_batches(rows, max_rows=8, max_tokens=250)
    assert all(padded_tokens(batch) <= 250 for batch in batches)
    assert sorted(r.index for batch in batches for r in batch) == list(range(6))


def test_a_single_row_over_the_token_budget_still_gets_a_batch() -> None:
    batches = length_batches(_rows([1000]), max_rows=8, max_tokens=10)
    assert len(batches) == 1 and len(batches[0]) == 1


def test_batch_planning_rejects_bad_limits() -> None:
    with pytest.raises(ValueError):
        length_batches(_rows([10]), max_rows=0, max_tokens=10)
    with pytest.raises(ValueError):
        length_batches(_rows([10]), max_rows=1, max_tokens=0)


def test_distribution_ok_checks_length_finiteness_and_the_sum() -> None:
    assert distribution_ok([0.25, 0.25, 0.25, 0.25], 4)
    assert not distribution_ok([0.25, 0.25, 0.25, 0.25], 3)
    assert distribution_ok([0.5, 0.5, 0.0], 3)
    assert distribution_ok([0.5, 0.5], 2)
    assert not distribution_ok([1.2, -0.2], 2)  # negative
    assert not distribution_ok([float("nan"), 1.0], 2)  # non-finite
    assert not distribution_ok([0.2, 0.2], 2)  # sum 0.4, outside the tolerance
    assert not distribution_ok(None, 2)


def test_non_constant_and_argmax() -> None:
    assert not non_constant([0.5, 0.5])
    assert non_constant([0.6, 0.4])
    assert non_constant([0.7, 0.2, 0.1])
    assert argmax_index([0.1, 0.7, 0.2]) == 1
    assert argmax_index([0.5, 0.5]) == 0  # first of the tie, like the readout
