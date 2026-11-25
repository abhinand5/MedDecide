"""The letter readout's pure helpers: per-item softmax over flat logits and the within-item letter index."""

from __future__ import annotations

import math

import pytest
import torch

from meddecide.model.meddecide_model import letter_positions, segment_softmax


def test_segment_softmax_normalises_each_item_separately() -> None:
    flat = torch.tensor([0.0, 1.0, 2.0, -1.0, 3.0], dtype=torch.float64)
    probs = segment_softmax(flat, [3, 2])
    assert torch.allclose(probs[:3].sum(), torch.tensor(1.0, dtype=torch.float64))
    assert torch.allclose(probs[3:].sum(), torch.tensor(1.0, dtype=torch.float64))
    expected_first = torch.softmax(torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64), 0)
    assert torch.allclose(probs[:3], expected_first)


def test_segment_softmax_rejects_mismatched_option_counts() -> None:
    with pytest.raises(ValueError):
        segment_softmax(torch.zeros(4), [3, 2])


def test_letter_positions_restart_for_each_item() -> None:
    assert letter_positions([3, 2]) == [0, 1, 2, 0, 1]
    assert letter_positions([]) == []


def test_segment_softmax_of_equal_logits_is_uniform() -> None:
    probs = segment_softmax(torch.zeros(4, dtype=torch.float64), [4])
    assert all(math.isclose(float(p), 0.25) for p in probs)
