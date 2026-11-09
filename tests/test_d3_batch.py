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


def test_criteria_for_builds_the_d3_request_shapes() -> None:
    from meddecide.distill.d3_batch import criteria_for, question_request

    options = [("A", "alpha"), ("B", "beta"), ("C", "gamma")]
    assert criteria_for("choice", options) == {"A": "alpha", "B": "beta", "C": "gamma"}
    assert criteria_for("noul", [("yes", "Yes"), ("no", "No")]) is None
    assert criteria_for("score", [("0", "low"), ("1", "mid"), ("2", "high")]) == ["low", "mid", "high"]
    with pytest.raises(ValueError):
        criteria_for("noul", [("A", "Yes"), ("B", "No")])
    with pytest.raises(ValueError):
        criteria_for("score", [("0", "only")])
    with pytest.raises(ValueError):
        criteria_for("box", options)
    assert question_request("choice", "pick", options)["criteria"]["B"] == "beta"
    assert "criteria" not in question_request("noul", "is it?", [("yes", "Yes"), ("no", "No")])


def test_align_probabilities_uses_each_qtype_own_order() -> None:
    from meddecide.distill.d3_batch import align_probabilities

    assert align_probabilities("choice", [("A", "a"), ("B", "b")], [0.25, 0.75]) == [0.25, 0.75]
    assert align_probabilities("noul", [("yes", "Yes"), ("no", "No")], [0.2, 0.8]) == [0.8, 0.2]
    assert align_probabilities("noul", [("no", "No"), ("yes", "Yes")], [0.2, 0.8]) == [0.2, 0.8]
    assert align_probabilities("score", [("0", "a"), ("1", "b"), ("2", "c")], [0.1, 0.3, 0.6]) == [0.1, 0.3, 0.6]
    with pytest.raises(ValueError):
        align_probabilities("choice", [("A", "a"), ("B", "b")], [1.0])
    with pytest.raises(ValueError):
        align_probabilities("noul", [("A", "a"), ("B", "b")], [0.5, 0.5])


def test_lora_starts_at_the_frozen_model_output() -> None:
    import torch
    from torch import nn

    from meddecide.distill.d3_train import LoRALinear

    torch.manual_seed(0)
    base = nn.Linear(16, 8).to(torch.bfloat16)
    adapter = LoRALinear(base, r=4, alpha=4)
    x = torch.randn(3, 5, 16, dtype=torch.bfloat16)
    assert torch.equal(adapter(x), base(x))
    assert not base.weight.requires_grad and adapter.A.requires_grad and adapter.B.requires_grad


def test_lora_update_is_rank_limited_and_learns() -> None:
    import torch
    from torch import nn

    from meddecide.distill.d3_train import LoRALinear

    torch.manual_seed(1)
    base = nn.Linear(8, 4)
    adapter = LoRALinear(base, r=2, alpha=2)
    with torch.no_grad():
        adapter.B.normal_()
    x = torch.randn(2, 3, 8)
    out = adapter(x)
    assert out.shape == (2, 3, 4)
    assert not torch.equal(out, base(x))


def test_masked_logits_and_loss_terms() -> None:
    import torch

    from meddecide.distill.d3_train import loss_from_logits, masked_logits

    logits = torch.tensor([[1.0, 2.0, 3.0, 9.0], [0.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
    masked = masked_logits(logits, [2, 3])
    assert masked[0, 2].item() == float("-inf") and masked[0, 3].item() == float("-inf")
    assert masked[1, 3].item() == float("-inf") and masked[1, 2].item() == 0.0
    # A uniform teacher equal to the student distribution has zero KL; CE picks the gold index.
    logits = torch.tensor([[0.0, 0.0], [5.0, 0.0]], dtype=torch.float32)
    ce, kl = loss_from_logits(logits, [2, 2], [0, 1], [[0.5, 0.5], None])
    assert ce.item() == pytest.approx(-torch.log_softmax(logits, -1)[0, 0].item() - torch.log_softmax(logits, -1)[1, 1].item())
    assert kl.item() == pytest.approx(0.0, abs=1e-6)
    # A confident wrong student has positive KL.
    _, kl = loss_from_logits(torch.tensor([[5.0, 0.0]]), [2], [1], [[0.9, 0.1]])
    assert kl.item() > 0


def test_batch_plan_is_deterministic_resumable_and_length_sorted() -> None:
    from meddecide.distill.d3_train import BatchPlan

    lengths = [100 + (i * 7) % 50 for i in range(640)]
    plan = BatchPlan(lengths, batch=8, window=64, seed=0)
    assert plan.steps_per_epoch == 80
    first = plan.batch(0)
    assert len(first) == 8 and len(set(first)) == 8
    assert first == plan.batch(0)  # same step, same batch
    assert plan.batch(1) != first
    assert sorted(plan.batch(0) + plan.batch(1)) == sorted(set(plan.batch(0) + plan.batch(1)))  # disjoint within a window
    # A new plan at the same seed reproduces the mapping (resume).
    assert BatchPlan(lengths, batch=8, window=64, seed=0).batch(37) == plan.batch(37)
    # A different seed changes the epoch order.
    assert BatchPlan(lengths, batch=8, window=64, seed=1).batch(37) != plan.batch(37)
    # Epoch 1 starts a different permutation but the same steps-per-epoch.
    assert plan.epoch_of(79) == 0 and plan.epoch_of(80) == 1
    # Within a batch the lengths are nondecreasing (length-sorted window).
    batch_lengths = [lengths[i] for i in plan.batch(3)]
    assert batch_lengths == sorted(batch_lengths)
