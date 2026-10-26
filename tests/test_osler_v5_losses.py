"""osler_v5 D51 (loss_weighting: per_item): the update after clipping is multiplied by n_items / batch_size, and KL and distillation are sums
over the masked items divided by batch_size. The mechanics are the trainer's own functions (clip, then scale)."""

from __future__ import annotations

import pytest
import torch

from meddecide.train.losses import item_weight, kl_to_reference, scale_gradients


def _applied_update(item_grads: list[torch.Tensor], batch_size: int, mode: str, clip: float = 1.0) -> torch.Tensor:
    """The gradient the optimiser would see for a batch whose per-item losses have the given gradients (linear items: loss_i = g_i . p),
    with the per-step mean as the loss (per_step and the clip of both modes), then the trainer's post-clip factor under per_item."""
    dim = item_grads[0].numel()
    p = torch.zeros(dim, requires_grad=True)
    loss = torch.stack([(g * p).sum() for g in item_grads]).mean()
    loss.backward()
    torch.nn.utils.clip_grad_norm_([p], clip)
    if mode == "per_item":
        scale_gradients([p], item_weight(len(item_grads), batch_size))
    assert p.grad is not None
    return p.grad.detach().clone()


def test_under_clipping_a_lone_item_gets_one_eighth_of_a_full_batch_under_per_item() -> None:
    torch.manual_seed(0)
    mimic = torch.randn(16) * 5.0  # pre-clip gradient norm well above the clip of 1.0
    replay = [torch.randn(16) * 5.0 for _ in range(8)]
    lone = _applied_update([mimic], batch_size=8, mode="per_item")
    full = _applied_update(replay, batch_size=8, mode="per_item")
    assert lone.norm().item() == pytest.approx(1.0 / 8.0, rel=1e-5)
    assert full.norm().item() == pytest.approx(1.0, rel=1e-5)
    assert (lone.norm() / full.norm()).item() == pytest.approx(0.125, rel=1e-5)
    # the lone item's update points along its own gradient (only the scale changed)
    assert torch.nn.functional.cosine_similarity(lone, mimic, dim=0).item() == pytest.approx(1.0, abs=1e-5)


def test_under_per_step_a_lone_item_and_a_full_batch_have_the_same_clipped_norm() -> None:
    torch.manual_seed(1)
    mimic = torch.randn(16) * 5.0
    replay = [torch.randn(16) * 5.0 for _ in range(8)]
    lone = _applied_update([mimic], batch_size=8, mode="per_step")
    full = _applied_update(replay, batch_size=8, mode="per_step")
    assert lone.norm().item() == pytest.approx(1.0, rel=1e-5)
    assert full.norm().item() == pytest.approx(1.0, rel=1e-5)


def test_a_batch_with_fewer_items_than_the_batch_size_is_scaled_by_its_item_count() -> None:
    assert item_weight(8, 8) == 1.0
    assert item_weight(2, 8) == 0.25
    assert item_weight(1, 8) == 0.125
    with pytest.raises(ValueError, match="does not fit"):
        item_weight(9, 8)
    with pytest.raises(ValueError, match="does not fit"):
        item_weight(0, 8)


def test_a_full_batch_is_unchanged_by_per_item() -> None:
    torch.manual_seed(2)
    grads = [torch.randn(16) for _ in range(8)]
    assert torch.allclose(_applied_update(grads, 8, "per_item"), _applied_update(grads, 8, "per_step"))


def test_kl_sums_over_the_anchor_items_and_divides_by_the_batch_size_under_per_item() -> None:
    probs = torch.tensor([0.5, 0.5, 0.2, 0.8, 0.9, 0.1])
    ref = torch.tensor([0.4, 0.6, 0.3, 0.7, 0.5, 0.5])
    index = torch.tensor([0, 0, 1, 1, 2, 2])
    anchor = torch.tensor([True, False, True])  # items 0 and 2 are anchors

    def kl_of(item: int) -> float:
        sel = index == item
        return float((ref[sel] * (torch.log(ref[sel]) - torch.log(probs[sel]))).sum())

    per_item = float(kl_to_reference(probs, ref, index, 3, anchor, per_item_normaliser=8))
    per_step = float(kl_to_reference(probs, ref, index, 3, anchor))
    assert per_item == pytest.approx((kl_of(0) + kl_of(2)) / 8, rel=1e-5)
    assert per_step == pytest.approx((kl_of(0) + kl_of(2)) / 2, rel=1e-5)


def test_kl_without_anchors_is_zero_under_both_reductions() -> None:
    probs = torch.tensor([0.5, 0.5])
    ref = torch.tensor([0.4, 0.6])
    index = torch.tensor([0, 0])
    anchor = torch.tensor([False])
    assert float(kl_to_reference(probs, ref, index, 1, anchor, per_item_normaliser=8)) == 0.0
    assert float(kl_to_reference(probs, ref, index, 1, anchor)) == 0.0


def test_a_non_positive_normaliser_is_refused() -> None:
    probs = torch.tensor([0.5, 0.5])
    with pytest.raises(ValueError, match="per_item_normaliser"):
        kl_to_reference(probs, torch.tensor([0.4, 0.6]), torch.tensor([0, 0]), 1, torch.tensor([True]), per_item_normaliser=0)
