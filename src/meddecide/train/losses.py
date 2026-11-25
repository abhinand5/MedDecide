"""Losses for the pointer head: cross-entropy plus lambda * Brier.

Both terms are computed over **exactly the offered options** of each item, on the flat
per-option layout the head produces (items may have different option counts in one batch).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class LossBreakdown:
    """Total loss and its parts. ``accuracy`` counts ties (gold among the maxima) as correct."""

    total: Tensor
    ce: Tensor
    brier: Tensor
    accuracy: float
    n_items: int
    n_options: int

    def to_dict(self) -> dict[str, float | int]:
        return {
            "loss": float(self.total.detach()),
            "ce": float(self.ce.detach()),
            "brier": float(self.brier.detach()),
            "accuracy": self.accuracy,
            "n_items": self.n_items,
            "n_options": self.n_options,
        }


def ce_plus_brier(
    probs: Tensor,
    gold_flat: Tensor,
    item_index: Tensor,
    n_items: int,
    *,
    lam: float = 1.0,
    eps: float = 1e-12,
) -> LossBreakdown:
    """``mean(-log p_gold) + lam * mean(sum_k (p_k - y_k)^2)`` over the batch.

    ``probs`` is flat over all offered options of the batch, ``gold_flat`` is the flat index of
    each item's gold option, ``item_index`` maps each flat position to its item.
    """
    if probs.ndim != 1 or gold_flat.shape != (n_items,) or item_index.shape != probs.shape:
        raise ValueError("ce_plus_brier expects flat probs, (n_items,) gold and matching index")
    if n_items < 1:
        raise ValueError("ce_plus_brier needs at least one item")
    log_probs = torch.log(probs.clamp_min(eps))
    ce = -log_probs[gold_flat].mean()
    target = torch.zeros_like(probs)
    target[gold_flat] = 1.0
    squared = (probs - target) ** 2
    per_item = torch.zeros(n_items, device=probs.device, dtype=probs.dtype)
    per_item = per_item.index_add(0, item_index, squared)
    brier = per_item.mean()
    total = ce + lam * brier
    max_per_item = torch.full((n_items,), float("-inf"), device=probs.device, dtype=probs.dtype)
    max_per_item = max_per_item.scatter_reduce(
        0, item_index, probs.detach(), reduce="amax", include_self=False
    )
    correct = float((probs.detach()[gold_flat] >= max_per_item).float().mean())
    return LossBreakdown(
        total=total,
        ce=ce,
        brier=brier,
        accuracy=correct,
        n_items=n_items,
        n_options=int(probs.numel()),
    )


def kl_to_reference(
    probs: Tensor,
    ref_probs: Tensor,
    item_index: Tensor,
    n_items: int,
    anchor: Tensor,
    *,
    eps: float = 1e-12,
    per_item_normaliser: int | None = None,
) -> Tensor:
    """Mean over anchor items of ``KL(ref || probs)`` across each item's offered options (osler_v3 KL-to-base).

    ``ref_probs`` are the frozen base model's probabilities for the same flat options (no gradient); ``anchor`` is a
    ``(n_items,)`` boolean mask of the items the penalty applies to. Returns a zero tensor when no item is an anchor.

    With ``per_item_normaliser`` (osler_v5 D51, ``loss_weighting: per_item``) the penalty is the **sum** over the anchor items divided
    by that normaliser (the batch size), not the mean over anchors.
    """
    if probs.shape != ref_probs.shape or item_index.shape != probs.shape or anchor.shape != (n_items,):
        raise ValueError("kl_to_reference expects matching flat probs, index and an (n_items,) anchor mask")
    ref = ref_probs.detach().to(probs.dtype)
    terms = ref * (torch.log(ref.clamp_min(eps)) - torch.log(probs.clamp_min(eps)))
    per_item = torch.zeros(n_items, device=probs.device, dtype=probs.dtype).index_add(0, item_index, terms)
    mask = anchor.to(probs.device)
    if not bool(mask.any()):
        return probs.sum() * 0.0
    if per_item_normaliser is not None:
        if per_item_normaliser < 1:
            raise ValueError("per_item_normaliser must be >= 1")
        return per_item[mask].sum() / per_item_normaliser
    return per_item[mask].mean()


def item_weight(n_items: int, batch_size: int) -> float:
    """The per-item factor of ``loss_weighting: per_item`` (osler_v5 D51): n_items / batch_size, applied after clipping."""
    if not 1 <= n_items <= batch_size:
        raise ValueError(f"a batch of {n_items} items does not fit batch_size {batch_size}")
    return n_items / batch_size


def scale_gradients(parameters: Sequence[Tensor], factor: float) -> None:
    """Multiply every present gradient in place (the post-clip step of ``per_item``)."""
    for parameter in parameters:
        if parameter.grad is not None:
            parameter.grad.mul_(factor)


def teacher_reference(
    teacher_rows: Sequence[Sequence[float] | None], n_options: Sequence[int]
) -> tuple[Tensor, Tensor]:
    """Flat teacher distributions over a batch's offered options, in display order, and the (n_items,) mask of the items
    that have one (osler_v4 Y6, D48). An item without a teacher gets a uniform placeholder, which the mask removes from
    ``kl_to_reference``. ``teacher_rows`` and ``n_options`` are the batch's items in batch order."""
    flat: list[float] = []
    mask: list[bool] = []
    for probs, n in zip(teacher_rows, n_options, strict=True):
        if probs is None:
            flat.extend([1.0 / n] * n)
            mask.append(False)
            continue
        if len(probs) != n:
            raise ValueError(f"teacher distribution has {len(probs)} values for {n} options")
        flat.extend(float(p) for p in probs)
        mask.append(True)
    return torch.tensor(flat, dtype=torch.float32), torch.tensor(mask, dtype=torch.bool)
