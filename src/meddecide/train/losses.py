"""Losses for the pointer head: cross-entropy plus lambda * Brier.

Both terms are computed over **exactly the offered options** of each item, on the flat
per-option layout the head produces (items may have different option counts in one batch).
"""

from __future__ import annotations

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
