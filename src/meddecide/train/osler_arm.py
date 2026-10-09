"""Shared pieces of the three matched Osler-4B arms (ADVISORY O6-O8; D23): budget, step count, dev subsets, tier-1 items.

The recipe itself lives in the committed configs (configs/osler_v0/arm_L.yaml, arm_P.yaml, arm_N.yaml); this module holds
the pure functions the driver uses, so they can be tested without a GPU.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from typing import Any

import numpy as np

from meddecide.bench.schema import Item

EXAMPLE_BUDGET_CAP = 200_000
TIER1_SOURCES = ("medqa", "medmcqa")
DEV_SUBSET_SIZE = 6_000


def example_budget(mix_rows: int, cap: int = EXAMPLE_BUDGET_CAP) -> int:
    """The ADVISORY budget: one pass over the mix or ``cap`` examples, whichever is smaller."""
    if mix_rows < 1:
        raise ValueError("the mix has no rows")
    return min(int(mix_rows), int(cap))


def step_count(examples: int, batch_size: int) -> int:
    """Optimiser steps for an example budget at a fixed batch size (the budget is a multiple of the batch size)."""
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if examples % batch_size:
        raise ValueError(f"the example budget {examples} is not a multiple of the batch size {batch_size}")
    return examples // batch_size


def stratified_subset[T: Item](items: Sequence[T], *, key: str, size: int, seed: int) -> list[T]:
    """A deterministic subset of ``size`` items, drawn up to an equal share per group of ``key``.

    The per-group cap is the smallest value whose capped total reaches ``size``; each group contributes its cap (or all of
    its items when it has fewer), chosen by a seeded permutation, and the result is trimmed to exactly ``size`` in input order.
    Raises if the items cannot supply ``size`` rows.
    """
    if size < 1:
        raise ValueError("size must be >= 1")
    if len(items) < size:
        raise ValueError(f"only {len(items)} items for a subset of {size}")
    groups: dict[str, list[int]] = defaultdict(list)
    for index, item in enumerate(items):
        groups[str(getattr(item, key))].append(index)
    counts = sorted(len(v) for v in groups.values())

    def capped_total(cap: int) -> int:
        return sum(min(c, cap) for c in counts)

    low, high = 1, max(counts)
    while low < high:
        mid = (low + high) // 2
        if capped_total(mid) >= size:
            high = mid
        else:
            low = mid + 1
    cap = low
    rng = np.random.default_rng([int(seed), 6000])
    chosen: list[int] = []
    for group in sorted(groups):
        positions = groups[group]
        if len(positions) <= cap:
            chosen.extend(positions)
        else:
            picked = rng.permutation(len(positions))[:cap]
            chosen.extend(positions[int(i)] for i in picked)
    chosen = sorted(chosen)
    if len(chosen) > size:
        kept = sorted(int(i) for i in rng.permutation(len(chosen))[:size])
        chosen = [chosen[i] for i in kept]
    return [items[i] for i in chosen]


def tier1_items[T: Item](items: Sequence[T]) -> list[T]:
    """The tier-1 diagnostic items (MedQA and MedMCQA dev rows): reported at every evaluation, never a selection signal."""
    return [item for item in items if item.source in TIER1_SOURCES]


def accuracy_by_source(decisions: Sequence[Any], items: Sequence[Item]) -> dict[str, float]:
    """Accuracy of argmax decisions by source (the tier-1 diagnostic)."""
    correct: dict[str, list[bool]] = defaultdict(list)
    for decision, item in zip(decisions, items, strict=True):
        correct[item.source].append(decision.argmax_key == item.gold)
    return {source: sum(v) / len(v) for source, v in sorted(correct.items())}


def log_scale(value: float) -> float:
    return math.log10(value) if value > 0 else float("-inf")
