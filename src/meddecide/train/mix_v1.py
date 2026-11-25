"""Mix rules for student_v1 (V4): per-template share cap and the template-stratified dev set.

Pure functions over counts and item lists. The assembly script reads and writes files; the rules
live here so they can be unit-tested.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

MAX_TEMPLATE_SHARE = 0.08


def cap_fixed_point(counts: Mapping[str, int], share: float = MAX_TEMPLATE_SHARE) -> int:
    """Largest per-template cap ``c`` with ``c == floor(share * sum(min(n_t, c)))``.

    Capping every template at ``c`` shrinks the mix, which lowers the share bound, so the cap is
    iterated to a fixed point. The sequence is non-increasing and bounded below, so it converges.
    Afterwards no template exceeds ``share`` of the capped mix.
    """
    if not 0 < share < 1:
        raise ValueError("share must be in (0, 1)")
    if not counts or sum(counts.values()) == 0:
        raise ValueError("cap_fixed_point needs at least one non-empty template")
    cap = int(share * sum(counts.values()))
    while True:
        mix = sum(min(n, cap) for n in counts.values())
        new_cap = int(share * mix)
        if new_cap == cap:
            return cap
        cap = new_cap


def cap_items[T](items_by_template: Mapping[str, Sequence[T]], cap: int, key) -> dict[str, list[T]]:
    """Keep the first ``cap`` items of each template in ``key`` order (deterministic)."""
    if cap < 1:
        raise ValueError("cap must be at least 1")
    return {t: sorted(items, key=key)[:cap] for t, items in sorted(items_by_template.items())}


def max_share(counts: Mapping[str, int]) -> float:
    total = sum(counts.values())
    return max(counts.values()) / total if total else 0.0


def stratified_sample[T](items_by_template: Mapping[str, Sequence[T]], per_template: int,
                         key) -> dict[str, list[T]]:
    """Up to ``per_template`` items per template, the first in ``key`` order (deterministic)."""
    return {t: sorted(items, key=key)[:per_template] for t, items in sorted(items_by_template.items())}


def describe(counts: Mapping[str, int]) -> dict[str, Any]:
    total = sum(counts.values())
    return {"total": total, "templates": len(counts), "max_template_share": max_share(counts)}


def balance_classes[T](items: Sequence[T], label, key) -> list[T]:
    """Keep the same number of items for every label: the smallest label's count, first in key order.

    Used for the binary (noul) templates, whose yes-rate can be a few per cent in the source records.
    Without it a template teaches its prior rather than the question.
    """
    by_label: dict[str, list[T]] = {}
    for item in items:
        by_label.setdefault(label(item), []).append(item)
    if not by_label:
        return []
    per_label = min(len(v) for v in by_label.values())
    kept: list[T] = []
    for name in sorted(by_label):
        kept.extend(sorted(by_label[name], key=key)[:per_label])
    return kept


def balanced_sample[T](items: Sequence[T], per_label: int, label, key) -> list[T]:
    """Up to ``per_label`` items per label (deterministic by key); smaller labels are not padded."""
    by_label: dict[str, list[T]] = {}
    for item in items:
        by_label.setdefault(label(item), []).append(item)
    kept: list[T] = []
    for name in sorted(by_label):
        kept.extend(sorted(by_label[name], key=key)[:per_label])
    return kept
