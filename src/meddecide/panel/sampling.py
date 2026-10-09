"""Deterministic subsampling for the external panel and the robustness pack (O1).

Items are ordered by a stable hash of their key (not by file order, not by a random seed), and the
first ``n`` are kept. The same population always gives the same sample on any machine, and the
recorded keys let anyone re-derive it.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from meddecide.utils.hashing import stable_hash


def sample_by_hash[T](items: Sequence[T], key: Callable[[T], str], n: int) -> list[T]:
    """Keep the ``n`` items with the smallest stable hash of ``key(item)``; all of them if fewer."""
    if n < 0:
        raise ValueError("n must be non-negative")
    ranked = sorted(items, key=lambda item: stable_hash(key(item), length=32))
    return ranked[:n]
