"""Per-template audit of training mix v2 (osler_v0 O4): a deterministic subsample for the bag-of-words diagnostic.

The diagnostic is additional (R4), not a gate: a template whose bag-of-words model does no better than its majority
class has no lexical cue for its label in the state text. That is evidence about the data, not a verdict on whether the
label is true, and it is reported beside the gold-in-state rate, which is the screen's own check.
"""

from __future__ import annotations

import heapq
from collections.abc import Iterable, Mapping
from typing import Any

from meddecide.utils.hashing import stable_hash


def smallest_hash_rows(rows: Iterable[Mapping[str, Any]], k: int) -> dict[str, list[Mapping[str, Any]]]:
    """For each template, the ``k`` rows whose item id hashes lowest: a deterministic subsample of a stream that
    does not depend on file order."""
    heaps: dict[str, list[tuple[int, int, Mapping[str, Any]]]] = {}
    for seen, row in enumerate(rows, start=1):
        h = int(stable_hash({"audit": row["item_id"]}, length=16), 16)
        heap = heaps.setdefault(row["template_id"], [])
        entry = (-h, seen, row)
        if len(heap) < k:
            heapq.heappush(heap, entry)
        elif -h > heap[0][0]:
            heapq.heapreplace(heap, entry)
    return {t: [e[2] for e in sorted(heap, key=lambda e: -e[0])] for t, heap in heaps.items()}
