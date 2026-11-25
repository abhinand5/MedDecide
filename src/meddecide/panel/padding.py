"""Padding sources for perturbation (c) of the robustness pack (osler_v0 O1).

Sentences come from unrelated records of the same set, so the padding is irrelevant to the item's
question. A sentence that echoes the gold label is skipped (labels longer than three characters).
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Mapping

from meddecide.panel.schema import EvalItem
from meddecide.utils.hashing import stable_hash

SENTENCE = re.compile(r"(?<=[.!?])\s+")


def pool_by_set(pool: list[EvalItem]) -> dict[str, list[EvalItem]]:
    """The padding pool per set, in a fixed order (by hash of item id)."""
    grouped: dict[str, list[EvalItem]] = defaultdict(list)
    for item in pool:
        grouped[item.set_name].append(item)
    return {name: sorted(items, key=lambda it: stable_hash(it.item_id, length=32)) for name, items in grouped.items()}


def padding_for(item: EvalItem, pool: Mapping[str, list[EvalItem]], scan: int = 60) -> list[str] | None:
    """Two sentences from another record of the same set. The start point is a per-item hash offset, so
    different bases get different padding; the walk is deterministic. Sentences that echo the gold label
    (labels longer than three characters) are skipped."""
    records = pool.get(item.set_name, [])
    if not records:
        return None
    gold_label = item.label_of(item.gold).lower()
    start = int(stable_hash(item.item_id, length=8), 16) % len(records)
    seen = 0
    for offset in range(len(records)):
        other = records[(start + offset) % len(records)]
        if other.source_record_id == item.source_record_id:
            continue
        seen += 1
        sentences = [s.strip() for s in SENTENCE.split(other.state) if 20 <= len(s.strip()) <= 300]
        if len(gold_label) > 3:
            sentences = [s for s in sentences if gold_label not in s.lower()]
        if sentences:
            return sentences[:2]
        if seen >= scan:
            break
    return None
