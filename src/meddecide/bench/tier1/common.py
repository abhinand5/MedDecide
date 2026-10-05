"""Shared helpers for the tier-1 (established) source loaders.

Every loader returns an :class:`~meddecide.bench.schema.Item` list plus the record of what
it dropped and why ("no silent drops", AGENTS.md). No loader ever derives gold from a
language model: gold is a lookup of a structured field of the source record.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from meddecide.bench.schema import Option, QuestionType, Split, Tier, make_item

TIER = Tier.ESTABLISHED
# Tier-1 records carry the date the dataset revision was published, not a per-record date:
# established test sets are not date-filtered (that is the fresh tier's job).
DEFAULT_LICENSE = "UNKNOWN"


@dataclass
class LoadResult:
    """Items produced by one loader, plus the accounting for everything that did not survive."""

    source: str
    rows: list[Any] = field(default_factory=list)
    dropped: dict[str, int] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    dataset_id: str = ""
    dataset_revision: str = ""
    license: str = DEFAULT_LICENSE
    split_map: dict[str, str] = field(default_factory=dict)

    def drop(self, reason: str, n: int = 1) -> None:
        self.dropped[reason] = self.dropped.get(reason, 0) + n

    @property
    def items(self) -> list[Any]:
        """The built items (read-only alias for ``rows``).

        The field is named ``rows`` so that ``LoadResult.items`` cannot shadow
        ``dict.items`` on this dataclass by accident.
        """
        return self.rows

    @property
    def n_items(self) -> int:
        return len(self.rows)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "dataset_id": self.dataset_id,
            "dataset_revision": self.dataset_revision,
            "license": self.license,
            "n_items": self.n_items,
            "dropped_by_reason": dict(sorted(self.dropped.items())),
            "split_map": self.split_map,
            "notes": self.notes,
        }


def caps(record: dict[str, Any]) -> tuple[int, int]:
    """``(max_test, max_dev)`` item caps from a benchmark config dict."""
    c = record.get("caps", {})
    return (
        int(c.get("tier1_max_test_items_per_source", 5000)),
        int(c.get("tier1_max_dev_items_per_source", 2000)),
    )


def subsample(rows: Sequence[Any], cap: int, seed: int, *, salt: str) -> tuple[list[Any], int]:
    """Deterministic random sample of at most ``cap`` rows; returns ``(rows, n_dropped)``.

    Sampling is seeded per ``(seed, salt)`` so a rebuild with the same config is identical.
    """
    if len(rows) <= cap:
        return list(rows), 0
    rng = random.Random(f"{seed}:{salt}")
    picked = rng.sample(range(len(rows)), cap)
    return [rows[i] for i in sorted(picked)], len(rows) - cap


def make_choice_item(
    *,
    source: str,
    record_id: str,
    url: str,
    license_: str,
    record_date: date,
    split: Split,
    template_id: str,
    skill: str,
    state: str,
    question: str,
    labels: Sequence[str],
    gold_index: int,
    option_order_seed: int,
    meta: dict[str, Any] | None = None,
    qtype: QuestionType = QuestionType.CHOICE,
    lead: str = "A",
):
    """Build a ``choice``/``score`` item with options keyed ``A``, ``B``, … or ``1``, ``2``, ….

    ``lead`` is ``"A"`` for lettered options and ``"1"`` for score levels.
    """
    if lead == "A":
        keys = [chr(ord("A") + i) for i in range(len(labels))]
    else:
        keys = [str(i + 1) for i in range(len(labels))]
    options = [Option(key=k, label=text) for k, text in zip(keys, labels, strict=True)]
    return make_item(
        tier=TIER,
        source=source,
        source_record_id=record_id,
        source_url=url,
        source_license=license_,
        record_date=record_date,
        split=split,
        template_id=template_id,
        skill=skill,
        qtype=qtype,
        state=state,
        question=question,
        options=options,
        gold=keys[gold_index],
        option_order_seed=option_order_seed,
        meta=meta or {},
    )


def dedupe_by_record(items: Iterable[Any], *, key: Callable[[Any], str] | None = None) -> tuple[list[Any], int]:
    """Drop duplicate items sharing a ``(source, source_record_id, split)`` key.

    Sources that repeat a record (e.g. MedQuAD asks many questions per document) must be
    deduplicated *per split* so the split-disjointness invariant holds; duplicates inside a
    split are dropped and counted.
    """
    seen: set[str] = set()
    kept: list[Any] = []
    dropped = 0
    for item in items:
        k = key(item) if key else f"{item.source}|{item.source_record_id}|{item.split}"
        if k in seen:
            dropped += 1
            continue
        seen.add(k)
        kept.append(item)
    return kept, dropped


def split_train_into_dev_train(
    rows: Sequence[Any],
    *,
    dev_cap: int,
    seed: int,
    salt: str,
) -> tuple[list[Any], list[Any], int]:
    """Split an official *train* split into dev (carved) and remaining train.

    Used where a source publishes no usable test labels (MedMCQA) or where a dev set is
    needed for teacher calibration. Returns ``(dev_rows, train_rows, n_dev_over_cap)``;
    ``dev_rows`` and ``train_rows`` are disjoint by construction (sampled indices).
    """
    if len(rows) <= dev_cap:
        return list(rows), [], 0
    rng = random.Random(f"{seed}:{salt}:dev")
    picked = set(rng.sample(range(len(rows)), dev_cap))
    dev_rows = [rows[i] for i in sorted(picked)]
    train_rows = [row for i, row in enumerate(rows) if i not in picked]
    return dev_rows, train_rows, 0


def finalize_splits(
    items: list[Any],
    *,
    cfg: dict[str, Any],
    source: str,
    salt: str,
    cap_test: int | None = None,
    cap_dev: int | None = None,
) -> tuple[list[Any], dict[str, int], list[str]]:
    """Enforce three invariants on a source's items, in this order:

    1. **No duplicate items.** Two items sharing ``(record, template, split)`` are the same
       item; the duplicates are dropped and counted.
    2. **No identical question text in two splits.** Sources contain the same stem twice
       with different answers (MMLU), or the same question attached to two documents
       (MedQuAD). Any content that appears in both dev and test is forced into the test
       split — never silently kept on both sides of the boundary.
    3. **Caps applied last**, by deterministic seeded sampling, so the exported item count
       equals ``min(cap, distinct items)`` rather than "cap minus whatever dedup removed".

    Item ids are recomputed after a forced split move, because the split is part of the id.

    Returns ``(items, counts, notes)`` where ``counts`` is a drop-reason → count mapping.
    """
    from meddecide.bench.schema import make_item  # local import: avoids a cycle at import time
    from meddecide.utils.hashing import stable_hash

    counts: dict[str, int] = {}
    notes: list[str] = []

    def bump(reason: str, n: int) -> None:
        if n:
            counts[reason] = counts.get(reason, 0) + n

    # 1. duplicates
    seen: set[str] = set()
    deduped: list[Any] = []
    for item in items:
        key = f"{item.source}|{item.source_record_id}|{item.template_id}|{item.split}"
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    bump("duplicate_item_identity", len(items) - len(deduped))

    # 2. identical content across splits -> force to test
    splits_by_content: dict[str, set[str]] = {}
    for item in deduped:
        splits_by_content.setdefault(stable_hash({"state": item.state, "q": item.question}), set()).add(
            str(item.split)
        )
    leaking = {h for h, s in splits_by_content.items() if len(s) > 1}
    moved = 0
    finalized: list[Any] = []
    for item in deduped:
        content = stable_hash({"state": item.state, "q": item.question})
        if content in leaking and str(item.split) != "test":
            moved += 1
            payload = item.model_dump()
            payload["split"] = "test"
            payload["item_id"] = make_item(
                tier=item.tier,
                source=item.source,
                source_record_id=item.source_record_id,
                source_url=item.source_url,
                source_license=item.source_license,
                record_date=item.record_date,
                split="test",
                template_id=item.template_id,
                skill=item.skill,
                qtype=item.qtype,
                state=item.state,
                question=item.question,
                options=item.options,
                gold=item.gold,
                option_order_seed=int(item.meta.get("option_order_seed", 0)),
                meta=item.meta,
            ).item_id
            item = type(item).model_validate(payload)
        finalized.append(item)
    if moved:
        bump("content_in_two_splits_moved_to_test", moved)
        notes.append(
            f"{moved} item(s) had identical question text in two splits and were moved to "
            "test, so no stem is on both sides of the dev/test boundary"
        )
    if leaking:
        notes.append(f"{len(leaking)} distinct question text(s) were present in two splits")

    # 3. caps
    seed = int(cfg.get("seed", 0))
    capped: list[Any] = []
    for split, cap in (("test", cap_test), ("dev", cap_dev), ("train", None)):
        in_split = [i for i in finalized if str(i.split) == split]
        if cap is None or len(in_split) <= cap:
            capped.extend(in_split)
            continue
        kept, over = subsample(in_split, cap, seed, salt=f"{salt}:{split}")
        capped.extend(kept)
        bump(f"{split}_over_cap", over)
    return capped, counts, notes
