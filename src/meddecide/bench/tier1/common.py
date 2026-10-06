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
    per_template: bool = True,
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
       With ``per_template`` (the default) the cap is per (template, split) — a source with
       several templates must not let one template's volume cap another's.

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

    # 3. caps. Fresh-tier sources carry several templates, and a cap is per (template, split):
    #    one template exporting 5,000 items must not silently cap another template to zero.
    seed = int(cfg.get("seed", 0))
    if per_template:
        groups: dict[str, list[Any]] = {}
        for item in finalized:
            groups.setdefault(f"{item.template_id}|{item.split}", []).append(item)
    else:
        groups = {}
        for split in ("test", "dev", "train"):
            rows = [i for i in finalized if str(i.split) == split]
            if rows:
                groups[f"*|{split}"] = rows
    capped: list[Any] = []
    for key, rows in sorted(groups.items()):
        template_id, split = key.split("|", 1)
        cap = {"test": cap_test, "dev": cap_dev, "train": None}[split]
        if cap is None or len(rows) <= cap:
            capped.extend(rows)
            continue
        kept, over = subsample(rows, cap, seed, salt=f"{salt}:{key}")
        capped.extend(kept)
        bump(f"{split}_over_cap" if template_id == "*" else f"{split}_over_cap:{template_id}", over)
    return capped, counts, notes


def balance_classes(
    items: list[Any],
    *,
    cfg: dict[str, Any],
    salt: str,
    target_per_class: int | None = None,
    min_class_size: int | None = None,
) -> tuple[list[Any], dict[str, Any]]:
    """Cap every gold class at ``K`` items per (template, split); return the new list and a report.

    Why: bench_v0's fresh templates were dominated by one class (`pubmed_observational_noul_v1`
    98.5 % "no", `pubmed_pubtype_choice_v1` 92.6 % "Other"), so micro accuracy could not distinguish
    a model from a majority-class predictor, and three templates had a single gold class in test.

    ``K`` is ``min(target_per_class, smallest class size)`` so no class is dropped, and the same
    ``K`` is applied to every class, keeping the split as balanced as the data allows. Templates
    whose smallest class is below ``min_class_size`` are reported (not silently dropped): the
    decision to drop a template belongs to the screen (F3), which has the majority baseline.

    Sampling within a class is seeded and deterministic; the report records the class counts before
    and after and the ``K`` used, so "balanced" is a checkable claim rather than a description.
    """
    counts = cfg.get("caps", {}) if isinstance(cfg.get("caps"), dict) else {}
    target = int(target_per_class or counts.get("fresh_target_items_per_template", 1000))
    floor = int(min_class_size or counts.get("fresh_balance_min_class", 200))

    report: dict[str, Any] = {"target_per_class": target, "min_class_size": floor, "templates": {}}
    kept: list[Any] = []
    groups: dict[tuple[str, str], list[Any]] = {}
    for item in items:
        groups.setdefault((item.template_id, str(item.split)), []).append(item)

    seed = int(cfg.get("seed", 0))
    dropped_groups: list[str] = []
    for (template_id, split), rows in sorted(groups.items()):
        by_class: dict[str, list[Any]] = {}
        for row in rows:
            by_class.setdefault(str(row.gold), []).append(row)
        if len(by_class) < 2:
            # A split with one gold class cannot produce a discriminating accuracy (every answer
            # scores the majority baseline), so the template is dropped here with a reason
            # rather than exported and screened later.
            dropped_groups.append(f"{template_id}|{split}")
            report["templates"][f"{template_id}|{split}"] = {
                "k": 0,
                "n_before": len(rows),
                "n_after": 0,
                "n_classes": len(by_class),
                "before": {cls: len(v) for cls, v in sorted(by_class.items())},
                "after": {},
                "smallest_class_before": len(rows),
                "below_min_class_size": True,
                "single_class": True,
                "dropped": True,
                "drop_reason": "single_gold_class_in_split",
            }
            continue
        smallest = min(len(v) for v in by_class.values())
        k = min(target, smallest)
        before = {cls: len(v) for cls, v in sorted(by_class.items())}
        picked: list[Any] = []
        for cls, class_rows in sorted(by_class.items()):
            if len(class_rows) <= k:
                picked.extend(class_rows)
                continue
            rng = random.Random(f"{seed}:{salt}:{template_id}:{split}:{cls}")
            indices = sorted(rng.sample(range(len(class_rows)), k))
            picked.extend(class_rows[i] for i in indices)
        kept.extend(picked)
        report["templates"][f"{template_id}|{split}"] = {
            "k": k,
            "n_before": len(rows),
            "n_after": len(picked),
            "n_classes": len(by_class),
            "before": before,
            "after": {cls: k if before[cls] > k else before[cls] for cls in sorted(by_class)},
            "smallest_class_before": smallest,
            "below_min_class_size": smallest < floor,
            "single_class": False,
            "dropped": False,
        }
    report["dropped_groups"] = sorted(dropped_groups)
    report["n_dropped_single_class"] = sum(
        entry["n_before"] for key, entry in report["templates"].items()
        if entry.get("dropped") and not key.endswith("|test")
    )
    report["n_before_total"] = len(items)
    report["n_after_total"] = len(kept)
    return kept, report
