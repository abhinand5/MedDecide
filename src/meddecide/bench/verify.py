"""Independent gold verification for tier-1 items (task F1).

Integrity audits (no split leaks, no duplicate ids) say nothing about whether a gold label is
*correct*. This module recomputes, for every item, the gold option **text** straight from the
raw source record using a minimal mapping written from the dataset card — deliberately not
reusing the loader's code path, so a loader bug cannot hide itself.

The mapping per source is deliberately explicit and small:

* **MedQA** — `answer_idx` (letter) selects from `options`; gold text is that option's value.
* **MedMCQA** — `cop` is the 0-based index into `opa..opd` (proven separately against `exp`).
* **PubMedQA** — `final_decision` (`yes`/`no`/`maybe`) is the gold label text.
* **MMLU** — `answer` is the 0-based index into `choices`.
* **MedQuAD** — `question_type` is the gold label text.
* **Relevance sources** — the gold is a property of the qrels (relevant / not), not of a text
  field, so verification recomputes relevance membership from the qrels and checks the item's
  gold against it; `score` items are checked against the qrels grade.

A mismatch is a hard failure: it means the benchmark's answer disagrees with the source.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class VerificationReport:
    """Result of comparing a built item set against independently recomputed gold."""

    source: str
    n_items: int = 0
    n_checked: int = 0
    n_mismatched: int = 0
    n_unverifiable: int = 0
    mismatches: list[dict[str, Any]] = field(default_factory=list)
    gold_class_counts: Counter[str] = field(default_factory=Counter)
    raw_class_counts: Counter[str] = field(default_factory=Counter)
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.n_mismatched == 0

    def add_mismatch(self, item_id: str, record_id: str, expected: str, found: str) -> None:
        self.n_mismatched += 1
        if len(self.mismatches) < 25:
            self.mismatches.append(
                {"item_id": item_id, "record_id": record_id, "expected_gold_text": expected,
                 "loader_gold_text": found}
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "n_items": self.n_items,
            "n_checked": self.n_checked,
            "n_mismatched": self.n_mismatched,
            "n_unverifiable": self.n_unverifiable,
            "ok": self.ok,
            "gold_class_counts": dict(sorted(self.gold_class_counts.items())),
            "raw_class_counts": dict(sorted(self.raw_class_counts.items())),
            "mismatch_examples": self.mismatches,
            "notes": self.notes,
        }


def normalize_label(text: str) -> str:
    """Whitespace/Unicode-normalised comparison key for label text."""
    import unicodedata

    return " ".join(unicodedata.normalize("NFKC", str(text)).split()).casefold()


def check_items_against_raw(
    source: str,
    items: list[Any],
    raw_by_id: dict[str, Any],
    expected_key: Callable[[Any, Any], str | None],
    *,
    expected_text: Callable[[Any, Any], str | None] | None = None,
) -> VerificationReport:
    """Compare every item's gold **key** with the key recomputed from the raw record.

    ``expected_key(raw, item)`` is the independent mapping from a raw record to the option key
    that must be gold. The key is what the benchmark stores, and for the relevance sources the
    key is a grade ("1"/"2"/"3") or a yes/no — comparing *label text* there is wrong and
    produced 549 false mismatches on the first run of this checker. ``expected_text`` is
    optional and only used to compare the gold option's text as a second, independent check.

    Items whose record is absent from ``raw_by_id``, or for which the mapping returns ``None``,
    are counted as unverifiable, never as passes.
    """
    report = VerificationReport(source=source, n_items=len(items))
    missing_raw = 0
    unmapped = 0
    text_checked = 0
    text_mismatches = 0
    for item in items:
        key = str(item.source_record_id)
        raw = raw_by_id.get(key)
        report.gold_class_counts[normalize_label(item.gold)] += 1
        if raw is None:
            missing_raw += 1
            report.n_unverifiable += 1
            continue
        want_key = expected_key(raw, item)
        if want_key is None:
            unmapped += 1
            report.n_unverifiable += 1
            continue
        report.raw_class_counts[normalize_label(want_key)] += 1
        report.n_checked += 1
        if normalize_label(item.gold) != normalize_label(want_key):
            report.add_mismatch(item.item_id, key, str(want_key), str(item.gold))
            continue
        if expected_text is not None:
            want_text = expected_text(raw, item)
            if want_text is not None:
                text_checked += 1
                if normalize_label(item.options[item.gold_index].label) != normalize_label(want_text):
                    text_mismatches += 1
                    report.add_mismatch(
                        item.item_id, key, f"text:{want_text}",
                        item.options[item.gold_index].label,
                    )
    if missing_raw:
        report.notes.append(f"{missing_raw} item(s) had no raw record with the expected id")
    if unmapped:
        report.notes.append(f"{unmapped} raw record(s) could not be mapped to a gold key")
    if expected_text is not None:
        report.notes.append(
            f"gold option text compared for {text_checked} item(s); {text_mismatches} text mismatch(es)"
        )
    report.notes.append(
        "gold-key distribution compared against the raw answer-field distribution: "
        f"{len(report.gold_class_counts)} classes in items, {len(report.raw_class_counts)} in raw"
    )
    return report
