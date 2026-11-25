"""Leakage keys for the external panel (osler_v0 O1 acceptance: overlap check must be 0).

Two keys per item: a normalised-text key over the state, the question and the option labels, and the
(set, record id) pair. Training rows from any mix are reduced to the same keys and compared.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from meddecide.panel.schema import EvalItem
from meddecide.utils.hashing import normalize_text, stable_hash


def text_key(state: str, question: str, labels: Iterable[str]) -> str:
    """Normalised-text key: NFC, whitespace-collapsed, case kept (case carries meaning in medicine)."""
    body = "\n".join([normalize_text(state), normalize_text(question),
                      "|".join(normalize_text(label) for label in labels)])
    return stable_hash(body, length=32)


def item_text_key(item: EvalItem) -> str:
    return text_key(item.state, item.question, [o.label for o in item.options])


def row_text_key(row: Mapping[str, Any]) -> str:
    """Key of a training row (student mixes store options as dicts with a ``label``)."""
    labels = [o["label"] for o in row["options"]]
    return text_key(str(row["state"]), str(row["question"]), labels)


@dataclass
class OverlapReport:
    """Counts per panel set: exact text matches and record matches against the scanned training rows."""

    scanned_rows: int = 0
    text_hits: dict[str, int] = field(default_factory=dict)
    record_hits: dict[str, int] = field(default_factory=dict)
    matched_text_items: list[str] = field(default_factory=list)

    def total_text_hits(self) -> int:
        return sum(self.text_hits.values())


class TrainingKeys:
    """Sets of training text keys and (source, record id) keys, built once from training rows."""

    def __init__(self) -> None:
        self.text: set[str] = set()
        self.records: set[tuple[str, str]] = set()
        self.rows = 0

    def add_row(self, row: Mapping[str, Any]) -> None:
        self.text.add(row_text_key(row))
        self.records.add((str(row.get("source", "")), str(row.get("source_record_id", ""))))
        self.rows += 1

    def compare(self, items: Iterable[EvalItem]) -> OverlapReport:
        report = OverlapReport(scanned_rows=self.rows)
        for item in items:
            if item_text_key(item) in self.text:
                report.text_hits[item.set_name] = report.text_hits.get(item.set_name, 0) + 1
                report.matched_text_items.append(item.item_id)
            if (item.set_name, item.source_record_id) in self.records:
                report.record_hits[item.set_name] = report.record_hits.get(item.set_name, 0) + 1
        return report
