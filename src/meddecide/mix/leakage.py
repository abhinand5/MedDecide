"""Leakage checks for training mix v2 (osler_v0 O4).

A training or dev row leaks when its normalised text (state, question and option labels) or its source record id equals
one in a protected set: the v0.2 test and dev items, the external panel, the robustness pack, the held-out generators,
and the held-out templates (D14). Counts are reported by kind; a non-zero count is a failure, not a filter (R4, R8).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from meddecide.panel.overlap import text_key


def row_text_key(row: Mapping[str, Any]) -> str:
    return text_key(row["state"], row["question"], [o["label"] for o in row["options"]])


def leak_counts(rows: Iterable[Mapping[str, Any]], protected_text: Mapping[str, set[str]],
                protected_records: Mapping[str, set[str]]) -> dict[str, int]:
    """Counts of rows whose text key or record id is in each protected set. Keys look like ``text:<name>`` and
    ``record:<name>``. A row can count under several kinds; every kind is reported."""
    counts: dict[str, int] = {}
    for name in protected_text:
        counts[f"text:{name}"] = 0
    for name in protected_records:
        counts[f"record:{name}"] = 0
    for row in rows:
        key = row_text_key(row)
        for name, keys in protected_text.items():
            if key in keys:
                counts[f"text:{name}"] += 1
        rec = str(row["source_record_id"])
        for name, ids in protected_records.items():
            if rec in ids:
                counts[f"record:{name}"] += 1
    return counts


def date_violations(rows: Iterable[Mapping[str, Any]], window_start: str) -> int:
    """Rows dated on or after the window start. Undated rows (empty record_date) are counted separately by the caller."""
    return sum(1 for r in rows if r["record_date"] and r["record_date"] >= window_start)
