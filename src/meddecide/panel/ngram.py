"""Word n-gram near-duplicate check against the external panel (osler_v4 Y4, ADVISORY §2: 13-gram, all seven panel datasets).

A candidate is a near-duplicate when any run of ``n`` consecutive normalised tokens of its text (state, question and option
labels) also occurs in the panel. Tokens are lowercase alphanumeric runs. A text with fewer than ``n`` tokens cannot hit.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any

TOKEN = re.compile(r"[a-z0-9]+")
DEFAULT_N = 13


def tokens(text: str) -> list[str]:
    return TOKEN.findall(text.lower())


def ngrams(text: str, n: int = DEFAULT_N) -> set[tuple[str, ...]]:
    toks = tokens(text)
    return {tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)}


def item_text(row: Mapping[str, Any]) -> str:
    """The text the check compares: state, question and the option labels (never the gold key alone)."""
    labels = " ".join(str(o.get("label", "")) for o in row.get("options", []))
    return f"{row.get('state', '')} {row.get('question', '')} {labels}"


def build_index(rows: Iterable[Mapping[str, Any]], n: int = DEFAULT_N) -> set[tuple[str, ...]]:
    index: set[tuple[str, ...]] = set()
    for row in rows:
        index |= ngrams(item_text(row), n)
    return index


def hits(row: Mapping[str, Any], index: set[tuple[str, ...]], n: int = DEFAULT_N) -> int:
    """How many n-grams of the row occur in the index (0 means no near-duplicate)."""
    return len(ngrams(item_text(row), n) & index)
