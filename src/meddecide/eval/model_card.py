"""Model-card examples for smoke tests (osler_v0 O0).

A model card's ```python blocks are run in order by a smoke script; each block's comment lines
carry the card's printed result (``# Yes  0.000``) and, optionally, its gold label
(``# correct: No``). The functions here parse those comments and compare them with what the code
returned. They do not execute anything.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass

BLOCK = re.compile(r"```python\n(.*?)```", re.S)
PRINTED = re.compile(r"^#\s+(.+?)\s+(\d\.\d{3})\s*$")
CORRECT = re.compile(r"^#\s*correct:\s*(.+?)\s*$")


@dataclass(frozen=True)
class CardBlock:
    index: int
    code: str
    printed: dict[str, float]
    correct: str | None


@dataclass(frozen=True)
class PrintedComparison:
    """Per-option comparison of the card's printed probabilities with the returned ones."""

    diffs: dict[str, dict[str, float | None]]
    missing: list[str]
    worst_abs_diff: float | None


def card_blocks(text: str) -> list[CardBlock]:
    """Every ```python block of a model card, with its printed values and gold label."""
    blocks = []
    for index, match in enumerate(BLOCK.finditer(text)):
        code = match.group(1)
        printed: dict[str, float] = {}
        correct: str | None = None
        for line in code.splitlines():
            printed_match = PRINTED.match(line)
            if printed_match:
                printed[printed_match.group(1).strip()] = float(printed_match.group(2))
            correct_match = CORRECT.match(line)
            if correct_match:
                correct = correct_match.group(1).strip()
        blocks.append(CardBlock(index=index, code=code, printed=printed, correct=correct))
    return blocks


def compare_printed(returned: Mapping[str, float], printed: Mapping[str, float]) -> PrintedComparison:
    """Match each printed label to a returned option (exact or prefix) and take the absolute gap.

    A printed label with no matching returned option is listed in ``missing``, never skipped.
    """
    diffs: dict[str, dict[str, float | None]] = {}
    missing: list[str] = []
    for label, expected in printed.items():
        matches = [key for key in returned if str(key).strip() == label or str(key).startswith(label)]
        if not matches:
            diffs[label] = {"expected": expected, "got": None, "abs_diff": None}
            missing.append(label)
            continue
        got = float(returned[matches[0]])
        diffs[label] = {"expected": expected, "got": got, "abs_diff": round(abs(got - expected), 4)}
    gaps = [d["abs_diff"] for d in diffs.values() if d["abs_diff"] is not None]
    return PrintedComparison(diffs=diffs, missing=missing,
                             worst_abs_diff=max(gaps) if gaps else None)


def argmax_label(values: Mapping[str, float]) -> str | None:
    """The key with the largest value, or None for an empty mapping."""
    return max(values, key=lambda k: values[k]) if values else None
