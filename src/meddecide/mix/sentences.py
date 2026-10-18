"""Sentence spans of a text, for the osler_v3 source converters (RAGTruth sentences, DrugProt same-sentence pairs).

A sentence ends after ``.``, ``!`` or ``?`` followed by whitespace; a newline is also a break. The spans are character offsets
into the text, trimmed of whitespace. The rule is deliberately simple and is the same in every converter that uses it.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def sentence_spans(text: str) -> list[tuple[int, int]]:
    """Character spans (start, end) of the sentences of ``text``, in order, whitespace trimmed; empty sentences are skipped."""
    spans: list[tuple[int, int]] = []
    for line_start, line in _lines(text):
        marks = [0]
        for match in _SENTENCE_END.finditer(line):
            marks.extend((match.start(), match.end()))
        marks.append(len(line))
        for begin, end in zip(marks[0::2], marks[1::2], strict=True):
            chunk = line[begin:end]
            stripped = chunk.strip()
            if stripped:
                lead = len(chunk) - len(chunk.lstrip())
                spans.append((line_start + begin + lead, line_start + begin + lead + len(stripped)))
    return spans


def sentence_index(spans: list[tuple[int, int]], offset: int) -> int | None:
    """The index of the sentence that contains character ``offset``, or None when it falls between sentences."""
    for index, (start, end) in enumerate(spans):
        if start <= offset < end:
            return index
    return None


def _lines(text: str) -> Iterable[tuple[int, str]]:
    offset = 0
    for line in text.split("\n"):
        yield offset, line
        offset += len(line) + 1
