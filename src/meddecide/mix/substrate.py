"""The real-text substrate of the osler_v3 X3 generators: document splits and passage windows (ADVISORY X3).

A document's split carries over to every item made from it. Documents are split by a stable hash of their key (so a MultiCaRe
article, a PMC-Patients patient or a CodiEsp case never crosses splits): 80 % train, 10 % dev, 10 % test. A passage is a window of
one to four consecutive paragraphs, cut at paragraph boundaries and capped in length, chosen by stable hash.
"""

from __future__ import annotations

import re

from meddecide.utils.hashing import stable_hash

PASSAGE_MAX_CHARS = 3000
_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")


def split_of(doc_key: str) -> str:
    """``dev`` for one tenth of the keys, ``test`` for another tenth, ``train`` for the rest (stable hash)."""
    bucket = int(stable_hash({"substrate": str(doc_key)}, length=8), 16) % 10
    if bucket == 0:
        return "dev"
    if bucket == 1:
        return "test"
    return "train"


def paragraphs(text: str) -> list[str]:
    """The non-empty paragraphs of a document: blank-line breaks, or single newlines when there are none."""
    parts = _PARAGRAPH_BREAK.split(str(text).strip())
    if len(parts) == 1:
        parts = str(text).strip().split("\n")
    return [p.strip() for p in parts if p.strip()]


def passage(text: str, doc_key: str, *, max_chars: int = PASSAGE_MAX_CHARS) -> str | None:
    """A window of one to four consecutive paragraphs, at most ``max_chars`` long (cut at a paragraph boundary), or None when
    the document has no usable paragraph."""
    paras = paragraphs(text)
    if not paras:
        return None
    start = int(stable_hash({"window": str(doc_key), "field": "start"}, length=8), 16) % len(paras)
    length = 1 + int(stable_hash({"window": str(doc_key), "field": "length"}, length=8), 16) % 4
    chosen: list[str] = []
    for para in paras[start:start + length]:
        if sum(len(p) for p in chosen) + len(para) + (2 if chosen else 0) > max_chars:
            break
        chosen.append(para)
    if not chosen:
        return None
    return "\n\n".join(chosen)
