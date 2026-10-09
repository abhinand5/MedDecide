"""Deterministic hashing helpers.

Item and template ids must be stable across machines and processes: same logical input →
same id. Everything here is pure and locale-independent.
"""

from __future__ import annotations

import hashlib
import json
import unicodedata
from pathlib import Path
from typing import Any


def normalize_text(value: str) -> str:
    """NFC-normalise and collapse insignificant whitespace.

    Applied before hashing so that visually identical inputs hash identically.
    """
    return " ".join(unicodedata.normalize("NFC", value).split())


def stable_hash(payload: dict[str, Any] | list[Any] | str, *, length: int = 16) -> str:
    """Hex digest of the canonical JSON encoding of ``payload``.

    Keys are sorted and text is NFC-normalised first, so the digest does not depend on
    dict insertion order, platform, or Unicode composition.
    """

    def canon(obj: Any) -> Any:
        if isinstance(obj, str):
            return normalize_text(obj)
        if isinstance(obj, dict):
            return {str(k): canon(v) for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))}
        if isinstance(obj, (list, tuple)):
            return [canon(v) for v in obj]
        return obj

    blob = json.dumps(canon(payload), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:length]


def file_sha256(path: Path, *, chunk_size: int = 1 << 20) -> str:
    """Streaming sha256 of a file's bytes."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        while chunk := fh.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()
