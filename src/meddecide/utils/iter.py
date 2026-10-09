"""Iteration helpers shared by the report scripts."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any


def iter_jsonl_dir(path: Path) -> Iterator[dict[str, Any]]:
    """Yield JSON rows from one JSONL file, skipping blank lines."""
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)
