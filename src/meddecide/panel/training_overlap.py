"""Training-overlap check for dev and CG candidates (osler_v4 Y4).

A candidate item is a training match when (a) its text (state, question and option labels) occurs in a training row, or (b)
its source record id occurs in a training row. Training input is every .jsonl under data/train/ except the dev.jsonl files:
a dev file is a dev input, not training (the first version of this check counted osler_v0's dev file and reported dev against
dev as training). A training row is compared on text only when its normalised question equals a candidate's question (an
exact pre-filter that does not change the result).
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from meddecide.panel.overlap import row_text_key, text_key
from meddecide.utils.hashing import normalize_text

TRAINING_ROOT = Path("data/train")


def training_files(root: Path = TRAINING_ROOT) -> list[Path]:
    """Every .jsonl under ``root`` except files named dev.jsonl, in a stable order."""
    return sorted(p for p in root.rglob("*.jsonl") if p.name != "dev.jsonl")


@dataclass
class Match:
    text: bool = False
    record: bool = False
    files: set[str] = field(default_factory=set)


def candidate_text_key(row: Mapping[str, Any]) -> str:
    return text_key(row["state"], row["question"], [o["label"] for o in row["options"]])


def match_training(candidates: Iterable[Mapping[str, Any]], paths: Iterable[Path]) -> dict[str, Match]:
    """item_id -> Match, for every candidate with a text or record match in the given training files."""
    by_text: dict[str, list[str]] = defaultdict(list)
    by_record: dict[str, list[str]] = defaultdict(list)
    questions: set[str] = set()
    for row in candidates:
        item_id = str(row["item_id"])
        by_text[candidate_text_key(row)].append(item_id)
        questions.add(normalize_text(str(row["question"])))
        if row.get("source_record_id"):
            by_record[str(row["source_record_id"])].append(item_id)
    found: dict[str, Match] = {}
    for path in paths:
        name = str(path)
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if normalize_text(str(row.get("question", ""))) in questions:
                    for item_id in by_text.get(row_text_key(row), []):
                        found.setdefault(item_id, Match()).text = True
                        found[item_id].files.add(name)
                rid = row.get("source_record_id")
                if rid is not None and str(rid) in by_record:
                    for item_id in by_record[str(rid)]:
                        found.setdefault(item_id, Match()).record = True
                        found[item_id].files.add(name)
    return found
