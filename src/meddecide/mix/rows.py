"""Training-row schema for osler_v0 mix v2 (O4).

A row has the same keys as a student_v1 training row, so the training code reads both without change. Every row is
built by ``make_row``, which checks the offered keys, the gold key and the state, and gives the row a stable id.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from meddecide.utils.hashing import stable_hash

WINDOW_START = date(2026, 3, 1)
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def make_row(*, source: str, template_id: str, skill: str, qtype: str, state: str, question: str,
             options: list[tuple[str, str]], gold: str, record_id: str, record_url: str, licence: str,
             record_date: str, split: str, tier: str, meta: dict[str, Any], variant: str = "") -> dict[str, Any]:
    keys = [k for k, _ in options]
    if len(keys) < 2 or len(set(keys)) != len(keys):
        raise ValueError(f"{template_id}/{record_id}: options must be two or more distinct keys")
    if gold not in keys:
        raise ValueError(f"{template_id}/{record_id}: gold {gold!r} is not an offered key")
    if not state.strip():
        raise ValueError(f"{template_id}/{record_id}: blank state")
    item_id = stable_hash({"source": source, "template": template_id, "record": record_id, "question": question,
                           "options": [list(o) for o in options], "gold": gold, "variant": variant}, length=16)
    return {
        "item_id": item_id, "tier": tier, "source": source, "source_record_id": record_id,
        "source_url": record_url, "source_license": licence, "record_date": record_date, "split": split,
        "template_id": template_id, "skill": skill, "qtype": qtype, "state": state, "question": question,
        "options": [{"key": k, "label": lab, "description": None} for k, lab in options],
        "gold": gold, "meta": dict(meta),
    }


def parse_pubhealth_date(text: str) -> str:
    """'April 26, 2015' -> '2015-04-26'. Raises ValueError for any other shape."""
    return datetime.strptime(text.strip(), "%B %d, %Y").date().isoformat()


def before_window(iso_date: str) -> bool:
    return date.fromisoformat(iso_date) < WINDOW_START
