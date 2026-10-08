"""MedDecide items as Unsloth decision rows (student_v1 V5 and V8).

Pure functions over plain dicts (the JSON form of a MedDecide item), so the same code runs in the
main environment and in ``envs/unsloth`` where ``pydantic`` may not be installed.

Unsloth's decision API reads a row as ``{"state", "questions", "gold"}``:

* ``questions`` maps a name to ``{"type", "instructions", "criteria"}``. ``choice`` needs a dict from
  option key to label; ``score`` needs an ordered list of level labels; ``noul`` takes no criteria.
* ``gold`` maps the same name to ``{"label": ...}``. For ``choice`` the label is the option key; for
  ``noul`` it is ``"true"`` or ``"false"``; for ``score`` it is the zero-based level index, because the
  API numbers score levels from 0 (``_option_keys`` in ``unsloth/models/decision.py``).

Both mappings are checked in the round-trip script (``scripts/student/v5_roundtrip.py``) by counting
the rows Unsloth's ``build_dataset`` skips; the expected count is zero.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

CHOICE = "choice"
SCORE = "score"
NOUL = "noul"
NAME = "q"


class ConversionError(ValueError):
    """An item that cannot be expressed as an Unsloth row; ``reason`` is a fixed code for counting."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def to_unsloth_row(item: Mapping[str, Any], *, name: str = NAME) -> dict[str, Any]:
    """One MedDecide item as an Unsloth row with a single question called ``name``."""
    qtype = str(item["qtype"])
    instructions = str(item["question"])
    options = list(item.get("options") or [])
    gold = str(item["gold"])
    if qtype == CHOICE:
        criteria = {str(o["key"]): str(o["label"]) for o in options}
        if len(criteria) != len(options) or len(criteria) < 2:
            raise ConversionError("choice_needs_two_unique_keys")
        if gold not in criteria:
            raise ConversionError("choice_gold_not_an_option")
        question: dict[str, Any] = {"type": CHOICE, "instructions": instructions, "criteria": criteria}
        gold_value: dict[str, Any] = {"label": gold}
    elif qtype == SCORE:
        ordered = sorted(options, key=lambda o: int(o["key"]))
        levels = [str(o["label"]) for o in ordered]
        if len(levels) < 2:
            raise ConversionError("score_needs_two_levels")
        level = int(gold) - 1
        if not 0 <= level < len(levels):
            raise ConversionError("score_gold_outside_levels")
        question = {"type": SCORE, "instructions": instructions, "criteria": levels}
        gold_value = {"label": str(level)}
    elif qtype == NOUL:
        if gold not in ("yes", "no"):
            raise ConversionError("noul_gold_not_yes_no")
        question = {"type": NOUL, "instructions": instructions}
        gold_value = {"label": "true" if gold == "yes" else "false"}
    else:
        raise ConversionError("unsupported_question_type")
    return {
        "state": str(item["state"]),
        "questions": {name: question},
        "gold": {name: gold_value},
        "item_id": str(item.get("item_id", "")),
        "template_id": str(item.get("template_id", "")),
    }


def to_unsloth_rows(items: list[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Convert many items; returns the rows and a count of items refused, by reason."""
    rows: list[dict[str, Any]] = []
    refused: dict[str, int] = {}
    for item in items:
        try:
            rows.append(to_unsloth_row(item))
        except ConversionError as exc:
            refused[exc.reason] = refused.get(exc.reason, 0) + 1
    return rows, refused


def dumps(row: Mapping[str, Any]) -> str:
    return json.dumps(row, ensure_ascii=False)
