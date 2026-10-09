"""Convert the JEV-9B decision-head rows (F7 format) into the harness prediction format.

F7 (``scripts/bench/run_jev_baseline.py``) writes one row per item with ``probs`` in *head slot*
order: ``noul`` is ``[false, true]``, ``score`` is digit order (= the item's option order) and
``choice`` is the offered-option order. The harness format names options by presentation letter
(``option_keys``) with the item's own labels in ``original_option_keys``, so G1 can pair rows with
other models. ``noul`` rows are re-keyed by the item's yes/no labels, never by slot position.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from meddecide.bench.schema import Item, QuestionType

VERBALIZER_VARIANT = "verbalizer-head"
LETTERS = "ABCDEFGHIJKLMNOP"


def _argmax(values: list[float]) -> int:
    return max(range(len(values)), key=lambda i: (values[i], -i))


def option_probabilities(item: Item, slot_probs: list[float]) -> list[float]:
    """Probabilities in the item's option order, from the head's slot order."""
    native = [o.key for o in item.options]
    if item.qtype is QuestionType.NOUL:
        if sorted(native) != ["no", "yes"]:
            raise ValueError(f"noul item {item.item_id} has option keys {native}, expected yes/no")
        if len(slot_probs) != 2:
            raise ValueError(f"noul item {item.item_id}: head returned {len(slot_probs)} slots")
        by_key = {"no": slot_probs[0], "yes": slot_probs[1]}
        return [by_key[key] for key in native]
    if len(slot_probs) != len(native):
        raise ValueError(
            f"{item.qtype} item {item.item_id}: head returned {len(slot_probs)} slots "
            f"for {len(native)} options"
        )
    return list(slot_probs)


def harness_row(f7: Mapping[str, Any], item: Item, *, run_id: str, model_id: str) -> dict[str, Any]:
    """One harness-format prediction row, with ``correct`` checked against the F7 row."""
    if f7["item_id"] != item.item_id:
        raise ValueError(f"F7 row {f7['item_id']} paired with item {item.item_id}")
    native = [o.key for o in item.options]
    values = option_probabilities(item, [float(p) for p in f7["probs"]])
    argmax = _argmax(values)
    gold_position = native.index(item.gold)
    correct = argmax == gold_position
    if correct != bool(f7["correct"]):
        raise ValueError(
            f"item {item.item_id}: harness correct={correct} disagrees with F7 correct={f7['correct']}"
        )
    keys = [LETTERS[i] for i in range(len(native))]
    return {
        "item_id": item.item_id,
        "model_id": model_id,
        "run_id": run_id,
        "source": item.source,
        "template_id": item.template_id,
        "split": str(item.split),
        "qtype": str(item.qtype),
        "option_keys": keys,
        "original_option_keys": native,
        "option_probs": values,
        "label_mass": float(f7["label_mass"]),
        "argmax_key": keys[argmax],
        "gold_key": keys[gold_position],
        "correct": correct,
        "variant": VERBALIZER_VARIANT,
        "transform": {"readout": VERBALIZER_VARIANT},
        "prompt_tokens": int(f7["prompt_tokens"]),
    }
