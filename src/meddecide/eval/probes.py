"""Item transforms used by the harness probes and by the recorded run variants.

Every transform is deterministic given a seed and returns the transform's parameters so
that a rerun can be reconstructed exactly (R7). None of them changes gold.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from meddecide.bench.schema import Item, Option, QuestionType
from meddecide.utils.hashing import stable_hash

# Extra options the candidate-count probe can add. They must never be the gold answer.
FILLER_OPTIONS = [
    ("None of these apply", "no matching category is described"),
    ("Insufficient information to classify", "the record does not state the field"),
    ("Not reported in the source record", "the source omits this field"),
    ("Unknown at time of posting", "the value is not final"),
    ("Not applicable to this record type", "the field does not apply"),
    ("Withheld by the sponsor", "the value is redacted"),
    ("Pending reclassification", "the value is under review"),
    ("Other, not elsewhere classified", "residual category"),
    ("Not stated in the record", "the source does not report this field"),
    ("Not assessed", "the field was not evaluated"),
    ("Not collected for this study", "the protocol omits the field"),
    ("No matching option", "none of the listed options applies"),
    ("Not determined", "the value could not be established"),
    ("Not available", "the value is unavailable"),
    ("Unclassified", "the record does not fall into a listed category"),
    ("Cannot be determined from the record", "insufficient information"),
    ("Not specified by the sponsor", "the sponsor did not state it"),
    ("Outside the stated categories", "the answer is not among the options"),
]
NONE_OF_THE_ABOVE_LABEL = "None of the above"


@dataclass(frozen=True)
class TransformRecord:
    """What a transform did, so the run can be reconstructed."""

    name: str
    seed: int
    details: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "seed": self.seed, "details": self.details}


def _rekey(options: list[Option], *, lead: str) -> list[Option]:
    """Re-assign option keys in display order, preserving content."""
    keys = (
        [chr(ord("A") + i) for i in range(len(options))]
        if lead == "A"
        else [str(i + 1) for i in range(len(options))]
    )
    return [Option(key=k, label=o.label, description=o.description) for k, o in zip(keys, options, strict=True)]


def shuffle_options(item: Item, *, seed: int, salt: str = "") -> tuple[Item, TransformRecord]:
    """Permute option display order; gold follows its content (never its position).

    The permuted item gets a new ``item_id`` because the option order is part of the id, so
    the two variants can never be confused in a prediction file. ``meta.shuffle`` records
    the permutation so the run can be reconstructed.
    """
    seed_value = int(stable_hash({"item": item.item_id, "seed": seed, "salt": salt}, length=8), 16)
    rng = random.Random(seed_value)
    order = list(range(item.n_options))
    rng.shuffle(order)
    gold_label = item.options[item.gold_index].label
    reordered = [item.options[i] for i in order]
    transformed = item.model_copy(
        update={
            "item_id": stable_hash(
                {"item_id": item.item_id, "perm": order, "seed": seed, "salt": salt}, length=16
            ),
            "options": _rekey(reordered, lead="1" if item.qtype is QuestionType.SCORE else "A"),
            "meta": {**item.meta, "shuffle": {"seed": seed, "salt": salt, "perm": order}},
        }
    )
    new_gold_index = next(i for i, o in enumerate(transformed.options) if o.label == gold_label)
    transformed = transformed.model_copy(update={"gold": transformed.options[new_gold_index].key})
    return transformed, TransformRecord(
        name="option_shuffle", seed=seed, details={"perm": order, "source_item_id": item.item_id}
    )


def scale_candidates(
    item: Item,
    *,
    target_n: int,
    seed: int,
    salt: str = "",
) -> tuple[Item | None, TransformRecord | None]:
    """Extend a ``choice`` item to ``target_n`` options with deterministic filler options.

    Returns ``(None, None)`` when the item cannot be extended (``noul``/``score`` items, or
    a target smaller than the item's own option count) — the caller records that as
    unsupported rather than inventing options. Gold is never one of the fillers.
    """
    if item.qtype is not QuestionType.CHOICE:
        return None, None
    if target_n < item.n_options or target_n > 255:
        return None, None
    n_filler = target_n - item.n_options
    if n_filler == 0:
        return item, TransformRecord(name="candidate_count", seed=seed,
                                     details={"target_n": target_n, "n_filler": 0})
    if n_filler > len(FILLER_OPTIONS):
        return None, None
    seed_value = int(stable_hash({"item": item.item_id, "seed": seed, "salt": salt}, length=8), 16)
    rng = random.Random(seed_value)
    fillers = rng.sample(FILLER_OPTIONS, n_filler)
    gold_label = item.options[item.gold_index].label
    # placeholder keys are re-assigned below; they must still be schema-valid
    combined = [
        *item.options,
        *[Option(key="Z", label=label, description=desc) for label, desc in fillers],
    ]
    order = list(range(len(combined)))
    rng.shuffle(order)
    reordered = _rekey([combined[i] for i in order], lead="A")
    transformed = item.model_copy(
        update={
            "item_id": stable_hash(
                {"item_id": item.item_id, "n": target_n, "perm": order, "seed": seed, "salt": salt},
                length=16,
            ),
            "options": reordered,
            "meta": {
                **item.meta,
                "candidate_scale": {
                    "seed": seed, "salt": salt, "target_n": target_n,
                    "perm": order, "fillers": [f[0] for f in fillers],
                },
            },
        }
    )
    new_gold_index = next(i for i, o in enumerate(transformed.options) if o.label == gold_label)
    transformed = transformed.model_copy(update={"gold": transformed.options[new_gold_index].key})
    return transformed, TransformRecord(
        name="candidate_count",
        seed=seed,
        details={"target_n": target_n, "n_filler": n_filler, "fillers": [f[0] for f in fillers]},
    )


def none_of_the_above(item: Item, *, seed: int, salt: str = "") -> tuple[Item, TransformRecord]:
    """Variant with the gold option removed and ``None of the above`` added.

    The item is *abstention-required*: the correct answer is the added option, and the
    original gold is gone. The transform is only meaningful for items whose labels are not
    already "none of the above".
    """
    gold_label = item.options[item.gold_index].label
    kept = [o for i, o in enumerate(item.options) if i != item.gold_index]
    order = list(range(len(kept)))
    seed_value = int(stable_hash({"item": item.item_id, "seed": seed, "salt": salt}, length=8), 16)
    rng = random.Random(seed_value)
    rng.shuffle(order)
    reordered = [kept[i] for i in order]
    keys = [chr(ord("A") + i) for i in range(len(reordered) + 1)]
    options = [Option(key=keys[i], label=o.label) for i, o in enumerate(reordered)]
    options.append(Option(key=keys[-1], label=NONE_OF_THE_ABOVE_LABEL))
    transformed = item.model_copy(
        update={
            "item_id": stable_hash(
                {"item_id": item.item_id, "nota": True, "perm": order, "seed": seed}, length=16
            ),
            "options": options,
            "gold": keys[-1],
            "meta": {
                **item.meta,
                "none_of_the_above": {
                    "removed_label": gold_label, "perm": order, "seed": seed, "salt": salt,
                },
            },
        }
    )
    return transformed, TransformRecord(
        name="none_of_the_above",
        seed=seed,
        details={"removed_label": gold_label, "perm": order},
    )


def detect_abstention(
    nota_probabilities: dict[str, list[float]],
    nota_gold_key: dict[str, str],
    nota_option_keys: dict[str, list[str]],
) -> dict[str, float]:
    """Detection rate on abstention items: did the model pick the added option?

    ``nota_probabilities`` maps item id → option probabilities for the modified item,
    ``nota_gold_key`` maps item id → the key of the added option in that item, and
    ``nota_option_keys`` maps item id → the option keys in display order (so the gold key can
    be located without assuming it is last). The false-rejection rate needs the *unmodified*
    items with the added option present, which is a different run; it is reported separately
    by the caller when that run exists, and `NOT MEASURED` otherwise.
    """
    if not nota_probabilities:
        raise ValueError("no abstention items")
    detections = 0
    for item_id, probs in nota_probabilities.items():
        gold_key = nota_gold_key[item_id]
        keys = nota_option_keys[item_id]
        gold_index = keys.index(gold_key)
        if probs.index(max(probs)) == gold_index:
            detections += 1
    return {
        "n_abstention_items": len(nota_probabilities),
        "detection_rate": detections / len(nota_probabilities),
    }
