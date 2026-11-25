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


# ---------------------------------------------------------------------------
# Contamination probes (T4/F5)
# ---------------------------------------------------------------------------
def min_k_prob(logprobs: list[float], k_fraction: float = 0.20) -> float:
    """Min-K%-Prob memorisation score: the mean log-probability of the least likely K% of tokens.

    A model that memorised a string assigns its tokens higher probability than a model seeing it
    for the first time, and the *tail* of the token distribution is the sensitive part: the
    easiest 80 % of tokens are predictable from general language statistics, so averaging over all
    tokens washes the signal out. This is the K=20 % variant from the T4 spec.

    ``logprobs`` must be the per-token log-probabilities of the text under the model, one per
    predicted token (the first token of the text has no predecessor and is excluded by the
    caller). Higher (less negative) means "more memorised".
    """
    if not logprobs:
        raise ValueError("logprobs must be non-empty")
    if not 0.0 < k_fraction <= 1.0:
        raise ValueError("k_fraction must be in (0, 1]")
    ordered = sorted(logprobs)
    n_tail = max(1, round(len(ordered) * k_fraction))
    return float(sum(ordered[:n_tail]) / n_tail)


def reorder_control(text: str, *, seed: int, salt: str = "") -> tuple[str, TransformRecord]:
    """Deterministic control paraphrase with three escalating granularities.

    Tier-1 items are often one or two sentences, so sentence reordering alone leaves many items
    without a control. The control therefore tries, in order:

    1. **sentences** — permute sentences (best: destroys verbatim sequence, keeps words);
    2. **clauses** — permute comma/semicolon-separated clauses within the text (medium);
    3. **lines** — permute non-empty lines (weakest, used when a record is a list of fields).

    The granularity actually used is recorded per item (``details.kind``), because a control built
    from lines is a weaker test than one built from sentences and the report must say which was
    used. Text that admits none of the three is returned unchanged with ``applied: False`` and is
    excluded from the gap rather than scored against itself.
    """
    import re

    def _permute(parts: list[str]) -> tuple[list[str], list[int]]:
        rng = random.Random(f"{seed}:{salt}")
        order = list(range(len(parts)))
        for _ in range(20):
            rng.shuffle(order)
            if order != list(range(len(parts))):
                break
        return [parts[i] for i in order], order

    sentences = [x for x in re.split(r"(?<=[.!?])\s+", text) if x.strip()]
    if len(sentences) >= 2:
        parts, order = _permute(sentences)
        return " ".join(parts), TransformRecord(
            name="reorder_control", seed=seed,
            details={"kind": "sentences", "n_units": len(sentences), "applied": True, "order": order},
        )

    clauses = [x for x in re.split(r"(?<=[,;:])\s+", text) if x.strip()]
    if len(clauses) >= 2:
        parts, order = _permute(clauses)
        return " ".join(parts), TransformRecord(
            name="reorder_control", seed=seed,
            details={"kind": "clauses", "n_units": len(clauses), "applied": True, "order": order},
        )

    lines = [x for x in text.splitlines() if x.strip()]
    if len(lines) >= 2:
        parts, order = _permute(lines)
        return "\n".join(parts), TransformRecord(
            name="reorder_control", seed=seed,
            details={"kind": "lines", "n_units": len(lines), "applied": True, "order": order},
        )

    return text, TransformRecord(
        name="reorder_control", seed=seed,
        details={"kind": "none", "n_units": 1, "applied": False,
                 "reason": "text has fewer than two sentences, clauses and lines"},
    )


def sentence_reorder(text: str, *, seed: int, salt: str = "") -> tuple[str, TransformRecord]:
    """Sentence-only reordering (kept for callers that need exactly that transform)."""
    return reorder_control(text, seed=seed, salt=salt)
