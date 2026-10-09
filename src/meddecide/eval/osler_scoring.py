"""Scoring Osler models on the O2 common set (ADVISORY O11), as O2-format prediction rows.

The O2 common set (``data/bench/v0.3_ext/o2_items.jsonl``) is the item set every model is scored on, so an Osler row pairs
with each baseline's row by item id. ``o2_row_to_item`` builds the model's ``Item`` from a common-set row. The model reads the
state, the question, the options and the gold only. The benchmark's record date and source URL are not read by the model, so
a row that lacks them gets the placeholders below. The placeholders are a scoring-only convention and are never written to an
output: the prediction rows carry the item id, the template, the option keys and the probabilities.

Both-order scoring (ADVISORY O11): a ``choice`` item is scored in its original option order and in the reversed order, and each
option's probability is averaged over the two orders, mapped back to the original keys. ``noul`` and ``score`` items have a fixed
canonical order (yes then no; levels lowest first, see ``meddecide.eval.readout.canonicalise_options``), so reversing them does
not change the scoring and only one pass is run for them.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date
from typing import Any

import numpy as np

from meddecide.bench.schema import Item, Option, QuestionType, Split, Tier, make_item
from meddecide.eval.o2_io import prediction_row  # the O2 row shape, shared by every runner
from meddecide.model.meddecide_model import reorder_options

SCORING_URL = "https://example.org/meddecide-scoring-only"
PLACEHOLDER_DATE = date(1900, 1, 1)
SCORING_LICENCE = "unrecorded (scoring-only placeholder)"
ORDERED_TYPES = frozenset({QuestionType.CHOICE})


def _tier(meta: Mapping[str, Any]) -> Tier:
    """The row's tier when it has one; the external and robustness sets carry none, and their tier is not read by the model."""
    value = meta.get("tier")
    return Tier(value) if value is not None else Tier.ESTABLISHED


def o2_row_to_item(row: Mapping[str, Any]) -> Item:
    """The model's Item for one O2 common-set row. Option keys, gold and text are copied exactly.

    ``make_item`` derives an id from the identity fields, and that derivation reproduces the row's id for the v0.2 and panel
    rows but not for the perturbed robustness items. The row's own id is therefore set after construction, because the
    baselines are joined to Osler's rows by that id.
    """
    meta = row.get("meta") or {}
    options = [Option(key=o["key"], label=o["label"], **({"description": o["description"]} if o.get("description")
                                                        else {})) for o in row["options"]]
    source_url = row.get("source_url") or ""
    item = make_item(
        tier=_tier(meta),
        source=row["set_name"],
        source_record_id=str(row.get("source_record_id") or row["item_id"]),
        source_url=source_url if source_url.startswith(("http://", "https://")) else SCORING_URL,
        source_license=row.get("licence") or SCORING_LICENCE,
        record_date=PLACEHOLDER_DATE,
        split=Split(row["split"]) if row.get("split") in {s.value for s in Split} else Split.TEST,
        template_id=row["template_id"],
        skill=str(meta.get("skill") or row["template_id"]),
        qtype=QuestionType(row["qtype"]),
        state=row["state"],
        question=row["question"],
        options=options,
        gold=row["gold"],
        option_order_seed=0,
    )
    item_id = str(row["item_id"])
    if len(item_id) != len(item.item_id):
        raise ValueError(f"item id {item_id!r} does not have the benchmark id length {len(item.item_id)}")
    return item.model_copy(update={"item_id": item_id})


def reversed_order(item: Item) -> Item | None:
    """The item with its options in reverse display order (gold follows its content), or None when the order is canonical."""
    if item.qtype not in ORDERED_TYPES:
        return None
    return reorder_options(item, list(range(item.n_options))[::-1])


def score_rows(scorer: Any, rows: Sequence[Mapping[str, Any]], *, rows_per_call: int,
               **score_kwargs: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Score common-set rows in their original order and, for choice items, in the reversed order.

    ``scorer`` is anything with ``score_items(items, **kwargs)`` returning per-item ``probs`` (a MedDecideModel); the keyword
    arguments (for example ``batch_size``) go to it unchanged. ``rows_per_call`` is how many rows each call receives. Returns two
    lists of O2-format prediction rows, aligned one to one with ``rows``: the both-order averages (``orders`` 2 for choice
    items, 1 otherwise) and the single-order rows (the original order only, ``orders`` 1).
    """
    both: list[dict[str, Any]] = []
    single: list[dict[str, Any]] = []
    for start in range(0, len(rows), rows_per_call):
        chunk_rows = rows[start : start + rows_per_call]
        chunk = [o2_row_to_item(r) for r in chunk_rows]
        scored = scorer.score_items(chunk, **score_kwargs)
        reversed_items = [reversed_order(it) for it in chunk]
        reversed_pairs = [(i, rev) for i, rev in enumerate(reversed_items) if rev is not None]
        reversed_probs: dict[int, np.ndarray] = {}
        if reversed_pairs:
            scored_reversed = scorer.score_items([rev for _, rev in reversed_pairs], **score_kwargs)
            reversed_probs = {i: np.asarray(scored_reversed.probs[j], dtype=np.float64)
                              for j, (i, _) in enumerate(reversed_pairs)}
        for i, row in enumerate(chunk_rows):
            original = np.asarray(scored.probs[i], dtype=np.float64)
            latency = scored.latency_s[i] if getattr(scored, "latency_s", None) is not None else None
            tokens = scored.prompt_tokens[i] if getattr(scored, "prompt_tokens", None) is not None else None
            if i in reversed_probs:
                combined = average_both_orders(original, reversed_probs[i])
                orders = 2
            else:
                combined = original
                orders = 1
            single.append(prediction_row(row, status="scored", probs=[float(p) for p in original], latency_s=latency,
                                         prompt_tokens=tokens, extras={"orders": 1}))
            both.append(prediction_row(row, status="scored", probs=[float(p) for p in combined], latency_s=latency,
                                       prompt_tokens=tokens, extras={"orders": orders}))
    return both, single


def average_both_orders(original: np.ndarray, reversed_display: np.ndarray) -> np.ndarray:
    """Average the probabilities of the two orders, aligned to the original options.

    ``reversed_display[j]`` is the probability of the option shown at position j in the reversed order, which is original
    option ``n - 1 - j``. Returns the mean over the two orders, aligned to the original option positions.
    """
    original = np.asarray(original, dtype=np.float64)
    reversed_display = np.asarray(reversed_display, dtype=np.float64)
    if original.shape != reversed_display.shape or original.ndim != 1:
        raise ValueError("both orders must score the same options: one 1-D vector each")
    reversed_by_original = reversed_display[::-1]
    averaged = (original + reversed_by_original) / 2.0
    return averaged
