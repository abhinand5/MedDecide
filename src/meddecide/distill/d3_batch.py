"""Batched text scoring through a d3 checkpoint's own runtime (d3_distill step 0 and labelling).

One request still holds one question, rendered and tokenized by ``d3_runtime.D3.prepare`` exactly as
``system_one`` does it (including the ``prefix_bidir`` state-block count). The batch is formed over
*requests*: their prompt token sequences are left-padded and ``D3.probabilities`` runs the model's own
block mask, forward pass, FP32 readout and softmax on the batch. This module adds only the batch planning
(length sort under a padded-token budget) and the probability checks; it re-implements no model step.

The teacher's ``decision_config`` declares ``temperature_by_questions`` with 1.046 for a one-question
request, so the caller must hold ``D3.request_temperature(1)`` around a batch of one-question requests.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

PROB_TOLERANCE = 1e-3


@dataclass(frozen=True)
class PreparedQuestion:
    """One request's prompt, ready to batch: ``sequence`` includes its state block."""

    index: int
    sequence: tuple[int, ...]
    count: int
    state: int | None = None

    @property
    def length(self) -> int:
        return len(self.sequence)


def length_batches(
    rows: Sequence[PreparedQuestion], *, max_rows: int, max_tokens: int
) -> list[list[PreparedQuestion]]:
    """Greedy length-sorted batches: each at most ``max_rows`` rows and ``max_tokens`` padded tokens.

    Rows are ordered by descending length, so the first row of a batch is its padded width and a batch's
    padded tokens are ``max(len) * len(batch)``. Every row appears exactly once.
    """
    if max_rows < 1 or max_tokens < 1:
        raise ValueError("max_rows and max_tokens must be positive")
    ordered = sorted(rows, key=lambda r: (-r.length, r.index))
    batches: list[list[PreparedQuestion]] = []
    current: list[PreparedQuestion] = []
    for row in ordered:
        if current:
            width = max(current[0].length, row.length)
            if len(current) >= max_rows or width * (len(current) + 1) > max_tokens:
                batches.append(current)
                current = []
        current.append(row)
    if current:
        batches.append(current)
    return batches


def padded_tokens(batch: Sequence[PreparedQuestion]) -> int:
    """The forward's padded width times its rows (what the runtime materializes)."""
    return max(r.length for r in batch) * len(batch)


def distribution_ok(
    probabilities: Sequence[float] | None, count: int, *, tolerance: float = PROB_TOLERANCE
) -> bool:
    """One option per criterion, finite and non-negative, summing to one within ``tolerance``."""
    if probabilities is None or len(probabilities) != count:
        return False
    values = [float(p) for p in probabilities]
    return (
        all(math.isfinite(p) and p >= 0 for p in values)
        and abs(sum(values) - 1.0) <= tolerance
        and count >= 1
    )


def non_constant(probabilities: Sequence[float]) -> bool:
    """True when the vector is not one repeated value (the gate's 'non-constant' check)."""
    return len({round(float(p), 6) for p in probabilities}) > 1


def argmax_index(probabilities: Sequence[float]) -> int:
    """First index of the maximum, the readout's answer."""
    best = 0
    for i in range(1, len(probabilities)):
        if probabilities[i] > probabilities[best]:
            best = i
    return best


def criteria_for(qtype: str, options: Sequence[Any]) -> Any:
    """The d3 request's criteria for a row, from its ``(key, label)`` options in order.

    ``choice``: ``{key: label}`` in row order (the answer comes back in the same order). ``noul``: omitted, the
    runtime supplies ``false`` / ``true``; the row's keys must be ``yes`` / ``no``. ``score``: the labels in level
    order (lowest first); the answer comes back one entry per level.
    """
    keys = [str(key) for key, _ in options]
    labels = [str(label) for _, label in options]
    if qtype == "choice":
        return dict(zip(keys, labels, strict=True))
    if qtype == "noul":
        if sorted(keys) != ["no", "yes"]:
            raise ValueError(f"a noul row needs the keys yes and no, got {keys}")
        return None
    if qtype == "score":
        if len(labels) < 2:
            raise ValueError("a score row needs at least two levels")
        return labels
    raise ValueError(f"unsupported qtype {qtype!r}")


def question_request(qtype: str, instructions: str, options: Sequence[Any]) -> dict[str, Any]:
    """The question object of one d3 request for a pool row (one question per request)."""
    criteria = criteria_for(qtype, options)
    spec: dict[str, Any] = {"type": qtype, "instructions": instructions}
    if criteria is not None:
        spec["criteria"] = criteria
    return spec


def align_probabilities(qtype: str, options: Sequence[Any], values: Sequence[float]) -> list[float]:
    """The runtime's option probabilities in the row's option order.

    ``choice``: the runtime answers in criteria order, which is the row's order. ``noul``: the runtime's two
    entries are ``false`` then ``true``. ``score``: one entry per level, in level order.
    """
    keys = [str(key) for key, _ in options]
    if qtype == "choice":
        if len(values) != len(keys):
            raise ValueError(f"expected {len(keys)} choice probabilities, got {len(values)}")
        return [float(v) for v in values]
    if qtype == "noul":
        if len(values) != 2 or sorted(keys) != ["no", "yes"]:
            raise ValueError("a noul row needs the keys yes and no and two probabilities")
        p_false, p_true = float(values[0]), float(values[1])
        return [p_true if key == "yes" else p_false for key in keys]
    if qtype == "score":
        if len(values) != len(keys):
            raise ValueError(f"expected {len(keys)} score probabilities, got {len(values)}")
        return [float(v) for v in values]
    raise ValueError(f"unsupported qtype {qtype!r}")


def prepare_request(
    runtime: Any,
    index: int,
    state: str,
    instructions: str,
    qtype: str,
    options: Sequence[Any],
) -> PreparedQuestion:
    """Prepare one one-question request through the runtime's own render/tokenize path."""
    prepared = runtime.prepare(state, {"q": question_request(qtype, instructions, options)})
    if "q" in prepared.errors:
        raise ValueError(f"row {index}: {prepared.errors['q']['message']}")
    return PreparedQuestion(
        index=index,
        sequence=tuple(prepared.sequences["q"]),
        count=len(options),
        state=prepared.states.get("q"),
    )


def prepare_choice(runtime: Any, index: int, state: str, instructions: str, criteria: dict[str, str]) -> PreparedQuestion:
    """Prepare one one-question choice request through the runtime's own render/tokenize path."""
    return prepare_request(runtime, index, state, instructions, "choice", list(criteria.items()))


def score_batch(runtime: Any, batch: Sequence[PreparedQuestion]) -> list[list[float]]:
    """Option probabilities per row, from the runtime's own batched forward and softmax."""
    sequences = [list(r.sequence) for r in batch]
    counts = [r.count for r in batch]
    states = None if all(r.state is None for r in batch) else [r.state for r in batch]
    if states is not None and any(s is None for s in states):
        raise ValueError("a batch mixes prepared rows with and without a state block")
    return runtime.probabilities(sequences, counts, states)
