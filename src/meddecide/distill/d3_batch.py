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


def prepare_choice(runtime: Any, index: int, state: str, instructions: str, criteria: dict[str, str]) -> PreparedQuestion:
    """Prepare one one-question request through the runtime's own render/tokenize path."""
    prepared = runtime.prepare(state, {"q": {"type": "choice", "instructions": instructions, "criteria": criteria}})
    if "q" in prepared.errors:
        raise ValueError(f"row {index}: {prepared.errors['q']['message']}")
    return PreparedQuestion(
        index=index,
        sequence=tuple(prepared.sequences["q"]),
        count=len(criteria),
        state=prepared.states.get("q"),
    )


def score_batch(runtime: Any, batch: Sequence[PreparedQuestion]) -> list[list[float]]:
    """Option probabilities per row, from the runtime's own batched forward and softmax."""
    sequences = [list(r.sequence) for r in batch]
    counts = [r.count for r in batch]
    states = None if all(r.state is None for r in batch) else [r.state for r in batch]
    if states is not None and any(s is None for s in states):
        raise ValueError("a batch mixes prepared rows with and without a state block")
    return runtime.probabilities(sequences, counts, states)
