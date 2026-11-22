"""The osler_v7 9B final's token-capped batch plan, for the 4B (`MedDecideModel`) path.

The 9B final (`scripts/osler_v7/t8_run_arm.py`) formed its batches over the **length-sorted** rows under two caps —
``batch`` items and ``max_tokens`` padded tokens — then shuffled the batch order with the seed and indexed it by step.
The 4B path (`meddecide.train.data.iter_batches`) buckets lengths only inside chunks of ``100 x batch_size`` items and
caps the **sum of real tokens**, so one long row still pads a whole batch: T6's 13,992-step arm averaged 5.43 items and
16,662 padded tokens a step, and its worst step held 57,856 padded tokens.

This module is that 9B plan expressed for the 4B collate. The differences are only the ones the 4B path forces:

* a row's padded width is its length rounded up to ``LINEAR_ATTENTION_CHUNK`` (64), because ``ModelBatch`` left-pads
  every row to a multiple of that chunk (``MedDecideModel.collate``); the 9B harness pads to the exact maximum;
* the cap is therefore checked on ``rounded width x items``, the padded-token count the 4B actually allocates.

Everything else is the 9B rule: greedy packing of one ascending-length pass, a batch closes at ``batch`` items or when
the next row would push the padded-token count over ``max_tokens``, a single row longer than the cap still gets its own
batch (never dropped), the batch list is shuffled with ``random.Random(seed)``, and steps index it cyclically.

The plan is a pure function of ``(lengths, batch_size, max_tokens, seed)``: a resumed run builds the same plan and the
same step -> batch map.
"""

from __future__ import annotations

import json
import random
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from meddecide.model.meddecide_model import LINEAR_ATTENTION_CHUNK


def padded_width(n_tokens: int, *, chunk: int = LINEAR_ATTENTION_CHUNK) -> int:
    """The row width ``ModelBatch`` allocates for an item of ``n_tokens`` tokens (rounded up to the attention chunk)."""
    if n_tokens < 1:
        raise ValueError(f"an item needs at least one token, got {n_tokens}")
    if chunk < 1:
        raise ValueError(f"chunk must be >= 1, got {chunk}")
    return -(-int(n_tokens) // int(chunk)) * int(chunk)


@dataclass(frozen=True)
class TokenBatchPlan:
    """A deterministic step -> batch map over length-sorted, token-capped batches (see the module docstring)."""

    batches: tuple[tuple[int, ...], ...]
    item_cap: int
    token_cap: int
    seed: int
    chunk: int = LINEAR_ATTENTION_CHUNK
    n_items: int = 0

    def __post_init__(self) -> None:
        if not self.batches:
            raise ValueError("a plan needs at least one batch")
        if self.item_cap < 1 or self.token_cap < 1:
            raise ValueError("need item_cap >= 1 and token_cap >= 1")
        seen = sorted(index for batch in self.batches for index in batch)
        if seen != list(range(self.n_items)):
            raise ValueError(f"the plan covers {len(seen)} positions, not 0..{self.n_items - 1}")

    def __len__(self) -> int:
        return len(self.batches)

    def batch(self, step: int) -> tuple[int, ...]:
        """The batch of a 1-based step; steps past the end cycle, as the 9B final's ``step % len(batches)`` did."""
        if step < 1:
            raise ValueError(f"steps are 1-based, got {step}")
        return self.batches[(step - 1) % len(self.batches)]

    def with_steps(self, steps: int) -> Iterator[tuple[int, tuple[int, ...]]]:
        """``(step, batch)`` for the first ``steps`` 1-based steps."""
        for step in range(1, int(steps) + 1):
            yield step, self.batch(step)

    def to_dict(self) -> dict[str, Any]:
        return {"batches": len(self.batches), "items": self.n_items, "item_cap": self.item_cap,
                "token_cap": self.token_cap, "seed": self.seed, "chunk": self.chunk}


def build_token_plan(
    lengths: Sequence[int],
    *,
    batch_size: int = 8,
    max_tokens: int = 8192,
    seed: int = 0,
    chunk: int = LINEAR_ATTENTION_CHUNK,
) -> TokenBatchPlan:
    """Pack ``lengths`` into token-capped batches and shuffle the batch order with ``seed`` (the 9B rule)."""
    if batch_size < 1 or max_tokens < 1:
        raise ValueError("need batch_size >= 1 and max_tokens >= 1")
    if len(lengths) == 0:
        raise ValueError("a plan needs at least one row")
    widths = [padded_width(int(n), chunk=chunk) for n in lengths]
    order = sorted(range(len(lengths)), key=lambda index: (lengths[index], index))
    batches: list[list[int]] = []
    current: list[int] = []
    width = 0
    for index in order:
        width = max(width, widths[index])
        would_use = width * (len(current) + 1)
        if current and (len(current) >= batch_size or would_use > max_tokens):
            batches.append(current)
            current = []
            width = widths[index]
        current.append(index)
    if current:
        batches.append(current)
    random.Random(int(seed)).shuffle(batches)
    return TokenBatchPlan(batches=tuple(tuple(batch) for batch in batches), item_cap=batch_size,
                          token_cap=max_tokens, seed=int(seed), chunk=int(chunk), n_items=len(lengths))


def plan_statistics(plan: TokenBatchPlan, lengths: Sequence[int]) -> dict[str, Any]:
    """The realised composition of a plan: items, real and padded tokens per step, and the packing extremes."""
    if len(lengths) != plan.n_items:
        raise ValueError(f"the plan covers {plan.n_items} rows, the length series {len(lengths)}")
    items = [len(batch) for batch in plan.batches]
    real = [sum(int(lengths[index]) for index in batch) for batch in plan.batches]
    padded = [padded_width(max(int(lengths[index]) for index in batch), chunk=plan.chunk) * len(batch)
              for batch in plan.batches]
    total_items = sum(items)
    total_real = sum(real)
    total_padded = sum(padded)
    return {
        "batches": len(plan.batches),
        "covered_items": total_items,
        "mean_items": total_items / len(items),
        "min_items": min(items),
        "max_items": max(items),
        "mean_real_tokens": total_real / len(real),
        "mean_padded_tokens": total_padded / len(padded),
        "max_padded_tokens": max(padded),
        "padding_ratio": total_padded / total_real if total_real else float("nan"),
        "batches_at_item_cap": sum(1 for n in items if n == plan.item_cap),
        "mean_item_tokens": total_real / total_items,
    }


def flagged_batch_share(plan: TokenBatchPlan, flags: Sequence[bool]) -> float:
    """The share of batches that hold at least one row whose ``flags`` entry is true (a step-level dose check)."""
    if len(flags) != plan.n_items:
        raise ValueError(f"the plan covers {plan.n_items} rows, the flag series {len(flags)}")
    hits = sum(1 for batch in plan.batches if any(bool(flags[index]) for index in batch))
    return hits / len(plan.batches)


def save_plan(plan: TokenBatchPlan, path: str | Path) -> Path:
    """Write a plan as one ``.npz`` (batch offsets + concatenated row indices) so a run's batch map is auditable."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    offsets = np.zeros(len(plan.batches) + 1, dtype=np.int64)
    for i, batch in enumerate(plan.batches):
        offsets[i + 1] = offsets[i] + len(batch)
    indices = np.fromiter((index for batch in plan.batches for index in batch), dtype=np.int32, count=int(offsets[-1]))
    np.savez(out, offsets=offsets, indices=indices,
             meta=np.array(json.dumps(plan.to_dict())))
    return out


def load_plan(path: str | Path) -> TokenBatchPlan:
    """Read a plan written by :func:`save_plan`."""
    with np.load(Path(path), allow_pickle=False) as data:
        meta = json.loads(str(data["meta"]))
        offsets = data["offsets"]
        indices = data["indices"]
    if int(offsets[-1]) != int(indices.shape[0]):
        raise ValueError(f"{path}: the plan offsets do not close on the index array")
    batches = tuple(tuple(int(j) for j in indices[offsets[i]:offsets[i + 1]]) for i in range(len(offsets) - 1))
    return TokenBatchPlan(batches=batches, item_cap=int(meta["item_cap"]), token_cap=int(meta["token_cap"]),
                          seed=int(meta["seed"]), chunk=int(meta["chunk"]), n_items=int(meta["items"]))
