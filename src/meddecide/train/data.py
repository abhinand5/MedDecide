"""Reading, sampling and batching training items.

Two concerns live here:

* **deterministic sampling** of items from the (large, gitignored) JSONL files, so a smoke run
  or an overfit test can name exactly which items it used; and
* **option-order augmentation** — the permutation for (seed, epoch, item) is derived from a
  hash-free counter-based generator, so the same epoch always sees the same augmentation and a
  rerun reproduces it exactly.

Batches are formed by streaming: items are ordered by a cheap character-count proxy, encoded
one at a time, and a batch is closed when adding the next item would break either the item
limit or the token limit. Nothing about a batch depends on wall-clock or on hash order.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from meddecide.bench.schema import Item

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids importing torch at module load
    from meddecide.model.meddecide_model import EncodedItem, MedDecideModel


def read_items(
    path: str | Path,
    *,
    limit: int | None = None,
    stride: int = 1,
    offset: int = 0,
) -> list[Item]:
    """Read items from a JSONL file, optionally taking every ``stride``-th line.

    ``stride`` sampling is how a small slice is drawn from a 200k-item file without reading
    all of it into memory: lines ``offset, offset+stride, ...`` are parsed until ``limit``
    items are collected. It is deterministic and spreads the sample over the file's sources.
    """
    if stride < 1:
        raise ValueError("stride must be >= 1")
    items: list[Item] = []
    with Path(path).open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh):
            if (lineno - offset) % stride != 0:
                continue
            items.append(Item.model_validate_json(line))
            if limit is not None and len(items) >= limit:
                break
    return items


def deterministic_sample(items: Sequence[Item], n: int, *, seed: int) -> list[Item]:
    """First ``n`` items of a seeded permutation, returned in the sample's own order."""
    if n >= len(items):
        return list(items)
    rng = np.random.default_rng(seed)
    chosen = rng.permutation(len(items))[:n]
    return [items[int(i)] for i in sorted(chosen)]


def stratified_sample(items: Sequence[Item], *, per_qtype: int, seed: int) -> list[Item]:
    """Up to ``per_qtype`` items of every question type, in the input order.

    Question types are very unbalanced in dev (16,421 ``choice``+``noul`` against 33 ``score``),
    and a temperature has to be fitted per qtype: this keeps the rare shape present instead of
    leaving it with a denominator of zero.
    """
    if per_qtype < 1:
        raise ValueError("per_qtype must be >= 1")
    rng = np.random.default_rng([int(seed), 5150])
    by_qtype: dict[str, list[int]] = {}
    for i, item in enumerate(items):
        by_qtype.setdefault(str(item.qtype), []).append(i)
    keep: list[int] = []
    for qtype in sorted(by_qtype):
        positions = by_qtype[qtype]
        if len(positions) <= per_qtype:
            keep.extend(positions)
        else:
            keep.extend(
                positions[int(p)] for p in rng.permutation(len(positions))[:per_qtype]
            )
    return [items[i] for i in sorted(keep)]


def char_proxy(item: Item) -> int:
    """Cheap length proxy used only for ordering batches (never for a reported number)."""
    return len(item.state) + len(item.question) + sum(len(o.label) for o in item.options)


def option_permutation(n_options: int, *, seed: int, epoch: int, index: int) -> list[int]:
    """Deterministic option-order permutation for one (epoch, item) pair.

    Counter-based: independent of iteration order and of how many items came before, so the
    augmentation of item *i* in epoch *e* is the same on any machine and in any process.
    """
    if n_options < 2:
        return list(range(n_options))
    rng = np.random.default_rng([int(seed), int(epoch), int(index)])
    return [int(i) for i in rng.permutation(n_options)]


@dataclass(frozen=True)
class BatchPlan:
    """Bookkeeping for one streamed training epoch."""

    n_items: int
    n_batches: int
    real_tokens: int
    padded_tokens: int
    truncated: int

    def to_dict(self) -> dict[str, int]:
        return {
            "n_items": self.n_items,
            "n_batches": self.n_batches,
            "real_tokens": self.real_tokens,
            "padded_tokens": self.padded_tokens,
            "truncated": self.truncated,
        }


def iter_batches(
    model: MedDecideModel,
    items: Sequence[Item],
    *,
    batch_size: int,
    max_batch_tokens: int,
    max_prompt_tokens: int,
    order: Sequence[int] | None = None,
    augment_options: bool = False,
    epoch: int = 0,
    seed: int = 0,
) -> Iterator[tuple[int, list[EncodedItem]]]:
    """Yield ``(batch_index, encoded items)`` with both limits respected.

    Items are visited in ``order`` (default: file order) after a stable sort by the character
    proxy, so batches hold similar-length prompts and padding waste stays low on a set whose
    states range from a sentence to a 75k-token record.
    """
    positions = list(order) if order is not None else list(range(len(items)))
    positions.sort(key=lambda i: char_proxy(items[i]))
    current: list[EncodedItem] = []
    current_tokens = 0
    batch_index = 0
    for position in positions:
        permutation = (
            option_permutation(
                items[position].n_options, seed=seed, epoch=epoch, index=int(position)
            )
            if augment_options
            else None
        )
        encoded = model.encode_item(
            items[position],
            permutation=permutation,
            max_prompt_tokens=max_prompt_tokens,
        )
        over_tokens = current_tokens + encoded.n_tokens > max_batch_tokens
        if current and (len(current) >= batch_size or over_tokens):
            yield batch_index, current
            batch_index += 1
            current = []
            current_tokens = 0
        current.append(encoded)
        current_tokens += encoded.n_tokens
    if current:
        yield batch_index, current


def plan_epoch(
    model: MedDecideModel,
    items: Sequence[Item],
    *,
    batch_size: int,
    max_batch_tokens: int,
    max_prompt_tokens: int,
    augment_options: bool = False,
    epoch: int = 0,
    seed: int = 0,
) -> BatchPlan:
    """Encode one epoch without training, to report its size and token mix."""
    real = 0
    padded = 0
    batches = 0
    truncated = 0
    count = 0
    for _, batch in iter_batches(
        model,
        items,
        batch_size=batch_size,
        max_batch_tokens=max_batch_tokens,
        max_prompt_tokens=max_prompt_tokens,
        augment_options=augment_options,
        epoch=epoch,
        seed=seed,
    ):
        real += sum(e.n_tokens for e in batch)
        padded += len(batch) * max(e.n_tokens for e in batch)
        truncated += sum(1 for e in batch if e.truncated)
        count += len(batch)
        batches += 1
    return BatchPlan(
        n_items=count, n_batches=batches, real_tokens=real, padded_tokens=padded,
        truncated=truncated,
    )


def shuffled_order(n_items: int, *, seed: int, epoch: int) -> list[int]:
    """Deterministic item order for one epoch (batch composition follows the length sort)."""
    rng = np.random.default_rng([int(seed), 977, int(epoch)])
    return [int(i) for i in rng.permutation(n_items)]
