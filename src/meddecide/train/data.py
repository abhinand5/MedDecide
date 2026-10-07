"""Reading, sampling and batching training items.

Two concerns live here:

* **deterministic sampling** of items from the (large, gitignored) JSONL files, so a smoke run
  or an overfit test can name exactly which items it used; and
* **option-order augmentation** — the permutation for (seed, epoch, item) is derived from a
  hash-free counter-based generator, so the same epoch always sees the same augmentation and a
  rerun reproduces it exactly.

Batches are formed by streaming: inside each chunk items are ordered by a cheap character-count
proxy, encoded one at a time, and a batch is closed when adding the next item would break either
the item limit or the token limit. Nothing about a batch depends on wall-clock or on hash order.

**The chunking is a bug fix, not decoration.** The first version sorted the whole epoch by the
character proxy *after* shuffling, so the epoch ran shortest-to-longest: an accidental length
curriculum. S9 run 1 trained under it, peaked on dev about 45 % of the way through the epoch, and
then degraded as batches grew long. Length bucketing now happens only *inside* chunks of
``batch_chunk_factor x batch_size`` items, and the chunks themselves are visited in a per-epoch
shuffled order, so batch length is stationary across the run. ``tests/test_train_student.py``
carries the regression test (Spearman |rho| between batch index and mean batch length < 0.1).
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from meddecide.bench.schema import Item

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids importing torch at module load
    from meddecide.model.meddecide_model import EncodedItem, MedDecideModel

# length-bucketing unit: batches are formed inside chunks of this many batches' worth of items,
# and the chunks are shuffled per epoch (see the module docstring)
DEFAULT_BATCH_CHUNK_FACTOR = 100


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


def stratified_sample_by(
    items: Sequence[Item], *, key: str, per_group: int, seed: int
) -> list[Item]:
    """Up to ``per_group`` items per value of ``key``, returned in the input order.

    Deterministic for a given ``(items, key, per_group, seed)``. A group with fewer items than
    ``per_group`` contributes all of them, and the realised per-group counts are reported by the
    caller rather than assumed (R5: a denominator of zero is not a fit).
    """
    if per_group < 1:
        raise ValueError("per_group must be >= 1")
    rng = np.random.default_rng([int(seed), 5150])
    by_group: dict[str, list[int]] = {}
    for i, item in enumerate(items):
        if not hasattr(item, key):
            raise ValueError(f"items have no attribute {key!r}")
        by_group.setdefault(str(getattr(item, key)), []).append(i)
    keep: list[int] = []
    for group in sorted(by_group):
        positions = by_group[group]
        if len(positions) <= per_group:
            keep.extend(positions)
        else:
            keep.extend(
                positions[int(p)] for p in rng.permutation(len(positions))[:per_group]
            )
    return [items[i] for i in sorted(keep)]


def stratified_sample(items: Sequence[Item], *, per_qtype: int, seed: int) -> list[Item]:
    """Up to ``per_qtype`` items of every question type, in the input order.

    Question types are very unbalanced in dev (16,421 ``choice``+``noul`` against 33 ``score``),
    and a temperature has to be fitted per qtype: this keeps the rare shape present instead of
    leaving it with a denominator of zero.
    """
    return stratified_sample_by(items, key="qtype", per_group=per_qtype, seed=seed)


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
    chunk_factor: int = DEFAULT_BATCH_CHUNK_FACTOR,
) -> Iterator[tuple[int, list[EncodedItem]]]:
    """Yield ``(batch_index, encoded items)`` with both limits respected.

    Items are visited in ``order`` (default: file order), cut into chunks of
    ``chunk_factor x batch_size`` items. Each chunk is sorted by the character proxy before
    batching (batches hold similar-length prompts, so padding waste stays low on a set whose
    states range from a sentence to a 75k-token record) and the chunks are visited in a
    per-``(seed, epoch)`` shuffled order, so the *global* batch order carries no length
    signal — see the module docstring for why that matters.
    """
    if chunk_factor < 1:
        raise ValueError("chunk_factor must be >= 1")
    positions = list(order) if order is not None else list(range(len(items)))
    chunk_size = max(int(batch_size), int(chunk_factor) * int(batch_size))
    chunks = [positions[i : i + chunk_size] for i in range(0, len(positions), chunk_size)]
    order_rng = np.random.default_rng([int(seed), 3131, int(epoch)])
    chunk_order = [int(i) for i in order_rng.permutation(len(chunks))]
    batch_index = 0
    for chunk_index in chunk_order:
        chunk = chunks[chunk_index]
        # 1. bucket by length inside the chunk, so a batch holds similar-length prompts
        chunk.sort(key=lambda i: char_proxy(items[i]))
        formed: list[list[EncodedItem]] = []
        current: list[EncodedItem] = []
        current_tokens = 0
        for position in chunk:
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
                formed.append(current)
                current = []
                current_tokens = 0
            current.append(encoded)
            current_tokens += encoded.n_tokens
        if current:
            formed.append(current)
        # 2. yield the chunk's batches in a shuffled order: bucketing decided the *membership*
        # (padding), never the order the optimiser sees them in
        batch_rng = np.random.default_rng([int(seed), 7717, int(epoch), int(chunk_index)])
        for pick in batch_rng.permutation(len(formed)):
            yield batch_index, formed[int(pick)]
            batch_index += 1


@dataclass
class EpochDetail:
    """A planned epoch with its per-batch length series (the anti-curriculum check).

    ``real_tokens`` is the token count after truncation to the prompt cap, i.e. what the model
    actually reads; ``batch_tokens`` and ``batch_mean_chars`` are per-batch series indexed by
    batch order, which is what a length-trend check needs.
    """

    plan: BatchPlan
    batch_tokens: list[int] = field(default_factory=list)
    batch_mean_chars: list[float] = field(default_factory=list)
    batch_items: list[int] = field(default_factory=list)

    def to_dict(self, *, include_series: bool = True) -> dict[str, Any]:
        out: dict[str, Any] = {"plan": self.plan.to_dict()}
        if include_series:
            out["batch_tokens"] = self.batch_tokens
            out["batch_mean_chars"] = self.batch_mean_chars
            out["batch_items"] = self.batch_items
        return out


def plan_epoch_detail(
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
    chunk_factor: int = DEFAULT_BATCH_CHUNK_FACTOR,
) -> EpochDetail:
    """Encode one epoch without training and keep the per-batch length series.

    One tokenizer pass; no model forward. The series lets a caller verify that batch length does
    not trend with batch index (the accidental-curriculum regression) on the *real* item set
    before paying for a training run.
    """
    real = 0
    padded = 0
    batches = 0
    truncated = 0
    count = 0
    tokens: list[int] = []
    mean_chars: list[float] = []
    items_per_batch: list[int] = []
    chars_by_id = {item.item_id: char_proxy(item) for item in items}
    for _, batch in iter_batches(
        model,
        items,
        batch_size=batch_size,
        max_batch_tokens=max_batch_tokens,
        max_prompt_tokens=max_prompt_tokens,
        order=order,
        augment_options=augment_options,
        epoch=epoch,
        seed=seed,
        chunk_factor=chunk_factor,
    ):
        real += sum(e.n_tokens for e in batch)
        padded += len(batch) * max(e.n_tokens for e in batch)
        truncated += sum(1 for e in batch if e.truncated)
        count += len(batch)
        batches += 1
        tokens.append(sum(e.n_tokens for e in batch))
        mean_chars.append(float(np.mean([chars_by_id[e.item_id] for e in batch])))
        items_per_batch.append(len(batch))
    plan = BatchPlan(
        n_items=count, n_batches=batches, real_tokens=real, padded_tokens=padded,
        truncated=truncated,
    )
    return EpochDetail(
        plan=plan, batch_tokens=tokens, batch_mean_chars=mean_chars, batch_items=items_per_batch
    )


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
    chunk_factor: int = DEFAULT_BATCH_CHUNK_FACTOR,
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
        chunk_factor=chunk_factor,
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
