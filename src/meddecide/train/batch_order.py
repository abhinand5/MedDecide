"""Batch order for the arm C (Unsloth) trainer: the arms A and B length bucketing, with fixed batches.

``meddecide.train.data.iter_batches`` (arms A and B) cuts the training file into chunks of
``chunk_factor x batch_size`` consecutive items, visits the chunks in a per-``(seed, epoch)`` shuffled
order, sorts each chunk by length so a batch holds similar-length prompts, and yields each chunk's batches
in a shuffled order. This module applies the same chunking, the same seeded streams and the same chunk
visiting order to a list of lengths. Two differences are recorded as deviations (D15): every batch holds
exactly ``batch_size`` items (arms A and B also close a batch at ``max_batch_tokens``), and the in-chunk
sort key is the token count of the built item rather than the character proxy.

Pure numpy, so the Unsloth environment loads this file by path, as it loads ``unsloth_rows.py``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

# the stream constants of meddecide.train.data.iter_batches (a test keeps them in step)
CHUNK_STREAM = 3131
BATCH_STREAM = 7717

# option-order augmentation: choice and score items are reordered, as arms A and B reorder them
AUGMENTED_QTYPES = ("choice", "score")


def length_bucketed_order(
    lengths: Sequence[int],
    *,
    batch_size: int,
    chunk_factor: int,
    seed: int,
    epoch: int = 0,
) -> list[int]:
    """Flat item order over one epoch. Consecutive runs of ``batch_size`` indices form one batch."""
    if batch_size < 1 or chunk_factor < 1:
        raise ValueError("batch_size and chunk_factor must be >= 1")
    n = len(lengths)
    chunk_size = chunk_factor * batch_size
    chunks = [list(range(start, min(start + chunk_size, n))) for start in range(0, n, chunk_size)]
    chunk_order = np.random.default_rng([int(seed), CHUNK_STREAM, int(epoch)]).permutation(len(chunks))
    order: list[int] = []
    for chunk_index in chunk_order:
        members = sorted(chunks[int(chunk_index)], key=lambda i: lengths[i])
        batches = [members[k : k + batch_size] for k in range(0, len(members), batch_size)]
        batch_rng = np.random.default_rng([int(seed), BATCH_STREAM, int(epoch), int(chunk_index)])
        for pick in batch_rng.permutation(len(batches)):
            order.extend(batches[int(pick)])
    return order


def option_permutation(n_options: int, *, seed: int, epoch: int, index: int) -> list[int]:
    """The arms A/B option-order permutation (``meddecide.train.data.option_permutation``; a test checks it)."""
    if n_options < 2:
        return list(range(n_options))
    rng = np.random.default_rng([int(seed), int(epoch), int(index)])
    return [int(i) for i in rng.permutation(n_options)]


def permute_options(
    rows: Sequence[Mapping[str, Any]], *, seed: int, epoch: int = 0
) -> tuple[list[dict[str, Any]], int]:
    """Option-order augmentation of the source items, applied as arms A and B apply it.

    The item at file index ``i`` is reordered by ``option_permutation(n, seed, epoch, i)``. Keys are then
    relabelled by display position (choice: A, B, ...; score: 1, 2, ..., so the level order holds), and the
    gold follows its content. Score items are first put in level order, which is how the converter reads
    them. Noul items are returned unchanged: the Unsloth noul question has no option list to reorder.
    Returns the items and the number of items whose order changed.
    """
    out: list[dict[str, Any]] = []
    changed = 0
    for index, row in enumerate(rows):
        options = [dict(o) for o in (row.get("options") or [])]
        qtype = str(row.get("qtype"))
        if qtype not in AUGMENTED_QTYPES or len(options) < 2:
            out.append(dict(row))
            continue
        if qtype == "score":
            options.sort(key=lambda o: int(o["key"]))
        keys = [str(o["key"]) for o in options]
        gold_position = keys.index(str(row["gold"]))
        perm = option_permutation(len(options), seed=seed, epoch=epoch, index=index)
        if perm != list(range(len(options))):
            changed += 1
        new_options = [{**options[p], "key": _position_key(qtype, j)} for j, p in enumerate(perm)]
        gold = _position_key(qtype, perm.index(gold_position))
        out.append({**row, "options": new_options, "gold": gold})
    return out, changed


def _position_key(qtype: str, position: int) -> str:
    """The key of the option shown at 0-based ``position``: choice letters A, B, ...; score levels 1, 2, ..."""
    return str(position + 1) if qtype == "score" else chr(ord("A") + position)
