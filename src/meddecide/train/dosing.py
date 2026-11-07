"""D57 dosing: a seeded stratified sampler that gives every batch at least three teacher-labelled anchors.

The osler_v4 A5 run carried a teacher distribution on 110,455 of its 973,305 mix rows, but its token-limited batches held
about 1.4 labelled rows on the 45 % of steps that had any at all (C116). D57 therefore replaces "whichever batch happens
to hold a labelled item" with a quota: each batch is drawn as ``min_labelled`` labelled anchors plus fill items up to
``batch_size``, the labelled anchors are drawn from a per-epoch permutation **without replacement**, and the fill items
are drawn from a second per-epoch permutation of the unlabelled rows. Everything is a pure function of
``(seed, epoch, items)``, so a run replays exactly.

Two properties matter for the pre-registered calibration (ADVISORY §2.4):

* **Every optimizer step carries the distillation term.** The per-step gradient share is averaged over steps, and a step
  with no labelled item would contribute a zero to that mean; the closed form ``λ = (0.4/0.6)·(1-s0)/s0`` assumes the
  share's steps are all present. The quota is therefore applied to the *formed* batch, not to a draw that a token limit
  may later split.
* **No item is silently dropped.** A fill item is added while the token cap allows. If the very first fill item after the
  quota does not fit (a long MIMIC record next to three short anchors), it is trained anyway — the cap is a soft guard for
  that one case — and the batch closes, so the item is never skipped and never deferred forever.

The token cap is otherwise the arm's ``max_batch_tokens`` (F2's 8,192). The sampler records the realised composition
(labelled items per batch and the share of batches meeting the minimum) in :class:`DosingStats`.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from meddecide.bench.schema import Item
from meddecide.train.data import option_permutation

if TYPE_CHECKING:  # pragma: no cover - typing only
    from meddecide.model.meddecide_model import EncodedItem, MedDecideModel

# separate streams for the two strata, so a change in one does not move the other
LABELLED_STREAM = 6060
FILL_STREAM = 7070


@dataclass
class DosingStats:
    """The sampler's realised batch statistics (D57: log labelled items per batch and the share meeting the minimum)."""

    min_labelled: int = 0
    batches: int = 0
    items: int = 0
    labelled_items: int = 0
    fill_items: int = 0
    batches_meeting_minimum: int = 0
    labelled_only_batches: int = 0
    soft_cap_batches: int = 0
    fill_cycles: int = 0
    max_labelled: int = 0
    max_items: int = 0
    max_tokens: int = 0
    labelled_histogram: dict[int, int] = field(default_factory=dict)

    def add_batch(self, *, n_labelled: int, n_items: int, tokens: int, soft_cap: bool) -> None:
        """Count one realised batch."""
        self.batches += 1
        self.items += n_items
        self.labelled_items += n_labelled
        self.fill_items += n_items - n_labelled
        if n_labelled >= self.min_labelled:
            self.batches_meeting_minimum += 1
        if n_items == n_labelled:
            self.labelled_only_batches += 1
        if soft_cap:
            self.soft_cap_batches += 1
        self.max_labelled = max(self.max_labelled, n_labelled)
        self.max_items = max(self.max_items, n_items)
        self.max_tokens = max(self.max_tokens, tokens)
        self.labelled_histogram[n_labelled] = self.labelled_histogram.get(n_labelled, 0) + 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "min_labelled": self.min_labelled,
            "batches": self.batches,
            "items": self.items,
            "labelled_items": self.labelled_items,
            "fill_items": self.fill_items,
            "batches_meeting_minimum": self.batches_meeting_minimum,
            "share_batches_meeting_minimum": (
                self.batches_meeting_minimum / self.batches if self.batches else 0.0
            ),
            "labelled_only_batches": self.labelled_only_batches,
            "soft_cap_batches": self.soft_cap_batches,
            "fill_cycles": self.fill_cycles,
            "max_labelled": self.max_labelled,
            "max_items": self.max_items,
            "max_tokens": self.max_tokens,
            "labelled_histogram": {str(k): v for k, v in sorted(self.labelled_histogram.items())},
            "mean_labelled_per_batch": (self.labelled_items / self.batches if self.batches else 0.0),
        }


def labelled_positions(items: Sequence[Item]) -> list[int]:
    """Positions of the items that carry a teacher distribution (the sampler's labelled stratum)."""
    return [i for i, item in enumerate(items) if getattr(item, "teacher_probs", None) is not None]


def dosed_batches(
    model: MedDecideModel,
    items: Sequence[Item],
    *,
    batch_size: int,
    min_labelled: int,
    max_batch_tokens: int,
    max_prompt_tokens: int,
    augment_options: bool = False,
    epoch: int = 0,
    seed: int = 0,
    stats: DosingStats | None = None,
) -> Iterator[tuple[int, list[EncodedItem]]]:
    """Yield ``(batch_index, encoded items)`` with the D57 quota; deterministic in ``(seed, epoch, items)``.

    Each batch starts with ``min_labelled`` labelled anchors (the last batch of an epoch may hold fewer when the
    labelled permutation runs out), then fill items from the unlabelled stratum while ``max_batch_tokens`` allows.
    A batch keeps at least its quota; the only batch over the token cap is one whose first fill item does not fit
    next to the quota (trained anyway so that no item is dropped).
    """
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if not 1 <= min_labelled <= batch_size:
        raise ValueError(f"min_labelled must be in [1, batch_size], got {min_labelled} for batch_size {batch_size}")
    if max_batch_tokens < 1:
        raise ValueError("max_batch_tokens must be >= 1")
    labelled = labelled_positions(items)
    if not labelled:
        raise ValueError("the dosing sampler needs at least one item with a teacher distribution")
    labelled_set = set(labelled)
    others = [i for i in range(len(items)) if i not in labelled_set]
    if not others:
        raise ValueError("the dosing sampler needs unlabelled items to fill batches")
    if stats is None:
        stats = DosingStats()
    stats.min_labelled = min_labelled

    lab_order = [
        int(i) for i in np.random.default_rng([int(seed), LABELLED_STREAM, int(epoch)]).permutation(len(labelled))
    ]
    fill_cycle = 0
    fill_order = [
        int(i)
        for i in np.random.default_rng([int(seed), FILL_STREAM, int(epoch), fill_cycle]).permutation(len(others))
    ]
    lab_cursor = 0
    fill_cursor = 0
    batch_index = 0

    def encode(position: int) -> EncodedItem:
        item = items[position]
        permutation = (
            option_permutation(item.n_options, seed=seed, epoch=epoch, index=int(position))
            if augment_options
            else None
        )
        return model.encode_item(item, permutation=permutation, max_prompt_tokens=max_prompt_tokens)

    while lab_cursor < len(lab_order):
        quota = min(min_labelled, len(lab_order) - lab_cursor)
        batch = [encode(labelled[int(lab_order[position])]) for position in range(lab_cursor, lab_cursor + quota)]
        lab_cursor += quota
        tokens = sum(encoded.n_tokens for encoded in batch)
        soft_cap = False
        if tokens < max_batch_tokens:
            while len(batch) < batch_size:
                if fill_cursor >= len(fill_order):
                    fill_cycle += 1
                    fill_order = [
                        int(i)
                        for i in np.random.default_rng(
                            [int(seed), FILL_STREAM, int(epoch), fill_cycle]
                        ).permutation(len(others))
                    ]
                    fill_cursor = 0
                    stats.fill_cycles += 1
                encoded = encode(others[fill_order[fill_cursor]])
                if tokens + encoded.n_tokens > max_batch_tokens:
                    if len(batch) > quota:
                        break  # normal cap: close here; the item stays at the head of the fill stream
                    batch.append(encoded)  # soft cap: the first fill item does not fit next to the quota
                    tokens += encoded.n_tokens
                    fill_cursor += 1
                    soft_cap = True
                    break
                batch.append(encoded)
                tokens += encoded.n_tokens
                fill_cursor += 1
        stats.add_batch(n_labelled=quota, n_items=len(batch), tokens=tokens, soft_cap=soft_cap)
        yield batch_index, batch
        batch_index += 1
