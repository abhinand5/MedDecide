"""osler_v5 D54 and D52(b): the dev composite S and the positional health check, as pure functions over per-item prediction rows.

The rows are the prediction rows of meddecide.train.osler_arm.dev_prediction_rows (item_id, source, template_id, n_options, gold_index,
correct, ...). Nothing here reads a test split, the external panel, the MIMIC audit suite or a held-out file; the callers pass dev rows.

S is the mean of three item accuracies on dev: the MR partition (osler_v3's keep-rule partition: the MR sources), the MIMIC partition of
fast dev v4 (the template ids that start with mimic_), and the CG v2 **micro** accuracy (item-pooled, not template macro, D54).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from meddecide.train.blocks import MR_SOURCES

MIMIC_PREFIX = "mimic_"
TWO_LETTER_FROM = 26  # option positions 26 and above take the two-letter codes (D52b)


PERMUTATION_TAG = 5353  # the positional check's own stream in the seed vector (independent of the training streams)


def permutation_for(n_options: int, *, seed: int, index: int) -> list[int]:
    """The seeded option permutation of one dev item for the positional check (D52b): source positions in display order. The
    original order is the identity; the check uses two seeds on top of it."""
    if n_options < 2:
        return list(range(n_options))
    rng = np.random.default_rng([int(seed), PERMUTATION_TAG, int(index)])
    return [int(i) for i in rng.permutation(n_options)]


def accuracy(rows: Sequence[dict[str, Any]]) -> float:
    """Item-pooled accuracy; raises on an empty set (a partition with no items is an error, never a zero)."""
    if not rows:
        raise ValueError("accuracy of an empty partition")
    return sum(bool(row["correct"]) for row in rows) / len(rows)


def mr_partition(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if str(row["source"]) in MR_SOURCES]


def mimic_partition(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if str(row["template_id"]).startswith(MIMIC_PREFIX)]


@dataclass(frozen=True)
class CompositeParts:
    """S and its three parts (dev accuracies). ``s`` is their mean."""

    mr: float
    mimic: float
    cg_micro: float
    n_mr: int
    n_mimic: int
    n_cg: int

    @property
    def s(self) -> float:
        return (self.mr + self.mimic + self.cg_micro) / 3.0

    def to_dict(self) -> dict[str, float | int]:
        return {"S": self.s, "mr": self.mr, "mimic": self.mimic, "cg_micro": self.cg_micro,
                "n_mr": self.n_mr, "n_mimic": self.n_mimic, "n_cg": self.n_cg}


def composite(fast_rows: Sequence[dict[str, Any]], cg_rows: Sequence[dict[str, Any]]) -> CompositeParts:
    """S over fast dev v4 rows (for the MR and MIMIC partitions) and CG v2 rows (micro)."""
    mr = mr_partition(fast_rows)
    mimic = mimic_partition(fast_rows)
    return CompositeParts(mr=accuracy(mr), mimic=accuracy(mimic), cg_micro=accuracy(cg_rows),
                          n_mr=len(mr), n_mimic=len(mimic), n_cg=len(cg_rows))


@dataclass(frozen=True)
class PositionalHealth:
    """D52b on one ordering: two-letter-gold accuracy against half the single-letter-gold accuracy, over the items of the templates
    that offer more than 26 options. ``passes`` is the qualifier; it is never a test-set gate."""

    two_letter_accuracy: float
    single_letter_accuracy: float
    n_two_letter: int
    n_single_letter: int
    templates: tuple[str, ...]

    @property
    def passes(self) -> bool:
        return self.two_letter_accuracy >= 0.5 * self.single_letter_accuracy

    def to_dict(self) -> dict[str, Any]:
        return {"two_letter_accuracy": self.two_letter_accuracy, "single_letter_accuracy": self.single_letter_accuracy,
                "n_two_letter": self.n_two_letter, "n_single_letter": self.n_single_letter,
                "templates": list(self.templates), "passes": self.passes}


def positional_health(rows: Sequence[dict[str, Any]]) -> PositionalHealth:
    """D52b on the given rows (one ordering of the dev items). The gold position decides the group: index 26 and above is two-letter."""
    wide = sorted({str(row["template_id"]) for row in rows if int(row["n_options"]) > TWO_LETTER_FROM})
    if not wide:
        raise ValueError("no dev template offers more than 26 options")
    items = [row for row in rows if str(row["template_id"]) in wide]
    if any(row.get("gold_index") is None for row in items):
        raise ValueError("a positional item has no gold position (its option keys are not the readout codes)")
    two = [row for row in items if int(row["gold_index"]) >= TWO_LETTER_FROM]
    single = [row for row in items if int(row["gold_index"]) < TWO_LETTER_FROM]
    if not two or not single:
        raise ValueError("the positional check needs both two-letter and single-letter gold items")
    return PositionalHealth(two_letter_accuracy=accuracy(two), single_letter_accuracy=accuracy(single),
                            n_two_letter=len(two), n_single_letter=len(single), templates=tuple(wide))
