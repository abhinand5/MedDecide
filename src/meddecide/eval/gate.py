"""Gate O1 decision rule (PROGRAM D24), as the pre-registered verdict logic over paired differences.

Each comparison is Osler minus one baseline, with an item-paired bootstrap interval (1,000 resamples, items within datasets
or templates). A comparison passes when the macro-accuracy interval's lower bound is above 0 and the mean-Brier interval's
upper bound is below 0 (lower Brier is better). The gate passes only when every required comparison passes, on the fresh
seen-template set and on the external clinical panel, against both baselines. A knowledge guard also applies: on MedQA plus
MedMCQA, the accuracy interval's lower bound of Osler minus the zero-shot base of the same size must be above -0.02.

The thresholds and the set of comparisons are fixed by D24 before any result. This module implements them and changes
nothing about them.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from meddecide.eval.paired import (
    Difference,
    item_brier,
    paired_mean_difference_stratified,
    stratified_macro_difference,
)

KNOWLEDGE_GUARD_FLOOR = -0.02


@dataclass(frozen=True)
class Comparison:
    """Osler minus one baseline on one scope (a set of items), with both paired intervals."""

    scope: str
    baseline: str
    accuracy: Difference  # macro accuracy, Osler minus baseline
    brier: Difference  # mean Brier, Osler minus baseline (lower is better)

    @property
    def passes(self) -> bool:
        return self.accuracy.lo > 0 and self.brier.hi < 0


def gate_verdict(comparisons: Sequence[Comparison]) -> bool:
    """PASS only when at least one comparison is required and every one passes."""
    return len(comparisons) > 0 and all(c.passes for c in comparisons)


def knowledge_guard(difference: Difference) -> bool:
    """The guard passes when the accuracy interval's lower bound (Osler minus zero-shot base) is above -0.02."""
    return difference.lo > KNOWLEDGE_GUARD_FLOOR


@dataclass(frozen=True)
class PairedRows:
    """A comparison built from prediction rows, with the item accounting the report prints (no silent drops)."""

    comparison: Comparison
    n_scope: int  # items in the scope (the common-set rows that define it)
    n_paired: int  # items scored by both Osler and the baseline: the comparison's items
    n_baseline_skipped: int  # scope items the baseline skipped (the rows carry the reasons)
    n_baseline_missing: int  # scope items with no baseline row
    n_osler_missing: int  # scope items with no Osler row (must be 0)


def _correct_and_brier(row: Mapping[str, object]) -> tuple[float, float]:
    keys = list(row["option_keys"])  # type: ignore[arg-type]
    probs = row["probs"]
    predicted = keys[int(np.argmax(np.asarray(probs, dtype=np.float64)))]
    return float(predicted == row["gold"]), item_brier(probs, keys.index(row["gold"]))  # type: ignore[arg-type]


def compare_rows(
    scope: str,
    baseline: str,
    scope_rows: Sequence[Mapping[str, object]],
    osler_rows: Mapping[str, Mapping[str, object]],
    baseline_rows: Mapping[str, Mapping[str, object]],
    *,
    n_resamples: int = 1000,
    seed: int = 0,
) -> PairedRows:
    """Osler minus one baseline on one scope, paired by item id.

    ``scope_rows`` are the common-set rows that define the scope (each has ``item_id`` and ``template_id``); ``osler_rows`` and
    ``baseline_rows`` map item ids to prediction rows. The accuracy is the template macro (the mean over templates of per-template
    accuracy; D24's "macro accuracy", as G1 defines it), bootstrapped within templates. The Brier is the item mean, bootstrapped
    within templates. Items a baseline skipped or lacks, and items Osler lacks, are counted and left out of the pairing.
    """
    by_template: dict[str, list[tuple[float, float, float, float]]] = {}
    n_skipped = n_missing_base = n_missing_osler = 0
    for scope_row in scope_rows:
        item_id = str(scope_row["item_id"])
        template = str(scope_row["template_id"])
        o = osler_rows.get(item_id)
        b = baseline_rows.get(item_id)
        if o is None or o.get("status") != "scored":
            n_missing_osler += 1
            continue
        if b is None:
            n_missing_base += 1
            continue
        if b.get("status") != "scored":
            n_skipped += 1
            continue
        co, bo = _correct_and_brier(o)
        cb, bb = _correct_and_brier(b)
        by_template.setdefault(template, []).append((co, cb, bo, bb))
    if not by_template:
        raise ValueError(f"{scope} vs {baseline}: no paired items")
    acc_a = {t: [v[0] for v in vals] for t, vals in by_template.items()}
    acc_b = {t: [v[1] for v in vals] for t, vals in by_template.items()}
    brier_a: list[float] = []
    brier_b: list[float] = []
    strata: list[str] = []
    for template, vals in by_template.items():
        for v in vals:
            brier_a.append(v[2])
            brier_b.append(v[3])
            strata.append(template)
    accuracy = stratified_macro_difference(acc_a, acc_b, n_resamples=n_resamples, seed=seed)
    brier = paired_mean_difference_stratified(brier_a, brier_b, strata, n_resamples=n_resamples, seed=seed)
    n_paired = len(brier_a)
    return PairedRows(
        comparison=Comparison(scope=scope, baseline=baseline, accuracy=accuracy, brier=brier),
        n_scope=len(scope_rows), n_paired=n_paired, n_baseline_skipped=n_skipped,
        n_baseline_missing=n_missing_base, n_osler_missing=n_missing_osler,
    )


def accuracy_difference(
    scope_rows: Sequence[Mapping[str, object]],
    osler_rows: Mapping[str, Mapping[str, object]],
    baseline_rows: Mapping[str, Mapping[str, object]],
    *,
    n_resamples: int = 1000,
    seed: int = 0,
) -> tuple[Difference, int]:
    """Plain (micro) accuracy difference, Osler minus the baseline, paired by item and bootstrapped within templates.

    This is the knowledge guard's statistic (D24: "the accuracy CI lower bound"). It returns the difference and the number of
    paired items; items missing or skipped on either side are left out, as in :func:`compare_rows`.
    """
    values_a: list[float] = []
    values_b: list[float] = []
    strata: list[str] = []
    for scope_row in scope_rows:
        item_id = str(scope_row["item_id"])
        o = osler_rows.get(item_id)
        b = baseline_rows.get(item_id)
        if o is None or b is None or o.get("status") != "scored" or b.get("status") != "scored":
            continue
        values_a.append(_correct_and_brier(o)[0])
        values_b.append(_correct_and_brier(b)[0])
        strata.append(str(scope_row["template_id"]))
    if not values_a:
        raise ValueError("accuracy_difference: no paired items")
    diff = paired_mean_difference_stratified(values_a, values_b, strata, n_resamples=n_resamples, seed=seed)
    return diff, len(values_a)
