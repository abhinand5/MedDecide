"""Paired comparisons of two scorings of the same items.

Pure functions: arrays or mappings in, numbers out. The report scripts parse files and format
output; anything that decides a number lives here so it can be unit-tested.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np


def item_brier(option_probs: Sequence[float], gold_index: int) -> float:
    """Multi-class Brier for one item: ``sum_k (p_k - y_k)**2``.

    Unlike :func:`meddecide.eval.metrics.brier_score`, this does not require the row to sum to
    one within 1e-6. Prediction files store probabilities rounded to six decimals, so a row's sum
    can be off by a few 1e-6.
    """
    p = np.asarray(option_probs, dtype=np.float64)
    if not 0 <= gold_index < p.size:
        raise ValueError(f"gold_index {gold_index} is outside the {p.size} options")
    onehot = np.zeros_like(p)
    onehot[gold_index] = 1.0
    return float(np.sum((p - onehot) ** 2))


@dataclass(frozen=True)
class Difference:
    """A mean paired difference with a percentile bootstrap interval, and its item count."""

    n: int
    point: float
    lo: float
    hi: float

    def as_dict(self) -> dict[str, float | int]:
        return {"n": self.n, "point": self.point, "ci_lo": self.lo, "ci_hi": self.hi}


def _paired_diff(a: Sequence[float], b: Sequence[float], *, label: str) -> np.ndarray:
    """``a - b`` for two non-empty 1-D sequences of equal length.

    The shapes are checked explicitly: NumPy would broadcast ``[1, 0]`` against ``[1]`` silently.
    """
    a_arr = np.asarray(a, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)
    if a_arr.ndim != 1 or a_arr.size == 0 or a_arr.shape != b_arr.shape:
        raise ValueError(f"{label}: needs two non-empty 1-D sequences of equal length")
    return a_arr - b_arr


def _interval(means: np.ndarray, alpha: float) -> tuple[float, float]:
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def paired_difference(
    a: Sequence[float],
    b: Sequence[float],
    *,
    n_resamples: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Difference:
    """Mean of ``a - b`` over paired items, with a seeded percentile bootstrap over items."""
    diff = _paired_diff(a, b, label="paired_difference")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, diff.size, size=(n_resamples, diff.size))
    lo, hi = _interval(diff[idx].mean(axis=1), alpha)
    return Difference(n=int(diff.size), point=float(diff.mean()), lo=lo, hi=hi)


def stratified_macro_difference(
    groups_a: Mapping[str, Sequence[float]],
    groups_b: Mapping[str, Sequence[float]],
    *,
    n_resamples: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Difference:
    """Macro difference: the unweighted mean over groups of each group's mean paired difference.

    Items are resampled within each group, so every group keeps its size and the bootstrap
    respects the group structure (for example one group per template).
    """
    if groups_a.keys() != groups_b.keys():
        raise ValueError("both scorings must cover the same groups")
    if not groups_a:
        raise ValueError("stratified_macro_difference needs at least one group")
    rng = np.random.default_rng(seed)
    point_terms = []
    boot_terms = []
    n_items = 0
    for key in sorted(groups_a):
        diff = _paired_diff(groups_a[key], groups_b[key], label=f"group {key!r}")
        n_items += int(diff.size)
        point_terms.append(diff.mean())
        idx = rng.integers(0, diff.size, size=(n_resamples, diff.size))
        boot_terms.append(diff[idx].mean(axis=1))
    lo, hi = _interval(np.mean(boot_terms, axis=0), alpha)
    return Difference(n=n_items, point=float(np.mean(point_terms)), lo=lo, hi=hi)


def _average_ranks(values: Sequence[float]) -> np.ndarray:
    """1-based ranks with ties sharing the average of the ranks they span."""
    arr = np.asarray(values, dtype=np.float64)
    order = np.argsort(arr, kind="mergesort")
    ranks = np.empty(arr.size, dtype=np.float64)
    start = 0
    while start < arr.size:
        end = start
        while end + 1 < arr.size and arr[order[end + 1]] == arr[order[start]]:
            end += 1
        ranks[order[start : end + 1]] = (start + end) / 2 + 1
        start = end + 1
    return ranks


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    """Spearman rank correlation (Pearson on average ranks); NaN when either side is constant."""
    if len(x) != len(y) or len(x) < 2:
        raise ValueError("spearman needs two equal-length sequences of at least two values")
    rx, ry = _average_ranks(x), _average_ranks(y)
    if np.ptp(rx) == 0 or np.ptp(ry) == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def bootstrap_spearman(
    x: Sequence[float],
    y: Sequence[float],
    *,
    n_resamples: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """``(point, lo, hi)``: Spearman with a seeded percentile bootstrap over the paired points.

    Resamples that make one side constant are dropped; ``n_valid`` is not returned, so a caller
    that needs it should recount. An all-constant resample set raises.
    """
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    if x_arr.shape != y_arr.shape or x_arr.size < 2:
        raise ValueError("bootstrap_spearman needs two equal-length sequences of at least two values")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x_arr.size, size=(n_resamples, x_arr.size))
    stats = np.array([spearman(x_arr[row], y_arr[row]) for row in idx], dtype=np.float64)
    stats = stats[np.isfinite(stats)]
    if stats.size == 0:
        raise ValueError("every bootstrap resample was constant on one side")
    lo, hi = _interval(stats, alpha)
    return spearman(x_arr, y_arr), lo, hi
