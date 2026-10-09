"""Metrics for the eval harness (task T6 fills this out; scoring primitives live here).

Every function is pure: it takes probabilities and labels and returns numbers. Unit
tests use hand-computed cases (AGENTS.md code style).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AccuracyReport:
    """Micro accuracy with an explicit denominator (R5) and the majority baseline."""

    n: int
    n_correct: int
    micro: float
    majority_baseline: float
    majority_label: str

    def to_dict(self) -> dict[str, object]:
        return {
            "n": self.n,
            "n_correct": self.n_correct,
            "micro": self.micro,
            "majority_baseline": self.majority_baseline,
            "majority_label": self.majority_label,
        }


def accuracy(predicted: Sequence[str], gold: Sequence[str]) -> AccuracyReport:
    """Micro accuracy plus the majority-class baseline over the same items.

    The majority baseline is the share of items whose gold label equals the most common
    gold label — the number every accuracy must be read against (failure mode 7).
    """
    if len(predicted) != len(gold):
        raise ValueError(f"length mismatch: {len(predicted)} predictions vs {len(gold)} gold")
    n = len(gold)
    if n == 0:
        raise ValueError("accuracy() needs at least one item")
    n_correct = sum(int(p == g) for p, g in zip(predicted, gold, strict=True))
    counts: dict[str, int] = {}
    for g in gold:
        counts[g] = counts.get(g, 0) + 1
    majority_label = max(sorted(counts), key=lambda k: counts[k])
    return AccuracyReport(
        n=n,
        n_correct=n_correct,
        micro=n_correct / n,
        majority_baseline=counts[majority_label] / n,
        majority_label=majority_label,
    )


def macro_accuracy(predicted: Sequence[str], gold: Sequence[str]) -> float:
    """Unweighted mean of per-class accuracy (recall) over classes present in gold."""
    if len(predicted) != len(gold):
        raise ValueError("length mismatch")
    classes = sorted(set(gold))
    recalls = []
    for cls in classes:
        idx = [i for i, g in enumerate(gold) if g == cls]
        recalls.append(sum(int(predicted[i] == cls) for i in idx) / len(idx))
    return float(np.mean(recalls)) if recalls else math.nan


def brier_score(probs: np.ndarray, gold_index: Sequence[int]) -> float:
    """Multi-class Brier score: mean over items of ``sum_k (p_k - y_k)**2``.

    ``probs`` is ``(n_items, n_options)``; rows must sum to 1 (within 1e-6).
    """
    probs = np.asarray(probs, dtype=np.float64)
    gold_index = np.asarray(gold_index, dtype=np.int64)
    if probs.ndim != 2:
        raise ValueError("probs must be 2-D (n_items, n_options)")
    if probs.shape[0] != gold_index.shape[0]:
        raise ValueError("probs and gold_index disagree on n_items")
    if not np.allclose(probs.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("probability rows must sum to 1")
    onehot = np.zeros_like(probs)
    onehot[np.arange(probs.shape[0]), gold_index] = 1.0
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def ece(probs: np.ndarray, gold_index: Sequence[int], *, n_bins: int = 15) -> float:
    """Expected calibration error over equal-width confidence bins.

    Confidence = probability assigned to the predicted (argmax) option; accuracy within
    a bin = share of items whose gold is that argmax. ECE is the bin-size-weighted mean
    of |accuracy - confidence|. Empty bins contribute nothing; bin sizes are returned by
    :func:`ece_detail`.
    """
    return ece_detail(probs, gold_index, n_bins=n_bins)["ece"]


def ece_from_confidence(
    confidence: Sequence[float], correct: Sequence[bool | float], *, n_bins: int = 15
) -> float:
    """Top-label ECE from per-item confidence and correctness, with no rectangular array.

    This is the same quantity :func:`ece` computes — ``ece`` only ever uses the maximum
    probability and whether the argmax was right — but it accepts items with **different
    option counts**, where a rectangular ``(items x options)`` array cannot be built (the HLE
    supplementary set mixes 5- to 16-option items, which made the earlier code raise).
    ``test_metrics.py`` asserts the two agree exactly on rectangular input.
    """
    confidence = np.asarray(confidence, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.float64)
    if confidence.ndim != 1 or confidence.shape != correct.shape:
        raise ValueError("confidence and correct must be 1-D and the same length")
    if not 0 < n_bins <= 1000:
        raise ValueError("n_bins must be in (0, 1000]")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(confidence, edges[1:-1], right=False), 0, n_bins - 1)
    n = len(confidence)
    if n == 0:
        return 0.0
    ece_value = 0.0
    for b in range(n_bins):
        mask = idx == b
        count = int(mask.sum())
        if count == 0:
            continue
        ece_value += (count / n) * abs(correct[mask].mean() - confidence[mask].mean())
    return float(ece_value)


def ece_detail(probs: np.ndarray, gold_index: Sequence[int], *, n_bins: int = 15) -> dict:
    """ECE plus the bin table it was computed from (so the number can be checked by eye)."""
    probs = np.asarray(probs, dtype=np.float64)
    gold_index = np.asarray(gold_index, dtype=np.int64)
    if probs.ndim != 2:
        raise ValueError("probs must be 2-D")
    if not 0 < n_bins <= 1000:
        raise ValueError("n_bins must be in (0, 1000]")
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == gold_index).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # right-closed bins except the first, so confidence == 0 and == 1 both land somewhere
    idx = np.clip(np.digitize(conf, edges[1:-1], right=False), 0, n_bins - 1)
    n = len(conf)
    total = 0.0
    bins = []
    for b in range(n_bins):
        mask = idx == b
        size = int(mask.sum())
        if size == 0:
            bins.append({"bin": b, "lo": edges[b], "hi": edges[b + 1], "n": 0,
                         "confidence": None, "accuracy": None, "gap": None})
            continue
        bin_conf = float(conf[mask].mean())
        bin_acc = float(correct[mask].mean())
        gap = abs(bin_acc - bin_conf)
        total += size / n * gap
        bins.append({"bin": b, "lo": edges[b], "hi": edges[b + 1], "n": size,
                     "confidence": bin_conf, "accuracy": bin_acc, "gap": gap})
    return {"ece": float(total), "n": n, "n_bins": n_bins, "bins": bins}


def bootstrap_ci(
    values: Sequence[float],
    *,
    statistic=np.mean,
    n_resamples: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Percentile bootstrap CI of ``statistic`` over items.

    Returns ``(point, lo, hi)``. Resampling is seeded, so the CI is reproducible.
    """
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        raise ValueError("bootstrap_ci needs at least one value")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_resamples, arr.size))
    stats = statistic(arr[idx], axis=1)
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return float(statistic(arr)), float(lo), float(hi)


def label_mass(option_probs: np.ndarray, option_masses: np.ndarray) -> dict[str, float]:
    """Full-vocabulary probability mass of the option tokens (readout sanity check).

    A model whose option tokens carry almost no mass is being read out wrongly
    (failure mode 3). ``option_probs`` is ``(n_items, n_options)`` renormalised over
    options; ``option_masses`` is the same shape but unnormalised, holding the model's
    raw probability for each option token.
    """
    total = float(np.asarray(option_masses, dtype=np.float64).sum(axis=1).mean())
    top = float(np.asarray(option_probs, dtype=np.float64).max(axis=1).mean())
    return {"mean_option_token_mass": total, "mean_top_prob": top}
