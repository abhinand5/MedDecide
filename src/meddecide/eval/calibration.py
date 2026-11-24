"""Temperature calibration of a decision model's option distribution (osler_rc0 R2).

The release candidates are calibrated on dev items only, per qtype (``choice``, ``noul``, ``score``). A temperature
``T > 0`` rescales the stored distribution as ``p_i ** (1/T) / sum_j p_j ** (1/T)``, which is exactly softmax(logit / T)
when ``p = softmax(logit)``. Because that map is strictly monotone, it can never change the argmax; :func:`apply_temperature`
therefore only changes the probabilities, never the accuracy.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.optimize import minimize_scalar

from meddecide.eval.metrics import brier_score, ece_from_confidence

PROB_FLOOR = 1e-12


def _clean(probs: np.ndarray | Sequence[Sequence[float]]) -> np.ndarray:
    p = np.asarray(probs, dtype=np.float64)
    if p.ndim != 2:
        raise ValueError("probs must be 2-D (n_items, n_options)")
    if not np.all(np.isfinite(p)):
        raise ValueError("probs contain a non-finite value")
    if np.any(p < 0):
        raise ValueError("probs contain a negative value")
    if np.any(p.sum(axis=1) <= 0):
        raise ValueError("a probability row sums to zero")
    return p


def negative_log_likelihood(probs: np.ndarray | Sequence[Sequence[float]], gold_index: Sequence[int]) -> float:
    """Mean NLL (natural log) of the gold options; rows are renormalised first so rounded files are accepted."""
    p = _clean(probs)
    gold = np.asarray(gold_index, dtype=np.int64)
    if p.shape[0] != gold.shape[0]:
        raise ValueError("probs and gold_index disagree on n_items")
    if np.any(gold < 0) or np.any(gold >= p.shape[1]):
        raise ValueError("gold_index outside the option range")
    p = np.clip(p, PROB_FLOOR, None)
    p = p / p.sum(axis=1, keepdims=True)
    return float(-np.mean(np.log(p[np.arange(p.shape[0]), gold])))


def apply_temperature(probs: np.ndarray | Sequence[Sequence[float]], temperature: float) -> np.ndarray:
    """``p ** (1/T)`` renormalised. Strictly monotone in ``p`` for ``T > 0``, so the argmax is unchanged."""
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError(f"temperature must be finite and positive, got {temperature}")
    p = _clean(probs)
    p = np.clip(p, PROB_FLOOR, None)
    scaled = p ** (1.0 / float(temperature))
    return scaled / scaled.sum(axis=1, keepdims=True)


def fit_temperature(probs: np.ndarray | Sequence[Sequence[float]], gold_index: Sequence[int],
                    *, lo: float = 0.02, hi: float = 100.0) -> float:
    """Temperature in ``[lo, hi]`` minimising the mean NLL of the gold options (dev items only)."""
    p = _clean(probs)
    gold = np.asarray(gold_index, dtype=np.int64)

    def objective(temperature: float) -> float:
        return negative_log_likelihood(apply_temperature(p, temperature), gold)

    result = minimize_scalar(objective, bounds=(lo, hi), method="bounded",
                             options={"xatol": 1e-4, "maxiter": 200})
    if not result.success:
        raise RuntimeError(f"temperature fit did not converge: {result.message}")
    return float(result.x)


def summary(probs: np.ndarray | Sequence[Sequence[float]], gold_index: Sequence[int], *, n_bins: int = 15) -> dict:
    """Dev metrics after renormalising the rows: NLL, accuracy, Brier and top-label ECE."""
    p = _clean(probs)
    gold = np.asarray(gold_index, dtype=np.int64)
    p = np.clip(p, PROB_FLOOR, None)
    p = p / p.sum(axis=1, keepdims=True)
    confidence = p.max(axis=1)
    correct = p.argmax(axis=1) == gold
    return {
        "n_items": int(p.shape[0]),
        "nll": negative_log_likelihood(p, gold),
        "accuracy": float(np.mean(correct)),
        "brier": brier_score(p, gold),
        "ece": ece_from_confidence(confidence, correct, n_bins=n_bins),
        "mean_confidence": float(np.mean(confidence)),
    }
