"""Temperature calibration of a decision model's option distribution (osler_rc0 R2).

The release candidates are calibrated on dev items only, per qtype (``choice``, ``noul``, ``score``). A temperature
``T > 0`` rescales the stored distribution as ``p_i ** (1/T) / sum_j p_j ** (1/T)``, which is exactly softmax(logit / T)
when ``p = softmax(logit)``. Because that map is strictly monotone, it can never change the argmax; :func:`apply_temperature`
therefore only changes the probabilities, never the accuracy.

Items inside a qtype have different option counts (2 to 255), so every function here accepts either a rectangular
``(n_items, n_options)`` array or a sequence of 1-D rows of differing lengths. A 1-D input is treated as one item.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy.optimize import minimize_scalar

from meddecide.eval.metrics import ece_from_confidence

PROB_FLOOR = 1e-12


def _rows(probs: Any) -> tuple[list[np.ndarray], str]:
    """Normalise the input to a list of 1-D arrays; the second value says how to restore the caller's shape."""
    if isinstance(probs, np.ndarray) and probs.ndim == 1:
        rows, kind = [np.asarray(probs, dtype=np.float64)], "single"
    elif isinstance(probs, np.ndarray) and probs.ndim == 2:
        rows, kind = [np.asarray(row, dtype=np.float64) for row in probs], "matrix"
    else:
        rows, kind = [np.asarray(row, dtype=np.float64) for row in probs], "list"
    for row in rows:
        if row.ndim != 1:
            raise ValueError("every probability row must be 1-D")
        if not np.all(np.isfinite(row)):
            raise ValueError("probs contain a non-finite value")
        if np.any(row < 0):
            raise ValueError("probs contain a negative value")
        if row.sum() <= 0:
            raise ValueError("a probability row sums to zero")
    return rows, kind


def _restore(kind: str, rows: list[np.ndarray]) -> Any:
    if kind == "single":
        return rows[0]
    if kind == "matrix":
        return np.stack(rows)
    return rows


def negative_log_likelihood(probs: Any, gold_index: Sequence[int]) -> float:
    """Mean NLL (natural log) of the gold options; rows are renormalised first so rounded files are accepted."""
    rows, _ = _rows(probs)
    gold = np.asarray(gold_index, dtype=np.int64)
    if len(rows) != gold.shape[0]:
        raise ValueError("probs and gold_index disagree on n_items")
    losses = []
    for row, index in zip(rows, gold.tolist(), strict=True):
        if not 0 <= index < row.size:
            raise ValueError("gold_index outside the option range")
        losses.append(-float(np.log(max(float(row[index] / row.sum()), PROB_FLOOR))))
    return float(np.mean(losses))


def apply_temperature(probs: Any, temperature: float) -> Any:
    """``p ** (1/T)`` renormalised. Strictly monotone in ``p`` for ``T > 0``, so the argmax is unchanged."""
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError(f"temperature must be finite and positive, got {temperature}")
    rows, kind = _rows(probs)
    calibrated = []
    for row in rows:
        scaled = np.clip(row, PROB_FLOOR, None) ** (1.0 / float(temperature))
        calibrated.append(scaled / scaled.sum())
    return _restore(kind, calibrated)


def fit_temperature(probs: Any, gold_index: Sequence[int], *, lo: float = 0.02, hi: float = 100.0) -> float:
    """Temperature in ``[lo, hi]`` minimising the mean NLL of the gold options (dev items only)."""
    rows, _kind = _rows(probs)
    gold = np.asarray(gold_index, dtype=np.int64)

    def objective(temperature: float) -> float:
        return _nll_rows(rows, gold, temperature)

    result = minimize_scalar(objective, bounds=(lo, hi), method="bounded",
                             options={"xatol": 1e-4, "maxiter": 200})
    if not result.success:
        raise RuntimeError(f"temperature fit did not converge: {result.message}")
    return float(result.x)


def _nll_rows(rows: list[np.ndarray], gold: np.ndarray, temperature: float) -> float:
    losses = []
    for row, index in zip(rows, gold.tolist(), strict=True):
        if not 0 <= index < row.size:
            raise ValueError("gold_index outside the option range")
        scaled = np.clip(row, PROB_FLOOR, None) ** (1.0 / float(temperature))
        scaled /= scaled.sum()
        losses.append(-float(np.log(max(float(scaled[index]), PROB_FLOOR))))
    return float(np.mean(losses))


def summary(probs: Any, gold_index: Sequence[int], *, n_bins: int = 15) -> dict:
    """Dev metrics after renormalising the rows: NLL, accuracy, Brier and top-label ECE."""
    rows, _ = _rows(probs)
    gold = np.asarray(gold_index, dtype=np.int64)
    normalised = [row / row.sum() for row in rows]
    confidence = np.asarray([float(row.max()) for row in normalised], dtype=np.float64)
    predicted = np.asarray([int(row.argmax()) for row in normalised], dtype=np.int64)
    correct = predicted == gold
    brier = float(np.mean([float(((row - np.eye(row.size)[index]) ** 2).sum())
                           for row, index in zip(normalised, gold.tolist(), strict=True)]))
    return {
        "n_items": len(rows),
        "nll": negative_log_likelihood(normalised, gold),
        "accuracy": float(np.mean(correct)),
        "brier": brier,
        "ece": ece_from_confidence(confidence, correct, n_bins=n_bins),
        "mean_confidence": float(np.mean(confidence)),
    }

