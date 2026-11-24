"""Unit tests for dev-only temperature calibration (osler_rc0 R2)."""

from __future__ import annotations

import numpy as np
import pytest

from meddecide.eval.calibration import (
    apply_temperature,
    fit_temperature,
    negative_log_likelihood,
    summary,
)


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def test_nll_of_uniform_is_log_options() -> None:
    probs = np.full((5, 4), 0.25)
    assert negative_log_likelihood(probs, [0, 1, 2, 3, 0]) == pytest.approx(np.log(4))


def test_apply_temperature_preserves_argmax_and_normalises() -> None:
    rng = np.random.default_rng(0)
    probs = _softmax(rng.normal(size=(50, 5)))
    for temperature in (0.5, 1.0, 3.0):
        calibrated = apply_temperature(probs, temperature)
        assert np.allclose(calibrated.sum(axis=1), 1.0)
        assert np.array_equal(calibrated.argmax(axis=1), probs.argmax(axis=1))


def test_temperature_one_is_identity() -> None:
    probs = _softmax(np.random.default_rng(1).normal(size=(10, 3)))
    assert np.allclose(apply_temperature(probs, 1.0), probs)


def test_fit_flattens_overconfident_predictions() -> None:
    rng = np.random.default_rng(2)
    z = rng.normal(size=(4000, 4))
    # the gold is drawn from the *honest* distribution softmax(z), but the stored probs are overconfident softmax(2z)
    honest = _softmax(z)
    gold = [int(rng.choice(4, p=honest[i])) for i in range(len(z))]
    overconfident = _softmax(2.0 * z)
    temperature = fit_temperature(overconfident, gold)
    assert temperature > 1.0
    pre = negative_log_likelihood(overconfident, gold)
    post = negative_log_likelihood(apply_temperature(overconfident, temperature), gold)
    assert post < pre


def test_fit_sharpens_underconfident_predictions() -> None:
    rng = np.random.default_rng(3)
    z = rng.normal(size=(4000, 4))
    sharp = _softmax(2.0 * z)
    gold = [int(rng.choice(4, p=sharp[i])) for i in range(len(z))]
    temperature = fit_temperature(_softmax(0.5 * z), gold)
    assert temperature < 1.0


def test_summary_of_perfect_predictions() -> None:
    probs = np.eye(3)
    result = summary(probs, [0, 1, 2])
    assert result["accuracy"] == 1.0
    assert result["brier"] == pytest.approx(0.0, abs=1e-9)
    assert result["ece"] == pytest.approx(0.0, abs=1e-9)
    assert result["nll"] == pytest.approx(0.0, abs=1e-9)


def test_rejects_invalid_probabilities() -> None:
    with pytest.raises(ValueError):
        negative_log_likelihood(np.zeros((2, 2)), [0, 1])
    with pytest.raises(ValueError):
        apply_temperature(np.array([[0.5, 0.5]]), 0.0)
    with pytest.raises(ValueError):
        negative_log_likelihood(np.array([[1.0, 0.0]]), [2])
