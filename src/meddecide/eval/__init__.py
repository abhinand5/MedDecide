"""Eval harness: readout, metrics, and probes (task T6)."""

from meddecide.eval.calibration import (
    apply_temperature,
    fit_temperature,
    negative_log_likelihood,
)
from meddecide.eval.metrics import (
    AccuracyReport,
    accuracy,
    bootstrap_ci,
    brier_score,
    ece,
    ece_detail,
    label_mass,
    macro_accuracy,
)

__all__ = [
    "AccuracyReport",
    "accuracy",
    "apply_temperature",
    "bootstrap_ci",
    "brier_score",
    "ece",
    "ece_detail",
    "fit_temperature",
    "label_mass",
    "macro_accuracy",
    "negative_log_likelihood",
]
