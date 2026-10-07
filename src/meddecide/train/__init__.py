"""Training package: config, data, losses, temperature calibration, trainer, smoke run."""

from meddecide.train.config import StudentConfig, load_config
from meddecide.train.losses import LossBreakdown, ce_plus_brier
from meddecide.train.temperature import TemperatureFit, fit_per_qtype, fit_temperature
from meddecide.train.trainer import StepRecord, Trainer, TrainResult

__all__ = [
    "LossBreakdown",
    "StepRecord",
    "StudentConfig",
    "TemperatureFit",
    "TrainResult",
    "Trainer",
    "ce_plus_brier",
    "fit_per_qtype",
    "fit_temperature",
    "load_config",
]
