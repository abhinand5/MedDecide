"""The temperatures a saved checkpoint is scored with (osler_v4 Y9 pre-launch fix; STATE §7 deviation, C191)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from meddecide.eval.osler_scoring import checkpoint_temperatures


def _model(calibration: dict[str, float]) -> Any:
    return SimpleNamespace(calibration=calibration)


def _arm_result(fits: dict[str, float]) -> dict[str, Any]:
    return {"temperature_fits": {qtype: {"temperature": value} for qtype, value in fits.items()}}


def test_returns_the_saved_calibration_as_floats() -> None:
    assert checkpoint_temperatures(_model({"choice": 1.5, "noul": 2})) == {"choice": 1.5, "noul": 2.0}


def test_refuses_an_empty_calibration() -> None:
    with pytest.raises(ValueError, match="no saved calibration"):
        checkpoint_temperatures(_model({}))


def test_agrees_with_the_arm_fits_to_six_decimals() -> None:
    saved = {"choice": 1.4907949814928816, "score": 1.218539172944011}
    assert checkpoint_temperatures(_model(saved), _arm_result({"choice": 1.490795, "score": 1.218539})) == saved


def test_refuses_a_calibration_that_differs_from_the_arm_fits() -> None:
    with pytest.raises(ValueError, match="differs from the arm's dev fits"):
        checkpoint_temperatures(_model({"choice": 1.5}), _arm_result({"choice": 1.6}))


def test_refuses_an_arm_result_without_fits() -> None:
    with pytest.raises(ValueError, match="no temperature_fits"):
        checkpoint_temperatures(_model({"choice": 1.5}), {})
