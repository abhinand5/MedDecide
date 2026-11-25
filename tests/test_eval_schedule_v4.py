"""osler_v4 evaluation schedule (ADVISORY Y8): the modulus schedule is unchanged, and the explicit grid ends at the step budget."""

from __future__ import annotations

from dataclasses import fields

import pytest

from meddecide.train.config import StudentConfig
from meddecide.train.trainer import evaluation_grid, is_eval_step


def test_modulus_schedule_is_unchanged() -> None:
    assert is_eval_step(400, 200)
    assert not is_eval_step(401, 200)
    assert not is_eval_step(400, 0)
    assert not is_eval_step(400, None)


def test_explicit_steps_replace_the_modulus() -> None:
    grid = (6330, 63304)
    assert is_eval_step(63304, 200, grid)
    assert is_eval_step(6330, None, grid)
    assert not is_eval_step(6400, 200, grid)


def test_osler_v4_grid_has_ten_evaluations_ending_at_the_budget() -> None:
    assert evaluation_grid(63304, 10) == (6330, 12661, 18991, 25322, 31652, 37982, 44313, 50643, 56974, 63304)
    assert evaluation_grid(13992, 4) == (3498, 6996, 10494, 13992)


def test_grid_rejects_impossible_counts() -> None:
    with pytest.raises(ValueError):
        evaluation_grid(3, 4)


def test_config_default_keeps_the_modulus_schedule() -> None:
    defaults = {f.name: f.default for f in fields(StudentConfig)}
    assert defaults["eval_steps"] is None
    assert defaults["eval_every"] == 200
