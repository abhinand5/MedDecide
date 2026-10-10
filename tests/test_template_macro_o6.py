"""Template macro accuracy (G1/D24's "macro"; osler_v0 Q20) and its place in the dev metrics."""

from __future__ import annotations

import math

import pytest

from meddecide.eval.metrics import template_macro_accuracy


def test_template_macro_is_the_mean_of_per_template_accuracy() -> None:
    correct = [1, 1, 1, 0, 0, 1]
    templates = ["a", "a", "a", "a", "b", "b"]
    assert template_macro_accuracy(correct, templates) == pytest.approx((0.75 + 0.5) / 2)


def test_template_macro_empty_and_mismatch() -> None:
    assert math.isnan(template_macro_accuracy([], []))
    with pytest.raises(ValueError):
        template_macro_accuracy([1], ["a", "b"])


def test_selection_metric_is_validated() -> None:
    from meddecide.train.config import StudentConfig

    with pytest.raises(ValueError):
        StudentConfig.from_dict({"selection_metric": "accuracy"})
