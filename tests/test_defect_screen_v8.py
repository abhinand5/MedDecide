"""The V8/V9 defect screen (pure functions, no GPU): each flag fires on the defect it names and stays off on a clean run."""

from __future__ import annotations

import pytest

from meddecide.train.blocks import defect_screen


def _row(i: int, *, template: str = "t1", n_options: int = 4, gold: int = 0, predicted: int | None = None) -> dict:
    predicted = gold if predicted is None else predicted
    return {"item_id": f"{template}-{i}", "template_id": template, "n_options": n_options, "gold_index": gold,
            "predicted_index": predicted, "correct": predicted == gold}


def _falling_losses(n: int = 200) -> list[float]:
    return [1.0 - 0.001 * k for k in range(n)]


def test_a_clean_run_raises_no_flag() -> None:
    # varied gold, right on 90 % of the items, two templates of 60 items each (both screened, neither too good)
    rows = [_row(i, template="t1" if i % 2 else "t2", gold=i % 4, predicted=(i % 4) if i % 10 else (i + 1) % 4)
            for i in range(120)]
    report = defect_screen(rows, _falling_losses())
    assert report["any_flag"] is False
    assert report["accuracy"] == pytest.approx(0.9)
    assert report["templates_screened"] == 2


def test_a_constant_answer_is_flagged_by_template() -> None:
    rows = [_row(i, template="c", gold=i % 4, predicted=0) for i in range(40)]
    report = defect_screen(rows, _falling_losses())
    assert report["flags"]["constant_answer"] is True
    assert report["constant_answer_templates"] == ["c"]


def test_an_accuracy_at_chance_or_below_is_flagged() -> None:
    rows = [_row(i, gold=i % 4, predicted=(i % 4 + 1) % 4) for i in range(120)]  # every answer wrong
    report = defect_screen(rows, _falling_losses())
    assert report["accuracy"] == 0.0
    assert report["flags"]["below_chance"] is True


def test_a_rising_loss_or_a_non_finite_loss_is_divergence() -> None:
    rows = [_row(i, gold=i % 4) for i in range(40)]
    rising = [0.1 + 0.001 * k for k in range(100)]
    assert defect_screen(rows, rising)["flags"]["divergence"] is True
    assert defect_screen(rows, [float("nan"), *_falling_losses(20)])["flags"]["divergence"] is True


def test_a_template_at_the_too_good_threshold_is_flagged_for_a_look() -> None:
    rows = [_row(i, template="g", gold=i % 4) for i in range(100)]  # all right, varied gold
    report = defect_screen(rows, _falling_losses())
    assert report["flags"]["too_good"] is True
    assert report["too_good_templates"] == ["g"]


def test_no_predictions_is_refused_not_screened() -> None:
    with pytest.raises(ValueError):
        defect_screen([], [])
