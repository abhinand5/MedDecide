"""Unsloth answers -> run_student-format prediction rows."""

from __future__ import annotations

import pytest

from meddecide.eval.unsloth_predictions import option_probabilities, prediction_row


def _item(qtype: str, options: list[dict], gold: str) -> dict:
    return {"item_id": "i1", "template_id": "t", "source": "s", "qtype": qtype, "options": options, "gold": gold}


def test_choice_probabilities_follow_the_item_option_order() -> None:
    raw = {"B": 0.2, "A": 0.5, "C": 0.3}
    probs = option_probabilities("choice", raw, ["A", "B", "C"])
    assert probs == pytest.approx([0.5, 0.2, 0.3])


def test_noul_maps_true_false_to_yes_no() -> None:
    raw = {"true": 0.75, "false": 0.25}
    assert option_probabilities("noul", raw, ["yes", "no"]) == pytest.approx([0.75, 0.25])


def test_missing_probability_is_an_error_not_a_silent_zero() -> None:
    with pytest.raises(KeyError):
        option_probabilities("choice", {"A": 1.0}, ["A", "B"])


def test_row_has_the_run_student_fields_and_a_correct_argmax() -> None:
    item = _item("choice", [{"key": "A", "label": "x"}, {"key": "B", "label": "y"}], "B")
    row = prediction_row(item, [0.3, 0.7], model_id="m", run_id="r", split="test", latency_s=0.01)
    assert row["argmax_key"] == "B" and row["correct"] is True
    assert row["variant"] == "unsloth-decision-head"
    assert row["label_mass"] is None  # not a letter readout: the D12 fields do not apply
    assert set(row) >= {"item_id", "option_keys", "option_probs", "gold_key", "correct", "template_id", "qtype"}


def test_score_row_reports_the_expected_level() -> None:
    options = [{"key": str(i), "label": f"l{i}"} for i in range(1, 4)]
    row = prediction_row(_item("score", options, "2"), [0.0, 0.5, 0.5], model_id="m", run_id="r", split="test")
    assert row["expected_level"] == pytest.approx(2.5)
    assert row["argmax_key"] == "2"


def test_probability_count_must_match_the_options() -> None:
    with pytest.raises(ValueError):
        prediction_row(_item("noul", [{"key": "yes", "label": "Yes"}, {"key": "no", "label": "No"}], "no"),
                       [1.0], model_id="m", run_id="r", split="test")


def test_score_levels_are_numbered_from_zero_in_unsloth() -> None:
    raw = {"0": 0.1, "1": 0.6, "2": 0.3}  # Unsloth's level indices for MedDecide levels 1, 2, 3
    assert option_probabilities("score", raw, ["1", "2", "3"]) == pytest.approx([0.1, 0.6, 0.3])
