from __future__ import annotations

from datetime import date

import pytest

from meddecide.bench.schema import make_item
from meddecide.eval.jev_harness import VERBALIZER_VARIANT, harness_row


def _item(qtype: str, options: list[str], gold: str):
    return make_item(
        tier="fresh",
        source="pubmed",
        source_record_id="rec-1",
        source_url="https://example.org/rec-1",
        source_license="CC0-1.0",
        record_date=date(2026, 5, 1),
        split="test",
        template_id="t_example_v1",
        skill="record-claim consistency",
        qtype=qtype,
        state="state text",
        question="question text",
        options=[{"key": key, "label": key} for key in options],
        gold=gold,
        option_order_seed=1,
    )


def _f7(item, probs: list[float], correct: bool) -> dict:
    return {
        "item_id": item.item_id,
        "probs": probs,
        "correct": correct,
        "label_mass": sum(probs),
        "prompt_tokens": 40,
    }


def test_noul_probabilities_are_rekeyed_by_yes_no_not_by_slot_position():
    item = _item("noul", ["yes", "no"], gold="no")
    # head slots are [false, true]; p(yes)=0.2 and p(no)=0.8 must land on yes and no
    row = harness_row(_f7(item, [0.8, 0.2], correct=True), item, run_id="r", model_id="m")
    assert row["original_option_keys"] == ["yes", "no"]
    assert row["option_probs"] == pytest.approx([0.2, 0.8])
    assert row["argmax_key"] == "B"
    assert row["gold_key"] == "B"
    assert row["correct"] is True
    assert row["transform"] == {"readout": VERBALIZER_VARIANT}


def test_choice_probabilities_keep_offered_order_and_letter_keys():
    item = _item("choice", ["A", "B", "C"], gold="C")
    row = harness_row(_f7(item, [0.1, 0.3, 0.6], correct=True), item, run_id="r", model_id="m")
    assert row["option_keys"] == ["A", "B", "C"]
    assert row["option_probs"] == pytest.approx([0.1, 0.3, 0.6])
    assert row["argmax_key"] == "C"
    assert row["gold_key"] == "C"


def test_tie_breaks_to_the_first_option_like_numpy_argmax():
    item = _item("choice", ["A", "B"], gold="B")
    row = harness_row(_f7(item, [0.5, 0.5], correct=False), item, run_id="r", model_id="m")
    assert row["argmax_key"] == "A"
    assert row["correct"] is False


def test_disagreement_with_the_f7_correct_flag_is_an_error():
    item = _item("choice", ["A", "B"], gold="A")
    with pytest.raises(ValueError, match="disagrees"):
        harness_row(_f7(item, [0.2, 0.8], correct=True), item, run_id="r", model_id="m")


def test_slot_count_mismatch_is_an_error():
    item = _item("choice", ["A", "B", "C"], gold="A")
    with pytest.raises(ValueError, match="slots"):
        harness_row(_f7(item, [0.5, 0.5], correct=True), item, run_id="r", model_id="m")

