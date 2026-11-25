"""MedDecide items -> Unsloth decision rows: gold mapping per question type and counted refusals."""

from __future__ import annotations

import pytest

from meddecide.train.unsloth_rows import ConversionError, to_unsloth_row, to_unsloth_rows


def _item(qtype: str, options: list[dict], gold: str, **extra) -> dict:
    return {"item_id": "x1", "template_id": "t", "qtype": qtype, "state": "A plain state.",
            "question": "What is it?", "options": options, "gold": gold, **extra}


def test_choice_row_keeps_keys_and_labels_and_the_gold_key() -> None:
    row = to_unsloth_row(_item("choice", [{"key": "A", "label": "Single"}, {"key": "B", "label": "Double"}], "B"))
    question = row["questions"]["q"]
    assert question == {"type": "choice", "instructions": "What is it?", "criteria": {"A": "Single", "B": "Double"}}
    assert row["gold"]["q"] == {"label": "B"}


def test_score_row_uses_zero_based_level_indices() -> None:
    options = [{"key": str(i), "label": f"level {i}"} for i in range(1, 6)]
    row = to_unsloth_row(_item("score", options, "4"))
    assert row["questions"]["q"]["criteria"] == [f"level {i}" for i in range(1, 6)]
    assert row["gold"]["q"] == {"label": "3"}  # level 4 is index 3


@pytest.mark.parametrize(("gold", "label"), [("yes", "true"), ("no", "false")])
def test_noul_row_maps_yes_no_to_true_false(gold: str, label: str) -> None:
    row = to_unsloth_row(_item("noul", [{"key": "yes", "label": "Yes"}, {"key": "no", "label": "No"}], gold))
    assert row["questions"]["q"] == {"type": "noul", "instructions": "What is it?"}
    assert row["gold"]["q"] == {"label": label}


def test_state_and_identifiers_are_carried_over() -> None:
    row = to_unsloth_row(_item("noul", [{"key": "yes", "label": "Yes"}, {"key": "no", "label": "No"}], "yes"))
    assert row["state"] == "A plain state."
    assert row["item_id"] == "x1" and row["template_id"] == "t"


@pytest.mark.parametrize(("qtype", "options", "gold", "reason"), [
    ("choice", [{"key": "A", "label": "x"}], "A", "choice_needs_two_unique_keys"),
    ("choice", [{"key": "A", "label": "x"}, {"key": "B", "label": "y"}], "C", "choice_gold_not_an_option"),
    ("score", [{"key": "1", "label": "x"}], "1", "score_needs_two_levels"),
    ("score", [{"key": "1", "label": "x"}, {"key": "2", "label": "y"}], "9", "score_gold_outside_levels"),
    ("noul", [{"key": "yes", "label": "Yes"}, {"key": "no", "label": "No"}], "maybe", "noul_gold_not_yes_no"),
    ("ranking", [{"key": "1", "label": "x"}, {"key": "2", "label": "y"}], "1", "unsupported_question_type"),
])
def test_refusals_carry_a_fixed_reason(qtype: str, options: list[dict], gold: str, reason: str) -> None:
    with pytest.raises(ConversionError) as caught:
        to_unsloth_row(_item(qtype, options, gold))
    assert caught.value.reason == reason


def test_batch_conversion_counts_refusals_and_keeps_the_rest() -> None:
    good = _item("noul", [{"key": "yes", "label": "Yes"}, {"key": "no", "label": "No"}], "no")
    bad = _item("noul", [{"key": "yes", "label": "Yes"}, {"key": "no", "label": "No"}], "maybe")
    rows, refused = to_unsloth_rows([good, bad, bad])
    assert len(rows) == 1
    assert refused == {"noul_gold_not_yes_no": 2}
