"""d1-2B: the card's question shapes and the answer mapping (pure; the checkpoint is not loaded here)."""

from __future__ import annotations

import math

import pytest

from meddecide.eval.d1_bench import (
    cell_statuses,
    item_micro_accuracy,
    scored_rows,
    template_macro_accuracy,
)
from meddecide.model.d1 import (
    REPO,
    REVISION,
    probabilities_from_answer,
    question_for,
    validate_probabilities,
)

YN = [("yes", "Yes"), ("no", "No")]
LETTERS3 = [("A", "Billing"), ("B", "Shipping"), ("C", "Technical")]
LEVELS = [("A", "Low"), ("B", "Medium"), ("C", "High")]


def test_pinned_revision_is_the_card_checkpoint() -> None:
    assert REPO == "LiquidAI/d1-2B"
    assert REVISION == "10e7671d44e4b37700020b1eed3f594c5f2af0b4"


def test_question_shapes_follow_the_card() -> None:
    assert question_for("noul", "Is it a refund?", YN) == {"type": "noul", "instructions": "Is it a refund?"}
    choice = question_for("choice", "Which team?", LETTERS3)
    assert choice == {"type": "choice", "instructions": "Which team?",
                      "criteria": {"A": "Billing", "B": "Shipping", "C": "Technical"}}
    assert question_for("score", "How urgent?", LEVELS)["criteria"] == ["Low", "Medium", "High"]


@pytest.mark.parametrize(
    ("qtype", "question", "options"),
    [
        ("noul", "q", [("A", "x"), ("B", "y")]),  # noul must offer yes and no
        ("choice", "q", [("A", "x")]),  # one option is not a choice
        ("choice", "q", [("A", "x"), ("A", "y")]),  # keys must be unique
        ("score", "q", [("A", "x")]),  # a score has at least two levels
        ("score", "q", [(str(i), str(i)) for i in range(11)]),  # and at most ten
        ("choice", "  ", LETTERS3),  # empty question
        ("text", "q", LETTERS3),  # not a card type for this harness
    ],
)
def test_question_refuses_malformed_items(qtype: str, question: str, options: list[tuple[str, str]]) -> None:
    with pytest.raises(ValueError):
        question_for(qtype, question, options)


def test_noul_answer_maps_p_yes_to_the_item_order() -> None:
    probs = probabilities_from_answer("noul", [("no", "No"), ("yes", "Yes")], {"type": "noul", "noul": 0.8})
    assert probs == pytest.approx([0.2, 0.8])
    validate_probabilities(probs)


def test_choice_answer_follows_the_item_order() -> None:
    answer = {"type": "choice", "choice": "B", "confidence": 0.7,
              "probabilities": {"A": 0.2, "B": 0.7, "C": 0.1}}
    assert probabilities_from_answer("choice", LETTERS3, answer) == pytest.approx([0.2, 0.7, 0.1])


def test_choice_answer_with_other_keys_is_refused() -> None:
    answer = {"type": "choice", "probabilities": {"B": 0.5, "A": 0.5}}  # order differs from the item's
    with pytest.raises(ValueError):
        probabilities_from_answer("choice", LETTERS3[:2], answer)


def test_score_answer_reads_levels_by_index() -> None:
    answer = {"type": "score", "score": 1.0, "probabilities": {"0": 0.1, "1": 0.6, "2": 0.3}}
    assert probabilities_from_answer("score", LEVELS, answer) == pytest.approx([0.1, 0.6, 0.3])


def test_validate_refuses_bad_vectors() -> None:
    for bad in ([0.5, 0.6], [1.2, -0.2], [math.nan, 1.0], []):
        with pytest.raises(ValueError):
            validate_probabilities(bad)


def _pred(item_id: str, probs: list[float], status: str = "scored") -> dict:
    return {"item_id": item_id, "status": status, "probs": probs}


ITEMS = {
    "i1": {"item_id": "i1", "template_id": "t1", "gold": "A", "options": [{"key": "A"}, {"key": "B"}]},
    "i2": {"item_id": "i2", "template_id": "t1", "gold": "B", "options": [{"key": "A"}, {"key": "B"}]},
    "i3": {"item_id": "i3", "template_id": "t2", "gold": "A", "options": [{"key": "A"}, {"key": "B"}]},
}


def test_macro_and_micro_use_the_scoreboard_definitions() -> None:
    preds = [_pred("i1", [0.9, 0.1]), _pred("i2", [0.9, 0.1]), _pred("i3", [0.2, 0.8])]
    rows = scored_rows(preds, ITEMS)
    assert [r["correct"] for r in rows] == [True, False, False]
    assert item_micro_accuracy(rows) == pytest.approx(1 / 3)
    assert template_macro_accuracy(rows) == pytest.approx((0.5 + 0.0) / 2)  # t1 = 1/2, t2 = 0/1


def test_a_skipped_or_unknown_prediction_is_an_error() -> None:
    with pytest.raises(ValueError):
        scored_rows([_pred("i1", [0.5, 0.5], status="skipped")], ITEMS)
    with pytest.raises(ValueError):
        scored_rows([_pred("zz", [0.5, 0.5])], ITEMS)


def test_cell_status_reports_constant_answers() -> None:
    items = {f"c{i}": {"item_id": f"c{i}", "template_id": "const", "gold": "A" if i % 2 else "B",
                       "options": [{"key": "A"}, {"key": "B"}]} for i in range(40)}
    rows = scored_rows([_pred(f"c{i}", [0.9, 0.1]) for i in range(40)], items)
    status = cell_statuses(rows)["const"]["status"]
    assert status.startswith("READOUT_FAIL")
    assert "constant answer" in status
