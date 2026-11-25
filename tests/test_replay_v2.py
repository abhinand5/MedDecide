"""Unit tests for the osler_v2 share-alike replay converters (ADVISORY W2; D36)."""

from __future__ import annotations

import pytest

from meddecide.mix.replay_v2 import (
    ARC_TEMPLATE,
    BOOLQ_TEMPLATE,
    SNLI_TEMPLATE,
    arc_row,
    boolq_row,
    snli_row,
)


def _arc(answer: str = "2", labels: list[str] | None = None) -> dict:
    labels = labels or ["1", "2", "3", "4"]
    return {"id": "Mercury_1", "question": "Which planet is closest to the sun?",
            "choices": {"label": labels, "text": ["Venus", "Mercury", "Mars", "Earth"][:len(labels)]},
            "answerKey": answer}


def test_arc_rekeys_numeric_labels_to_letters_and_keeps_the_gold_text() -> None:
    row, reason = arc_row(_arc(answer="2"), split="train", config="ARC-Easy")
    assert reason == "kept"
    assert row is not None
    assert row["template_id"] == ARC_TEMPLATE
    assert row["qtype"] == "choice"
    assert row["gold"] == "B"
    assert [(o["key"], o["label"]) for o in row["options"]] == [
        ("A", "Venus"), ("B", "Mercury"), ("C", "Mars"), ("D", "Earth")]


def test_arc_drops_rows_without_a_key_offered() -> None:
    row, reason = arc_row(_arc(answer="9"), split="train", config="ARC-Challenge")
    assert row is None
    assert reason == "answer_missing"


def test_arc_drops_rows_with_mismatched_choice_lists() -> None:
    rec = _arc()
    rec["choices"]["text"] = ["only one"]
    row, reason = arc_row(rec, split="train", config="ARC-Easy")
    assert row is None
    assert reason == "bad_choices"


def test_boolq_is_a_yes_no_item_with_the_human_answer() -> None:
    rec = {"question": "do iran and afghanistan speak the same language", "answer": True,
           "passage": "Persian language. Persian is also known as Farsi and is spoken in Iran and Afghanistan."}
    row, reason = boolq_row(rec, split="train", index=7)
    assert reason == "kept"
    assert row is not None
    assert row["template_id"] == BOOLQ_TEMPLATE
    assert row["qtype"] == "noul"
    assert row["gold"] == "yes"
    assert row["question"].endswith("?")


def test_boolq_false_answer_maps_to_no() -> None:
    rec = {"question": "is x", "answer": False, "passage": "Some passage."}
    row, _ = boolq_row(rec, split="train", index=0)
    assert row is not None
    assert row["gold"] == "no"


def test_boolq_drops_a_missing_answer() -> None:
    row, reason = boolq_row({"question": "is x", "answer": None, "passage": "p"}, split="train", index=0)
    assert row is None
    assert reason == "answer_missing"


@pytest.mark.parametrize("label, gold", [(0, "A"), (1, "B"), (2, "C")])
def test_snli_maps_the_gold_label_to_its_option(label: int, gold: str) -> None:
    row, reason = snli_row({"premise": "A man runs.", "hypothesis": "A person moves.", "label": label},
                           split="train", index=3)
    assert reason == "kept"
    assert row is not None
    assert row["template_id"] == SNLI_TEMPLATE
    assert row["gold"] == gold
    assert len(row["options"]) == 3


def test_snli_drops_rows_without_a_gold_label() -> None:
    row, reason = snli_row({"premise": "p", "hypothesis": "h", "label": -1}, split="train", index=0)
    assert row is None
    assert reason == "no_gold_label"
