"""Scoring Osler on the O2 common set (ADVISORY O11): row conversion, option reversal and both-order averaging (CPU)."""

from __future__ import annotations

import numpy as np
import pytest

from meddecide.bench.schema import QuestionType
from meddecide.eval.osler_scoring import average_both_orders, o2_row_to_item, reversed_order


def _row(**overrides) -> dict:
    row = {
        "item_id": "a1b2c3d4e5f60718",
        "benchmark": "v0.2",
        "set_name": "pubmed",
        "template_id": "pubmed_example_choice_v1",
        "qtype": "choice",
        "state": "A patient presents with a headache after a fall.",
        "question": "Which option applies?",
        "gold": "B",
        "options": [{"key": "A", "label": "alpha"}, {"key": "B", "label": "beta"}, {"key": "C", "label": "gamma"}],
        "split": "test",
        "licence": "CC0",
        "source_url": "https://example.org/record/1",
        "source_record_id": "rec-1",
        "meta": {"tier": "fresh", "skill": "triage"},
    }
    row.update(overrides)
    return row


def test_a_choice_row_converts_with_its_keys_gold_tier_and_id() -> None:
    item = o2_row_to_item(_row())
    assert item.qtype is QuestionType.CHOICE
    assert item.option_keys == ["A", "B", "C"]
    assert item.gold == "B" and item.gold_index == 1
    assert item.item_id == "a1b2c3d4e5f60718"
    assert item.tier == "fresh"


def test_noul_and_score_rows_convert() -> None:
    noul = o2_row_to_item(_row(qtype="noul", gold="yes", options=[{"key": "yes", "label": "Yes"},
                                                                 {"key": "no", "label": "No"}]))
    assert noul.qtype is QuestionType.NOUL and noul.option_keys == ["yes", "no"]
    score = o2_row_to_item(_row(qtype="score", gold="2", options=[{"key": "1", "label": "low"},
                                                                 {"key": "2", "label": "mid"},
                                                                 {"key": "3", "label": "high"}]))
    assert score.qtype is QuestionType.SCORE and score.option_keys == ["1", "2", "3"]


def test_a_robustness_style_id_is_kept_as_the_row_says() -> None:
    # the benchmark's derivation gives a different id for perturbed items; the row's id is the join key
    item = o2_row_to_item(_row(item_id="934351d694fd137b", benchmark="robustness", set_name="pubmed"))
    assert item.item_id == "934351d694fd137b"


def test_a_row_with_a_wrong_id_length_is_refused() -> None:
    with pytest.raises(ValueError, match="benchmark id length"):
        o2_row_to_item(_row(item_id="short"))


def test_reversal_moves_the_options_and_keeps_the_gold_with_its_content() -> None:
    item = o2_row_to_item(_row())
    reversed_item = reversed_order(item)
    assert reversed_item is not None
    assert reversed_item.option_keys == ["C", "B", "A"]  # display order reversed, keys travel with the options
    assert reversed_item.gold == "B"  # gold follows its content ("beta")
    assert reversed_item.options[reversed_item.gold_index].label == "beta"


def test_noul_and_score_are_not_reversed() -> None:
    noul = o2_row_to_item(_row(qtype="noul", gold="yes", options=[{"key": "yes", "label": "Yes"},
                                                                 {"key": "no", "label": "No"}]))
    assert reversed_order(noul) is None


def test_both_orders_average_back_to_the_original_positions() -> None:
    original = np.array([0.2, 0.5, 0.3])
    reversed_display = np.array([0.6, 0.2, 0.2])  # display order C, B, A: C is original option 2
    averaged = average_both_orders(original, reversed_display)
    # original option 0 ("A") is shown last in the reversed order (0.2); option 2 ("C") is shown first (0.6)
    assert averaged == pytest.approx(np.array([0.2, 0.35, 0.45]))
    assert averaged.sum() == pytest.approx(1.0)


def test_both_orders_reject_mismatched_vectors() -> None:
    with pytest.raises(ValueError, match="same options"):
        average_both_orders(np.array([0.5, 0.5]), np.array([0.2, 0.3, 0.5]))


def test_the_average_of_two_distributions_is_a_distribution() -> None:
    rng = np.random.default_rng(0)
    for _ in range(20):
        a = rng.dirichlet(np.ones(5))
        b = rng.dirichlet(np.ones(5))
        assert average_both_orders(a, b).sum() == pytest.approx(1.0)
