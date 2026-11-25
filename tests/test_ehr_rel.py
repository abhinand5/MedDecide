"""Unit tests for the EHR-Rel score converter (osler_v2 ADVISORY W2 step 2)."""

from __future__ import annotations

import pytest

from meddecide.mix.ehr_rel import TEMPLATE, ehr_rel_row, gold_level, pair_key, split_of


@pytest.mark.parametrize("mean, level", [(0.0, "1"), (0.49, "1"), (0.5, "2"), (1.5, "3"), (2.5, "4"), (3.0, "4")])
def test_gold_is_the_mean_rounded_half_up(mean: float, level: str) -> None:
    assert gold_level(mean) == level


@pytest.mark.parametrize("mean", [-0.1, 3.01])
def test_a_mean_off_the_scale_is_an_error(mean: float) -> None:
    with pytest.raises(ValueError):
        gold_level(mean)


def test_pair_key_ignores_order() -> None:
    assert pair_key("13645005", "79955004") == pair_key("79955004", "13645005")


def test_split_is_stable_and_by_pair() -> None:
    key = pair_key("13645005", "79955004")
    assert split_of(key) == split_of(key)
    assert split_of(key) in {"train", "dev", "test"}


def _rec(**over: str) -> dict[str, str]:
    base = {"snomed_id_1": "13645005", "snomed_label_1": "Chronic obstructive lung disease",
            "snomed_id_2": "79955004", "snomed_label_2": "Chronic cor pulmonale", "mean_rating": "2.0"}
    base.update(over)
    return base


def test_a_pair_becomes_a_score_row_with_four_ordered_levels() -> None:
    row, reason = ehr_rel_row(_rec(), subset="RelB")
    assert reason == "kept"
    assert row is not None
    assert row["template_id"] == TEMPLATE
    assert row["qtype"] == "score"
    assert [o["key"] for o in row["options"]] == ["1", "2", "3", "4"]
    assert row["gold"] == "3"
    assert "Chronic cor pulmonale" in row["state"]


@pytest.mark.parametrize("over, reason", [
    ({"mean_rating": ""}, "no_mean_rating"),
    ({"mean_rating": "x"}, "bad_mean_rating"),
    ({"mean_rating": "3.5"}, "rating_off_scale"),
    ({"snomed_id_2": "13645005"}, "same_concept"),
    ({"snomed_label_2": " "}, "blank_concept"),
])
def test_bad_lines_are_dropped_with_a_reason(over: dict[str, str], reason: str) -> None:
    row, got = ehr_rel_row(_rec(**over), subset="RelA")
    assert row is None
    assert got == reason
