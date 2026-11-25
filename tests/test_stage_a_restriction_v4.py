"""The Stage A choice's D12 restriction (STATE §7, Y6) on synthetic rows: qualification, passing templates, the common set, pairs."""

import pytest

from meddecide.train.stage_a_choice import (
    common_templates,
    passing_templates,
    qualifies,
    restrict_pairs,
)


def _row(template: str, gold: str, predicted: str) -> dict[str, object]:
    return {"template_id": template, "gold_key": gold, "predicted_key": predicted, "correct": gold == predicted,
            "probs": [0.5, 0.5]}


def test_qualification_needs_both_lower_bounds_above_the_limit() -> None:
    assert qualifies(-0.005, -0.009)
    assert not qualifies(-0.01, 0.02)  # the limit itself does not qualify
    assert not qualifies(0.02, -0.02)


def test_a_template_with_a_constant_answer_does_not_pass() -> None:
    balanced = [_row("t_ok", g, g) for g in "AB" * 10]
    constant = [_row("t_bad", g, "A") for g in "AB" * 10]  # balanced gold (majority 0.5), every answer A
    passing = passing_templates(balanced + constant)
    assert "t_ok" in passing
    assert "t_bad" not in passing


def test_the_common_set_is_the_intersection_and_an_empty_one_is_refused() -> None:
    assert common_templates([{"a", "b"}, {"b", "c"}]) == {"b"}
    with pytest.raises(ValueError):
        common_templates([{"a"}, {"b"}])
    with pytest.raises(ValueError):
        common_templates([])


def test_restrict_pairs_keeps_order_and_only_the_given_templates() -> None:
    pairs = [({"template_id": "x", "n": 1}, {"template_id": "x"}), ({"template_id": "y", "n": 2}, {"template_id": "y"}),
             ({"template_id": "x", "n": 3}, {"template_id": "x"})]
    kept = restrict_pairs(pairs, {"x"})
    assert [candidate["n"] for candidate, _ in kept] == [1, 3]
