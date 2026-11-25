"""Unit tests for the osler_v6 pre-registered choice rule (ADVISORY §2.6)."""

from __future__ import annotations

import pytest

from meddecide.eval.osler_v6_choice import ArmChoice, ControlChoice, choose_recipe, failed_checks


def arm(name: str = "G1s0", **overrides) -> ArmChoice:
    values = {"knowledge_dev": 0.70, "knowledge_lower": 0.005, "mr_lower": -0.01, "mimic_lower": -0.01,
              "cg_micro": 0.80, "positional_all_orders": True}
    values.update(overrides)
    return ArmChoice(name=name, **values)


CONTROL = ControlChoice(cg_micro=0.79, knowledge_dev=0.69)


def test_a_clean_arm_qualifies_and_is_chosen() -> None:
    report = choose_recipe([arm()], CONTROL)
    assert report.chosen == "G1s0" and report.rows["G1s0"]["failed"] == []


def test_the_knowledge_limit_is_strict() -> None:
    assert failed_checks(arm(knowledge_lower=-0.01), CONTROL)  # exactly the limit does not pass
    assert not failed_checks(arm(knowledge_lower=-0.0099), CONTROL)


def test_the_mr_and_mimic_limits_are_strict() -> None:
    assert any("MR lower" in reason for reason in failed_checks(arm(mr_lower=-0.03), CONTROL))
    assert any("MIMIC lower" in reason for reason in failed_checks(arm(mimic_lower=-0.03), CONTROL))
    assert not failed_checks(arm(mr_lower=-0.0299, mimic_lower=-0.0299), CONTROL)


def test_cg_must_not_be_below_the_control_and_may_be_equal() -> None:
    assert any("CG v2" in reason for reason in failed_checks(arm(cg_micro=0.7899), CONTROL))
    assert not failed_checks(arm(cg_micro=CONTROL.cg_micro), CONTROL)


def test_positional_health_is_required() -> None:
    assert any("positional" in reason for reason in failed_checks(arm(positional_all_orders=False), CONTROL))


def test_no_qualifier_is_outcome_4() -> None:
    report = choose_recipe([arm(positional_all_orders=False), arm(name="G1s1", knowledge_lower=-0.02)], CONTROL)
    assert report.chosen is None
    assert report.outcome.startswith("(4)")


def test_the_highest_knowledge_dev_v2_qualifier_wins() -> None:
    report = choose_recipe([arm(name="A", knowledge_dev=0.70), arm(name="B", knowledge_dev=0.71)], CONTROL)
    assert report.chosen == "B"


def test_a_tie_is_rejected_not_broken() -> None:
    with pytest.raises(ValueError, match="tied"):
        choose_recipe([arm(name="A"), arm(name="B")], CONTROL)


def test_an_empty_candidate_list_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least one"):
        choose_recipe([], CONTROL)
