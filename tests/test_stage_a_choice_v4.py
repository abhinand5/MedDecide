"""osler_v4 Stage A choice rule (ADVISORY Y6): qualification by the lower bounds, the highest CG v2 run, and the tie to A0."""

from __future__ import annotations

import pytest

from meddecide.train.stage_a_choice import ArmResult, choose


def _a0(cg: float = 0.80) -> ArmResult:
    return ArmResult("A0", None, None, cg)


def test_the_reference_alone_is_chosen():
    assert choose([_a0()])["chosen"] == "A0"


def test_a_qualifying_run_with_a_clear_cg_gain_is_chosen():
    result = choose([_a0(0.800), ArmResult("A4", 0.001, -0.002, 0.815)])
    assert result["chosen"] == "A4"
    assert result["status"]["A4"] == "qualifies"


def test_a_gain_under_the_tie_margin_goes_to_the_reference():
    result = choose([_a0(0.800), ArmResult("A4", 0.001, -0.002, 0.802)])
    assert result["chosen"] == "A0"
    assert "a tie" in result["reason"]


def test_a_run_with_a_lower_bound_at_the_limit_does_not_qualify():
    # the limit is strict: a lower bound equal to -0.01 fails
    result = choose([_a0(0.800), ArmResult("A4", -0.01, 0.0, 0.900)])
    assert result["chosen"] == "A0"
    assert result["status"]["A4"].startswith("does not qualify: MR")


def test_a_run_that_fails_on_knowledge_is_excluded_even_with_the_best_cg():
    results = [_a0(0.800), ArmResult("A4", 0.0, -0.02, 0.950), ArmResult("A5", 0.0, 0.0, 0.820)]
    result = choose(results)
    assert result["chosen"] == "A5"
    assert "knowledge" in result["status"]["A4"]


def test_the_best_qualifying_run_wins_among_several():
    results = [_a0(0.800), ArmResult("A1", 0.0, 0.0, 0.810), ArmResult("A3", 0.0, 0.0, 0.830),
               ArmResult("A4", 0.0, 0.0, 0.820)]
    assert choose(results)["chosen"] == "A3"


def test_a_missing_reference_or_missing_bounds_or_a_repeated_arm_is_an_error():
    with pytest.raises(ValueError, match="reference A0 is missing"):
        choose([ArmResult("A4", 0.0, 0.0, 0.9)])
    with pytest.raises(ValueError, match="lower bounds"):
        choose([_a0(), ArmResult("A4", None, 0.0, 0.9)])
    with pytest.raises(ValueError, match="appears twice"):
        choose([_a0(), _a0()])
