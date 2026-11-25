"""osler_v5 Z2 choice rule (pre-registered): qualifiers, the highest CG micro, the tie band and the fuller-fix tie-break."""

from __future__ import annotations

import pytest

from meddecide.eval.arm_choice import ArmStats, choose

ORDER = ("R0", "R0p", "R2", "R3")


def _arm(name: str, cg: float, *, pos: bool = True, know: float | None = 0.0, mr: float | None = 0.0,
         mimic: float | None = 0.0) -> ArmStats:
    return ArmStats(name=name, cg_micro=cg, positional_passes=pos, knowledge_lower=know, mr_lower=mr, mimic_lower=mimic)


def test_the_bounds_are_strict_at_the_margins() -> None:
    assert _arm("R2", 0.8, know=-0.01).qualifies() is False  # not above -0.01
    assert _arm("R2", 0.8, know=-0.0099).qualifies() is True
    assert _arm("R2", 0.8, mr=-0.03).qualifies() is False
    assert _arm("R2", 0.8, mimic=-0.0299).qualifies() is True
    assert _arm("R2", 0.8, pos=False).qualifies() is False


def test_the_control_is_its_own_reference_and_qualifies_on_positional_health_alone() -> None:
    control = ArmStats("R0", 0.79, True, None, None, None)
    assert control.qualifies() is True
    assert ArmStats("R0", 0.79, False, None, None, None).qualifies() is False


def test_the_highest_cg_micro_among_the_qualifiers_is_chosen() -> None:
    arms = [_arm("R0", 0.74), _arm("R0p", 0.74, know=-0.02), _arm("R2", 0.78), _arm("R3", 0.77)]
    out = choose(arms, control="R0", repeat="R0p", order=ORDER)
    assert out["chosen"] == "R2" and out["outcome"] == "chosen"
    assert "R0p" not in out["tied"]  # R0p fails the knowledge bound


def test_within_the_tie_band_the_fuller_fix_wins() -> None:
    # R0 and R0' differ by 0.004, so the band is the 0.005 floor; R2 and R3 are 0.003 apart, inside the band
    arms = [_arm("R0", 0.740), _arm("R0p", 0.744), _arm("R2", 0.780), _arm("R3", 0.777)]
    out = choose(arms, control="R0", repeat="R0p", order=ORDER)
    assert out["chosen"] == "R3" and set(out["tied"]) == {"R2", "R3"}


def test_a_seed_repeat_further_apart_widens_the_band() -> None:
    # |R0 - R0'| = 0.02 widens the band to 0.02, so R3 (0.015 below R2) is tied and wins
    arms = [_arm("R0", 0.700), _arm("R0p", 0.720), _arm("R2", 0.780), _arm("R3", 0.765)]
    out = choose(arms, control="R0", repeat="R0p", order=ORDER)
    assert out["rule"]["tie_band"] == pytest.approx(0.02)
    assert out["chosen"] == "R3"


def test_no_qualifier_is_outcome_four_and_nothing_is_chosen() -> None:
    arms = [_arm("R0", 0.74, pos=False), _arm("R0p", 0.74, pos=False), _arm("R2", 0.78, pos=False), _arm("R3", 0.77, mr=-0.05)]
    out = choose(arms, control="R0", repeat="R0p", order=ORDER)
    assert out["chosen"] is None and "outcome 4" in out["outcome"]


def test_the_choice_refuses_missing_or_repeated_arms() -> None:
    with pytest.raises(ValueError, match="must be among"):
        choose([_arm("R0", 0.7), _arm("R2", 0.7)], control="R0", repeat="R0p", order=ORDER)
    with pytest.raises(ValueError, match="named twice"):
        choose([_arm("R0", 0.7), _arm("R0", 0.7), _arm("R0p", 0.7)], control="R0", repeat="R0p", order=ORDER)
    with pytest.raises(ValueError, match="missing from the tie order"):
        choose([_arm("R0", 0.7), _arm("R0p", 0.7), _arm("X", 0.7)], control="R0", repeat="R0p", order=ORDER)
