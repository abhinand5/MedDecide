import pytest

from meddecide.eval.d56c_choice import ArmResult, choose_arm, failed_checks


def arm(name: str, fix_level: int = 1, positional: bool = True, knowledge: float = 0.0, mr: float = 0.0, mimic: float = 0.0,
        cg: float = 0.78) -> ArmResult:
    return ArmResult(name=name, fix_level=fix_level, positional_all_orders=positional, knowledge_lower=knowledge, mr_lower=mr,
                     mimic_lower=mimic, cg_micro=cg)


def test_the_corrected_arms_at_13992_leave_no_qualifier() -> None:
    f0 = arm("F0", fix_level=0, positional=False, cg=0.7740)
    f2 = arm("F2", positional=True, knowledge=-0.0150, mr=-0.0029, mimic=-0.0343, cg=0.7844)
    f2s1 = arm("F2s1", positional=True, knowledge=-0.0115, mr=0.0019, mimic=-0.0420, cg=0.7791)
    report = choose_arm([f0, f2, f2s1])
    assert report["qualifiers"] == []
    assert report["chosen"] is None
    assert report["rows"]["F0"]["failed"] == ["positional health in all orders"]
    assert len(report["rows"]["F2"]["failed"]) == 2  # knowledge and MIMIC; MR passes
    assert len(report["rows"]["F2s1"]["failed"]) == 2  # knowledge and MIMIC; MR passes
    assert report["outcome"].startswith("(4)")


def test_a_bound_exactly_at_the_limit_does_not_qualify() -> None:
    assert failed_checks(arm("A", knowledge=-0.01, mr=-0.03, mimic=-0.03)) == [
        "knowledge lower bound -0.0100 is not above -0.01",
        "MR lower bound -0.0300 is not above -0.03",
        "MIMIC lower bound -0.0300 is not above -0.03",
    ]
    assert choose_arm([arm("A", knowledge=-0.0099, mr=-0.0299, mimic=-0.0299)])["chosen"] == "A"


def test_the_highest_cg_micro_among_qualifiers_is_chosen() -> None:
    report = choose_arm([arm("low", fix_level=0, cg=0.78), arm("high", fix_level=1, cg=0.80)])
    assert report["chosen"] == "high"
    assert report["qualifiers"] == ["low", "high"]


def test_the_fuller_fix_wins_a_tie_within_the_band() -> None:
    control = arm("control", fix_level=0, cg=0.8000)
    fuller = arm("fuller", fix_level=1, cg=0.7950)  # 0.005 below the top, inside the 0.010 band
    assert choose_arm([control, fuller])["chosen"] == "fuller"


def test_a_qualifier_outside_the_band_does_not_win_on_fix_level() -> None:
    control = arm("control", fix_level=0, cg=0.8000)
    fuller = arm("fuller", fix_level=1, cg=0.7800)  # 0.020 below the top, outside the band
    assert choose_arm([control, fuller])["chosen"] == "control"


def test_two_arms_of_the_same_fix_level_in_one_tie_are_refused() -> None:
    with pytest.raises(ValueError, match="does not order them"):
        choose_arm([arm("F2", cg=0.7844), arm("F2s1", cg=0.7791)])


def test_a_non_qualifying_fuller_fix_does_not_take_a_tie() -> None:
    control = arm("control", fix_level=0, cg=0.8000)
    fuller = arm("fuller", fix_level=1, cg=0.7950, positional=False)
    assert choose_arm([control, fuller])["chosen"] == "control"


def test_a_duplicate_arm_name_is_refused() -> None:
    with pytest.raises(ValueError, match="appears twice"):
        choose_arm([arm("F2"), arm("F2")])
