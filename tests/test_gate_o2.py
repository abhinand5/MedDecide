"""Gate O2 verdict rules (PROGRAM D31), tested on synthetic differences: the boundaries and the NOT MEASURED path."""

from __future__ import annotations

from meddecide.eval.gate import Comparison
from meddecide.eval.gate_o2 import o2_verdict, panel_non_inferior, panel_superior
from meddecide.eval.paired import Difference


def diff(lo: float, hi: float) -> Difference:
    return Difference(n=100, point=(lo + hi) / 2, lo=lo, hi=hi)


def comparison(acc_lo: float, brier_hi: float, baseline: str) -> Comparison:
    return Comparison(scope="scope", baseline=baseline, accuracy=diff(acc_lo, acc_lo + 0.1),
                      brier=diff(brier_hi - 0.1, brier_hi))


def passing(scope: dict[str, float]) -> dict[str, Comparison | None]:
    return {name: comparison(0.01, -0.01, name) for name in scope}


def test_panel_rule_is_non_inferiority_with_a_strict_floor() -> None:
    assert panel_non_inferior(diff(-0.0199, 0.01))
    assert not panel_non_inferior(diff(-0.02, 0.01))  # the floor itself fails
    assert not panel_non_inferior(diff(-0.05, -0.03))


def test_panel_superiority_is_not_needed() -> None:
    # the interval may contain zero: non-inferiority is enough for D31 (ii)
    assert panel_non_inferior(diff(-0.01, 0.03))


def test_panel_superiority_is_a_separate_check_at_zero() -> None:
    assert panel_superior(diff(0.001, 0.03))
    assert not panel_superior(diff(0.0, 0.03))  # the lower bound at zero is not superior
    assert not panel_superior(diff(-0.01, 0.03))


def test_superiority_never_enters_the_verdict() -> None:
    # non-inferior to both baselines but not superior (the interval contains zero): the verdict is still PASS
    mr = {name: comparison(0.01, -0.01, name) for name in ("MedDecider-4B", "zero-shot Qwen3.5-4B")}
    panel = {name: Comparison(scope="external_panel", baseline=name, accuracy=diff(-0.01, 0.03), brier=diff(-0.1, 0.0))
             for name in ("MedDecider-4B", "zero-shot Qwen3.5-4B")}
    verdict, _ = o2_verdict(mr, panel, diff(0.01, 0.05))
    assert verdict == "PASS"


def test_mr_rule_needs_both_bounds_strictly() -> None:
    assert comparison(0.001, -0.001, "b").passes
    assert not comparison(0.0, -0.001, "b").passes  # accuracy lower bound at zero fails
    assert not comparison(0.001, 0.0, "b").passes  # Brier upper bound at zero fails


BASELINES = ("MedDecider-4B", "zero-shot Qwen3.5-4B")


def test_pass_when_every_primary_check_passes() -> None:
    verdict, reason = o2_verdict(passing(dict.fromkeys(BASELINES)), passing(dict.fromkeys(BASELINES)), diff(-0.01, 0.02))
    assert verdict == "PASS"
    assert "every primary check passes" in reason


def test_fail_when_one_mr_comparison_fails() -> None:
    mr = passing(dict.fromkeys(BASELINES))
    mr["MedDecider-4B"] = comparison(0.01, 0.0, "MedDecider-4B")  # Brier upper bound at zero
    verdict, reason = o2_verdict(mr, passing(dict.fromkeys(BASELINES)), diff(-0.01, 0.02))
    assert verdict == "FAIL"
    assert "MR suite vs MedDecider-4B" in reason


def test_fail_when_the_panel_is_not_non_inferior() -> None:
    panel = passing(dict.fromkeys(BASELINES))
    panel["zero-shot Qwen3.5-4B"] = comparison(-0.03, 0.0, "zero-shot Qwen3.5-4B")
    verdict, reason = o2_verdict(passing(dict.fromkeys(BASELINES)), panel, diff(-0.01, 0.02))
    assert verdict == "FAIL"
    assert "external panel vs zero-shot Qwen3.5-4B" in reason


def test_fail_when_the_knowledge_guard_fails() -> None:
    verdict, reason = o2_verdict(passing(dict.fromkeys(BASELINES)), passing(dict.fromkeys(BASELINES)), diff(-0.03, 0.0))
    assert verdict == "FAIL"
    assert "knowledge guard" in reason


def test_not_measured_when_a_primary_comparison_is_missing() -> None:
    mr = passing(dict.fromkeys(BASELINES))
    mr["MedDecider-4B"] = None
    verdict, reason = o2_verdict(mr, passing(dict.fromkeys(BASELINES)), diff(-0.01, 0.02))
    assert verdict == "NOT MEASURED"
    assert "MR suite vs MedDecider-4B" in reason


def test_not_measured_when_the_guard_is_missing() -> None:
    verdict, reason = o2_verdict(passing(dict.fromkeys(BASELINES)), passing(dict.fromkeys(BASELINES)), None)
    assert verdict == "NOT MEASURED"
    assert "knowledge guard" in reason
