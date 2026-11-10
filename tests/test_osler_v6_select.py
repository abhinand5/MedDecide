"""Unit tests for the osler_v6 final's checkpoint selection: **A1** (the final at 63,304 only) and the retired
best-S rule (kept for the loop's record; the Z6 driver no longer calls it)."""

from __future__ import annotations

import pytest

from meddecide.eval.osler_v6_select import (
    FINAL_STEP,
    failed_positional_orders,
    select_by_s,
    select_final_step,
)


def evaluation(s: float, passes: bool, *, mr: float | None = None, mimic: float | None = None,
               cg: float | None = None, knowledge: float = 0.5) -> dict:
    mr = s if mr is None else mr
    mimic = s if mimic is None else mimic
    cg = s if cg is None else cg
    return {"composite": {"S": s, "mr": mr, "mimic": mimic, "cg_micro": cg},
            "positional_passes_all_orders": passes, "knowledge_dev_accuracy": knowledge}


def test_the_highest_s_among_the_passes_is_chosen() -> None:
    selection = select_by_s({"1000": evaluation(0.70, True), "2000": evaluation(0.75, True)})
    assert selection.chosen_step == 2000 and selection.chosen_s == pytest.approx(0.75)
    assert selection.eligible == (1000, 2000)


def test_a_higher_s_that_fails_positional_health_is_not_eligible() -> None:
    selection = select_by_s({"1000": evaluation(0.70, True), "2000": evaluation(0.90, False)})
    assert selection.chosen_step == 1000
    assert selection.table[1]["positional_passes_all_orders"] is False


def test_no_pass_is_blocked_not_a_fallback() -> None:
    selection = select_by_s({"1000": evaluation(0.70, False), "2000": evaluation(0.75, False)})
    assert selection.chosen_step is None
    assert selection.outcome.startswith("BLOCKED")


def test_a_tie_is_rejected_not_broken() -> None:
    with pytest.raises(ValueError, match="tied"):
        select_by_s({"1000": evaluation(0.70, True), "2000": evaluation(0.70, True)})


def test_an_empty_table_is_rejected() -> None:
    with pytest.raises(ValueError, match="no evaluations"):
        select_by_s({})


def test_the_table_keeps_every_evaluation_in_step_order() -> None:
    selection = select_by_s({"2000": evaluation(0.75, True), "1000": evaluation(0.70, True)})
    assert [row["step"] for row in selection.table] == [1000, 2000]
    assert selection.table[0]["S"] == pytest.approx(0.70)

# --------------------------------------------------------------------------- A1: the final at 63,304 only
def a1_evaluation(s: float, passes: bool, *, failed: list[str] | None = None, knowledge: float = 0.5) -> dict:
    """An evaluation record in the shape the driver writes (original + two permuted orders)."""
    failed = failed or []
    return {
        "composite": {"S": s, "mr": s, "mimic": s, "cg_micro": s},
        "positional_passes_all_orders": passes,
        "knowledge_dev_accuracy": knowledge,
        "positional_original": {"passes": "original" not in failed},
        "positional_permuted": {"1": {"passes": "seed1" not in failed}, "2": {"passes": "seed2" not in failed}},
    }


def test_a1_the_final_step_is_chosen_when_it_passes_positional_health() -> None:
    evaluations = {"6330": a1_evaluation(0.79, True), str(FINAL_STEP): a1_evaluation(0.83, True)}
    selection = select_final_step(evaluations)
    assert selection.chosen_step == FINAL_STEP
    assert selection.chosen_s == pytest.approx(0.83)
    assert selection.eligible == (FINAL_STEP,)


def test_a1_an_earlier_pass_never_rescues_a_final_failure() -> None:
    """The retired rule would have chosen the best intermediate S; A1 must return BLOCKED instead."""
    evaluations = {"18991": a1_evaluation(0.90, True), str(FINAL_STEP): a1_evaluation(0.80, False, failed=["original"])}
    selection = select_final_step(evaluations)
    assert selection.chosen_step is None
    assert selection.outcome.startswith("BLOCKED")
    assert "63,304" in selection.outcome


def test_a1_a_missing_final_step_is_blocked() -> None:
    selection = select_final_step({"6330": a1_evaluation(0.79, True), "12661": a1_evaluation(0.80, True)})
    assert selection.chosen_step is None
    assert "not present" in selection.outcome


def test_a1_the_table_keeps_every_evaluation_including_ineligible_ones() -> None:
    evaluations = {"6330": a1_evaluation(0.79, True), "18991": a1_evaluation(0.90, False, failed=["original"]),
                   str(FINAL_STEP): a1_evaluation(0.83, True)}
    selection = select_final_step(evaluations)
    assert [row["step"] for row in selection.table] == [6330, 18991, FINAL_STEP]
    assert [row["positional_passes_all_orders"] for row in selection.table] == [True, False, True]


def test_a1_the_failing_orders_are_named() -> None:
    assert failed_positional_orders(a1_evaluation(0.8, False, failed=["original", "seed2"])) == ["original", "seed2"]
    assert failed_positional_orders(a1_evaluation(0.8, True)) == []


def test_a1_an_empty_table_is_rejected() -> None:
    with pytest.raises(ValueError, match="no evaluations"):
        select_final_step({})

