"""Unit tests for the osler_v6 final's checkpoint selection by S (ADVISORY §2.7)."""

from __future__ import annotations

import pytest

from meddecide.eval.osler_v6_select import select_by_s


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
