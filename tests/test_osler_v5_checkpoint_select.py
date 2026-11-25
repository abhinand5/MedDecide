"""osler_v5 Z3 checkpoint selection: the highest S among the evaluations that pass positional health; ties go to the later step."""

from __future__ import annotations

from meddecide.eval.checkpoint_select import select_checkpoint


def _record(s: float, passes: bool) -> dict:
    return {"composite": {"S": s}, "positional_passes_all_orders": passes}


def test_the_highest_s_among_passing_evaluations_is_selected() -> None:
    records = {6330: _record(0.85, True), 12660: _record(0.90, False), 18990: _record(0.88, True), 25320: _record(0.86, True)}
    out = select_checkpoint(records)
    assert out["selected"] == 18990 and out["eligible_steps"] == [6330, 18990, 25320]


def test_an_exact_tie_goes_to_the_later_step() -> None:
    out = select_checkpoint({100: _record(0.5, True), 200: _record(0.5, True)})
    assert out["selected"] == 200


def test_no_passing_evaluation_selects_nothing() -> None:
    out = select_checkpoint({100: _record(0.9, False), 200: _record(0.8, False)})
    assert out["selected"] is None and "outcome 4" in out["outcome"]
