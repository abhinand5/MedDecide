"""Gate O1 paired comparisons from prediction rows (PROGRAM D24): item accounting, signs and the knowledge statistic (CPU)."""

from __future__ import annotations

import numpy as np
import pytest

from meddecide.eval.gate import accuracy_difference, compare_rows, gate_verdict, knowledge_guard


def _row(item_id: str, probs: list[float], gold: str, status: str = "scored") -> dict:
    keys = ["A", "B", "C"]
    return {"item_id": item_id, "option_keys": keys, "probs": probs, "gold": gold, "status": status,
            "template_id": "t"}


def _scope(n_per_template: int = 60, templates: tuple[str, ...] = ("t1", "t2", "t3")) -> list[dict]:
    rows = []
    for t in templates:
        for i in range(n_per_template):
            rows.append({"item_id": f"{t}-{i:04d}", "template_id": t})
    return rows


def _osler_and_baseline(scope: list[dict], *, p_osler: float, p_base: float, seed: int = 0):
    rng = np.random.default_rng(seed)
    osler, base = {}, {}
    for s in scope:
        gold = "B"
        ok_o = rng.random() < p_osler
        ok_b = rng.random() < p_base
        pred_o = gold if ok_o else "A"
        pred_b = gold if ok_b else "C"
        osler[s["item_id"]] = _row(s["item_id"], [0.8 if k == pred_o else 0.1 for k in "ABC"], gold)
        base[s["item_id"]] = _row(s["item_id"], [0.8 if k == pred_b else 0.1 for k in "ABC"], gold)
    return osler, base


def test_a_clearly_better_osler_passes_both_bounds() -> None:
    scope = _scope()
    osler, base = _osler_and_baseline(scope, p_osler=0.95, p_base=0.5)
    result = compare_rows("headline_seen", "MedDecider-4B", scope, osler, base, n_resamples=300, seed=1)
    assert result.n_paired == len(scope)
    assert result.comparison.accuracy.point > 0 and result.comparison.accuracy.lo > 0
    assert result.comparison.brier.point < 0 and result.comparison.brier.hi < 0
    assert result.comparison.passes
    assert gate_verdict([result.comparison])


def test_a_baseline_that_is_better_fails_the_gate() -> None:
    scope = _scope()
    osler, base = _osler_and_baseline(scope, p_osler=0.4, p_base=0.9, seed=2)
    result = compare_rows("headline_seen", "zero-shot", scope, osler, base, n_resamples=300, seed=1)
    assert result.comparison.accuracy.point < 0
    assert not result.comparison.passes
    assert not gate_verdict([result.comparison])


def test_skipped_missing_and_unscored_items_are_counted_not_dropped_silently() -> None:
    scope = _scope(n_per_template=10, templates=("t1", "t2"))
    osler, base = _osler_and_baseline(scope, p_osler=0.9, p_base=0.5, seed=3)
    base[scope[0]["item_id"]]["status"] = "skipped"  # the baseline skipped one item
    del base[scope[1]["item_id"]]  # the baseline has no row for another
    del osler[scope[2]["item_id"]]  # Osler has no row for a third
    result = compare_rows("s", "b", scope, osler, base, n_resamples=50, seed=0)
    assert result.n_scope == 20
    assert result.n_baseline_skipped == 1
    assert result.n_baseline_missing == 1
    assert result.n_osler_missing == 1
    assert result.n_paired == 17
    assert result.n_paired + result.n_baseline_skipped + result.n_baseline_missing + result.n_osler_missing == result.n_scope


def test_a_scope_with_no_paired_items_is_refused() -> None:
    scope = _scope(n_per_template=3, templates=("t1",))
    with pytest.raises(ValueError, match="no paired items"):
        compare_rows("s", "b", scope, {}, {}, n_resamples=10)


def test_accuracy_difference_is_the_micro_accuracy_and_the_guard_threshold_applies() -> None:
    scope = _scope(n_per_template=50, templates=("t1", "t2"))
    osler, base = _osler_and_baseline(scope, p_osler=0.7, p_base=0.7, seed=4)
    diff, n = accuracy_difference(scope, osler, base, n_resamples=200, seed=0)
    assert n == len(scope)
    micro_o = np.mean([osler[s["item_id"]]["probs"][1] > 0.5 for s in scope])
    micro_b = np.mean([base[s["item_id"]]["probs"][1] > 0.5 for s in scope])
    assert diff.point == pytest.approx(micro_o - micro_b)
    assert knowledge_guard(diff) == (diff.lo > -0.02)
