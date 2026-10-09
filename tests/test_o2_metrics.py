"""Unit tests for the osler_v0 O2 scoreboard metrics (hand-computed cases)."""

from __future__ import annotations

import pytest

from meddecide.eval.o2_metrics import (
    cell_health,
    flip_rates,
    label_for,
    macro_over_templates,
    score_row,
    summarise,
)


def _row(item_id, template, gold, keys, probs=None, *, status="scored", reason=None, latency=0.1, extras=None):
    return {"item_id": item_id, "benchmark": "ext_panel", "set_name": "s", "template_id": template,
            "qtype": "choice", "option_keys": keys, "gold": gold, "status": status, "reason": reason,
            "probs": probs if status == "scored" else None, "latency_s": latency if status == "scored" else None,
            "prompt_tokens": 10, "extras": extras or {}}


def test_score_row_argmax_correctness_brier_and_confidence() -> None:
    it = score_row(_row("a", "t1", "B", ["A", "B", "C"], [0.2, 0.5, 0.3]))
    assert it.predicted == "B" and it.correct and it.confidence == pytest.approx(0.5)
    # Brier: (0.2-0)^2 + (0.5-1)^2 + (0.3-0)^2 = 0.04 + 0.25 + 0.09
    assert it.brier == pytest.approx(0.38)


def test_macro_is_the_mean_over_templates_not_over_items() -> None:
    # template t1: 1 of 1 correct (accuracy 1.0); template t2: 1 of 3 correct (accuracy 1/3)
    items = [score_row(_row("a", "t1", "A", ["A", "B"], [0.9, 0.1])),
             score_row(_row("b", "t2", "A", ["A", "B"], [0.9, 0.1])),
             score_row(_row("c", "t2", "A", ["A", "B"], [0.1, 0.9])),
             score_row(_row("d", "t2", "A", ["A", "B"], [0.1, 0.9]))]
    assert macro_over_templates(items, lambda it: it.correct) == pytest.approx((1.0 + 1 / 3) / 2)


def test_summarise_counts_skips_by_reason_and_coverage() -> None:
    rows = [_row("a", "t1", "A", ["A", "B"], [0.7, 0.3]),
            _row("b", "t1", "B", ["A", "B"], [0.7, 0.3]),
            _row("c", "t1", "A", ["A", "B"], status="skipped", reason="over the cap")]
    s = summarise(rows)
    assert s.n_in_scope == 3 and s.n_scored == 2 and s.coverage == pytest.approx(2 / 3)
    assert s.skipped_by_reason == {"over the cap": 1}
    assert s.accuracy == pytest.approx(0.5)
    assert s.wall_clock_s_per_1k == pytest.approx(100.0)


def test_summarise_with_nothing_scored_reports_none_not_zero() -> None:
    s = summarise([_row("a", "t1", "A", ["A", "B"], status="skipped", reason="x")])
    assert s.accuracy is None and s.macro_accuracy is None and s.coverage == 0.0


def test_flip_rates_compare_labels_not_letters() -> None:
    items = {
        "base": {"item_id": "base", "options": [{"key": "A", "label": "x"}, {"key": "B", "label": "y"}]},
        "pert": {"item_id": "pert", "options": [{"key": "A", "label": "y"}, {"key": "B", "label": "x"}]},
    }
    base = _row("base", "t1", "A", ["A", "B"], [0.9, 0.1])   # predicts "x"
    pert = _row("pert", "t1", "B", ["A", "B"], [0.2, 0.8])   # predicts key B = label "x": no flip
    out = flip_rates([("reverse_options", base, pert)], items)
    assert out["reverse_options"]["flips"] == 0 and out["reverse_options"]["pairs"] == 1
    assert label_for(items, "pert", "A") == "y"


def test_cell_health_d21_flags_a_constant_answer_on_a_balanced_template() -> None:
    # template with balanced gold (50/50) but the model always answers A: degenerate under D21
    rows = [_row(f"i{k}", "t1", "A" if k % 2 == 0 else "B", ["A", "B"], [0.9, 0.1]) for k in range(40)]
    health = cell_health("m", "t1", rows, zero_shot=False, n_resamples=200)
    assert health["status"].startswith("READOUT_FAIL")
    assert any("constant answer" in f for f in health["failures"])


def test_cell_health_d21_passes_a_discriminating_cell() -> None:
    rows = []
    for k in range(40):
        gold = "A" if k % 2 == 0 else "B"
        probs = [0.9, 0.1] if gold == "A" else [0.1, 0.9]
        rows.append(_row(f"i{k}", "t1", gold, ["A", "B"], probs))
    health = cell_health("m", "t1", rows, zero_shot=False, n_resamples=200)
    assert health["status"] == "PASS (D21)" and health["accuracy"] == pytest.approx(1.0)


def test_cell_health_d12_uses_label_mass_for_zero_shot_cells() -> None:
    rows = []
    for k in range(40):
        gold = "A" if k % 2 == 0 else "B"
        probs = [0.9, 0.1] if gold == "A" else [0.1, 0.9]
        rows.append(_row(f"i{k}", "t1", gold, ["A", "B"], probs,
                         extras={"label_mass": 0.01, "vocab_argmax_is_option": False}))
    health = cell_health("zs", "t1", rows, zero_shot=True, n_resamples=200)
    assert health["status"].startswith("READOUT_FAIL") and "label mass" in health["status"]
