"""The read-only run-log summaries used while the O6-O10 arms train (src/meddecide/train/monitor.py)."""

from __future__ import annotations

import pytest

from meddecide.train.monitor import (
    block_summaries,
    eval_rows,
    multi_item_steps,
    step_rows,
    tripwire_count,
    window_summary,
)


def _eval(step: int, template: float, per_letter: float) -> dict:
    return {"event": "eval", "step": step,
            "metrics": {"n": 6000, "template_macro_accuracy": template, "macro_accuracy": per_letter,
                        "accuracy": 0.85, "brier": 0.22}}


def _step(step: int, loss: float, grad: float) -> dict:
    return {"event": "step", "step": step, "loss": loss, "ce": loss - 0.1, "brier": 0.2, "accuracy": 0.8,
            "grad_norm": grad, "n_items": 8}


def test_eval_rows_keep_both_macros_and_skip_other_events() -> None:
    rows = [_step(50, 1.0, 9.0), _eval(2500, 0.82, 0.60), {"event": "tripwire", "step": 2600}, _eval(5000, 0.81, 0.54)]
    evals = eval_rows(rows)
    assert [e["step"] for e in evals] == [2500, 5000]
    assert evals[0]["template_macro_accuracy"] == pytest.approx(0.82)
    assert evals[1]["macro_accuracy"] == pytest.approx(0.54)
    assert tripwire_count(rows) == 1
    assert [r["step"] for r in step_rows(rows)] == [50]


def test_window_summary_means_and_grad_norm_quantiles() -> None:
    window = [_step(50 * (i + 1), 1.0 + i, float(i + 1)) for i in range(20)]
    summary = window_summary(window)
    assert summary["records"] == 20 and summary["first_step"] == 50 and summary["last_step"] == 1000
    assert summary["loss"] == pytest.approx(10.5)
    assert summary["grad_norm_max"] == 20.0
    assert summary["grad_norm_p50"] == 11.0  # index 10 of the sorted grad norms 1..20
    assert summary["grad_norm_p95"] == 20.0
    with pytest.raises(ValueError, match="at least one"):
        window_summary([])


def test_multi_item_steps_drop_the_one_item_batches() -> None:
    steps = [{**_step(50, 1.0, 1.0), "n_items": 4}, {**_step(100, 0.0, 0.0), "n_items": 1},
             {**_step(150, 2.0, 2.0), "n_items": 8}]
    assert [r["step"] for r in multi_item_steps(steps)] == [50, 150]
    assert [r["step"] for r in multi_item_steps(steps, min_items=8)] == [150]


def test_block_summaries_average_the_multi_item_records_of_each_block() -> None:
    steps = [{**_step(50, 1.0, 2.0), "n_items": 8}, {**_step(100, 3.0, 4.0), "n_items": 2},
             {**_step(2550, 0.5, 6.0), "n_items": 8}]
    blocks = block_summaries(steps, width=2500)
    assert [b["from_step"] for b in blocks] == [0, 2500]
    assert blocks[0]["multi_item_records"] == 1 and blocks[0]["loss"] == pytest.approx(1.0)
    assert blocks[0]["one_item_records"] == 0 and blocks[0]["grad_norm_p50"] == 2.0
    assert blocks[1]["loss"] == pytest.approx(0.5) and blocks[1]["grad_norm_p50"] == 6.0
