"""V0 speed recipe (osler_v1 ADVISORY V0 step 4): the summary arithmetic and the pre-registered choice rule, without a GPU."""

from __future__ import annotations

import pytest

from meddecide.train.speed import LOSS_WINDOW, MEMORY_CAP_GB, WARMUP_STEPS, choose_config, summarise


def _records(n: int, *, n_items: int = 8, dt: float = 2.0, peak: float = 10.0) -> list[dict]:
    """n steps, each taking dt seconds of wall clock; loss equals the step number, so the tail mean is easy to check."""
    return [
        {"step": s, "n_items": n_items, "real_tokens": 1000, "padded_tokens": 1250, "loss": float(s),
         "elapsed_s": 0.8 * dt, "t_end_s": s * dt, "peak_reserved_gb": peak}
        for s in range(1, n + 1)
    ]


def test_summarise_uses_the_steady_state_window_and_the_last_fifty_losses() -> None:
    records = _records(150, n_items=8, dt=2.0)
    s = summarise(records)
    # window is steps 11..150: 140 steps of 8 items over 140 * 2.0 s of wall clock
    assert s["window_steps"] == 150 - WARMUP_STEPS == 140
    assert s["window_items"] == 140 * 8
    assert s["items_per_s"] == pytest.approx(8 / 2.0)
    assert s["real_tokens_per_s"] == pytest.approx(1000 / 2.0)
    assert s["padding_efficiency"] == pytest.approx(1000 / 1250)
    assert s["items_per_batch"] == pytest.approx(8.0)
    # the last 50 steps are 101..150 (loss = step number): mean 125.5
    assert s["loss_last_window_steps"] == LOSS_WINDOW == 50
    assert s["loss_last_window_mean"] == pytest.approx(125.5)
    assert s["peak_reserved_gb"] == pytest.approx(10.0)


def test_summarise_takes_the_maximum_peak_over_all_steps() -> None:
    records = _records(30)
    records[17]["peak_reserved_gb"] = 61.25
    assert summarise(records)["peak_reserved_gb"] == pytest.approx(61.25)


def test_summarise_accepts_records_in_any_order() -> None:
    records = _records(30)
    shuffled = list(reversed(records))
    assert summarise(shuffled) == summarise(records)


def test_summarise_rejects_gaps_and_short_runs() -> None:
    records = _records(30)
    del records[5]
    with pytest.raises(ValueError, match="no gaps"):
        summarise(records)
    with pytest.raises(ValueError, match="warm-up"):
        summarise(_records(WARMUP_STEPS))
    with pytest.raises(ValueError, match="no step records"):
        summarise([])


def _summary(items_per_s: float, *, n_steps: int = 150, peak: float = 40.0) -> dict:
    return {"items_per_s": items_per_s, "n_steps": n_steps, "peak_reserved_gb": peak}


def test_choose_takes_the_fastest_eligible_configuration() -> None:
    chosen, reason = choose_config({"a": _summary(10.0), "b": _summary(25.0), "c": _summary(18.0)}, target_steps=150)
    assert chosen == "b"
    assert "25.000" in reason


def test_choose_excludes_out_of_memory_short_and_over_cap_runs() -> None:
    summaries = {
        "a": _summary(10.0),
        "b": {"oom": True, "oom_at_step": 3},
        "c": _summary(99.0, n_steps=120),          # finished fewer steps than the target
        "d": _summary(80.0, peak=MEMORY_CAP_GB + 0.5),  # over the 85 GB cap
    }
    chosen, reason = choose_config(summaries, target_steps=150)
    assert chosen == "a"
    assert "b: out of memory at step 3" in reason
    assert "c: finished 120 of 150 steps" in reason
    assert "d: peak reserved" in reason


def test_choose_is_none_when_nothing_is_eligible() -> None:
    chosen, reason = choose_config({"b": {"oom": True, "oom_at_step": 1}}, target_steps=150)
    assert chosen is None
    assert "no configuration is eligible" in reason


def test_choose_breaks_ties_by_name() -> None:
    chosen, _ = choose_config({"c": _summary(12.0), "a": _summary(12.0)}, target_steps=150)
    assert chosen == "a"


def test_peak_exactly_at_the_cap_is_eligible() -> None:
    chosen, _ = choose_config({"x": _summary(5.0, peak=MEMORY_CAP_GB)}, target_steps=150)
    assert chosen == "x"
