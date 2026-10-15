"""The teacher spend guard and request rate limit (osler_v1 ADVISORY V4, §3.8), tested without the network."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from meddecide.teacher.budget import (
    LOOP_CAP_USD,
    WEEK_CAP_USD,
    WINDOW_CAP_USD,
    RateLimiter,
    SpendLedger,
)


def test_rate_limiter_spaces_grants_evenly_on_a_fake_clock() -> None:
    now = [100.0]
    slept: list[float] = []

    def clock() -> float:
        return now[0]

    def sleep(seconds: float) -> None:
        slept.append(seconds)
        now[0] += seconds

    limiter = RateLimiter(2.0, clock=clock, sleep=sleep)  # 2 requests per second: spacing 0.5 s
    waits = [limiter.acquire() for _ in range(4)]
    assert waits == [0.0, pytest.approx(0.5), pytest.approx(0.5), pytest.approx(0.5)]
    assert slept == [pytest.approx(0.5)] * 3


def test_rate_limiter_does_not_wait_after_a_long_idle_gap() -> None:
    now = [0.0]
    limiter = RateLimiter(2.0, clock=lambda: now[0], sleep=lambda s: None)
    limiter.acquire()
    now[0] = 60.0
    assert limiter.acquire() == 0.0


def test_rate_limiter_rejects_a_non_positive_rate() -> None:
    with pytest.raises(ValueError):
        RateLimiter(0.0)


def _at(hours_ago: float, now: datetime) -> datetime:
    return now - timedelta(hours=hours_ago)


def test_ledger_sums_the_window_week_and_loop_separately(tmp_path) -> None:
    ledger = SpendLedger(tmp_path / "ledger.jsonl")
    now = datetime(2026, 10, 13, 12, 0, tzinfo=UTC)
    ledger.record(3.0, label="old", when=_at(200, now))     # 8+ days ago: loop only
    ledger.record(10.0, label="week", when=_at(60, now))    # 2.5 days ago: week and loop
    ledger.record(4.0, label="window", when=_at(1, now))    # inside the 5-hour window
    check = ledger.check(now)
    assert check.ok
    assert check.window_usd == pytest.approx(4.0)
    assert check.week_usd == pytest.approx(14.0)
    assert check.loop_usd == pytest.approx(17.0)


def test_window_cap_stops_further_requests(tmp_path) -> None:
    ledger = SpendLedger(tmp_path / "ledger.jsonl")
    now = datetime(2026, 10, 13, 12, 0, tzinfo=UTC)
    ledger.record(WINDOW_CAP_USD, label="burst", when=_at(0.5, now))
    check = ledger.check(now)
    assert not check.ok
    assert "5-hour window" in check.reason
    # five hours later the window has emptied; the same spend still counts toward the week
    later = now + timedelta(hours=5, minutes=1)
    assert ledger.check(later).window_usd == 0.0
    assert ledger.check(later).week_usd == pytest.approx(WINDOW_CAP_USD)


def test_week_and_loop_caps_stop_requests(tmp_path) -> None:
    ledger = SpendLedger(tmp_path / "ledger.jsonl")
    now = datetime(2026, 10, 13, 12, 0, tzinfo=UTC)
    ledger.record(WEEK_CAP_USD, label="week", when=_at(30, now))
    check = ledger.check(now)
    assert not check.ok and "week" in check.reason
    ledger.record(LOOP_CAP_USD, label="loop", when=_at(400, now))
    assert "loop" in ledger.check(now).reason


def test_the_ledger_survives_a_restart(tmp_path) -> None:
    path = tmp_path / "ledger.jsonl"
    now = datetime(2026, 10, 13, 12, 0, tzinfo=UTC)
    SpendLedger(path).record(2.5, label="first process", when=_at(1, now))
    assert SpendLedger(path).check(now).window_usd == pytest.approx(2.5)


def test_a_missing_ledger_is_zero_spend(tmp_path) -> None:
    check = SpendLedger(tmp_path / "none.jsonl").check(datetime(2026, 10, 13, tzinfo=UTC))
    assert check.ok and check.loop_usd == 0.0


def test_a_ledger_total_cap_refuses_before_its_own_total_reaches_the_cap(tmp_path) -> None:
    # V4-D7: the DeepSeek ledger has its own total cap of $9.00; the other caps are the same as before
    ledger = SpendLedger(tmp_path / "deepseek.jsonl", total_cap_usd=9.0)
    now = datetime(2026, 10, 15, 6, 0, tzinfo=UTC)
    ledger.record(8.99, label="narrative:accepted:deepseek", when=now)
    assert ledger.check(now).ok is True
    ledger.record(0.02, label="narrative:accepted:deepseek", when=now)
    check = ledger.check(now)
    assert check.ok is False and "ledger total" in check.reason and "$9" in check.reason


def test_without_a_total_cap_only_the_window_week_and_loop_caps_apply(tmp_path) -> None:
    ledger = SpendLedger(tmp_path / "plain.jsonl")
    now = datetime(2026, 10, 15, 6, 0, tzinfo=UTC)
    ledger.record(10.0, label="narrative:accepted", when=now)  # under the window ($14), week ($36) and loop ($60) caps
    assert ledger.check(now).ok is True
