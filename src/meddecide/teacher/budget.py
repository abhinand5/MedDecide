"""Spend guard and request rate limit for the osler_v1 teacher build (ADVISORY §3.8 and V4).

The hosted teacher is billed per token. Every request's cost is written to an append-only ledger (timestamp and USD only,
never any prompt or output), so the guard survives a restart: the 5-hour window cap, the 7-day cap and the loop cap are
all computed from the ledger, not from the memory of one process. The rate limit is shared by every worker thread.

Caps (ADVISORY V4): $14 per 5-hour window (the plan allows $16), $36 per week (allowance $40), $60 for the loop.
"""

from __future__ import annotations

import bisect
import itertools
import json
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

WINDOW_HOURS = 5
WINDOW_CAP_USD = 14.0
WEEK_DAYS = 7
WEEK_CAP_USD = 36.0
LOOP_CAP_USD = 60.0


class RateLimiter:
    """At most ``rate_per_s`` acquisitions per second across all threads (a minimum spacing between grants)."""

    def __init__(self, rate_per_s: float, *, clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        if rate_per_s <= 0:
            raise ValueError("rate_per_s must be positive")
        self.spacing = 1.0 / rate_per_s
        self._clock = clock
        self._sleep = sleep
        self._lock = threading.Lock()
        self._next_free = 0.0

    def acquire(self) -> float:
        """Block until this caller may send a request; returns the seconds it waited."""
        with self._lock:
            now = self._clock()
            start = max(now, self._next_free)
            self._next_free = start + self.spacing
        wait = start - now
        if wait > 0:
            self._sleep(wait)
        return max(wait, 0.0)


@dataclass(frozen=True)
class SpendCheck:
    ok: bool
    reason: str
    window_usd: float
    week_usd: float
    loop_usd: float


class SpendLedger:
    """Append-only JSONL of request costs: one line per request, with a UTC timestamp, the USD cost and a label.

    The rows are also held in memory, sorted by time with prefix sums, so a check costs O(log n) however long the
    ledger grows (the build checks before every submission).
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._times: list[datetime] = []
        self._costs: list[float] = []
        self._prefix: list[float] = [0.0]
        for when, cost in self._read():
            self._insert(when, cost)

    def _read(self) -> list[tuple[datetime, float]]:
        if not self.path.exists():
            return []
        out: list[tuple[datetime, float]] = []
        with self.path.open(encoding="utf-8") as fh:
            for line in fh:
                row = json.loads(line)
                out.append((datetime.fromisoformat(row["utc"]), float(row["cost_usd"])))
        return out

    def _insert(self, when: datetime, cost: float) -> None:
        i = bisect.bisect_right(self._times, when)
        self._times.insert(i, when)
        self._costs.insert(i, cost)
        if i == len(self._times) - 1:
            self._prefix.append(self._prefix[-1] + cost)
        else:  # an out-of-order entry (rare): rebuild the prefix sums
            self._prefix = [0.0, *itertools.accumulate(self._costs)]

    def record(self, cost_usd: float, *, label: str, when: datetime | None = None) -> None:
        when = when or datetime.now(UTC)
        stamp = when.isoformat(timespec="seconds")
        cost = round(float(cost_usd), 8)
        line = json.dumps({"utc": stamp, "cost_usd": cost, "label": label})
        with self._lock, self.path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
            # the in-memory index holds exactly what a later process reads back from the file
            self._insert(datetime.fromisoformat(stamp), cost)

    def total(self) -> float:
        return self._prefix[-1]

    def spent_since(self, since: datetime) -> float:
        i = bisect.bisect_left(self._times, since)
        return self._prefix[-1] - self._prefix[i]

    def check(self, now: datetime | None = None) -> SpendCheck:
        """Whether another request may be sent under the window, week and loop caps."""
        now = now or datetime.now(UTC)
        window = self.spent_since(now - timedelta(hours=WINDOW_HOURS))
        week = self.spent_since(now - timedelta(days=WEEK_DAYS))
        loop = self.total()
        reasons = []
        if window >= WINDOW_CAP_USD:
            reasons.append(f"5-hour window at ${window:.2f} (cap ${WINDOW_CAP_USD:g})")
        if week >= WEEK_CAP_USD:
            reasons.append(f"week at ${week:.2f} (cap ${WEEK_CAP_USD:g})")
        if loop >= LOOP_CAP_USD:
            reasons.append(f"loop at ${loop:.2f} (cap ${LOOP_CAP_USD:g})")
        return SpendCheck(ok=not reasons, reason="; ".join(reasons) or "within caps",
                          window_usd=window, week_usd=week, loop_usd=loop)
