"""Prompt-length summaries for throughput planning (osler_v0 O0).

The quantile rule is the one the O0 throughput run used: the element at index ``int(p * n)`` of the
sorted lengths, clamped to the last element. The median is the usual statistical median, rounded
down to an integer.
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class LengthSummary:
    n: int
    median: int
    mean: float
    p90: int
    p95: int
    max: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def summarize_lengths(lengths: Sequence[int]) -> LengthSummary:
    if not lengths:
        raise ValueError("no lengths to summarise")
    ordered = sorted(lengths)
    n = len(ordered)

    def quantile(p: float) -> int:
        return ordered[min(n - 1, int(p * n))]

    return LengthSummary(
        n=n,
        median=int(statistics.median(ordered)),
        mean=round(statistics.fmean(ordered), 1),
        p90=quantile(0.90),
        p95=quantile(0.95),
        max=ordered[-1],
    )
