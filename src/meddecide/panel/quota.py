"""Quota-based construction of an evaluation set with counted drops (osler_v0 O1).

Rows are visited in a fixed order (the caller ranks them by a stable hash). Each row is converted; a
``Dropped`` error, a malformed row, or a text key already present in ``exclude_keys`` is counted by
reason. The walk stops once ``quota`` items are accepted. Every examined row is accepted or dropped,
and the rows after the stop are reported as not examined, so the accounting always closes.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass, field
from typing import Any

from meddecide.panel.converters import Dropped
from meddecide.panel.schema import EvalItem


@dataclass
class QuotaResult:
    accepted: list[EvalItem]
    population: int
    examined: int
    dropped_by_reason: dict[str, int] = field(default_factory=dict)

    @property
    def not_examined(self) -> int:
        return self.population - self.examined

    def summary(self, quota: int) -> dict[str, Any]:
        dropped = self.examined - len(self.accepted)
        return {
            "population": self.population,
            "examined": self.examined,
            "accepted": len(self.accepted),
            "dropped": dropped,
            "dropped_by_reason": dict(sorted(self.dropped_by_reason.items())),
            "not_examined_after_quota": self.not_examined,
            "quota": quota,
        }


def take_valid(
    rows: Sequence[Any],
    convert: Callable[[Any], EvalItem],
    quota: int,
    exclude_keys: Collection[str] = (),
    key_of: Callable[[EvalItem], str] | None = None,
) -> QuotaResult:
    """Convert rows in order until ``quota`` items are accepted.

    ``exclude_keys`` holds text keys of items that must not enter the set (for example, items already in
    the benchmark). ``key_of`` maps an accepted candidate to its key; without it, nothing is excluded.
    """
    accepted: list[EvalItem] = []
    drops: Counter[str] = Counter()
    examined = 0
    for raw in rows:
        if len(accepted) >= quota:
            break
        examined += 1
        try:
            item = convert(raw)
        except Dropped as exc:
            drops[exc.reason] += 1
            continue
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            drops[f"malformed_{type(exc).__name__}"] += 1
            continue
        if key_of is not None and key_of(item) in exclude_keys:
            drops["duplicate_of_v0_2_item"] += 1
            continue
        accepted.append(item)
    return QuotaResult(accepted=accepted, population=len(rows), examined=examined,
                       dropped_by_reason=dict(drops))
