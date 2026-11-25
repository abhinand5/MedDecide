"""Test-split denominators of a benchmark build (no silent drops: every drop is counted).

A screen (``screen.json``) marks each template kept or dropped. The headline denominator is the
test rows of the kept templates; the dropped rows are returned per template, with their count.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field


@dataclass(frozen=True)
class TestDenominator:
    raw_test_rows: int
    kept_test_rows: int
    kept_fresh_rows: int
    kept_tier1_rows: int
    dropped_test_rows: int
    dropped_by_template: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kept_test_rows + self.dropped_test_rows != self.raw_test_rows:
            raise ValueError("kept + dropped must equal the raw test rows")
        if self.kept_fresh_rows + self.kept_tier1_rows != self.kept_test_rows:
            raise ValueError("fresh + tier1 must equal the kept test rows")


def count_test_denominator(
    rows: Iterable[Mapping[str, object]], templates: Sequence[Mapping[str, object]]
) -> TestDenominator:
    """Count test rows against a screen's kept and dropped templates.

    ``templates`` needs ``template_id``, ``tier`` ("fresh" or "tier1") and ``drop``. A row whose
    template is not in the screen raises KeyError: an unscreened template is never counted silently.
    """
    status = {str(t["template_id"]): t for t in templates}
    raw = kept = fresh = tier1 = dropped = 0
    dropped_by: Counter[str] = Counter()
    for row in rows:
        if row["split"] != "test":
            continue
        raw += 1
        template = status[str(row["template_id"])]
        if template["drop"]:
            dropped += 1
            dropped_by[str(row["template_id"])] += 1
            continue
        kept += 1
        if template["tier"] == "fresh":
            fresh += 1
        else:
            tier1 += 1
    return TestDenominator(raw_test_rows=raw, kept_test_rows=kept, kept_fresh_rows=fresh,
                           kept_tier1_rows=tier1, dropped_test_rows=dropped,
                           dropped_by_template=dict(sorted(dropped_by.items())))
