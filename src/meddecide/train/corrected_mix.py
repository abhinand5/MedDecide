"""osler_v5 D56a: the corrected training mix. Every row of a template named in DROP_ALLOWED is dropped whole, because the defect condition
(the gold is routes[0] while other routes are also true) cannot be verified per row (ADVISORY §8.2). Nothing else changes: the drop is
a filter on the composed mix, and the rows that remain keep their order, their ids and their templates.

The allowed list is fixed by code. A config that names any other template is refused, so the corrected mix cannot be widened silently.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

DROP_ALLOWED: tuple[str, ...] = ("fda_route_choice_v1",)
# the composition hash of osler_v4's mix (the entry check of every run) and of the corrected mix (scripts/osler_v5/build_corrected_mix.py)
ORIGINAL_COMPOSITION_SHA256 = "9d25e30f53d7823d051b9c71334c248660318c86ce904c26f855b9a9dbc618b8"
CORRECTED_COMPOSITION_SHA256 = "78d7e5478060fc828784cb3365569e177c6961270d38a76a9cff144684e9f3be"


def validate_drop(templates: Iterable[str]) -> tuple[str, ...]:
    """The requested drops, checked against the fixed list; an unknown name or a repeated name raises."""
    names = tuple(templates)
    unknown = sorted(set(names) - set(DROP_ALLOWED))
    if unknown:
        raise ValueError(f"templates not allowed to be dropped: {unknown}")
    if len(set(names)) != len(names):
        raise ValueError("a template is named twice in the drop list")
    return names


@dataclass(frozen=True)
class DropReport:
    """What the drop did: rows before and after, the rows dropped per template, and the template counts of the kept rows."""

    rows_before: int
    rows_dropped: int
    rows_remaining: int
    dropped_by_template: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"rows_before": self.rows_before, "rows_dropped": self.rows_dropped, "rows_remaining": self.rows_remaining,
                "dropped_by_template": dict(sorted(self.dropped_by_template.items()))}


def drop_template_rows(rows: Sequence[Mapping[str, Any]], templates: Iterable[str]) -> tuple[list[Mapping[str, Any]], DropReport]:
    """The rows whose template is not dropped, in their original order, and the report of the drop."""
    names = set(validate_drop(templates))
    kept = [row for row in rows if str(row["template_id"]) not in names]
    dropped = Counter(str(row["template_id"]) for row in rows if str(row["template_id"]) in names)
    report = DropReport(rows_before=len(rows), rows_dropped=len(rows) - len(kept), rows_remaining=len(kept),
                        dropped_by_template=dict(dropped))
    return kept, report


def template_counts(rows: Iterable[Mapping[str, Any]]) -> Counter:
    return Counter(str(row["template_id"]) for row in rows)
