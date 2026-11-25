"""Aggregates for the d1-2B local benchmark (bench/d1-2b): the template macro accuracy, the item micro accuracy and the
D12 readout status per template, over predictions already written by ``scripts/bench/d1_score.py``.

The definitions are those of ``scripts/osler_v4/y9_gate.py::scoreboard_line``: the template macro accuracy is the mean over
templates of each template's mean correctness; the item micro accuracy is the mean correctness over all scored items. The
readout gate is ``meddecide.eval.readout_gate`` (D12, with the D21 variant for a trained readout that normalises over the
offered options), applied to every template cell.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from meddecide.eval.gate_o4 import normalise_row
from meddecide.eval.readout_gate import gate_by_template


def scored_rows(predictions: Sequence[Mapping[str, Any]], items: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Normalised rows (argmax, correctness) for the scored predictions, each checked against its item.

    A prediction whose item is missing is an error, not a drop. A row that is not ``scored`` is an error here: the runner
    only writes scored rows, and a skipped one would silently change a denominator.
    """
    out: list[dict[str, Any]] = []
    for pred in predictions:
        if pred.get("status") != "scored":
            raise ValueError(f"prediction {pred.get('item_id')!r} has status {pred.get('status')!r}, not scored")
        item = items.get(str(pred["item_id"]))
        if item is None:
            raise ValueError(f"prediction {pred['item_id']!r} has no item")
        out.append(normalise_row(pred, item))
    return out


def template_macro_accuracy(rows: Sequence[Mapping[str, Any]]) -> float:
    """Mean over templates of the mean correctness within each template."""
    if not rows:
        raise ValueError("no rows")
    per: dict[str, list[float]] = {}
    for r in rows:
        per.setdefault(str(r["template_id"]), []).append(1.0 if r["correct"] else 0.0)
    return sum(sum(v) / len(v) for v in per.values()) / len(per)


def item_micro_accuracy(rows: Sequence[Mapping[str, Any]]) -> float:
    """Mean correctness over all scored items."""
    if not rows:
        raise ValueError("no rows")
    return sum(1.0 for r in rows if r["correct"]) / len(rows)


def cell_statuses(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Per template: item count, accuracy and the D12 readout status (PASS, or READOUT_FAIL with its checks)."""
    out: dict[str, dict[str, Any]] = {}
    for template, cell in gate_by_template(rows).items():
        out[template] = {"items": cell.items, "accuracy": round(cell.accuracy, 6), "chance": round(cell.chance, 6),
                         "modal_share": round(cell.modal_share, 6), "ci_upper": round(cell.ci_upper, 6),
                         "status": cell.status}
    return out
