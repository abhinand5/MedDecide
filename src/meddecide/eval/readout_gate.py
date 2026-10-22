"""The readout checks of D12 and D21, per (model, template) cell, for readouts that normalise over the offered options.

A cell fails the gate when either check fails. The thresholds are those of ``meddecide.eval.health``:

- constant answer: the modal answer is at least ``DEFAULT_MAX_MODAL_SHARE`` of the cell's items, on a template whose majority
  share of gold answers is at most ``DEFAULT_DEGENERATE_MAJORITY_CEILING`` (a template that is mostly one answer is not flagged);
- below chance: the 95 % bootstrap upper bound of the accuracy is below the mean chance 1/n.

The label-mass check is not applied here: for trained readouts that normalise over the offered options, label mass is a
diagnostic (D21). A failing cell's accuracy is not reported as a result (D12).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from meddecide.eval.health import DEFAULT_DEGENERATE_MAJORITY_CEILING, DEFAULT_MAX_MODAL_SHARE
from meddecide.eval.metrics import bootstrap_ci


@dataclass(frozen=True)
class CellGate:
    """One cell's figures and its verdict."""

    items: int
    accuracy: float
    chance: float
    majority_share: float
    modal_share: float
    modal_key: str
    ci_upper: float
    failures: tuple[str, ...]

    @property
    def passes(self) -> bool:
        return not self.failures

    @property
    def status(self) -> str:
        return "PASS" if self.passes else "READOUT_FAIL — " + "; ".join(self.failures)


def cell_gate(gold: Sequence[str], predicted: Sequence[str], correct: Sequence[bool], chances: Sequence[float]) -> CellGate:
    """The gate for one cell: the gold and predicted option keys, the correct flags and each item's chance 1/n."""
    n = len(gold)
    if not n or not (len(predicted) == len(correct) == len(chances) == n):
        raise ValueError("the cell needs one gold key, prediction, correct flag and chance per item, and at least one item")
    accuracy = sum(bool(c) for c in correct) / n
    chance = sum(chances) / n
    ci_upper = bootstrap_ci([1.0 if c else 0.0 for c in correct])[2]
    majority_share = Counter(gold).most_common(1)[0][1] / n
    modal_key, modal_count = Counter(predicted).most_common(1)[0]
    modal_share = modal_count / n
    failures: list[str] = []
    if modal_share >= DEFAULT_MAX_MODAL_SHARE and majority_share <= DEFAULT_DEGENERATE_MAJORITY_CEILING:
        failures.append(f"constant answer ({modal_key}) on {modal_share:.3f} of items")
    if ci_upper < chance:
        failures.append(f"below chance (CI upper {ci_upper:.4f} < chance {chance:.4f})")
    return CellGate(items=n, accuracy=accuracy, chance=chance, majority_share=majority_share, modal_share=modal_share,
                    modal_key=modal_key, ci_upper=ci_upper, failures=tuple(failures))


def gate_by_template(rows: Iterable[Mapping[str, object]]) -> dict[str, CellGate]:
    """One cell per template, from rows with ``template_id``, ``gold_key``, ``predicted_key``, ``correct`` and ``probs``."""
    groups: dict[str, list[Mapping[str, object]]] = {}
    for row in rows:
        groups.setdefault(str(row["template_id"]), []).append(row)
    return {
        template: cell_gate(
            gold=[str(r["gold_key"]) for r in cell],
            predicted=[str(r["predicted_key"]) for r in cell],
            correct=[bool(r["correct"]) for r in cell],
            chances=[1.0 / len(r["probs"]) for r in cell],  # type: ignore[arg-type]
        )
        for template, cell in sorted(groups.items())
    }
