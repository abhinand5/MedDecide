"""Per-qtype temperature calibration on the dev split.

One temperature per question type, fitted by minimising the negative log-likelihood of the
gold option on dev. ``noul``, ``choice`` and ``score`` have different option counts and
therefore different confidence scales, which is why the fit is per qtype rather than global.
Only dev is used — never a test split (AGENTS.md data rules).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar


@dataclass(frozen=True)
class TemperatureFit:
    """One qtype's fitted temperature, with the NLL it was fitted on."""

    qtype: str
    temperature: float
    n_items: int
    nll_before: float
    nll_after: float

    def to_dict(self) -> dict[str, float | int | str]:
        return {
            "qtype": self.qtype,
            "temperature": self.temperature,
            "n_items": self.n_items,
            "nll_before": self.nll_before,
            "nll_after": self.nll_after,
        }


def _nll(logits: Sequence[np.ndarray], gold: np.ndarray, temperature: float) -> float:
    """Mean NLL of the gold option. Items are looped over, never stacked.

    Items of one qtype do **not** all have the same option count (the dev `choice` items have
    3, 4, 6, 9, 11 and 17 options), so a rectangular array is impossible — the S7 smoke run
    crashed on exactly this before the loop replaced ``np.stack``.
    """
    total = 0.0
    for row, g in zip(logits, gold, strict=True):
        scaled = np.asarray(row, dtype=np.float64) / temperature
        top = float(scaled.max())
        logsumexp = top + float(np.log(np.exp(scaled - top).sum()))
        total += logsumexp - float(scaled[int(g)])
    return total / len(logits)


def fit_temperature(
    logits: Sequence[np.ndarray],
    gold_indices: Sequence[int],
    *,
    qtype: str,
    bounds: tuple[float, float] = (0.02, 50.0),
) -> TemperatureFit:
    """Fit one temperature for a group of items by minimising NLL.

    A bounded scalar search on a smooth objective: with T -> 0 the NLL goes to 0 on separable
    data, so the lower bound is what keeps the fit meaningful; the value it lands on is
    reported, never hidden.
    """
    if len(logits) != len(gold_indices):
        raise ValueError("logits and gold_indices must have one entry per item")
    if not logits:
        raise ValueError("cannot fit a temperature on zero items")
    rows = [np.asarray(v, dtype=np.float64) for v in logits]
    gold = np.asarray(gold_indices, dtype=np.int64)
    for i, row in enumerate(rows):
        if not 0 <= int(gold[i]) < row.size:
            raise ValueError(f"gold index {int(gold[i])} is outside {row.size} options")
    before = _nll(rows, gold, 1.0)
    result = minimize_scalar(
        lambda t: _nll(rows, gold, float(t)),
        bounds=bounds,
        method="bounded",
        options={"xatol": 1e-4},
    )
    temperature = float(result.x)
    after = _nll(rows, gold, temperature)
    if after > before:  # pragma: no cover - the optimiser should never do worse than T=1
        temperature, after = 1.0, before
    return TemperatureFit(
        qtype=qtype,
        temperature=temperature,
        n_items=len(logits),
        nll_before=before,
        nll_after=after,
    )


def fit_per_qtype(scored: object, *, qtypes: Sequence[str] | None = None) -> dict[str, TemperatureFit]:
    """Fit one temperature per qtype over a :class:`~meddecide.model.ScoredItems` result."""
    logits = list(scored.logits)
    gold = list(scored.gold_indices)
    item_qtypes = list(qtypes) if qtypes is not None else list(scored.qtypes)
    if not (len(logits) == len(gold) == len(item_qtypes)):
        raise ValueError("scored object has mismatched logits/gold/qtype lengths")
    out: dict[str, TemperatureFit] = {}
    for qtype in sorted(set(item_qtypes)):
        idx = [i for i, q in enumerate(item_qtypes) if q == qtype]
        out[qtype] = fit_temperature(
            [logits[i] for i in idx], [gold[i] for i in idx], qtype=qtype
        )
    return out
