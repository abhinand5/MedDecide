"""Scoreboard metrics for osler_v0 O2 (pure functions; tested in tests/test_o2_metrics.py).

Definitions follow the project's gate definitions (D16 / D24 and the G1 scripts):
  - accuracy: micro, over scored items;
  - macro: unweighted mean over templates of the per-template value (the template is the grouping the gates use);
  - Brier: per item sum_k (p_k - y_k)^2, macro over templates;
  - ECE: top-label, 15 equal-width bins (metrics.ece_from_confidence, which takes items with different option counts);
  - coverage: scored / items in scope, with every skip reason counted;
  - wall-clock per 1,000 items: mean latency of scored items times 1,000.
Readout health per cell follows D12 for zero-shot readouts (health.evaluate_cell with the full-vocabulary
label mass and greedy agreement) and D21 for trained or competitor readouts (constant-answer and
accuracy-CI-not-below-chance checks; label mass and greedy agreement are diagnostics there).
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from meddecide.eval.health import (
    DEFAULT_DEGENERATE_MAJORITY_CEILING,
    DEFAULT_MAX_MODAL_SHARE,
    evaluate_cell,
)
from meddecide.eval.metrics import bootstrap_ci, ece_from_confidence


@dataclass
class ScoredItem:
    item_id: str
    template_id: str
    gold: str
    option_keys: list[str]
    probs: list[float]
    correct: bool
    predicted: str
    confidence: float
    brier: float
    latency_s: float | None
    extras: dict[str, Any] = field(default_factory=dict)


def score_row(row: Mapping[str, Any]) -> ScoredItem:
    """A scored prediction row as a ScoredItem; argmax ties resolve to the first option (as np.argmax does)."""
    probs = [float(p) for p in row["probs"]]
    keys = list(row["option_keys"])
    k = int(np.argmax(probs))
    gold_index = keys.index(row["gold"])
    onehot = [1.0 if i == gold_index else 0.0 for i in range(len(keys))]
    return ScoredItem(
        item_id=row["item_id"], template_id=row["template_id"], gold=row["gold"], option_keys=keys,
        probs=probs, correct=keys[k] == row["gold"], predicted=keys[k], confidence=probs[k],
        brier=float(sum((p - y) ** 2 for p, y in zip(probs, onehot, strict=True))),
        latency_s=row.get("latency_s"), extras=dict(row.get("extras") or {}),
    )


def macro_over_templates(items: Sequence[ScoredItem], value) -> float:
    """Unweighted mean over templates of the per-template mean of ``value(item)``."""
    groups: dict[str, list[float]] = defaultdict(list)
    for it in items:
        groups[it.template_id].append(float(value(it)))
    return float(np.mean([np.mean(v) for v in groups.values()])) if groups else float("nan")


@dataclass
class SetSummary:
    n_in_scope: int
    n_scored: int
    skipped_by_reason: dict[str, int]
    coverage: float
    accuracy: float | None
    accuracy_ci95: tuple[float, float] | None
    macro_accuracy: float | None
    macro_brier: float | None
    brier: float | None
    ece: float | None
    wall_clock_s_per_1k: float | None
    n_templates: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_in_scope": self.n_in_scope, "n_scored": self.n_scored,
            "skipped_by_reason": dict(sorted(self.skipped_by_reason.items())),
            "coverage": round(self.coverage, 4),
            "accuracy": _round(self.accuracy), "accuracy_ci95": [_round(x) for x in self.accuracy_ci95]
            if self.accuracy_ci95 else None,
            "macro_accuracy": _round(self.macro_accuracy), "macro_brier": _round(self.macro_brier),
            "brier": _round(self.brier), "ece": _round(self.ece),
            "wall_clock_s_per_1k": _round(self.wall_clock_s_per_1k, 1), "n_templates": self.n_templates,
        }


def _round(x: float | None, nd: int = 4) -> float | None:
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), nd)


def summarise(rows: Sequence[Mapping[str, Any]], *, seed: int = 0) -> SetSummary:
    """Summary of one (model, set) cell from its prediction rows (scored and skipped)."""
    skipped = Counter(r["reason"] for r in rows if r["status"] == "skipped")
    scored = [score_row(r) for r in rows if r["status"] == "scored"]
    n_in = len(rows)
    n_scored = len(scored)
    coverage = n_scored / n_in if n_in else float("nan")
    if not scored:
        return SetSummary(n_in, 0, dict(skipped), coverage, None, None, None, None, None, None, None,
                          len({r["template_id"] for r in rows}))
    correct = [1.0 if it.correct else 0.0 for it in scored]
    _, lo, hi = bootstrap_ci(correct, n_resamples=1000, seed=seed)
    latencies = [it.latency_s for it in scored if it.latency_s is not None]
    return SetSummary(
        n_in_scope=n_in, n_scored=n_scored, skipped_by_reason=dict(skipped), coverage=coverage,
        accuracy=float(np.mean(correct)), accuracy_ci95=(lo, hi),
        macro_accuracy=macro_over_templates(scored, lambda it: it.correct),
        macro_brier=macro_over_templates(scored, lambda it: it.brier),
        brier=float(np.mean([it.brier for it in scored])),
        ece=ece_from_confidence([it.confidence for it in scored], [it.correct for it in scored]),
        wall_clock_s_per_1k=(float(np.mean(latencies)) * 1000.0 if latencies else None),
        n_templates=len({it.template_id for it in scored}),
    )


def label_for(items: Mapping[str, Mapping[str, Any]], item_id: str, key: str) -> str:
    """The option label behind an option key of a common-set item (flips compare labels, not letters)."""
    for option in items[item_id]["options"]:
        if option["key"] == key:
            return option["label"]
    raise KeyError(f"{item_id}: no option {key!r}")


def flip_rates(pairs: Iterable[tuple[str, Mapping[str, Any], Mapping[str, Any]]],
               items: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Prediction flips between a base item and its perturbed copy, by perturbation name.

    ``pairs`` yields (perturbation, base prediction row, perturbed prediction row); both are scored. The
    comparison is on the predicted option's label, so a reversed option order (which re-letters options) is
    compared by content. ``items`` maps item id to its common-set item (for the labels).
    """
    out: dict[str, Counter[str]] = defaultdict(Counter)
    for name, base, pert in pairs:
        b = score_row(base)
        p = score_row(pert)
        b_label = label_for(items, base["item_id"], b.predicted)
        p_label = label_for(items, pert["item_id"], p.predicted)
        out[name]["pairs"] += 1
        out[name]["flips"] += int(b_label != p_label)
        out[name]["perturbed_correct"] += int(p.correct)
    return {name: {"pairs": c["pairs"], "flips": c["flips"],
                   "flip_rate": round(c["flips"] / c["pairs"], 4) if c["pairs"] else None,
                   "perturbed_accuracy": round(c["perturbed_correct"] / c["pairs"], 4) if c["pairs"] else None}
            for name, c in sorted(out.items())}


def cell_health(model_id: str, template_id: str, rows: Sequence[Mapping[str, Any]], *, zero_shot: bool,
                n_resamples: int = 1000, seed: int = 0) -> dict[str, Any]:
    """Readout-health verdict for one (model, template) cell.

    zero-shot: D12 via health.evaluate_cell (label mass, greedy agreement, accuracy CI, constant answer).
    trained or competitor: D21 — constant-answer and accuracy-CI-not-below-chance checks; label mass and
    greedy agreement are reported as diagnostics (not measured here for competitors).
    """
    scored = [score_row(r) for r in rows if r["status"] == "scored"]
    if not scored:
        return {"status": "NOT MEASURED — no scored items", "failures": []}
    n_options = round(float(np.mean([len(it.option_keys) for it in scored])))
    golds = [it.gold for it in scored]
    majority_share = Counter(golds).most_common(1)[0][1] / len(golds)
    predicted = [it.predicted for it in scored]
    correct = [it.correct for it in scored]
    if zero_shot:
        masses = [float(r["extras"]["label_mass"]) for r in rows if r["status"] == "scored"]
        greedy = [bool(r["extras"]["vocab_argmax_is_option"]) for r in rows if r["status"] == "scored"]
        report = evaluate_cell(
            model_id=model_id, template_id=template_id, label_masses=masses, correct=correct,
            greedy_matches=greedy, n_options=max(2, n_options), predicted_labels=predicted,
            majority_share=majority_share, n_resamples=n_resamples, seed=seed,
        )
        failures = list(report.failures)
        return {"status": "READOUT_FAIL — " + "; ".join(failures) if failures else "PASS (D12)",
                "failures": failures, "median_label_mass": _round(report.median_label_mass),
                "greedy_agreement": _round(report.greedy_agreement), "n": report.n,
                "accuracy": _round(report.accuracy), "accuracy_ci95": [_round(x) for x in report.accuracy_ci95]}
    # D21 checks for trained or competitor readouts
    failures: list[str] = []
    counts = Counter(predicted)
    modal_option, modal_count = max(sorted(counts.items()), key=lambda kv: kv[1])
    modal_share = modal_count / len(predicted)
    if majority_share <= DEFAULT_DEGENERATE_MAJORITY_CEILING and modal_share >= DEFAULT_MAX_MODAL_SHARE:
        failures.append(f"constant answer ({modal_option}) for {modal_share:.3f} of items on a template whose "
                        f"majority share is {majority_share:.3f}")
    _, lo, hi = bootstrap_ci([1.0 if c else 0.0 for c in correct], n_resamples=n_resamples, seed=seed)
    chance = 1.0 / max(2, n_options)
    if hi < chance:
        failures.append(f"accuracy CI upper bound {hi:.4f} < chance {chance:.4f}")
    return {"status": "READOUT_FAIL — " + "; ".join(failures) if failures else "PASS (D21)",
            "failures": failures, "n": len(correct), "accuracy": _round(float(np.mean(correct))),
            "accuracy_ci95": [_round(lo), _round(hi)], "modal_share": _round(modal_share),
            "majority_share": _round(majority_share), "greedy_check": "not applicable (no greedy path)",
            "label_mass_check": "diagnostic only (D21)"}
