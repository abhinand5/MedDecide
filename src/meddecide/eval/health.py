"""Readout-health gate (decision D12).

A (model, template) cell may only be reported as an accuracy when its readout is provably
reading the model's answer. bench_v0 reported a broken `noul` readout (label mass ~0.002) and an
off-by-one MedMCQA key as *findings*; this gate is the mechanical check that stops a pipeline
defect from ever being reported as a result again.

Three checks, all recomputed from the prediction rows plus a greedy-agreement sample:

* **median label mass** — the full-vocabulary probability the option tokens carry, per item;
  a readout that reads the wrong tokens leaves almost no mass on them.
* **greedy agreement** — on a seeded sample of items, the readout's argmax option must equal the
  option the model *generates* greedily on the same prompt. This catches a readout that is
  internally consistent but disagrees with what the model would actually say.
* **accuracy CI** — the bootstrap CI's upper bound must not be below chance for the item's option
  count, so a cell cannot be reported as a result when the measurement is indistinguishable from
  guessing (or worse).

A fourth check was added in loop 1 (S1): **constant answer**. A cell whose most-predicted option
is the argmax for ≥ 0.90 of items, on a template whose majority share is ≤ 0.60, is answering
without reading the state — LFM2.5-350M sat at exactly 0.50 on the balanced fresh `noul`
templates while answering "yes" every time, and the three checks above passed it.

A failing cell is reported as ``READOUT_FAIL — <which check>`` with the measured values, never as
an accuracy.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from meddecide.eval.metrics import bootstrap_ci

DEFAULT_MIN_MEDIAN_LABEL_MASS = 0.5
DEFAULT_MIN_GREEDY_AGREEMENT = 0.9
# Fourth check (added in loop 1, S1): a cell that gives the same answer to (almost) every
# item is not reading the state at all, and it passes the three checks above when the
# template is balanced and the label mass is high. LFM2.5-350M sits at exactly 0.50 on the
# balanced fresh `noul` templates while answering "yes" every time — the pattern this
# catches. The ceiling on the template's majority share keeps the check from firing on a
# template where one answer genuinely dominates.
DEFAULT_MAX_MODAL_SHARE = 0.90
DEFAULT_DEGENERATE_MAJORITY_CEILING = 0.60


@dataclass
class HealthReport:
    """Outcome of the gate for one (model, template) cell."""

    model_id: str
    template_id: str
    n: int
    n_options: int
    chance: float
    median_label_mass: float | None
    greedy_agreement: float | None
    greedy_sample_size: int
    accuracy: float | None
    accuracy_ci95: tuple[float, float] | None
    modal_option: str | None = None
    modal_share: float | None = None
    majority_share: float | None = None
    failures: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.failures

    @property
    def status(self) -> str:
        return "PASS" if self.passed else "READOUT_FAIL — " + "; ".join(self.failures)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "template_id": self.template_id,
            "n": self.n,
            "n_options": self.n_options,
            "chance": self.chance,
            "median_label_mass": self.median_label_mass,
            "greedy_agreement": self.greedy_agreement,
            "greedy_sample_size": self.greedy_sample_size,
            "accuracy": self.accuracy,
            "accuracy_ci95": list(self.accuracy_ci95) if self.accuracy_ci95 else None,
            "modal_option": self.modal_option,
            "modal_share": self.modal_share,
            "majority_share": self.majority_share,
            "passed": self.passed,
            "status": self.status,
            "failures": self.failures,
            "notes": self.notes,
        }


def evaluate_cell(
    *,
    model_id: str,
    template_id: str,
    label_masses: Sequence[float],
    correct: Sequence[bool],
    greedy_matches: Sequence[bool] | None,
    n_options: int,
    min_median_label_mass: float = DEFAULT_MIN_MEDIAN_LABEL_MASS,
    min_greedy_agreement: float = DEFAULT_MIN_GREEDY_AGREEMENT,
    n_resamples: int = 1000,
    seed: int = 0,
    predicted_labels: Sequence[str] | None = None,
    majority_share: float | None = None,
    max_modal_share: float = DEFAULT_MAX_MODAL_SHARE,
    degenerate_majority_ceiling: float = DEFAULT_DEGENERATE_MAJORITY_CEILING,
) -> HealthReport:
    """Run the four checks on one cell.

    ``greedy_matches`` is ``None`` when the greedy sample was not run for this cell; the gate
    then records the greedy check as not measured rather than passing it silently.

    ``predicted_labels`` (the readout's argmax option key per item) enables the
    **constant-answer** check; it needs the template's ``majority_share`` to know whether a
    dominant answer is degenerate or simply correct. When either is missing the check is
    recorded as not measured.
    """
    if not label_masses or len(label_masses) != len(correct):
        raise ValueError("label_masses and correct must be non-empty and the same length")
    if n_options < 2:
        raise ValueError("n_options must be >= 2")
    if predicted_labels is not None and len(predicted_labels) != len(correct):
        raise ValueError("predicted_labels must be the same length as correct")
    chance = 1.0 / n_options
    median_mass = float(np.median(np.asarray(label_masses, dtype=np.float64)))
    accuracy = float(np.mean([1.0 if c else 0.0 for c in correct]))
    _, lo, hi = bootstrap_ci([1.0 if c else 0.0 for c in correct], n_resamples=n_resamples, seed=seed)

    modal_option = modal_share = None
    if predicted_labels:
        counts: dict[str, int] = {}
        for label in predicted_labels:
            counts[label] = counts.get(label, 0) + 1
        modal_option, modal_count = max(sorted(counts.items()), key=lambda kv: kv[1])
        modal_share = modal_count / len(predicted_labels)

    report = HealthReport(
        model_id=model_id,
        template_id=template_id,
        n=len(correct),
        n_options=n_options,
        chance=chance,
        median_label_mass=median_mass,
        greedy_agreement=(
            float(np.mean([1.0 if m else 0.0 for m in greedy_matches]))
            if greedy_matches
            else None
        ),
        greedy_sample_size=len(greedy_matches) if greedy_matches else 0,
        accuracy=accuracy,
        accuracy_ci95=(lo, hi),
        modal_option=modal_option,
        modal_share=modal_share,
        majority_share=majority_share,
    )
    if median_mass < min_median_label_mass:
        report.failures.append(
            f"median label mass {median_mass:.4f} < {min_median_label_mass}"
        )
    if greedy_matches:
        if report.greedy_agreement is not None and report.greedy_agreement < min_greedy_agreement:
            report.failures.append(
                f"greedy agreement {report.greedy_agreement:.3f} < {min_greedy_agreement} "
                f"(n={len(greedy_matches)})"
            )
    else:
        report.notes.append("greedy agreement NOT MEASURED for this cell")
    if hi < chance:
        report.failures.append(
            f"accuracy CI upper bound {hi:.4f} < chance {chance:.4f} (below-chance cell)"
        )
    if modal_share is not None and majority_share is not None:
        if majority_share <= degenerate_majority_ceiling and modal_share >= max_modal_share:
            report.failures.append(
                f"degenerate: one answer ({modal_option}) for {modal_share:.3f} of items "
                f">= {max_modal_share} on a template whose majority share {majority_share:.3f} "
                f"<= {degenerate_majority_ceiling}"
            )
    elif modal_share is not None or majority_share is not None:
        report.notes.append(
            "constant-answer check NOT MEASURED (needs both predicted labels and the "
            "template's majority share)"
        )
    return report


def gate_table(
    cells: Sequence[dict[str, Any]],
    *,
    min_median_label_mass: float = DEFAULT_MIN_MEDIAN_LABEL_MASS,
    min_greedy_agreement: float = DEFAULT_MIN_GREEDY_AGREEMENT,
    max_modal_share: float = DEFAULT_MAX_MODAL_SHARE,
    degenerate_majority_ceiling: float = DEFAULT_DEGENERATE_MAJORITY_CEILING,
    seed: int = 0,
) -> list[HealthReport]:
    """Run the gate over many cells; each cell is a mapping with the keys ``evaluate_cell`` takes."""
    return [
        evaluate_cell(
            min_median_label_mass=min_median_label_mass,
            min_greedy_agreement=min_greedy_agreement,
            max_modal_share=max_modal_share,
            degenerate_majority_ceiling=degenerate_majority_ceiling,
            seed=seed,
            **cell,
        )
        for cell in cells
    ]
