"""osler_v4 Stage A choice rule (ADVISORY Y6; pre-registered before any Stage A run): which recipe the Stage B blocks start from.

A run qualifies when its lower bounds against A0 are both above -0.01: the MR partition (template macro) and the knowledge
partition (micro), from the paired bootstrap within templates of meddecide.train.blocks.keep_statistics (1,000 resamples,
seed 0). Among the qualifying runs the highest CG v2 point estimate wins (the zero-shot baseline is common to all runs, so it
does not change the order). A difference under 0.003 from A0 is a tie, and A0 wins it.

A readout cell that fails the gate of meddecide.eval.readout_gate is not a result (D12, D21). The driver therefore compares
each run on the templates that pass for it and for A0 (`passing_templates`, `restrict_pairs`), and ranks the qualifying runs on
the CG v2 templates that pass for all of them (`common_templates`). When every cell passes, these are the full sets. The
restriction is STATE §7's Y6 entry.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from meddecide.eval.readout_gate import gate_by_template
from meddecide.train.blocks import Pair

REFERENCE = "A0"
LOWER_BOUND_LIMIT = -0.01  # ADVISORY Y6: MR and knowledge lower bounds against A0
TIE_MARGIN = 0.003  # ADVISORY Y6: a CG v2 difference under this from A0 is a tie, and A0 wins it


@dataclass(frozen=True)
class ArmResult:
    """One run's numbers against A0: the MR and knowledge lower bounds (None for A0 itself) and its CG v2 accuracy."""

    arm: str
    mr_lower: float | None
    knowledge_lower: float | None
    cg_accuracy: float


def qualifies(mr_lower: float, knowledge_lower: float) -> bool:
    """The rule's qualification: both lower bounds against A0 are above the limit."""
    return mr_lower > LOWER_BOUND_LIMIT and knowledge_lower > LOWER_BOUND_LIMIT


def passing_templates(rows: Iterable[Mapping[str, object]]) -> set[str]:
    """The templates whose cell passes the readout gate (D12, D21) in these prediction rows."""
    return {template for template, gate in gate_by_template(rows).items() if gate.passes}


def common_templates(template_sets: Sequence[set[str]]) -> set[str]:
    """The templates that pass for every given set; an empty intersection is refused (nothing could be compared)."""
    if not template_sets:
        raise ValueError("no template sets to intersect")
    common = set.intersection(*template_sets)
    if not common:
        raise ValueError("no template passes the gate for every compared run: nothing can be compared")
    return common


def restrict_pairs(pairs: Sequence[Pair], templates: set[str]) -> list[Pair]:
    """The paired items (candidate, reference) whose template is among ``templates``; order kept."""
    return [(x, c) for x, c in pairs if str(x["template_id"]) in templates]


def choose(results: Sequence[ArmResult]) -> dict[str, Any]:
    """The chosen arm and the qualification of every arm, from the pre-registered rule."""
    by_arm = {r.arm: r for r in results}
    if REFERENCE not in by_arm:
        raise ValueError(f"the reference {REFERENCE} is missing: nothing can be chosen")
    if len(by_arm) != len(results):
        raise ValueError("an arm appears twice: stop")
    reference = by_arm[REFERENCE]
    status: dict[str, str] = {REFERENCE: "reference"}
    qualifying = [REFERENCE]
    for r in results:
        if r.arm == REFERENCE:
            continue
        if r.mr_lower is None or r.knowledge_lower is None:
            raise ValueError(f"{r.arm}: the lower bounds against {REFERENCE} are missing")
        if qualifies(r.mr_lower, r.knowledge_lower):
            status[r.arm] = "qualifies"
            qualifying.append(r.arm)
        else:
            failed = [name for name, value in (("MR", r.mr_lower), ("knowledge", r.knowledge_lower))
                      if value <= LOWER_BOUND_LIMIT]
            status[r.arm] = "does not qualify: " + " and ".join(failed) + f" lower bound <= {LOWER_BOUND_LIMIT}"
    best = max(qualifying, key=lambda arm: by_arm[arm].cg_accuracy)
    margin = by_arm[best].cg_accuracy - reference.cg_accuracy
    if best != REFERENCE and margin < TIE_MARGIN:
        chosen = REFERENCE
        reason = f"the best qualifying run {best} is {margin:.4f} above {REFERENCE} on CG v2: a tie, and {REFERENCE} wins it"
    else:
        chosen = best
        reason = "the qualifying run with the highest CG v2 accuracy" if best != REFERENCE else f"{REFERENCE} is the best"
    return {"chosen": chosen, "reason": reason, "status": status, "qualifying": qualifying,
            "best_qualifying": best, "best_margin_over_reference": margin}
