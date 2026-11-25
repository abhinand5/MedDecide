"""osler_v4 Gate O4 (D49): the paired comparison of the final model against one baseline, and the gate's verdict.

Pairing (D44(a), applied to Gate O4 by the pre-registration in loops/osler_v4/STATE.md §7, reading 3): a final-model row is paired with
the baseline row of the same item. An item is unpaired when the baseline cannot take it (a ten-letter interface and more options) or
when the baseline did not score it. Above 1 % unpaired, the set's check against that baseline is NOT MEASURED; at or below, the
comparison uses the paired items, and a template with no paired item drops out and is reported.

Statistics (D49): accuracy is the template macro of the paired differences (final model minus baseline); Brier is the item mean of the
paired Brier differences (final model minus baseline; a negative value favours the final model). Both are item-paired percentile
bootstraps within templates (1,000 resamples, seed 0; meddecide.eval.paired).

Readout (D12): a template whose readout cell fails the gate for either model is not a result. A check that includes such a cell is
NOT MEASURED, unless a caller passes an explicit restriction that the pre-registration allows (the MIMIC audit suite against
MedDecider-4B, STATE §6 question 6).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from meddecide.eval.paired import (
    Difference,
    item_brier,
    paired_mean_difference_stratified,
    stratified_macro_difference,
)
from meddecide.eval.readout_gate import gate_by_template

UNPAIRED_LIMIT = 0.01  # D44(a): above this share of a set, its comparison is NOT MEASURED
PANEL_LIMIT = -0.02  # D49 (iii) and (iv): the accuracy lower bound must be above this
BRIER_UPPER_LIMIT = 0.0  # D49 (i) and (ii): the Brier upper bound must be below this
ACCURACY_LOWER_LIMIT = 0.0  # D49 (i) and (ii): the accuracy lower bound must be above this
BOOTSTRAP_RESAMPLES = 1000
BOOTSTRAP_SEED = 0
MEDDECIDER_OPTION_CAP = 10  # the released prompt letters options A to J (STATE §7, reading 3)

Row = Mapping[str, Any]


@dataclass(frozen=True)
class Pairing:
    """The paired items of one set, in the set's order, and the unpaired ones with their reasons.

    ``pairs`` holds (final-model row, baseline row, gold option index) for every paired item.
    """

    items: int
    pairs: tuple[tuple[Row, Row, int], ...]
    unpaired: dict[str, int] = field(default_factory=dict)
    unpaired_templates: dict[str, int] = field(default_factory=dict)
    set_templates: frozenset[str] = frozenset()

    @property
    def unpaired_total(self) -> int:
        return sum(self.unpaired.values())

    @property
    def unpaired_share(self) -> float:
        return self.unpaired_total / self.items if self.items else 0.0

    @property
    def paired_templates(self) -> frozenset[str]:
        return frozenset(str(m["template_id"]) for m, _, _ in self.pairs)

    @property
    def dropped_templates(self) -> frozenset[str]:
        return self.set_templates - self.paired_templates


@dataclass(frozen=True)
class Comparison:
    """Accuracy (template macro) and Brier (item mean) differences, final model minus baseline, on the paired items."""

    items: int
    templates: tuple[str, ...]
    accuracy: Difference
    brier: Difference


@dataclass(frozen=True)
class CheckResult:
    """One Gate O4 check against one baseline: PASS, FAIL or NOT MEASURED, with the reason and the figures."""

    name: str
    verdict: str
    reason: str
    comparison: Comparison | None = None


def option_index(options: Sequence[Mapping[str, Any]], gold_key: str) -> int:
    """The position of the gold key among an item's options (the order of the probability vector)."""
    keys = [str(o["key"]) for o in options]
    if gold_key not in keys:
        raise ValueError(f"gold key {gold_key!r} is not among the options {keys}")
    return keys.index(gold_key)


def normalise_row(row: Row, item: Mapping[str, Any]) -> dict[str, Any]:
    """A prediction row in the common shape: the item's gold and template, the argmax answer, and correctness from the probabilities.

    The runners write probabilities aligned to the item's option keys. ``predicted_key`` is the first maximum, as Python's ``max``
    takes it. The item is the authority for the template, the gold and the option keys.
    """
    keys = [str(o["key"]) for o in item["options"]]
    probs = [float(p) for p in row["probs"]]
    if len(probs) != len(keys):
        raise ValueError(f"{item['item_id']}: {len(probs)} probabilities for {len(keys)} options")
    best = max(range(len(probs)), key=lambda i: probs[i])
    gold = str(item["gold"])
    return {"item_id": str(item["item_id"]), "template_id": str(item["template_id"]), "gold_key": gold,
            "predicted_key": keys[best], "correct": keys[best] == gold, "probs": probs}


def pair_rows(model_rows: Sequence[Row], base_rows: Sequence[Row], items: Sequence[Row], *,
              option_cap: int | None = None) -> Pairing:
    """Pair the final model's rows with the baseline's rows over the set's items.

    ``items`` is the set file (one row per item, with ``item_id``, ``template_id``, ``gold`` and ``options``). An item with more
    options than ``option_cap`` is unpaired by the interface; an item the baseline did not score is unpaired. A final-model row
    missing for an item is counted as unpaired too, so nothing is dropped silently.
    """
    model_index = {str(r["item_id"]): r for r in model_rows}
    base_index = {str(r["item_id"]): r for r in base_rows}
    if len(model_index) != len(model_rows) or len(base_index) != len(base_rows):
        raise ValueError("duplicate item ids in a prediction file")
    unpaired: dict[str, int] = defaultdict(int)
    unpaired_templates: dict[str, int] = defaultdict(int)
    pairs: list[tuple[Row, Row, int]] = []
    templates: set[str] = set()
    for item in items:
        item_id = str(item["item_id"])
        template = str(item["template_id"])
        templates.add(template)
        reason = None
        if option_cap is not None and len(item["options"]) > option_cap:
            reason = f"more than {option_cap} options"
        elif item_id not in base_index:
            reason = "not scored by the baseline"
        elif item_id not in model_index:
            reason = "not scored by the final model"
        if reason is not None:
            unpaired[reason] += 1
            unpaired_templates[template] += 1
            continue
        index = option_index(item["options"], str(item["gold"]))
        pairs.append((model_index[item_id], base_index[item_id], index))
    return Pairing(items=len(items), pairs=tuple(pairs), unpaired=dict(sorted(unpaired.items())),
                   unpaired_templates=dict(sorted(unpaired_templates.items())), set_templates=frozenset(templates))


def readout_failures(rows: Sequence[Row]) -> list[str]:
    """The templates whose readout cell fails the gate (D12), sorted."""
    return sorted(t for t, gate in gate_by_template(rows).items() if not gate.passes)


def comparison(pairing: Pairing, *, templates: frozenset[str] | None = None) -> Comparison:
    """The paired differences, final model minus baseline, optionally over a restriction to some templates (labelled by the caller)."""
    chosen = [p for p in pairing.pairs if templates is None or str(p[0]["template_id"]) in templates]
    if not chosen:
        raise ValueError("no paired items to compare")
    groups_m: dict[str, list[float]] = defaultdict(list)
    groups_b: dict[str, list[float]] = defaultdict(list)
    brier_m: list[float] = []
    brier_b: list[float] = []
    strata: list[str] = []
    for m, b, gold in chosen:
        template = str(m["template_id"])
        groups_m[template].append(1.0 if m["correct"] else 0.0)
        groups_b[template].append(1.0 if b["correct"] else 0.0)
        brier_m.append(item_brier(m["probs"], gold))
        brier_b.append(item_brier(b["probs"], gold))
        strata.append(template)
    accuracy = stratified_macro_difference(dict(groups_m), dict(groups_b), n_resamples=BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED)
    brier = paired_mean_difference_stratified(brier_m, brier_b, strata, n_resamples=BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED)
    return Comparison(items=len(chosen), templates=tuple(sorted(groups_m)), accuracy=accuracy, brier=brier)


def check(name: str, pairing: Pairing, *, kind: str, model_failures: Sequence[str] = (), base_failures: Sequence[str] = (),
          restriction: frozenset[str] | None = None, restriction_note: str = "") -> CheckResult:
    """One Gate O4 check. ``kind`` is ``"mr_or_mimic"`` (accuracy and Brier, D49 (i) and (ii)) or ``"accuracy"`` (D49 (iii) and (iv)).

    NOT MEASURED when the unpaired share is above 1 %, or when a readout cell fails for either model and no restriction is given.
    With a restriction (the one the pre-registration allows), the comparison uses only those templates, and the reason says so.
    """
    if pairing.unpaired_share > UNPAIRED_LIMIT:
        return CheckResult(name, "NOT MEASURED", f"{pairing.unpaired_total} of {pairing.items} items unpaired "
                           f"({pairing.unpaired_share:.2%}, above 1 %; D44(a))")
    failing = sorted(set(model_failures) | set(base_failures))
    if failing and restriction is None:
        return CheckResult(name, "NOT MEASURED", f"readout cells fail for a model: {', '.join(failing)} (D12)")
    note = ""
    if restriction is not None:
        if not failing:
            raise ValueError(f"{name}: a restriction is given but no readout cell fails")
        note = f"restricted to the {len(restriction)} templates that pass for both models ({restriction_note})"
    scored = comparison(pairing, templates=restriction)
    acc = scored.accuracy
    if kind == "mr_or_mimic":
        passed = acc.lo > ACCURACY_LOWER_LIMIT and scored.brier.hi < BRIER_UPPER_LIMIT
        rule = "accuracy lower > 0 and Brier upper < 0"
    elif kind == "accuracy":
        passed = acc.lo > PANEL_LIMIT
        rule = "accuracy lower > -0.02"
    else:
        raise ValueError(f"unknown check kind {kind!r}")
    reason = rule + (f"; {note}" if note else "")
    return CheckResult(name, "PASS" if passed else "FAIL", reason, scored)


def gate(results: Mapping[str, CheckResult]) -> str:
    """The gate against one baseline: PASS iff every check passes; FAIL if any measured check fails; else NOT MEASURED."""
    verdicts = [r.verdict for r in results.values()]
    if not verdicts:
        raise ValueError("no checks")
    if any(v == "FAIL" for v in verdicts):
        return "FAIL"
    if all(v == "PASS" for v in verdicts):
        return "PASS"
    return "NOT MEASURED"
