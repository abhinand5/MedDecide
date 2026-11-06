"""Seeded subsamples of a frozen set and the aggregate scores of one model on them (d3_bench quick benchmarks).

A quick benchmark scores a fixed, seeded subsample of a frozen set instead of the whole set. The subsample is drawn from the set's
item ids sorted as strings, so the same seed and the same set always give the same ids:

- ``seeded_sample``: a simple random sample of ``k`` ids (``random.Random(seed).sample``);
- ``stratified_sample``: ``k`` ids allocated to strata in proportion to their sizes (largest remainder, ties broken by the
  stratum name), then a simple random sample within each stratum, strata visited in name order with one generator.

``aggregate`` gives the same figures the gates use elsewhere: item micro accuracy, template macro accuracy (the mean of the
per-template accuracies) and the D12/D21 readout gate per (model, template) cell (``meddecide.eval.readout_gate``). A set whose
cells all pass reads ``PASS``; otherwise it reads ``READOUT_FAIL`` with the failing templates, and its accuracy is not a result.
"""

from __future__ import annotations

import hashlib
import random
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from meddecide.eval.readout_gate import gate_by_template


def seeded_sample(ids: Sequence[str], k: int, seed: int) -> list[str]:
    """``k`` distinct ids drawn with ``random.Random(seed)`` from the ids sorted as strings; returned in sorted order."""
    unique = sorted({str(i) for i in ids})
    if len(unique) != len(ids):
        raise ValueError("duplicate ids")
    if not 0 < k <= len(unique):
        raise ValueError(f"cannot draw {k} of {len(unique)} ids")
    return sorted(random.Random(seed).sample(unique, k))


def proportional_allocation(sizes: Mapping[str, int], k: int) -> dict[str, int]:
    """``k`` split over strata in proportion to their sizes (largest remainder; ties by stratum name)."""
    total = sum(sizes.values())
    if not 0 < k <= total:
        raise ValueError(f"cannot allocate {k} of {total}")
    quotas = {name: k * size / total for name, size in sizes.items()}
    alloc = {name: int(q) for name, q in quotas.items()}
    left = k - sum(alloc.values())
    order = sorted(sizes, key=lambda name: (-(quotas[name] - alloc[name]), name))
    for name in order[:left]:
        alloc[name] += 1
    if any(alloc[name] > sizes[name] for name in sizes):
        raise ValueError("an allocation exceeds its stratum")
    return dict(sorted(alloc.items()))


def stratified_sample(items: Sequence[Mapping[str, Any]], stratum: Callable[[Mapping[str, Any]], str], k: int,
                      seed: int) -> tuple[list[str], dict[str, int]]:
    """``k`` ids stratified by ``stratum`` with proportional allocation; returns (sorted ids, allocation per stratum)."""
    groups: dict[str, list[str]] = defaultdict(list)
    for item in items:
        groups[str(stratum(item))].append(str(item["item_id"]))
    alloc = proportional_allocation({name: len(ids) for name, ids in groups.items()}, k)
    rng = random.Random(seed)
    chosen: list[str] = []
    for name in sorted(groups):
        chosen.extend(rng.sample(sorted(groups[name]), alloc[name]))
    if len(set(chosen)) != k:
        raise ValueError("the stratified sample is not k distinct ids")
    return sorted(chosen), alloc


def id_list_sha256(ids: Sequence[str]) -> str:
    """SHA-256 of the sorted ids joined by newlines (the subsample's fingerprint)."""
    return hashlib.sha256("\n".join(sorted(str(i) for i in ids)).encode("utf-8")).hexdigest()


def probs_from_answer(option_keys: Sequence[str], qtype: str, answer: Mapping[str, Any]) -> list[float]:
    """One decision-model answer as a probability vector in the item's option order.

    ``choice``: ``answer["probabilities"]`` maps each offered key to its probability. ``noul``: ``answer["noul"]`` is p(yes)
    and the item's keys must be ``yes`` and ``no`` (in either order).
    """
    if qtype == "choice":
        probabilities = answer["probabilities"]
        missing = [k for k in option_keys if k not in probabilities]
        if missing or len(probabilities) != len(option_keys):
            raise ValueError(f"the answer's keys {sorted(probabilities)} are not the offered keys {list(option_keys)}")
        return [float(probabilities[k]) for k in option_keys]
    if qtype == "noul":
        if sorted(option_keys) != ["no", "yes"]:
            raise ValueError(f"a noul item needs the keys yes and no, not {list(option_keys)}")
        p_yes = float(answer["noul"])
        return [p_yes if k == "yes" else 1.0 - p_yes for k in option_keys]
    raise ValueError(f"unsupported qtype {qtype!r}")


def aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Micro and template-macro accuracy and the readout gate per template, from rows in the gate_o4 common shape."""
    if not rows:
        return {"items": 0, "correct": 0, "accuracy_micro": None, "accuracy_template_macro": None, "templates": 0,
                "gate": "NOT MEASURED", "failing_templates": {}, "per_template": {}}
    groups: dict[str, list[bool]] = defaultdict(list)
    for row in rows:
        groups[str(row["template_id"])].append(bool(row["correct"]))
    gates = gate_by_template(rows)
    failing = {t: g.status for t, g in gates.items() if not g.passes}
    per_template = {t: {"items": len(v), "correct": sum(v), "accuracy": sum(v) / len(v), "gate": gates[t].status}
                    for t, v in sorted(groups.items())}
    correct = sum(sum(v) for v in groups.values())
    return {
        "items": len(rows),
        "correct": correct,
        "accuracy_micro": correct / len(rows),
        "accuracy_template_macro": sum(e["accuracy"] for e in per_template.values()) / len(per_template),
        "templates": len(per_template),
        "gate": "PASS" if not failing else "READOUT_FAIL",
        "failing_templates": failing,
        "per_template": per_template,
    }
