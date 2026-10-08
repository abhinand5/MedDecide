#!/usr/bin/env python
"""S8: score MedDecide-Bench v0.2 with a trained MedDecide checkpoint (pointer head).

This is the evaluation path for a *trained* model. It loads a checkpoint written by
:meth:`meddecide.model.meddecide_model.MedDecideModel.save` (adapter + pointer head +
calibration), scores the benchmark with the **pointer head** — batched, no text generation —
and reports the same metrics, on the same item sets and with the same filters, as the
zero-shot runner ``scripts/bench/run_baselines_v0_1.py``.

What is shared with the zero-shot path (deliberately, so the two are comparable):

* item loading and the ``--tier1`` / ``--fresh`` / ``--keep-screen`` / ``--only-template``
  filters, and ``--split {test,dev}`` (test = the benchmark's test split, dev = its dev split);
* the prediction log: rows are :class:`meddecide.eval.harness.Prediction` JSON objects with a
  ``run_id``, appended to ``preds_<slug><suffix>.jsonl`` and read back through
  :mod:`meddecide.eval.predlog`, which deduplicates by ``(run_id, item_id)``;
* the report shape: ``model_<slug><suffix>.json`` with ``tiers`` and ``cells``, beside a
  ``*_provenance.json``;
* the per-cell metrics: accuracy (seeded percentile bootstrap CI), macro accuracy, Brier, top-label
  ECE (per-item form, so cells with mixed option counts work), the template's majority share,
  coverage (share of *this template's* items the model could take), latency and prompt tokens.

What is different, and why (D12 for a pointer head):

* there is no letter readout here. The head's softmax is over exactly the offered options, so the
  two readout-specific D12 checks — **median label mass** and **greedy agreement** — have no
  analogue. They are recorded per cell as
  ``NOT APPLICABLE — pointer head reads the option states directly; no letter readout exists to
  validate`` and are never reported as passed. They are not silently dropped: the cell carries
  them under ``gate.checks`` and in ``gate_notes``.
* the two checks that *are* meaningful are applied unchanged, with the library's thresholds
  (``meddecide.eval.health``): **accuracy CI vs chance** and the **constant-answer** check. A
  failing cell reads ``READOUT_FAIL — <check>`` and no accuracy is presented as a result.
* one **additional** check is measured and explicitly labelled as additional (not a replacement):
  **option-permutation consistency** — the share of sampled items whose chosen option *content*
  changes when the options are reordered (the zero-shot path reports the same probe as its
  ``shuffle_flip_rate``). Its threshold is new in S8 and is stated in the cell; it never changes
  the D12 status.

``--split dev`` fits one temperature per question type on the dev items (dev only — fitting on a
test split is refused) and writes them, with the item counts and the NLL before/after, into the
checkpoint directory as ``temperature.json``. A qtype with too few items, or a fit that lands on
the search bound, is reported as not fitted rather than quoted as if it were.

Usage::

    uv run python scripts/bench/run_student.py --split test --out outputs/student_v0/S8
    uv run python scripts/bench/run_student.py --split dev  --out outputs/student_v0/S8

Predictions and item text stay in the gitignored ``outputs/``; only aggregates are committed.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from meddecide.bench.schema import Item, QuestionType
from meddecide.eval.harness import Prediction
from meddecide.eval.health import (
    DEFAULT_DEGENERATE_MAJORITY_CEILING,
    DEFAULT_MAX_MODAL_SHARE,
)
from meddecide.eval.metrics import (
    bootstrap_ci,
    brier_score,
    ece_from_confidence,
    macro_accuracy,
)
from meddecide.eval.predlog import read_prediction_log
from meddecide.eval.probes import shuffle_options
from meddecide.eval.readout import expected_level
from meddecide.model.meddecide_model import Decision, MedDecideModel, ScoredItems
from meddecide.train.temperature import fit_per_qtype
from meddecide.utils.io import read_jsonl
from meddecide.utils.provenance import Provenance, gpu_name, utcnow

DEFAULT_CHECKPOINT = Path("outputs/student_v0/S7/checkpoint")
DEFAULT_TIER1 = Path("data/bench/v0.2/tier1")
DEFAULT_FRESH = Path("data/bench/v0.2/fresh")
DEFAULT_SCREEN = Path("data/bench/v0.2/screen.json")
DEFAULT_MODEL_ID = "meddecide-0.8b-lora-pointer"
POINTER_VARIANT = "pointer-head"
# the one string every inapplicable D12 check carries, so a reader can tell "not measured" from
# "not applicable" from "passed" without reading this file
NO_LETTER_READOUT_REASON = (
    "NOT APPLICABLE — pointer head reads the option states directly; no letter readout exists "
    "to validate"
)
# Additional check (new in S8; NOT part of D12, NOT a replacement for any D12 check): a readout
# whose choice is driven by option content keeps its choice on a reordered prompt most of the
# time. A pure display-position reader flips with probability 1 - 1/n, i.e. >= 0.5 for every
# offered count (0.5 at n=2, 0.75 at n=4), so a flip rate at or above 0.5 is not distinguishable
# from position guessing. The threshold is recorded in every cell that uses it.
FLIP_CHECK_THRESHOLD = 0.5
# selectivity fractions the ADVISORY asks for (most-confident 50 %, 80 %, 90 %)
SELECTIVE_FRACTIONS = (0.5, 0.8, 0.9)
# a temperature fitted on fewer items than this is reported as NOT FITTED (the value is kept in
# the artifact as a diagnostic, never as the fitted temperature)
DEFAULT_MIN_TEMPERATURE_ITEMS = 50
TEMPERATURE_BOUNDS = (0.02, 50.0)


def slug(model_id: str) -> str:
    """File-name-safe model slug, the same transform the zero-shot runner uses."""
    return model_id.split("/")[-1].lower().replace(".", "p")


# --------------------------------------------------------------------------- loading
def load_benchmark(
    tier1: Path, fresh: Path, keep: set[str] | None, split: str
) -> dict[str, list[Item]]:
    """All items of ``split``, grouped by tier, filtered to kept templates.

    The same loader as the zero-shot runner, with the split as a parameter instead of the
    hardcoded ``test``: ``--split dev`` reads the benchmark's dev split, which is the only split
    anything is ever fitted on.
    """
    if split not in {"test", "dev"}:
        raise ValueError(f"split must be 'test' or 'dev', got {split!r}")
    out: dict[str, list[Item]] = {}
    for tier, directory in (("tier1", tier1), ("fresh", fresh)):
        rows: list[Item] = []
        for path in sorted(directory.glob("*.jsonl")):
            loaded, report = read_jsonl(path, Item)
            report.check_closes()
            rows.extend(i for i in loaded if str(i.split) == split)
        if keep is not None:
            rows = [i for i in rows if i.template_id in keep]
        out[tier] = rows
    return out


def per_template_totals(items: Sequence[Item]) -> dict[str, int]:
    return dict(Counter(i.template_id for i in items))


def strict_slice_ids(fresh_items: Sequence[Item], acceptance_path: Path) -> set[str] | None:
    """Item ids of the strict slice (records dated on/after ``strict_slice_start``)."""
    if not acceptance_path.exists():
        return None
    strict_start = json.loads(acceptance_path.read_text(encoding="utf-8")).get("strict_slice_start")
    if not strict_start:
        return None
    return {i.item_id for i in fresh_items if str(i.record_date) >= str(strict_start)}


def load_keep_set(screen_path: Path | None) -> set[str] | None:
    """Template ids kept by the screen, or ``None`` when no screen is applied."""
    if screen_path is None or not Path(screen_path).exists():
        return None
    screen = json.loads(Path(screen_path).read_text(encoding="utf-8"))
    return {t["template_id"] for t in screen["templates"] if not t["drop"]}


# --------------------------------------------------------------------------- scoring
def decisions_from_scored(
    model: MedDecideModel,
    scored: ScoredItems,
    *,
    temperature: dict[str, float] | None = None,
) -> list[Decision]:
    """Turn a :class:`ScoredItems` result into :class:`Decision` objects.

    This mirrors :meth:`MedDecideModel.predict` exactly (calibration applied to the probabilities,
    argmax taken after it, expected level for ``score`` items) but takes the already-computed
    ``ScoredItems``, so a dev run can fit temperatures and produce predictions from **one** forward
    pass. ``test_decisions_from_scored_matches_predict`` pins the equivalence against the model's
    own ``predict`` implementation.
    """
    temps = dict(model.calibration) if temperature is None else dict(temperature)
    calibrated = model.apply_temperature(scored, temperature) if temps else scored
    decisions: list[Decision] = []
    for i, item in enumerate(scored.items):
        probs = calibrated.probs[i]
        argmax = int(np.argmax(probs))
        gold_index = scored.gold_indices[i]
        keys = scored.option_keys[i]
        decisions.append(
            Decision(
                item_id=item.item_id,
                qtype=str(item.qtype),
                template_id=item.template_id,
                source=item.source,
                option_keys=list(keys),
                original_option_keys=list(scored.original_option_keys[i]),
                probs=[float(p) for p in probs],
                argmax_index=argmax,
                gold_key=keys[gold_index],
                correct=argmax == gold_index,
                expected_level=(
                    expected_level(probs) if item.qtype is QuestionType.SCORE else None
                ),
                n_options=len(keys),
                prompt_tokens=scored.prompt_tokens[i],
                latency_s=scored.latency_s[i],
                temperature=float(temps.get(str(item.qtype), temps.get("*", 1.0))),
            )
        )
    return decisions


def decision_to_prediction(
    decision: Decision, *, model_id: str, run_id: str, split: str
) -> Prediction:
    """Convert the model API's :class:`Decision` into the shared prediction row.

    The columns the reports read (``item_id``, ``model_id``, ``run_id``, ``source``,
    ``template_id``, ``split``, ``qtype``, ``option_keys``, ``option_probs``, ``argmax_key``,
    ``gold_key``, ``correct``, ``expected_level``, ``latency_s``, ``prompt_tokens``) all come from
    the decision. Its letter-readout-only diagnostics (``label_mass`` and the vocabulary
    top-k fields) have no measured value for a pointer head: the softmax is over exactly the
    offered options, so the mass on them is 1.0 **by construction**. ``label_mass`` is written as
    1.0 for schema compatibility and ``variant`` is ``pointer-head``, so a consumer can see at a
    glance that the row did not come from the letter readout; the D12 mass check is *not*
    applied to it (see :func:`pointer_head_gate`).
    """
    return Prediction(
        item_id=decision.item_id,
        model_id=model_id,
        run_id=run_id,
        source=decision.source,
        template_id=decision.template_id,
        split=split,
        qtype=decision.qtype,
        option_keys=list(decision.option_keys),
        option_probs=[float(p) for p in decision.probs],
        # not a measurement for this readout: see the docstring (the pointer softmax lies on the
        # offered options by construction). The report never reads this as a health signal.
        label_mass=1.0,
        argmax_index=decision.argmax_index,
        gold_key=decision.gold_key,
        correct=decision.correct,
        expected_level=decision.expected_level,
        latency_s=decision.latency_s,
        variant=POINTER_VARIANT,
        prompt_tokens=decision.prompt_tokens,
        original_option_keys=list(decision.original_option_keys),
        transform={
            "readout": POINTER_VARIANT,
            "letter_readout_fields": "not applicable — pointer head scores option states directly",
        },
    )


def option_permutation_flip(
    model: MedDecideModel,
    items: Sequence[Item],
    predictions: Sequence[Prediction],
    *,
    n_items: int,
    seed: int,
    score_kwargs: dict[str, Any],
) -> tuple[float | None, int, str | None]:
    """Share of sampled items whose chosen option *content* changes when options are reordered.

    The same probe the zero-shot runner reports as ``shuffle_flip_rate``: options are permuted
    deterministically (:func:`meddecide.eval.probes.shuffle_options`, which re-keys them in display
    order and keeps gold with its content), the model is scored again, and the chosen display
    position is mapped back through the permutation to the original option it selected.

    Returns ``(flip_rate, n_scored, error)``; ``error`` is a one-line reason when the probe could
    not run (recorded, never silently dropped).
    """
    if n_items <= 0 or len(items) < 2:
        return None, 0, "option-shuffle probe disabled or fewer than 2 items"
    rng = random.Random(f"{seed}:shuffle:{items[0].template_id}")
    idx = rng.sample(range(len(items)), min(n_items, len(items)))
    transformed: list[Item] = []
    perms: list[list[int]] = []
    original_pick: list[int] = []
    for i in idx:
        shuffled, record = shuffle_options(items[i], seed=seed, salt=items[i].item_id)
        transformed.append(shuffled)
        perms.append([int(x) for x in record.details["perm"]])
        original_pick.append(predictions[i].argmax_index)
    try:
        scored = model.score_items(transformed, **score_kwargs)
    except Exception as exc:  # recorded, not hidden: the cell says NOT MEASURED with the error
        return None, 0, f"{type(exc).__name__}: {exc}"
    decisions = decisions_from_scored(model, scored, temperature={})
    if len(decisions) != len(transformed):
        return None, 0, f"scored {len(decisions)} of {len(transformed)} permuted items"
    flips = 0
    for n, decision in enumerate(decisions):
        original_position = perms[n][decision.argmax_index]
        flips += int(original_position != original_pick[n])
    return flips / len(decisions), len(decisions), None


# --------------------------------------------------------------------------- metrics
def selective_accuracy(
    confidence: Sequence[float],
    correct: Sequence[bool],
    *,
    fractions: Sequence[float] = SELECTIVE_FRACTIONS,
) -> dict[str, dict[str, float | int]]:
    """Accuracy on the most-confident ``f`` share of items, per fraction.

    Items are ranked by their top probability (descending); the subset is the first
    ``ceil(f * n)`` items. Ties are broken by input order, so the result is deterministic.
    Returns ``{fraction: {"n", "accuracy", "min_confidence"}}``.
    """
    if len(confidence) != len(correct):
        raise ValueError("confidence and correct must be the same length")
    if not confidence:
        raise ValueError("selective_accuracy needs at least one item")
    conf = np.asarray(confidence, dtype=np.float64)
    flags = np.asarray([1.0 if c else 0.0 for c in correct], dtype=np.float64)
    order = np.argsort(-conf, kind="stable")
    out: dict[str, dict[str, float | int]] = {}
    for fraction in fractions:
        if not 0.0 < float(fraction) <= 1.0:
            raise ValueError(f"fraction must be in (0, 1], got {fraction}")
        n_keep = max(1, int(np.ceil(float(fraction) * len(conf))))
        picked = order[:n_keep]
        out[f"{float(fraction):.2f}"] = {
            "n": int(n_keep),
            "accuracy": float(flags[picked].mean()),
            "min_confidence": float(conf[picked].min()),
        }
    return out


def brier_from_rows(probs: Sequence[np.ndarray], gold_indices: Sequence[int]) -> float:
    """Multi-class Brier over rows with possibly different option counts.

    ``brier_score`` needs a rectangular array, and one cell's items need not share an option
    count, so the metric is computed per option-count group and combined with explicit weights
    (R5: the parts sum to the total).
    """
    if len(probs) != len(gold_indices):
        raise ValueError("probs and gold_indices must have one entry per item")
    groups: dict[int, list[int]] = {}
    for i, row in enumerate(probs):
        groups.setdefault(len(row), []).append(i)
    weighted = 0.0
    counted = 0
    for width, indices in sorted(groups.items()):
        rows = np.stack([np.asarray(probs[i], dtype=np.float64) for i in indices])
        if rows.shape[1] != width:  # pragma: no cover - shape guard
            raise AssertionError("option-count grouping is inconsistent")
        weighted += len(indices) * brier_score(rows, [gold_indices[i] for i in indices])
        counted += len(indices)
    if counted != len(probs):  # pragma: no cover - shape guard
        raise AssertionError("Brier denominator disagrees with the item count")
    return weighted / counted


def brier_from_predictions(predictions: Sequence[Prediction]) -> float:
    """Brier over prediction rows (see :func:`brier_from_rows` for the option-count grouping)."""
    return brier_from_rows(
        [p.option_probs for p in predictions],
        [p.option_keys.index(p.gold_key) for p in predictions],
    )


def uncalibrated_metrics(scored: ScoredItems, correct: Sequence[bool]) -> dict[str, Any]:
    """Additional block: the same pass's raw probabilities (T=1 for every qtype).

    A per-qtype temperature is monotone, so it cannot change accuracy, macro accuracy or the
    argmax; it does change Brier, ECE and the mean gold probability. Reporting both makes the
    effect of the checkpoint's calibration visible instead of implicit.
    """
    probs = [np.asarray(p, dtype=np.float64) for p in scored.probs]
    gold_probs = [float(p[g]) for p, g in zip(probs, scored.gold_indices, strict=True)]
    return {
        "brier": brier_from_rows(probs, scored.gold_indices),
        "ece": ece_from_confidence([float(p.max()) for p in probs], correct),
        "mean_gold_prob": float(np.mean(gold_probs)),
        "note": (
            "additional: same forward pass at T=1 for every qtype; accuracy and macro accuracy "
            "are unchanged by a per-qtype temperature, Brier/ECE are not"
        ),
    }


def score_expected_level_error(predictions: Sequence[Prediction]) -> dict[str, Any]:
    """``score``-cell metric: |expected level - gold level| in level units.

    Levels are 1-based and follow the canonical (lowest-level-first) option order, which is the
    order ``canonicalise_options`` renders and the order the harness's ``expected_level`` uses.
    """
    if not predictions:
        raise ValueError("no predictions")
    levels = np.asarray([float(p.expected_level) for p in predictions], dtype=np.float64)
    gold = np.asarray(
        [float(p.option_keys.index(p.gold_key) + 1) for p in predictions], dtype=np.float64
    )
    abs_error = np.abs(levels - gold)
    return {
        "n": len(predictions),
        "mean_expected_level": float(levels.mean()),
        "mean_gold_level": float(gold.mean()),
        "mean_abs_error": float(abs_error.mean()),
        "median_abs_error": float(np.median(abs_error)),
    }


# --------------------------------------------------------------------------- D12 for a pointer head
def pointer_head_gate(
    *,
    correct: Sequence[bool],
    predicted_labels: Sequence[str],
    gold_labels: Sequence[str],
    n_options: int,
    seed: int = 0,
    n_resamples: int = 1000,
    chance: float | None = None,
    shuffle_flip: float | None = None,
    shuffle_n: int = 0,
) -> dict[str, Any]:
    """The D12 readout-health gate for a pointer-head cell, honestly bounded.

    Applied (unchanged thresholds, from :mod:`meddecide.eval.health`):

    * ``accuracy_ci_vs_chance`` — the bootstrap CI's upper bound must not be below chance;
    * ``constant_answer`` — one answer for >= 0.90 of items on a template whose majority share is
      <= 0.60 means the model is not reading the state.

    Recorded as inapplicable (never as passed): ``median_label_mass`` and ``greedy_agreement``.
    There is no vocabulary readout and no generation anywhere in the decision path, so neither
    quantity exists to measure.

    Measured as an **additional**, explicitly labelled check: ``option_permutation_consistency``
    (see ``FLIP_CHECK_THRESHOLD``). It is reported beside the D12 status and never changes it.
    """
    if not correct:
        raise ValueError("the gate needs at least one item")
    if not (len(correct) == len(predicted_labels) == len(gold_labels)):
        raise ValueError("correct, predicted_labels and gold_labels must be the same length")
    if n_options < 2:
        raise ValueError("n_options must be >= 2")
    chance_value = (1.0 / n_options) if chance is None else float(chance)
    flags = [1.0 if c else 0.0 for c in correct]
    accuracy = float(np.mean(flags))
    _, lo, hi = bootstrap_ci(flags, n_resamples=n_resamples, seed=seed)

    counts = Counter(predicted_labels)
    modal_option, modal_count = max(sorted(counts.items()), key=lambda kv: kv[1])
    modal_share = modal_count / len(predicted_labels)
    gold_counts = Counter(gold_labels)
    majority_label, majority_count = max(sorted(gold_counts.items()), key=lambda kv: kv[1])
    majority_share = majority_count / len(gold_labels)

    failures: list[str] = []
    failing_checks: list[str] = []
    checks: dict[str, Any] = {}

    ci_failure = None
    if hi < chance_value:
        ci_failure = (
            f"accuracy CI upper bound {hi:.4f} < chance {chance_value:.4f} (below-chance cell)"
        )
        failures.append(ci_failure)
        failing_checks.append("accuracy_ci_vs_chance")
    checks["accuracy_ci_vs_chance"] = {
        "applied": True,
        "kind": "d12",
        "status": "FAIL" if ci_failure else "PASS",
        "n": len(correct),
        "accuracy": accuracy,
        "accuracy_ci95": [lo, hi],
        "chance": chance_value,
        "failure": ci_failure,
    }

    ca_failure = None
    if majority_share <= DEFAULT_DEGENERATE_MAJORITY_CEILING and (
        modal_share >= DEFAULT_MAX_MODAL_SHARE
    ):
        ca_failure = (
            f"degenerate: one answer ({modal_option}) for {modal_share:.3f} of items "
            f">= {DEFAULT_MAX_MODAL_SHARE} on a template whose majority share "
            f"{majority_share:.3f} <= {DEFAULT_DEGENERATE_MAJORITY_CEILING}"
        )
        failures.append(ca_failure)
        failing_checks.append("constant_answer")
    checks["constant_answer"] = {
        "applied": True,
        "kind": "d12",
        "status": "FAIL" if ca_failure else "PASS",
        "modal_option": modal_option,
        "modal_share": modal_share,
        "majority_label": majority_label,
        "majority_share": majority_share,
        "max_modal_share": DEFAULT_MAX_MODAL_SHARE,
        "degenerate_majority_ceiling": DEFAULT_DEGENERATE_MAJORITY_CEILING,
        "failure": ca_failure,
    }

    checks["median_label_mass"] = {
        "applied": False,
        "kind": "d12",
        "status": NO_LETTER_READOUT_REASON,
        "reason": NO_LETTER_READOUT_REASON,
        "detail": (
            "the pointer head's softmax is over exactly the offered options, so 'mass on the "
            "option tokens' is 1.0 by construction and is not a measurement here"
        ),
        "failure": None,
    }
    checks["greedy_agreement"] = {
        "applied": False,
        "kind": "d12",
        "status": NO_LETTER_READOUT_REASON,
        "reason": NO_LETTER_READOUT_REASON,
        "detail": "no text generation is part of the decision path, so there is nothing to agree with",
        "failure": None,
    }

    if shuffle_flip is None:
        flip_status = "NOT MEASURED — " + (
            "the option-shuffle probe did not run for this cell"
        )
        flip_failure = None
    else:
        flip_failure = None
        if shuffle_flip >= FLIP_CHECK_THRESHOLD:
            flip_failure = (
                f"option-permutation flip rate {shuffle_flip:.3f} >= {FLIP_CHECK_THRESHOLD} "
                f"(n={shuffle_n}): the chosen option content changes on a reordered prompt as "
                f"often as a display-position reader would"
            )
            flip_status = "FAIL"
        else:
            flip_status = "PASS"
    checks["option_permutation_consistency"] = {
        "applied": True,
        "kind": "additional",
        "label": (
            "additional check — new in S8, NOT part of D12 and not a replacement for any D12 check"
        ),
        "status": flip_status,
        "flip_rate": shuffle_flip,
        "n": shuffle_n,
        "threshold": FLIP_CHECK_THRESHOLD,
        "failure": flip_failure,
    }

    notes = [
        f"median label mass: {NO_LETTER_READOUT_REASON}",
        f"greedy agreement: {NO_LETTER_READOUT_REASON}",
    ]
    if flip_failure:
        notes.append(f"additional check FAILED — {flip_failure}")
    elif shuffle_flip is not None:
        notes.append(
            f"additional check passed: option-permutation flip rate {shuffle_flip:.3f} "
            f"< {FLIP_CHECK_THRESHOLD} (n={shuffle_n})"
        )
    else:
        notes.append("additional check NOT MEASURED — option-shuffle probe did not run")

    return {
        "model_readout": POINTER_VARIANT,
        "applied_checks": ["accuracy_ci_vs_chance", "constant_answer"],
        "inapplicable_checks": ["median_label_mass", "greedy_agreement"],
        "inapplicable_reason": NO_LETTER_READOUT_REASON,
        "checks": checks,
        "failures": failures,
        "status": "PASS" if not failures else "READOUT_FAIL — " + "; ".join(failing_checks),
        "notes": notes,
    }


# --------------------------------------------------------------------------- cells
def empty_cell(
    *, model_id: str, tier: str, template_id: str, split: str, template_total: int, reason: str
) -> dict[str, Any]:
    """A cell that could not be measured, with the reason spelled out (R3)."""
    return {
        "model_id": model_id,
        "tier": tier,
        "template_id": template_id,
        "split": split,
        "n": 0,
        "template_total": template_total,
        "coverage": 0.0,
        "accuracy": None,
        "gate_status": f"NOT MEASURED — {reason}",
        "gate_failures": [],
        "gate_notes": [f"NOT MEASURED — {reason}"],
        "gate": None,
    }


def summarise_cell(
    *,
    model_id: str,
    tier: str,
    template_id: str,
    split: str,
    predictions: Sequence[Prediction],
    template_total: int,
    template_available: int,
    sampled: bool,
    strict_ids: set[str] | None,
    seed: int,
    shuffle_flip: float | None,
    shuffle_n: int,
    shuffle_error: str | None,
    temperature: dict[str, float],
    uncalibrated: dict[str, Any] | None,
    wall_seconds: float,
) -> dict[str, Any]:
    """One (model, tier, template) cell with its metrics and its gate."""
    n = len(predictions)
    if n == 0:
        return empty_cell(
            model_id=model_id,
            tier=tier,
            template_id=template_id,
            split=split,
            template_total=template_total,
            reason="the model could not take any item of this template",
        )
    qtype = str(predictions[0].qtype)
    n_options = max(len(p.option_keys) for p in predictions)
    chance = 1.0 / n_options
    correct = [p.correct for p in predictions]
    gold = [p.gold_key for p in predictions]
    predicted = [p.option_keys[p.argmax_index] for p in predictions]
    confidence = [max(p.option_probs) for p in predictions]
    gold_probs = np.asarray(
        [p.option_probs[p.option_keys.index(p.gold_key)] for p in predictions], dtype=np.float64
    )
    gate = pointer_head_gate(
        correct=correct,
        predicted_labels=predicted,
        gold_labels=gold,
        n_options=n_options,
        seed=seed,
        shuffle_flip=shuffle_flip,
        shuffle_n=shuffle_n,
    )
    cell: dict[str, Any] = {
        "model_id": model_id,
        "tier": tier,
        "template_id": template_id,
        "split": split,
        "qtype": qtype,
        "source": predictions[0].source,
        "n": n,
        "template_total": template_total,
        "template_available": template_available,
        "sampled": sampled,
        "coverage": n / template_total if template_total else None,
        "n_options": n_options,
        "chance": chance,
        # additional, labelled: mean per-item chance (1/n_i). The gate uses `chance` above, the
        # zero-shot runner's convention (1 / the cell's largest option count), so the two paths'
        # gates stay comparable; this number is reported beside it, never used as the threshold.
        "chance_item_mean": float(np.mean([1.0 / len(p.option_keys) for p in predictions])),
        "accuracy": float(np.mean([1.0 if c else 0.0 for c in correct])),
        "accuracy_ci95": list(bootstrap_ci([1.0 if c else 0.0 for c in correct], seed=seed)[1:]),
        "n_correct": int(sum(correct)),
        "majority_label": Counter(gold).most_common(1)[0][0],
        "majority_share": Counter(gold).most_common(1)[0][1] / n,
        "macro_accuracy": macro_accuracy(predicted, gold),
        "mean_gold_prob": float(gold_probs.mean()),
        "brier": brier_from_predictions(predictions),
        "ece": ece_from_confidence(confidence, correct),
        "selective_accuracy": selective_accuracy(confidence, correct),
        "median_label_mass": None,  # not a measurement for this readout: see `gate`
        "greedy_agreement": None,  # idem
        "shuffle_flip_rate": shuffle_flip,
        "shuffle_items_scored": shuffle_n,
        "shuffle_error": shuffle_error,
        "p50_seconds_per_item": float(np.median([p.latency_s for p in predictions])),
        "p95_seconds_per_item": float(np.percentile([p.latency_s for p in predictions], 95)),
        "prompt_tokens_p50": float(np.median([p.prompt_tokens for p in predictions])),
        "gate_status": gate["status"],
        "gate_failures": gate["failures"],
        "gate_notes": gate["notes"],
        "gate": gate,
        "temperature": {
            q: temperature.get(q, 1.0) for q in sorted({p.qtype for p in predictions})
        },
        "uncalibrated": uncalibrated,
        "wall_seconds": wall_seconds,
    }
    if qtype == QuestionType.SCORE.value:
        cell["score"] = score_expected_level_error(predictions)
    if strict_ids is not None:
        strict = [p for p in predictions if p.item_id in strict_ids]
        if strict:
            s_correct = sum(1 for p in strict if p.correct)
            cell["strict_slice"] = {
                "n": len(strict),
                "n_correct": s_correct,
                "accuracy": s_correct / len(strict),
            }
        else:
            cell["strict_slice"] = {"n": 0, "status": "NOT MEASURED — no strict-slice items"}
    return cell


# --------------------------------------------------------------------------- temperature fitting
def _not_fitted_reason(
    temperature: float,
    nll_after: float,
    bounds: tuple[float, float] = TEMPERATURE_BOUNDS,
    *,
    tolerance: float = 1e-3,
) -> str | None:
    """Why a fitted value must not be read as a fitted temperature, or ``None`` when it can.

    Two ways a bounded scalar fit fails to identify a temperature, both reported rather than
    hidden:

    * a **bound hit** — the search's own absolute x tolerance is ``1e-4``, so a value within
      ``tolerance`` of either bound means the optimum is not inside the search box;
    * a **numerically zero NLL** — the objective is flat at zero because the model is saturated on
      the sample, so any small temperature fits equally well and the returned value is arbitrary.
    """
    lo, hi = bounds
    if nll_after <= 1e-6:
        return (
            f"the fit reached a numerically zero NLL ({nll_after:.2e}): the model is saturated on "
            f"this sample, the objective is flat, and the temperature is not identified"
        )
    if temperature <= lo + tolerance:
        return f"the search landed on the lower bound {lo}"
    if temperature >= hi - tolerance:
        return f"the search landed on the upper bound {hi}"
    return None


def fit_temperatures(
    *,
    split: str,
    logits_by_qtype: dict[str, list[np.ndarray]],
    gold_by_qtype: dict[str, list[int]],
    seed: int,
    min_items: int = DEFAULT_MIN_TEMPERATURE_ITEMS,
    bounds: tuple[float, float] = TEMPERATURE_BOUNDS,
    expected_qtypes: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Fit one temperature per qtype on **dev** items.

    Refuses any split other than ``dev``: temperatures are fitted on dev, never on test
    (AGENTS.md data rules). A qtype with fewer than ``min_items`` items, or a fit that lands on
    the search bound, is reported with an explicit status; the measured numbers are kept beside
    it as a diagnostic and are never presented as a fitted value. ``expected_qtypes`` makes the
    report carry a qtype the run has **no** items for (``score`` on the screened dev split) as
    an explicit ``NOT FITTED`` entry instead of silently omitting it.
    """
    if split != "dev":
        raise ValueError(
            f"refusing to fit temperatures on split {split!r}: temperature fitting is dev-only"
        )
    per_qtype: dict[str, Any] = {}
    for qtype in sorted(set(logits_by_qtype) | set(expected_qtypes or ())):
        logits = logits_by_qtype.get(qtype, [])
        gold = gold_by_qtype.get(qtype, [])
        if not logits:
            per_qtype[qtype] = {
                "status": "NOT FITTED — no dev items of this qtype in this run",
                "n_items": 0,
                "fitted": False,
            }
            continue
        fits = fit_per_qtype(
            _Scored(logits=logits, gold_indices=gold, qtypes=[qtype] * len(logits)),
        )
        fit = fits[qtype]
        entry: dict[str, Any] = {
            "temperature": fit.temperature,
            "n_items": fit.n_items,
            "nll_before": fit.nll_before,
            "nll_after": fit.nll_after,
            "bounds": list(bounds),
        }
        if fit.n_items < min_items:
            entry["status"] = (
                f"NOT FITTED — too few dev items (n={fit.n_items} < {min_items}); the numbers "
                f"beside this are a diagnostic on that small sample, not a fitted temperature"
            )
            entry["fitted"] = False
        elif (reason := _not_fitted_reason(fit.temperature, fit.nll_after, bounds)) is not None:
            entry["status"] = (
                f"NOT FITTED — {reason}; the value must not be read as a fitted temperature"
            )
            entry["fitted"] = False
        else:
            entry["status"] = "FITTED"
            entry["fitted"] = True
        per_qtype[qtype] = entry
    return {
        "task": "S8",
        "kind": "temperature_fit",
        "split": split,
        "fitted_at_utc": utcnow(),
        "seed": seed,
        "min_items": min_items,
        "bounds": list(bounds),
        "per_qtype": per_qtype,
        "n_qtype_fitted": sum(1 for v in per_qtype.values() if v.get("fitted")),
        "n_qtype_not_fitted": sum(1 for v in per_qtype.values() if not v.get("fitted")),
    }


class _Scored:
    """Minimal ``ScoredItems``-shaped holder for :func:`meddecide.train.temperature.fit_per_qtype`."""

    def __init__(self, *, logits: list[np.ndarray], gold_indices: list[int], qtypes: list[str]):
        self.logits = logits
        self.gold_indices = gold_indices
        self.qtypes = qtypes


def write_temperature_fit(
    path: Path,
    fit: dict[str, Any],
    *,
    checkpoint: Path,
    split: str,
    model_id: str,
    sampled: bool,
    nll_by_qtype_uncalibrated: dict[str, float] | None = None,
) -> Path:
    """Write the fit beside the checkpoint, with its item counts and NLL before/after."""
    if split != "dev":
        raise ValueError(f"refusing to write a temperature fit for split {split!r} (dev-only)")
    payload = dict(fit)
    payload["checkpoint"] = str(checkpoint)
    payload["model_id"] = model_id
    payload["limited_sample"] = bool(sampled)
    if sampled:
        payload["limited_sample_note"] = (
            "--limit-per-template was used, so the fit is on a sample of each template, not the "
            "whole dev split"
        )
    if nll_by_qtype_uncalibrated:
        payload["mean_nll_uncalibrated_no_temperature"] = nll_by_qtype_uncalibrated
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def load_temperature_file(path: Path) -> tuple[dict[str, float], dict[str, Any]]:
    """Read a `temperature.json`, returning the applyable temperatures and a provenance block.

    Only entries whose status says they were fitted are applied; a qtype reported as NOT FITTED
    stays at T=1.0 for that qtype (never silently using a number the fit itself rejected).
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    applied: dict[str, float] = {}
    skipped: dict[str, str] = {}
    for qtype, entry in sorted(payload.get("per_qtype", {}).items()):
        if entry.get("fitted") and entry.get("temperature") is not None:
            applied[qtype] = float(entry["temperature"])
        else:
            skipped[qtype] = str(entry.get("status", "no status"))
    return applied, {
        "path": str(path),
        "fitted_at_utc": payload.get("fitted_at_utc"),
        "split": payload.get("split"),
        "applied": applied,
        "skipped_not_fitted": skipped,
    }


# --------------------------------------------------------------------------- run
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score MedDecide-Bench v0.2 with a trained checkpoint's pointer head"
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT,
                        help="checkpoint directory written by MedDecideModel.save")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID,
                        help="model id stamped on every prediction row")
    parser.add_argument("--tier1", type=Path, default=DEFAULT_TIER1)
    parser.add_argument("--fresh", type=Path, default=DEFAULT_FRESH)
    parser.add_argument("--keep-screen", type=Path, default=DEFAULT_SCREEN,
                        help="screen.json whose kept templates are scored")
    parser.add_argument("--no-screen", action="store_true",
                        help="ignore --keep-screen and score every template in the split")
    parser.add_argument("--only-template", default=None, help="debug: score one template only")
    parser.add_argument("--tag", default=None,
                        help="suffix for the report/prediction file names, e.g. dev or S9-step4000")
    parser.add_argument("--split", choices=("test", "dev"), default="test",
                        help="test = the benchmark's test split; dev = its dev split (fit-only)")
    parser.add_argument("--run-id", default=None,
                        help="stamp on every prediction row; defaults to <model>:<utc>")
    parser.add_argument("--out", type=Path, default=Path("outputs/student_v0/S8"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-batch-tokens", type=int, default=32768)
    parser.add_argument("--max-prompt-tokens", type=int, default=None,
                        help="override the checkpoint's prompt cap (truncation keeps the tail)")
    parser.add_argument("--allow-marker-mismatch", action="store_true",
                        help="score items whose located marker token disagrees with the harness key "
                             "token instead of refusing them (recorded per cell)")
    parser.add_argument("--shuffle-items", type=int, default=200,
                        help="items per template for the option-permutation probe (0 disables)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temperature-file", type=Path, default=None,
                        help="apply a previously fitted temperature.json (default: the checkpoint's)")
    parser.add_argument("--no-temperature", action="store_true",
                        help="score uncalibrated (T=1 everywhere), ignoring stored calibration")
    parser.add_argument("--temperature-out", type=Path, default=None,
                        help="where --split dev writes the fit (default <checkpoint>/temperature.json)")
    parser.add_argument("--min-temperature-items", type=int, default=DEFAULT_MIN_TEMPERATURE_ITEMS)
    parser.add_argument("--limit-per-template", type=int, default=None,
                        help="debug: score at most N items per template (recorded in every cell)")
    parser.add_argument("--write-predictions", action="store_true", default=True)
    parser.add_argument("--no-write-predictions", dest="write_predictions", action="store_false")
    parser.add_argument("--no-chunk-rounding", action="store_true",
                        help="V2 canonical path: no chunk rounding of the batch length (use with "
                             "--batch-size 1, where every item is scored with no padding)")
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> int:
    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(args.checkpoint)
    if not (checkpoint / "model.json").exists():
        print(
            f"BLOCKED — no checkpoint at {checkpoint}: {checkpoint / 'model.json'} does not exist",
            file=sys.stderr,
        )
        return 3
    try:
        model = MedDecideModel.load(checkpoint, device=args.device)
    except Exception as exc:  # the exact error is the point of the BLOCKED report
        print(
            f"BLOCKED — could not load the checkpoint at {checkpoint}: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 3

    keep = None if args.no_screen else load_keep_set(args.keep_screen)
    benchmark = load_benchmark(args.tier1, args.fresh, keep, args.split)
    strict_ids = strict_slice_ids(benchmark["fresh"], Path(args.fresh) / "acceptance.json")

    # which temperatures are applied: explicit file > checkpoint calibration > none
    temperature_provenance: dict[str, Any]
    if args.no_temperature:
        temperatures: dict[str, float] = {}
        temperature_provenance = {"source": "disabled by --no-temperature", "applied": {}}
    elif args.temperature_file is not None:
        temperatures, temperature_provenance = load_temperature_file(args.temperature_file)
        temperature_provenance["source"] = "temperature-file"
    else:
        temperatures = dict(model.calibration)
        temperature_provenance = {
            "source": "checkpoint calibration (model.json)",
            "applied": dict(temperatures),
        }
        if not temperatures:
            temperature_provenance["source"] = "none — the checkpoint carries no calibration"

    run_id = args.run_id or f"{slug(args.model_id)}:{utcnow()}"
    suffix_parts = [x for x in (args.tag, args.only_template) if x]
    suffix = f"__{'__'.join(suffix_parts)}" if suffix_parts else ""
    # unnamed runs keep the zero-shot runner's exact file names; --tag/--only-template add a
    # suffix so a test run and a dev run (or per-template debug runs) never overwrite each other
    preds_path = out_dir / f"preds_{slug(args.model_id)}{suffix}.jsonl"
    score_kwargs: dict[str, Any] = {
        "batch_size": args.batch_size,
        "max_batch_tokens": args.max_batch_tokens,
        "max_prompt_tokens": args.max_prompt_tokens,
        "allow_marker_mismatch": args.allow_marker_mismatch,
        "round_to_chunk": not args.no_chunk_rounding,
    }
    report: dict[str, Any] = {
        "task": "S8",
        "kind": "student_eval",
        "model_id": args.model_id,
        "run_id": run_id,
        "checkpoint": str(checkpoint),
        "checkpoint_meta": model.describe(),
        "split": args.split,
        "generated_at_utc": utcnow(),
        "tier1": str(args.tier1),
        "fresh": str(args.fresh),
        "keep_screen": None if args.no_screen else str(args.keep_screen),
        "n_templates_kept": None if keep is None else len(keep),
        "only_template": args.only_template,
        "batch_size": args.batch_size,
        "max_batch_tokens": args.max_batch_tokens,
        "max_prompt_tokens": (
            args.max_prompt_tokens if args.max_prompt_tokens is not None
            else model.max_prompt_tokens
        ),
        "allow_marker_mismatch": args.allow_marker_mismatch,
        "round_to_chunk": not args.no_chunk_rounding,
        "shuffle_items": args.shuffle_items,
        "limit_per_template": args.limit_per_template,
        "temperature": temperature_provenance,
        "gpu": gpu_name(),
        "tiers": {},
        "cells": [],
    }
    started = time.time()
    # built BEFORE the scoring loop: `started_at` is when the run started, not when the report was
    # written (the first version constructed it at the end, so the file read wall_clock_s = 0.0)
    prov = Provenance(
        run_name=f"S8_student_{slug(args.model_id)}",
        command=" ".join([sys.executable, *sys.argv]),
        gpu=report["gpu"],
        seed=args.seed,
        config={
            "checkpoint": str(checkpoint),
            "split": args.split,
            "model_id": args.model_id,
            "batch_size": args.batch_size,
            "max_batch_tokens": args.max_batch_tokens,
            "shuffle_items": args.shuffle_items,
        },
        models=[{"model_id": args.model_id, "checkpoint": str(checkpoint)}],
        datasets=[{"name": "MedDecide-Bench v0.2", "tier1": str(args.tier1),
                   "fresh": str(args.fresh), "split": args.split}],
    )
    logits_by_qtype: dict[str, list[np.ndarray]] = defaultdict(list)
    gold_by_qtype: dict[str, list[int]] = defaultdict(list)
    any_limited = False

    for tier, items in benchmark.items():
        if not items:
            continue
        totals = per_template_totals(items)
        by_template: dict[str, list[Item]] = defaultdict(list)
        for item in items:
            by_template[item.template_id].append(item)
        tier_cells: list[dict[str, Any]] = []
        for template_id, template_items in sorted(by_template.items()):
            if args.only_template and template_id != args.only_template:
                continue
            available = len(template_items)
            sampled = bool(args.limit_per_template) and available > args.limit_per_template
            if sampled:
                any_limited = True
                template_items = template_items[: args.limit_per_template]
            t0 = time.time()
            try:
                scored = model.score_items(template_items, **score_kwargs)
            except Exception as exc:
                reason = f"{type(exc).__name__}: {exc}"
                cell = empty_cell(
                    model_id=args.model_id, tier=tier, template_id=template_id, split=args.split,
                    template_total=totals.get(template_id, available), reason=reason,
                )
                cell["wall_seconds"] = time.time() - t0
                tier_cells.append(cell)
                print(f"{args.model_id} {tier}/{template_id}: {cell['gate_status']}", flush=True)
                continue
            decisions = decisions_from_scored(model, scored, temperature=temperatures)
            predictions = [
                decision_to_prediction(d, model_id=args.model_id, run_id=run_id, split=args.split)
                for d in decisions
            ]
            for logits, gold, qtype in zip(
                scored.logits, scored.gold_indices, scored.qtypes, strict=True
            ):
                logits_by_qtype[str(qtype)].append(logits)
                gold_by_qtype[str(qtype)].append(int(gold))
            if len(predictions) < len(template_items):
                print(
                    f"  WARNING {template_id}: scored {len(predictions)}/{len(template_items)}",
                    flush=True,
                )
            flip, flip_n, flip_error = option_permutation_flip(
                model, template_items, predictions,
                n_items=args.shuffle_items, seed=args.seed, score_kwargs=score_kwargs,
            )
            cell = summarise_cell(
                model_id=args.model_id,
                tier=tier,
                template_id=template_id,
                split=args.split,
                predictions=predictions,
                template_total=totals.get(template_id, available),
                template_available=available,
                sampled=sampled,
                strict_ids=strict_ids if tier == "fresh" else None,
                seed=args.seed,
                shuffle_flip=flip,
                shuffle_n=flip_n,
                shuffle_error=flip_error,
                temperature=temperatures,
                uncalibrated=uncalibrated_metrics(scored, [p.correct for p in predictions]),
                wall_seconds=time.time() - t0,
            )
            tier_cells.append(cell)
            flip_txt = "NOT MEASURED" if cell.get("shuffle_flip_rate") is None else (
                f"{cell['shuffle_flip_rate']:.3f}"
            )
            print(
                f"{args.model_id} {tier}/{template_id}: n={cell['n']}/{cell['template_total']} "
                f"acc={cell['accuracy']:.4f} macro={cell['macro_accuracy']:.4f} "
                f"brier={cell['brier']:.4f} ece={cell['ece']:.4f} flip={flip_txt} "
                f"[{cell['gate_status']}]",
                flush=True,
            )
            if args.write_predictions:
                preds_path.parent.mkdir(parents=True, exist_ok=True)
                with preds_path.open("a", encoding="utf-8") as fh:
                    for pred in predictions:
                        fh.write(json.dumps(pred.to_json(), ensure_ascii=False) + "\n")
                report["preds_path"] = str(preds_path)
        report["tiers"][tier] = {
            "n_items": len(items),
            "n_scored": sum(c["n"] for c in tier_cells),
            "n_cells": len(tier_cells),
            "coverage": sum(c["n"] for c in tier_cells) / len(items) if items else None,
            "n_gate_pass": sum(1 for c in tier_cells if c.get("gate_status") == "PASS"),
            "n_gate_fail": sum(
                1 for c in tier_cells if str(c.get("gate_status", "")).startswith("READOUT_FAIL")
            ),
            "n_not_measured": sum(
                1 for c in tier_cells if str(c.get("gate_status", "")).startswith("NOT MEASURED")
            ),
            "n_additional_check_fail": sum(
                1
                for c in tier_cells
                if (c.get("gate") or {}).get("checks", {})
                .get("option_permutation_consistency", {})
                .get("status") == "FAIL"
            ),
        }
        report["cells"].extend(tier_cells)

    report["wall_seconds"] = time.time() - started
    report["n_prediction_rows"] = sum(c["n"] for c in report["cells"])
    if report.get("preds_path"):
        report["prediction_log"] = read_prediction_log(Path(report["preds_path"])).report()

    if args.split == "dev":
        fit = fit_temperatures(
            split=args.split,
            logits_by_qtype=dict(logits_by_qtype),
            gold_by_qtype=dict(gold_by_qtype),
            seed=args.seed,
            min_items=args.min_temperature_items,
            # every qtype the schema defines is reported, so a qtype this split has no items for
            # (the screened dev split has no `score` item) reads NOT FITTED instead of vanishing
            expected_qtypes=[q.value for q in QuestionType],
        )
        temperature_out = (
            Path(args.temperature_out) if args.temperature_out is not None
            else checkpoint / "temperature.json"
        )
        # test-split safety is enforced inside both calls (a test run can never write a fit)
        written = write_temperature_fit(
            temperature_out, fit, checkpoint=checkpoint, split=args.split,
            model_id=args.model_id, sampled=any_limited,
        )
        report["temperature_fit"] = {"path": str(written), **fit}
        print(
            f"[temperature] wrote {written}: "
            + ", ".join(
                f"{q}={v.get('temperature')} ({v['status']})"
                if v.get("temperature") is not None
                else f"{q} ({v['status']})"
                for q, v in sorted(fit["per_qtype"].items())
            ),
            flush=True,
        )

    out_path = out_dir / f"model_{slug(args.model_id)}{suffix}.json"
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    prov.finish().write(out_dir / f"model_{slug(args.model_id)}{suffix}_provenance.json")
    print(json.dumps(report["tiers"], indent=2))
    print(f"wall: {report['wall_seconds']:.1f}s; wrote {out_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
