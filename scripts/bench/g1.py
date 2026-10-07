#!/usr/bin/env python
"""S11 — Gate G1 evaluation machinery (decision D16, ADVISORY section S11).

This script is the **evaluation machinery** for gate G1. It never trains and never scores a
model: it reads prediction JSONL files already written in the ``meddecide.eval.harness``
format and turns them into the G1 report. The real S11 run is therefore one command once the
prediction files exist; until then every comparison prints ``NOT MEASURED — <reason>`` (R3).

What D16 fixes (implemented here verbatim, no thresholds invented):

* headline fresh set = fresh test items on templates the v0.2 screen kept, excluding the
  superseded ``pubmed_mesh_major_choice_v1`` / ``fda_class_choice_v1``;
* for each baseline B ∈ {zero-shot Qwen3.5-0.8B, JEV-9B}: the **item-paired** difference
  MedDecide - B in **macro accuracy** and in **mean Brier**, with a paired bootstrap 95 % CI
  (items resampled **within templates**, 1,000 resamples by default);
* PASS iff, against both baselines, the accuracy-difference CI lower bound is > 0 **and** the
  Brier-difference CI upper bound is < 0, on **seen** templates; held-out templates get the
  same statistics and the same rule, reported separately.

Extra requirements from the ADVISORY/task brief, all mechanical here:

* accuracy (micro) and Brier as well as macro accuracy; paired CIs share one resample draw per
  pair, plus each model's own marginal CI and an explicitly unpaired difference CI;
* every comparison uses the **intersection** of the items both models scored, and the report
  states that item count;
* the strict slice (``record_date >= strict_slice_start``), the held-out-template split (D14),
  and optional tier 1 are computed from dates and template ids, never assumed;
* each ``(model, template)`` cell carries its D12 readout-health status from
  :func:`meddecide.eval.health.evaluate_cell`; a failing cell is printed as
  ``READOUT_FAIL — <check>`` and **excluded** from the verdict.

Where the code comes from (the report repeats this):

* reused from :mod:`meddecide.eval.metrics`: ``accuracy`` (micro + majority baseline),
  ``macro_accuracy``, ``brier_score``, ``bootstrap_ci``;
* reused from :mod:`meddecide.eval.health`: ``evaluate_cell`` (unchanged thresholds);
* reused from :mod:`meddecide.eval.predlog`: ``read_prediction_log`` (dedupe by
  ``(run_id, item_id)``);
* **new here**: the template-stratified resample plan, the paired/unpaired difference
  machinery, the per-item Brier for items with different option counts, the D12 cell wrapper
  for prediction logs (letter-readout-only checks marked ``NOT APPLICABLE``, the
  constant-answer check applied), and the item-set construction.

Predictions and item text stay in the gitignored ``outputs/``; only the aggregate report is
written to the committed path (``loops/student_v0/g1.md`` by default).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import numpy as np

from meddecide.eval.health import (
    DEFAULT_DEGENERATE_MAJORITY_CEILING,
    DEFAULT_MAX_MODAL_SHARE,
    DEFAULT_MIN_GREEDY_AGREEMENT,
    DEFAULT_MIN_MEDIAN_LABEL_MASS,
    evaluate_cell,
)
from meddecide.eval.metrics import accuracy as accuracy_report
from meddecide.eval.metrics import bootstrap_ci, brier_score, macro_accuracy
from meddecide.eval.predlog import read_prediction_log

# --------------------------------------------------------------------------- D14 / D16 constants
# Held-out templates: exactly the D14 list this loop fixed (same tuple as
# scripts/bench/build_training_mix.py HELD_OUT_TEMPLATES). Never extended here.
HELD_OUT_TEMPLATES: tuple[str, ...] = (
    "ct_phase_choice_v1",
    "fda_boxed_warning_noul_v1",
    "pubmed_humans_noul_v1",
    "ct_arm_role_noul_v1",
)

# D16: superseded templates are not part of the headline fresh set.
SUPERSEDED_TEMPLATES: tuple[str, ...] = (
    "pubmed_mesh_major_choice_v1",
    "fda_class_choice_v1",
)

DEFAULT_STRICT_SLICE_START = date(2026, 9, 10)  # D11
ALPHA = 0.05
POINTER_VARIANT = "pointer-head"
NO_LETTER_READOUT_REASON = (
    "NOT APPLICABLE — pointer head reads the option states directly; no letter readout exists "
    "to validate"
)
GREEDY_NOT_MEASURED_REASON = (
    "NOT MEASURED — a prediction log carries no greedy generation sample; the greedy check "
    "needs the model and is not run by this analysis"
)
UNPAIRED_SEED_OFFSET = 1_000_003
DEFAULT_SEED = 0
DEFAULT_RESAMPLES = 1000

ITEM_SETS: tuple[tuple[str, str], ...] = (
    ("fresh", "headline fresh test set (screen-kept, superseded excluded)"),
    ("fresh_seen", "headline fresh set, seen templates (not in the D14 held-out list)"),
    ("fresh_heldout", "headline fresh set, held-out templates (D14), reported separately"),
    ("fresh_strict", "headline fresh set, strict slice (record_date >= strict_slice_start)"),
    ("fresh_strict_seen", "strict slice, seen templates"),
    ("fresh_strict_heldout", "strict slice, held-out templates"),
    ("tier1", "tier-1 established test set (only when --tier1 is given)"),
)


# --------------------------------------------------------------------------- inputs
@dataclass(frozen=True)
class BenchItem:
    """The fields of a benchmark item this analysis needs (never the item text)."""

    item_id: str
    tier: str
    template_id: str
    source: str
    qtype: str
    record_date: date
    split: str
    gold_key: str
    n_options: int
    strict_flag: bool | None


@dataclass
class ModelSource:
    """One ``--models`` label: its file, the run chosen, and the rows kept."""

    label: str
    path: str
    exists: bool
    run_id: str = ""
    run_ids: list[str] = field(default_factory=list)
    rows: dict[str, dict[str, Any]] = field(default_factory=dict)
    n_raw: int = 0
    n_unique: int = 0
    n_without_run_id: int = 0
    unreadable_lines: int = 0
    model_ids: list[str] = field(default_factory=list)
    note: str = ""

    @property
    def n_items(self) -> int:
        return len(self.rows)

    @property
    def missing_reason(self) -> str | None:
        """Why this source cannot be used, or ``None`` when it can."""
        if not self.exists:
            return f"prediction file does not exist: {self.path}"
        if not self.rows:
            return f"prediction file has no readable rows: {self.path}"
        return None


def parse_models(specs: Sequence[str]) -> list[tuple[str, Path]]:
    """``LABEL=PATH`` entries, repeatable and comma-separated; duplicate labels are an error."""
    out: list[tuple[str, Path]] = []
    seen: set[str] = set()
    for spec in specs:
        for part in str(spec).split(","):
            part = part.strip()
            if not part:
                continue
            label, sep, path = part.partition("=")
            label, path = label.strip(), path.strip()
            if not sep or not label or not path:
                raise SystemExit(f"--models expects LABEL=PATH, got {part!r}")
            if label in seen:
                raise SystemExit(f"--models label {label!r} given twice")
            seen.add(label)
            out.append((label, Path(path)))
    return out


def load_bench_items(
    *, fresh_dir: Path | None, tier1_dir: Path | None
) -> tuple[dict[str, BenchItem], dict[str, int]]:
    """Item index keyed by ``item_id`` — only the fields G1 needs, streamed line by line."""
    items: dict[str, BenchItem] = {}
    per_file: dict[str, int] = {}
    for fallback_tier, directory in (("fresh", fresh_dir), ("established", tier1_dir)):
        if directory is None or not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.jsonl")):
            n = 0
            with path.open(encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    item_id = str(row.get("item_id"))
                    options = row.get("options") or []
                    meta = row.get("meta") or {}
                    flags = meta.get("strict_post_teacher")
                    items[item_id] = BenchItem(
                        item_id=item_id,
                        tier=str(row.get("tier") or fallback_tier),
                        template_id=str(row.get("template_id")),
                        source=str(row.get("source")),
                        qtype=str(row.get("qtype")),
                        record_date=date.fromisoformat(str(row.get("record_date"))[:10]),
                        split=str(row.get("split")),
                        gold_key=str(row.get("gold")),
                        n_options=len(options),
                        strict_flag=bool(flags) if flags is not None else None,
                    )
                    n += 1
            per_file[str(path)] = n
    return items, per_file


def load_screen(path: Path) -> dict[str, dict[str, Any]]:
    """Screen entries keyed by template id (``drop`` flag, ``majority_share``, tier, qtype)."""
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {str(t["template_id"]): t for t in data.get("templates", [])}


def resolve_strict_start(paths: Sequence[Path]) -> tuple[date, str]:
    """The strict-slice start from the first file carrying ``strict_slice_start``.

    The task brief pointed at ``data/bench/v0.2/fresh/acceptance.json``; that file does not
    carry the key (checked 2026-10-07) — ``data/bench/v0.2/fresh/manifest.json`` does. The
    lookup therefore walks the acceptance file, the fresh manifest and the top-level manifest
    in order and records which one answered, falling back to the D11 constant.
    """
    for path in paths:
        if not path.is_file():
            continue
        try:
            value = json.loads(path.read_text(encoding="utf-8")).get("strict_slice_start")
        except json.JSONDecodeError:
            continue
        if value:
            return date.fromisoformat(str(value)[:10]), str(path)
    return DEFAULT_STRICT_SLICE_START, "hardcoded D11 default 2026-09-10 (no manifest key found)"


def load_model_source(label: str, path: Path) -> ModelSource:
    """Read one prediction file and pick the run to analyse.

    ``read_prediction_log`` dedupes by ``(run_id, item_id)``; a file that accumulated several
    runs then holds one row per item **per run**. G1 works on one run per model, so the run
    with the most distinct items wins (ties: the lexicographically last run id, i.e. the later
    run), and every other run is named in the report — never silently dropped.
    """
    log = read_prediction_log(path)
    source = ModelSource(
        label=label,
        path=str(path),
        exists=path.is_file(),
        n_raw=log.n_raw,
        n_unique=log.n_unique,
        n_without_run_id=log.n_without_run_id,
        unreadable_lines=log.unreadable_lines,
    )
    if not source.exists:
        return source
    source.run_ids = log.run_ids
    if not log.rows:
        return source
    per_run = log.per_run()
    chosen = max(per_run, key=lambda run: (per_run[run]["n_distinct_items"], run))
    source.run_id = chosen
    source.rows = {str(row.get("item_id")): row for row in log.rows_for_run(chosen)}
    source.model_ids = sorted({str(row.get("model_id") or "") for row in source.rows.values()})
    if len(source.run_ids) > 1:
        others = ", ".join(
            f"{run or '(no run_id)'}={per_run[run]['n_distinct_items']}" for run in source.run_ids
        )
        source.note = (
            f"{len(source.run_ids)} runs in the file ({others}); analysed run "
            f"{chosen or '(no run_id)'} with {len(source.rows)} items"
        )
    return source


# --------------------------------------------------------------------------- D12 readout health
def readout_kind(rows: Sequence[dict[str, Any]]) -> str:
    """``letter``, ``pointer-head`` or ``mixed`` — from the row's own declaration."""
    pointer = sum(
        1
        for row in rows
        if str(row.get("variant") or "") == POINTER_VARIANT
        or str((row.get("transform") or {}).get("readout") or "") == POINTER_VARIANT
    )
    if pointer == 0:
        return "letter"
    if pointer == len(rows):
        return "pointer-head"
    return "mixed"


def check_name(failure: str) -> str:
    """Map ``evaluate_cell``'s failure sentence back to the D12 check name."""
    if failure.startswith("median label mass"):
        return "median_label_mass"
    if failure.startswith("greedy agreement"):
        return "greedy_agreement"
    if failure.startswith("accuracy CI"):
        return "accuracy_ci_vs_chance"
    if failure.startswith("degenerate"):
        return "constant_answer"
    return failure


@dataclass
class CellGate:
    """One ``(model, tier, template)`` cell: the D12 status the verdict has to respect."""

    model: str
    tier: str
    template_id: str
    readout: str
    n: int
    n_options: int
    status: str
    failing_checks: list[str]
    failures: list[str]
    applied_checks: list[str]
    inapplicable_checks: dict[str, str]
    not_measured_checks: dict[str, str]
    accuracy: float
    accuracy_ci95: tuple[float, float]
    median_label_mass: float | None
    modal_option: str | None
    modal_share: float | None
    majority_share: float | None
    notes: list[str]

    @property
    def passed(self) -> bool:
        return not self.failing_checks

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "tier": self.tier,
            "template_id": self.template_id,
            "readout": self.readout,
            "n": self.n,
            "n_options": self.n_options,
            "status": self.status,
            "failing_checks": self.failing_checks,
            "failures": self.failures,
            "applied_checks": self.applied_checks,
            "inapplicable_checks": self.inapplicable_checks,
            "not_measured_checks": self.not_measured_checks,
            "accuracy": self.accuracy,
            "accuracy_ci95": list(self.accuracy_ci95),
            "median_label_mass": self.median_label_mass,
            "modal_option": self.modal_option,
            "modal_share": self.modal_share,
            "majority_share": self.majority_share,
            "notes": self.notes,
        }


def gate_cell(
    *,
    model: str,
    tier: str,
    template_id: str,
    rows: Sequence[dict[str, Any]],
    majority_share: float | None,
    seed: int,
    n_resamples: int,
) -> CellGate:
    """Run D12 on one prediction-derived cell.

    :func:`meddecide.eval.health.evaluate_cell` supplies the checks with unchanged thresholds.
    Two of its four checks are letter-readout-only and are **recorded as inapplicable**, never
    as passed, when the cell has no letter readout:

    * ``median_label_mass`` — the pointer head's softmax is over exactly the offered options,
      so "mass on the option tokens" is 1.0 by construction, not a measurement;
    * ``greedy_agreement`` — there is no generation in the decision path to agree with.

    The two remaining checks are applied to every cell: the below-chance accuracy CI and the
    constant-answer check (``majority_share`` comes from the template's gold distribution in
    ``screen.json``). A letter-readout cell keeps ``median_label_mass``; its greedy check is
    recorded as ``NOT MEASURED`` because a prediction log has no generation sample.
    """
    if not rows:
        raise ValueError("gate_cell needs at least one row")
    kind = readout_kind(rows)
    correct = [bool(row.get("correct")) for row in rows]
    predicted = [str(row.get("argmax_key")) for row in rows]
    n_options = max(len(row.get("option_keys") or []) for row in rows)
    letter = kind == "letter"
    masses = (
        [float(row.get("label_mass") or 0.0) for row in rows] if letter else [1.0] * len(rows)
    )
    # The constant-answer check compares the cell's modal answer share with the template's gold
    # majority share. screen.json carries that per template; when a template has no screen entry
    # the cell's own golds supply it and the source is recorded, never silently assumed.
    majority_source = "screen.json template majority_share"
    if majority_share is None:
        golds = [str(row.get("gold_key")) for row in rows]
        majority_share = Counter(golds).most_common(1)[0][1] / len(golds)
        majority_source = "cell gold majority_share (no screen.json entry for this template)"
    report = evaluate_cell(
        model_id=model,
        template_id=template_id,
        label_masses=masses,
        correct=correct,
        greedy_matches=None,
        n_options=n_options,
        predicted_labels=predicted,
        majority_share=majority_share,
        min_median_label_mass=DEFAULT_MIN_MEDIAN_LABEL_MASS,
        min_greedy_agreement=DEFAULT_MIN_GREEDY_AGREEMENT,
        max_modal_share=DEFAULT_MAX_MODAL_SHARE,
        degenerate_majority_ceiling=DEFAULT_DEGENERATE_MAJORITY_CEILING,
        seed=seed,
        n_resamples=n_resamples,
    )
    failures = list(report.failures)
    if not letter:
        # The placeholder mass above is not a measurement: drop any mass failure so it can
        # never enter the verdict, and say so.
        failures = [f for f in failures if not f.startswith("median label mass")]

    applied = ["accuracy_ci_vs_chance", "constant_answer"]
    inapplicable: dict[str, str] = {}
    not_measured: dict[str, str] = {}
    if letter:
        applied.append("median_label_mass")
        not_measured["greedy_agreement"] = GREEDY_NOT_MEASURED_REASON
    elif kind == "pointer-head":
        inapplicable["median_label_mass"] = NO_LETTER_READOUT_REASON
        inapplicable["greedy_agreement"] = NO_LETTER_READOUT_REASON
    else:
        reason = (
            "NOT APPLICABLE — the cell mixes readout kinds "
            f"({kind}); a letter-readout check is not a measurement for all its items"
        )
        inapplicable["median_label_mass"] = reason
        inapplicable["greedy_agreement"] = reason

    failing = [check_name(f) for f in failures]
    notes = list(report.notes)
    notes.append(f"constant-answer check majority share: {majority_source}")
    notes.extend(f"{check}: {reason}" for check, reason in inapplicable.items())
    notes.extend(f"{check}: {reason}" for check, reason in not_measured.items())
    return CellGate(
        model=model,
        tier=tier,
        template_id=template_id,
        readout=kind,
        n=len(rows),
        n_options=n_options,
        status="PASS" if not failing else "READOUT_FAIL — " + ", ".join(failing),
        failing_checks=failing,
        failures=failures,
        applied_checks=applied,
        inapplicable_checks=inapplicable,
        not_measured_checks=not_measured,
        accuracy=report.accuracy if report.accuracy is not None else float("nan"),
        accuracy_ci95=report.accuracy_ci95 or (float("nan"), float("nan")),
        median_label_mass=report.median_label_mass if letter else None,
        modal_option=report.modal_option,
        modal_share=report.modal_share,
        majority_share=majority_share,
        notes=notes,
    )


def compute_cells(
    *,
    source: ModelSource,
    items: dict[str, BenchItem],
    screen: dict[str, dict[str, Any]],
    seed: int,
    n_resamples: int,
) -> tuple[dict[tuple[str, str], CellGate], dict[str, int]]:
    """Every ``(tier, template)`` cell of one model, over the benchmark items it scored.

    Rows whose item is not in the v0.2 test universe (dev items, HLE, other loops' files) are
    counted and named, never silently dropped.
    """
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    skipped: Counter[str] = Counter()
    for item_id, row in source.rows.items():
        item = items.get(item_id)
        if item is None:
            skipped["item not in the v0.2 fresh/tier-1 index"] += 1
            continue
        if item.split != "test":
            skipped[f"item is in split {item.split!r}, not test"] += 1
            continue
        groups[(item.tier, item.template_id)].append(row)
    cells: dict[tuple[str, str], CellGate] = {}
    for (tier, template_id), rows in sorted(groups.items()):
        entry = screen.get(template_id) or {}
        majority = entry.get("majority_share")
        cells[(tier, template_id)] = gate_cell(
            model=source.label,
            tier=tier,
            template_id=template_id,
            rows=rows,
            majority_share=float(majority) if majority is not None else None,
            seed=seed,
            n_resamples=n_resamples,
        )
    return cells, dict(skipped)


# --------------------------------------------------------------------------- item sets
@dataclass(frozen=True)
class ItemSets:
    """The G1 item sets, each a list of item ids in a stable order, plus how they were built."""

    sets: dict[str, list[str]]
    notes: list[str]
    n_universe: int
    n_excluded_dev: int
    n_excluded_not_kept_by_screen: int
    n_excluded_superseded: int
    n_excluded_split_disagreement: int


def build_item_sets(
    items: dict[str, BenchItem],
    *,
    screen: dict[str, dict[str, Any]],
    strict_start: date,
    include_strict: bool,
    include_tier1: bool,
) -> ItemSets:
    """Compute the fresh/seen/held-out/strict/tier-1 sets from dates and template ids."""
    fresh: list[str] = []
    tier1: list[str] = []
    n_dev = n_not_kept = n_superseded = 0
    for item_id, item in sorted(items.items()):
        if item.split != "test":
            n_dev += 1
            continue
        if item.tier == "established":
            if include_tier1:
                tier1.append(item_id)
            continue
        if item.template_id in SUPERSEDED_TEMPLATES:
            n_superseded += 1
            continue
        entry = screen.get(item.template_id)
        if entry is None or entry.get("drop"):
            n_not_kept += 1
            continue
        fresh.append(item_id)

    by_template = {item_id: items[item_id].template_id for item_id in fresh}
    heldout = {item_id for item_id in fresh if by_template[item_id] in HELD_OUT_TEMPLATES}
    seen = [item_id for item_id in fresh if item_id not in heldout]
    strict = [item_id for item_id in fresh if items[item_id].record_date >= strict_start]
    strict_heldout = [item_id for item_id in strict if item_id in heldout]
    strict_seen = [item_id for item_id in strict if item_id not in heldout]
    strict_flag_disagreements = sum(
        1
        for item_id in strict
        if items[item_id].strict_flag is not None and items[item_id].strict_flag != bool(
            items[item_id].record_date >= strict_start
        )
    )

    sets = {
        "fresh": fresh,
        "fresh_seen": seen,
        "fresh_heldout": [i for i in fresh if i in heldout],
        "fresh_strict": strict if include_strict else [],
        "fresh_strict_seen": strict_seen if include_strict else [],
        "fresh_strict_heldout": strict_heldout if include_strict else [],
        "tier1": tier1,
    }
    notes = [
        f"universe: {len(items)} items in the fresh/tier-1 index; headline fresh set "
        f"{len(fresh)} test items on screen-kept templates",
        f"excluded from the headline set: {n_dev} non-test items, {n_superseded} on superseded "
        f"templates {SUPERSEDED_TEMPLATES}, {n_not_kept} on templates the screen dropped or "
        "does not list",
        f"held-out list (D14): {HELD_OUT_TEMPLATES}",
        f"strict slice start: {strict_start.isoformat()}",
    ]
    if not include_strict:
        notes.append("--no-strict-slice: the strict-slice sets are empty by request")
    if strict and strict_flag_disagreements:
        notes.append(
            f"WARNING: {strict_flag_disagreements} strict-slice items disagree with their "
            "meta.strict_post_teacher flag"
        )
    if tier1:
        dropped_templates = sorted(
            {
                items[item_id].template_id
                for item_id in tier1
                if (screen.get(items[item_id].template_id) or {}).get("drop")
            }
        )
        notes.append(
            f"tier-1 set: {len(tier1)} test items over "
            f"{len({items[i].template_id for i in tier1})} templates, kept whole (no screen "
            "filter); templates the screen dropped and that are therefore inside it: "
            f"{dropped_templates or 'none'}"
        )
    return ItemSets(
        sets=sets,
        notes=notes,
        n_universe=len(items),
        n_excluded_dev=n_dev,
        n_excluded_not_kept_by_screen=n_not_kept,
        n_excluded_superseded=n_superseded,
        n_excluded_split_disagreement=0,
    )


# --------------------------------------------------------------------------- vectors
@dataclass
class Vectors:
    """Per-item arrays for one model over the item ids it scored and D12 lets us report."""

    item_ids: list[str]
    template_ids: list[str]
    correct: np.ndarray
    item_brier: np.ndarray
    probs: list[np.ndarray]
    gold_index: list[int]
    predicted: list[str]
    gold: list[str]
    n_options: list[int]
    index: dict[str, int]
    dropped: dict[str, int]


def item_brier(probs: Sequence[float], gold_index: int) -> float:
    """Multi-class Brier for one item: ``sum_k (p_k - y_k)**2`` (same formula as metrics)."""
    p = np.asarray(probs, dtype=np.float64)
    y = np.zeros_like(p)
    y[gold_index] = 1.0
    return float(np.sum((p - y) ** 2))


def build_vectors(
    *,
    source: ModelSource,
    items: dict[str, BenchItem],
    cells: dict[tuple[str, str], CellGate],
    item_ids: Sequence[str],
) -> Vectors:
    """Arrays over ``item_ids`` for rows of ``source`` whose D12 cell passed."""
    ids: list[str] = []
    templates: list[str] = []
    correct: list[float] = []
    briers: list[float] = []
    probs: list[np.ndarray] = []
    gold_indices: list[int] = []
    predicted: list[str] = []
    gold: list[str] = []
    n_options: list[int] = []
    dropped: Counter[str] = Counter()
    for item_id in item_ids:
        row = source.rows.get(item_id)
        item = items.get(item_id)
        if row is None or item is None:
            continue
        cell = cells.get((item.tier, item.template_id))
        if cell is None:
            dropped["no D12 cell for this (tier, template)"] += 1
            continue
        if not cell.passed:
            dropped[f"D12 failing cell — {cell.status}"] += 1
            continue
        option_keys = [str(k) for k in (row.get("option_keys") or [])]
        row_template = str(row.get("template_id"))
        if row_template != item.template_id:
            dropped["row template_id disagrees with the benchmark item"] += 1
            continue
        gold_key = str(row.get("gold_key"))
        if gold_key not in option_keys:
            dropped["gold_key is not among the scored option_keys"] += 1
            continue
        values = np.asarray(row.get("option_probs") or [], dtype=np.float64)
        if values.size != len(option_keys):
            dropped["option_probs length disagrees with option_keys"] += 1
            continue
        if not np.isclose(values.sum(), 1.0, atol=1e-3):
            dropped["option_probs do not sum to 1"] += 1
            continue
        gold_index = option_keys.index(gold_key)
        ids.append(item_id)
        templates.append(item.template_id)
        correct.append(1.0 if row.get("correct") else 0.0)
        briers.append(item_brier(values, gold_index))
        probs.append(values)
        gold_indices.append(gold_index)
        predicted.append(str(row.get("argmax_key")))
        gold.append(gold_key)
        n_options.append(len(option_keys))
    return Vectors(
        item_ids=ids,
        template_ids=templates,
        correct=np.asarray(correct, dtype=np.float64),
        item_brier=np.asarray(briers, dtype=np.float64),
        probs=probs,
        gold_index=gold_indices,
        predicted=predicted,
        gold=gold,
        n_options=n_options,
        index={item_id: i for i, item_id in enumerate(ids)},
        dropped=dict(dropped),
    )


# --------------------------------------------------------------------------- statistics
def stratified_resample_indices(
    template_ids: Sequence[str], *, n_resamples: int, seed: int
) -> dict[str, np.ndarray]:
    """One resample draw per template: positions resampled **within** templates (D16).

    Returns ``template -> (n_resamples, n_items_in_template)`` integer positions into the
    comparison's item order. The same draw is reused for both models of a pair, which is what
    makes the difference paired.
    """
    groups: dict[str, list[int]] = defaultdict(list)
    for position, template in enumerate(template_ids):
        groups[template].append(position)
    rng = np.random.default_rng(seed)
    out: dict[str, np.ndarray] = {}
    for template in sorted(groups):
        positions = np.asarray(groups[template], dtype=np.int64)
        draw = rng.integers(0, positions.size, size=(n_resamples, positions.size))
        out[template] = positions[draw]
    return out


def macro_accuracy_resamples(
    correct: np.ndarray, draw: dict[str, np.ndarray], templates: Sequence[str]
) -> np.ndarray:
    """Per-resample macro accuracy: mean over templates of the resampled per-template mean."""
    total: np.ndarray | None = None
    for template in templates:
        values = correct[draw[template]]
        mean = values.mean(axis=1)
        total = mean if total is None else total + mean
    assert total is not None
    return total / len(templates)


def mean_brier_resamples(
    item_briers: np.ndarray, draw: dict[str, np.ndarray], templates: Sequence[str]
) -> np.ndarray:
    """Per-resample mean Brier: pooled over every resampled item (template sizes preserved)."""
    total: np.ndarray | None = None
    count = 0
    for template in templates:
        sums = item_briers[draw[template]].sum(axis=1)
        total = sums if total is None else total + sums
        count += draw[template].shape[1]
    assert total is not None
    return total / count


def template_brier(
    *,
    probs: Sequence[np.ndarray],
    gold_index: Sequence[int],
    item_briers: np.ndarray,
    subset_positions: Sequence[int],
    mask: np.ndarray,
) -> tuple[float, str]:
    """Per-template Brier: ``metrics.brier_score`` on the rectangular case, else the item mean.

    Items of one template normally offer the same options, so the ``(n_items, n_options)``
    array ``brier_score`` wants can be built; when a template mixes option counts (allowed by
    the schema) that array does not exist and the mean of the per-item Brier is used instead -
    the same quantity, computed per item. The method is returned so the report can say which
    one produced a number.
    """
    selected = np.flatnonzero(mask)
    rows = [probs[subset_positions[i]] for i in selected]
    golds = [gold_index[subset_positions[i]] for i in selected]
    if rows and len({len(row) for row in rows}) == 1:
        return float(brier_score(np.vstack(rows), golds)), "metrics.brier_score"
    return float(np.mean(item_briers[mask])), "mean of per-item Brier (mixed option counts)"


def point_stats(
    correct: np.ndarray, item_briers: np.ndarray, template_of: Sequence[str], templates: Sequence[str]
) -> dict:
    """The un-resampled macro accuracy and mean Brier of one model on one comparison."""
    per_template = {}
    for template in templates:
        mask = np.asarray([t == template for t in template_of])
        per_template[template] = {
            "n": int(mask.sum()),
            "correct": float(correct[mask].mean()),
            "brier": float(item_briers[mask].mean()),
        }
    return {
        "macro_accuracy": float(np.mean([v["correct"] for v in per_template.values()])),
        "mean_brier": float(item_briers.mean()),
        "macro_brier": float(np.mean([v["brier"] for v in per_template.values()])),
        "per_template": per_template,
    }


def _difference_summary(
    diff: np.ndarray, *, point: float, alpha: float
) -> dict[str, Any]:
    lo, hi = np.quantile(diff, [alpha / 2, 1 - alpha / 2])
    return {
        "point": point,
        "ci95": [float(lo), float(hi)],
        "frac_gt_zero": float(np.mean(diff > 0)),
        "frac_lt_zero": float(np.mean(diff < 0)),
        "frac_eq_zero": float(np.mean(diff == 0)),
        "resamples": int(diff.size),
    }


def paired_comparison(
    *,
    set_name: str,
    reference: str,
    baseline: str,
    set_ids: Sequence[str],
    common_ids: Sequence[str],
    ref: Vectors,
    base: Vectors,
    seed: int,
    n_resamples: int,
    alpha: float = ALPHA,
) -> dict[str, Any]:
    """The paired and unpaired bootstrap for one ``(set, reference - baseline)`` comparison.

    Paired: one resample draw, shared by both models (identical resample indices), so the
    difference distribution reflects the models' correlated errors. Unpaired: independent
    draws per model - reported beside the paired CI, never used for the verdict.

    ``common_ids`` is the intersection of the two models' scored items within the set; the
    counts of items each model scored on its own are returned beside it, so a small
    intersection is visible with its reason instead of looking like a small item set.
    """
    ref_scored = [item_id for item_id in set_ids if item_id in ref.index]
    base_scored = [item_id for item_id in set_ids if item_id in base.index]
    counts = {
        "n_reference_scored": len(ref_scored),
        "n_baseline_scored": len(base_scored),
        "n_reference_only": len([i for i in ref_scored if i not in base.index]),
        "n_baseline_only": len([i for i in base_scored if i not in ref.index]),
    }
    if not common_ids:
        return {
            "set": set_name,
            "reference": reference,
            "baseline": baseline,
            "n_items": 0,
            "n_templates": 0,
            "templates": [],
            "metrics": {},
            "per_template": [],
            **counts,
        }
    ref_pos = [ref.index[item_id] for item_id in common_ids]
    base_pos = [base.index[item_id] for item_id in common_ids]
    template_of = [ref.template_ids[p] for p in ref_pos]
    templates = sorted(set(template_of))
    ref_correct = ref.correct[ref_pos]
    base_correct = base.correct[base_pos]
    ref_brier = ref.item_brier[ref_pos]
    base_brier = base.item_brier[base_pos]

    draw = stratified_resample_indices(
        template_of, n_resamples=n_resamples, seed=seed
    )
    # The unpaired difference must draw **independently per model** - one shared draw would
    # be the paired bootstrap under another seed.
    draw_unpaired_ref = stratified_resample_indices(
        template_of, n_resamples=n_resamples, seed=seed + UNPAIRED_SEED_OFFSET
    )
    draw_unpaired_base = stratified_resample_indices(
        template_of, n_resamples=n_resamples, seed=seed + 2 * UNPAIRED_SEED_OFFSET
    )
    ref_point = point_stats(ref_correct, ref_brier, template_of, templates)
    base_point = point_stats(base_correct, base_brier, template_of, templates)
    ref_stats = {
        "macro_accuracy": macro_accuracy_resamples(ref_correct, draw, templates),
        "mean_brier": mean_brier_resamples(ref_brier, draw, templates),
    }
    base_stats = {
        "macro_accuracy": macro_accuracy_resamples(base_correct, draw, templates),
        "mean_brier": mean_brier_resamples(base_brier, draw, templates),
    }
    unpaired_ref = {
        "macro_accuracy": macro_accuracy_resamples(ref_correct, draw_unpaired_ref, templates),
        "mean_brier": mean_brier_resamples(ref_brier, draw_unpaired_ref, templates),
    }
    unpaired_base = {
        "macro_accuracy": macro_accuracy_resamples(base_correct, draw_unpaired_base, templates),
        "mean_brier": mean_brier_resamples(base_brier, draw_unpaired_base, templates),
    }

    metrics_out: dict[str, Any] = {}
    for key in ("macro_accuracy", "mean_brier"):
        diff = ref_stats[key] - base_stats[key]
        unpaired_diff = unpaired_ref[key] - unpaired_base[key]
        summary = _difference_summary(
            diff, point=ref_point[key] - base_point[key], alpha=alpha
        )
        summary["unpaired_ci95"] = [
            float(v) for v in np.quantile(unpaired_diff, [alpha / 2, 1 - alpha / 2])
        ]
        summary["reference_marginal_ci95"] = [
            float(v) for v in np.quantile(ref_stats[key], [alpha / 2, 1 - alpha / 2])
        ]
        summary["baseline_marginal_ci95"] = [
            float(v) for v in np.quantile(base_stats[key], [alpha / 2, 1 - alpha / 2])
        ]
        summary["reference_marginal_point"] = ref_point[key]
        summary["baseline_marginal_point"] = base_point[key]
        metrics_out[key] = summary
    metrics_out["macro_brier_additional"] = {
        "reference_point": ref_point["macro_brier"],
        "baseline_point": base_point["macro_brier"],
        "difference": ref_point["macro_brier"] - base_point["macro_brier"],
        "label": (
            "additional — template-mean Brier (each template weighted equally). D16's verdict "
            "uses mean Brier above; this column is reported beside it, never instead of it"
        ),
    }

    per_template = []
    for template in templates:
        mask = np.asarray([t == template for t in template_of])
        ref_value, ref_method = template_brier(
            probs=ref.probs,
            gold_index=ref.gold_index,
            item_briers=ref_brier,
            subset_positions=ref_pos,
            mask=mask,
        )
        base_value, base_method = template_brier(
            probs=base.probs,
            gold_index=base.gold_index,
            item_briers=base_brier,
            subset_positions=base_pos,
            mask=mask,
        )
        per_template.append(
            {
                "template_id": template,
                "n": int(mask.sum()),
                "reference_accuracy": float(ref_correct[mask].mean()),
                "baseline_accuracy": float(base_correct[mask].mean()),
                "difference": float(ref_correct[mask].mean() - base_correct[mask].mean()),
                "reference_brier": ref_value,
                "baseline_brier": base_value,
                "brier_method": ref_method if ref_method == base_method else f"{ref_method} / {base_method}",
            }
        )
    return {
        "set": set_name,
        "reference": reference,
        "baseline": baseline,
        "n_items": len(common_ids),
        "n_templates": len(templates),
        "templates": templates,
        "metrics": metrics_out,
        "per_template": per_template,
        **counts,
    }


def comparison_verdict(comparison: dict[str, Any]) -> tuple[str, str]:
    """D16's rule for one comparison (PASS / FAIL / NOT MEASURED + reason)."""
    if comparison["n_items"] == 0:
        return "NOT MEASURED", "no items scored by both models in this set"
    accuracy = comparison["metrics"]["macro_accuracy"]
    brier = comparison["metrics"]["mean_brier"]
    accuracy_ok = accuracy["ci95"][0] > 0
    brier_ok = brier["ci95"][1] < 0
    if accuracy_ok and brier_ok:
        return "PASS", ""
    parts = []
    if not accuracy_ok:
        parts.append(
            f"accuracy CI lower bound {accuracy['ci95'][0]:.4f} is not > 0"
        )
    if not brier_ok:
        parts.append(f"Brier CI upper bound {brier['ci95'][1]:.4f} is not < 0")
    return "FAIL", "; ".join(parts)


def overall_verdict(
    comparisons: dict[str, dict[str, dict[str, Any]]], baselines: Sequence[str], set_name: str
) -> tuple[str, str]:
    """Conjunction of the per-baseline comparison verdicts on one item set.

    A measured FAIL dominates a missing comparison: an unwelcome measured result is never
    relabelled as "not measured" (R4); a missing baseline is named in the detail instead.
    """
    parts: list[str] = []
    n_fail = n_missing = 0
    for baseline in baselines:
        comparison = comparisons.get(set_name, {}).get(baseline)
        if comparison is None:
            n_missing += 1
            parts.append(f"{baseline}: no comparison")
            continue
        verdict, reason = comparison_verdict(comparison)
        if verdict == "NOT MEASURED":
            n_missing += 1
            parts.append(f"{baseline}: {reason}")
        elif verdict == "FAIL":
            n_fail += 1
            parts.append(f"{baseline}: {reason}")
    if n_fail:
        outcome = "FAIL"
    elif n_missing:
        outcome = "NOT MEASURED"
    elif not baselines:
        outcome = "NOT MEASURED"
        parts.append("no baseline was given")
    else:
        outcome = "PASS"
    if outcome == "PASS":
        return "PASS", "every baseline's accuracy CI lower bound > 0 and Brier CI upper bound < 0"
    return outcome, "; ".join(parts)


# --------------------------------------------------------------------------- rendering
def fmt(value: float | None, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "—"
    return f"{value:.{digits}f}"


def fmt_ci(ci: Sequence[float] | None, digits: int = 4) -> str:
    if not ci:
        return "—"
    return f"[{fmt(ci[0], digits)}, {fmt(ci[1], digits)}]"


def render_report(report: dict[str, Any]) -> str:
    """The committed ``g1.md``: verdict first, then every paired difference with its CIs."""
    lines: list[str] = []
    lines.append("# G1 — MedDecide vs baselines on MedDecide-Bench v0.2 (S11 machinery)")
    lines.append("")
    lines.append(
        "Generated by `scripts/bench/g1.py` (decision D16, ADVISORY §S11). This file is an "
        "aggregate only: no item text, no predictions."
    )
    lines.append("")
    prov = report["provenance"]
    lines.append(f"- Generated (UTC): `{prov['generated_at_utc']}`")
    lines.append(f"- Command: `{prov['command']}`")
    lines.append(f"- Git commit: `{prov['git_commit']}`")
    lines.append(
        f"- Seed: `{prov['seed']}`, resamples: `{prov['resamples']}`, alpha: `{prov['alpha']}`"
    )
    lines.append(f"- Reference model: `{prov['reference']}`")
    lines.append(f"- Verdict baselines: {', '.join(f'`{b}`' for b in prov['baselines']) or '—'}")
    lines.append("")

    lines.append("## Verdict (D16)")
    lines.append("")
    lines.append("| rule | result | detail |")
    lines.append("|---|---|---|")
    for set_name, label in (
        ("fresh_seen", "**G1 on seen templates** (decides the branch of §2)"),
        ("fresh_heldout", "G1 on held-out templates (D14, reported separately)"),
        ("fresh_strict_seen", "strict slice, seen templates (additional, same rule)"),
        ("fresh_strict_heldout", "strict slice, held-out templates (additional, same rule)"),
    ):
        verdict, reason = report["verdicts"][set_name]
        lines.append(f"| {label} | **{verdict}** | {reason or '—'} |")
    lines.append("")
    lines.append(
        "D16 rule: PASS iff, against **both** baselines, the accuracy-difference CI lower "
        "bound is > 0 **and** the Brier-difference CI upper bound is < 0. A missing baseline "
        "or an empty item intersection is `NOT MEASURED` (R3), never a pass."
    )
    lines.append("")

    lines.append("## Inputs")
    lines.append("")
    lines.append("| label | file | exists | run analysed | runs in file | rows | items | model ids | note |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for source in report["sources"]:
        lines.append(
            f"| `{source['label']}` | `{source['path']}` | {'yes' if source['exists'] else 'NO'} "
            f"| `{source['run_id'] or '—'}` | {source['n_runs']} | {source['n_rows']} "
            f"| {source['n_items']} | {', '.join(source['model_ids']) or '—'} "
            f"| {source['note'] or '—'} |"
        )
    lines.append("")
    lines.append("| input | path | state |")
    lines.append("|---|---|---|")
    for name, state in report["inputs"].items():
        lines.append(f"| {name} | `{state['path']}` | {state['state']} |")
    lines.append("")

    lines.append("## Item sets")
    lines.append("")
    lines.append("| set | what it is | items | templates |")
    lines.append("|---|---|---|---|")
    for set_name, description in ITEM_SETS:
        n_items, n_templates = report["set_sizes"][set_name]
        lines.append(f"| `{set_name}` | {description} | {n_items} | {n_templates} |")
    lines.append("")
    for note in report["item_set_notes"]:
        lines.append(f"- {note}")
    lines.append("")

    lines.append("## Model metrics per item set")
    lines.append("")
    lines.append(
        "Micro accuracy is `meddecide.eval.metrics.accuracy`; its CI is "
        "`metrics.bootstrap_ci` (plain item bootstrap). Macro accuracy and mean Brier come "
        "from the template-stratified resample plan **written in `g1.py`** (see Method). "
        "Marginal CI = the model's own stratified resample distribution."
    )
    lines.append("")
    lines.append(
        "| set | model | items | coverage | D12-excluded | micro acc | micro 95% CI | majority "
        "| macro acc | macro 95% CI | mean Brier | Brier 95% CI |",
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for set_name, _ in ITEM_SETS:
        for model in report["models"]:
            stats = report["model_stats"][set_name][model]
            if stats is None:
                lines.append(
                    f"| `{set_name}` | `{model}` | 0 | — | — | NOT MEASURED | — | — | — "
                    "| — | — | — | — |"
                )
                continue
            lines.append(
                f"| `{set_name}` | `{model}` | {stats['n_items']} | {fmt(stats['coverage'], 3)} "
                f"| {stats['n_excluded_by_d12']} | {fmt(stats['micro_accuracy'])} "
                f"| {fmt_ci(stats['micro_ci95'])} | {fmt(stats['majority_baseline'])} "
                f"| {fmt(stats['macro_accuracy'])} | {fmt_ci(stats['macro_ci95'])} "
                f"| {fmt(stats['mean_brier'])} | {fmt_ci(stats['brier_ci95'])} |"
            )
    lines.append("")
    lines.append(
        "The table above is **marginal**: each model is measured on the items it scored in "
        "that set (minus its own D12-excluded cells). It is not a comparison - the paired "
        "tables below use the intersection and state its item count. `coverage` is scored "
        "items / set items **before** D12 exclusions; `D12-excluded` is the number of the "
        "model's scored items dropped from the verdict by a failing cell."
    )
    lines.append("")

    lines.append("## Paired comparisons (MedDecide - baseline)")
    lines.append("")
    lines.append(
        "Every comparison uses exactly the **intersection** of the items both models scored "
        "(after D12 exclusions), restricted to the item set; the item count is stated. Paired "
        "CIs share one resample draw across the pair (identical resample indices); the "
        "unpaired CI draws independently per model and is reported beside the paired one, "
        "never used for the verdict."
    )
    lines.append("")
    for set_name, _ in ITEM_SETS:
        for baseline in report["provenance"]["baselines"]:
            comparison = report["comparisons"].get(set_name, {}).get(baseline)
            lines.append(f"### `{set_name}` — MedDecide - `{baseline}`")
            lines.append("")
            if comparison is None:
                reason = report["comparison_missing"].get(
                    f"{set_name}|{baseline}", "no comparison"
                )
                lines.append(f"`NOT MEASURED — {reason}`")
                lines.append("")
                continue
            if comparison["n_items"] == 0:
                lines.append(
                    "`NOT MEASURED — no items scored by both models in this set` "
                    f"(reference scored {comparison['n_reference_scored']} of the set's items, "
                    f"baseline {comparison['n_baseline_scored']}, before the intersection and "
                    "D12 exclusions)"
                )
                lines.append("")
                continue
            verdict, reason = comparison_verdict(comparison)
            lines.append(
                f"- item count: **n = {comparison['n_items']}** across "
                f"{comparison['n_templates']} templates; reference scored "
                f"{comparison['n_reference_scored']} of the set's items, baseline "
                f"{comparison['n_baseline_scored']}, both {comparison['n_items']}. "
                f"Comparison verdict: **{verdict}**{' — ' + reason if reason else ''}"
            )
            lines.append("")
            lines.append(
                "| metric | MedDecide | baseline | difference | paired 95% CI | frac > 0 "
                "| frac < 0 | frac = 0 | unpaired 95% CI | MedDecide marginal CI "
                "| baseline marginal CI |"
            )
            lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
            for key, label in (
                ("macro_accuracy", "macro accuracy"),
                ("mean_brier", "mean Brier (lower is better)"),
            ):
                metric = comparison["metrics"][key]
                lines.append(
                    f"| {label} | {fmt(metric['reference_marginal_point'])} "
                    f"| {fmt(metric['baseline_marginal_point'])} | {fmt(metric['point'])} "
                    f"| {fmt_ci(metric['ci95'])} | {fmt(metric['frac_gt_zero'], 3)} "
                    f"| {fmt(metric['frac_lt_zero'], 3)} | {fmt(metric['frac_eq_zero'], 3)} "
                    f"| {fmt_ci(metric['unpaired_ci95'])} "
                    f"| {fmt_ci(metric['reference_marginal_ci95'])} "
                    f"| {fmt_ci(metric['baseline_marginal_ci95'])} |"
                )
            additional = comparison["metrics"]["macro_brier_additional"]
            lines.append(
                f"| macro Brier (additional) | {fmt(additional['reference_point'])} "
                f"| {fmt(additional['baseline_point'])} | {fmt(additional['difference'])} "
                "| — | — | — | — | — | — | — |"
            )
            lines.append("")
            lines.append(
                "| template | n | MedDecide acc | baseline acc | difference "
                "| MedDecide Brier | baseline Brier |"
            )
            lines.append("|---|---|---|---|---|---|---|")
            for row in comparison["per_template"]:
                lines.append(
                    f"| `{row['template_id']}` | {row['n']} | {fmt(row['reference_accuracy'])} "
                    f"| {fmt(row['baseline_accuracy'])} | {fmt(row['difference'])} "
                    f"| {fmt(row['reference_brier'])} | {fmt(row['baseline_brier'])} |"
                )
            lines.append("")

    lines.append("## D12 readout health")
    lines.append("")
    lines.append(
        "Every cell is `meddecide.eval.health.evaluate_cell` with unchanged thresholds. "
        "Letter-readout-only checks (`median_label_mass`, `greedy_agreement`) are recorded as "
        "**NOT APPLICABLE** for a pointer-head cell — never as passed. The constant-answer "
        "check **is** applied to every cell."
    )
    lines.append("")
    lines.append("| model | tier | template | n | readout | status | applied checks | not applicable / not measured |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for cell in report["cells"]:
        other = "; ".join(
            f"{check}: {reason}" for check, reason in cell["inapplicable_checks"].items()
        )
        extra = "; ".join(
            f"{check}: {reason}" for check, reason in cell["not_measured_checks"].items()
        )
        if extra:
            other = f"{other}; {extra}" if other else extra
        lines.append(
            f"| `{cell['model']}` | {cell['tier']} | `{cell['template_id']}` | {cell['n']} "
            f"| {cell['readout']} | {cell['status']} "
            f"| {', '.join(cell['applied_checks'])} | {other or '—'} |"
        )
    lines.append("")

    lines.append("### Cells excluded from the verdict")
    lines.append("")
    if report["excluded_cells"]:
        lines.append("| model | tier | template | check(s) | failure |")
        lines.append("|---|---|---|---|---|")
        for cell in report["excluded_cells"]:
            lines.append(
                f"| `{cell['model']}` | {cell['tier']} | `{cell['template_id']}` "
                f"| {', '.join(cell['failing_checks'])} | {'; '.join(cell['failures'])} |"
            )
        lines.append("")
        lines.append(
            "A failing `(model, template)` cell drops **that model's** items of that template "
            "from every comparison, so the template leaves the intersection entirely. Item "
            "counts in the comparison tables already exclude them."
        )
    else:
        lines.append("None — every scored cell passed D12.")
    lines.append("")

    lines.append("## Items dropped, with counts (no silent drops)")
    lines.append("")
    lines.append("| model | reason | items |")
    lines.append("|---|---|---|")
    for model, dropped in report["dropped"].items():
        if not dropped:
            lines.append(f"| `{model}` | none | 0 |")
            continue
        for reason, count in dropped.items():
            lines.append(f"| `{model}` | {reason} | {count} |")
    lines.append("")

    lines.append("## Method — what is reused and what is new here")
    lines.append("")
    for note in report["method_notes"]:
        lines.append(f"- {note}")
    lines.append("")
    lines.append("## NOT MEASURED reasons")
    lines.append("")
    if report["not_measured"]:
        for reason in report["not_measured"]:
            lines.append(f"- `NOT MEASURED — {reason}`")
    else:
        lines.append("- none")
    lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------- main
def model_stats(
    *,
    vectors: Vectors,
    item_ids: Sequence[str],
    templates: Sequence[str],
    n_resamples: int,
    seed: int,
    alpha: float,
) -> dict[str, Any] | None:
    """One model's metrics on one item set: micro/macro accuracy and mean Brier, with CIs."""
    positions = [vectors.index[i] for i in item_ids if i in vectors.index]
    if not positions:
        return None
    correct = vectors.correct[positions]
    briers = vectors.item_brier[positions]
    template_of = [vectors.template_ids[p] for p in positions]
    present = sorted(set(template_of))
    micro = accuracy_report(
        [vectors.predicted[p] for p in positions], [vectors.gold[p] for p in positions]
    )
    _, micro_lo, micro_hi = bootstrap_ci(
        [1.0 if vectors.predicted[p] == vectors.gold[p] else 0.0 for p in positions],
        n_resamples=n_resamples,
        seed=seed,
    )
    draw = stratified_resample_indices(template_of, n_resamples=n_resamples, seed=seed)
    macro = macro_accuracy_resamples(correct, draw, present)
    brier = mean_brier_resamples(briers, draw, present)
    per_template: dict[str, Any] = {}
    for template in present:
        mask = np.asarray([t == template for t in template_of])
        golds = [
            vectors.gold[p]
            for p, t in zip(positions, template_of, strict=True)
            if t == template
        ]
        value, method = template_brier(
            probs=vectors.probs,
            gold_index=vectors.gold_index,
            item_briers=briers,
            subset_positions=positions,
            mask=mask,
        )
        per_template[template] = {
            "n": int(mask.sum()),
            "accuracy": float(correct[mask].mean()),
            "macro_accuracy": macro_accuracy(
                [
                    vectors.predicted[p]
                    for p, t in zip(positions, template_of, strict=True)
                    if t == template
                ],
                golds,
            ),
            "brier": value,
            "brier_method": method,
            "majority_share": Counter(golds).most_common(1)[0][1] / len(golds),
        }
    return {
        "n_items": len(positions),
        "coverage": len(positions) / len(item_ids) if item_ids else 0.0,
        "n_templates": len(present),
        "micro_accuracy": micro.micro,
        "micro_ci95": [micro_lo, micro_hi],
        "majority_baseline": micro.majority_baseline,
        "majority_label": micro.majority_label,
        "macro_accuracy": float(
            np.mean([correct[np.asarray([t == x for x in template_of])].mean() for t in present])
        ),
        "macro_ci95": [float(v) for v in np.quantile(macro, [alpha / 2, 1 - alpha / 2])],
        "mean_brier": float(briers.mean()),
        "brier_ci95": [float(v) for v in np.quantile(brier, [alpha / 2, 1 - alpha / 2])],
        "per_template": per_template,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Build the whole report as a dictionary (rendered and/or written by :func:`main`)."""
    bench_dir = Path(args.bench_dir)
    fresh_dir = Path(args.fresh) if args.fresh else bench_dir / "fresh"
    tier1_dir = Path(args.tier1) if args.tier1 else None
    screen_path = Path(args.screen) if args.screen else bench_dir / "screen.json"
    acceptance_path = (
        Path(args.acceptance) if args.acceptance else fresh_dir / "acceptance.json"
    )
    out_path = Path(args.out)

    models = parse_models(args.models)
    if not models:
        raise SystemExit("--models is required, e.g. --models meddecide=outputs/.../preds.jsonl")
    labels = [label for label, _ in models]
    if args.reference not in labels:
        raise SystemExit(f"--reference {args.reference!r} is not among --models {labels}")
    baselines = (
        [b.strip() for b in args.baselines.split(",") if b.strip()]
        if args.baselines
        else [label for label in labels if label != args.reference]
    )
    unknown = [b for b in baselines if b not in labels]
    if unknown:
        raise SystemExit(f"--baselines {unknown} are not among --models {labels}")

    items, per_file = load_bench_items(fresh_dir=fresh_dir, tier1_dir=tier1_dir)
    screen = load_screen(screen_path)
    strict_start, strict_source = resolve_strict_start(
        [acceptance_path, fresh_dir / "manifest.json", bench_dir / "manifest.json"]
    )
    sets = build_item_sets(
        items,
        screen=screen,
        strict_start=strict_start,
        include_strict=bool(args.strict_slice),
        include_tier1=tier1_dir is not None,
    )

    sources = [load_model_source(label, path) for label, path in models]
    by_label = {source.label: source for source in sources}

    cells_by_model: dict[str, dict[tuple[str, str], CellGate]] = {}
    skipped_by_model: dict[str, dict[str, int]] = {}
    for source in sources:
        if source.missing_reason:
            cells_by_model[source.label] = {}
            skipped_by_model[source.label] = {}
            continue
        cells, skipped = compute_cells(
            source=source,
            items=items,
            screen=screen,
            seed=args.seed,
            n_resamples=args.resamples,
        )
        cells_by_model[source.label] = cells
        skipped_by_model[source.label] = skipped

    vectors: dict[str, Vectors] = {}
    union_ids = sorted({item_id for ids in sets.sets.values() for item_id in ids})
    for source in sources:
        if source.missing_reason:
            continue
        # Vectors are built on the union of the reported sets; a comparison then subsets to
        # the intersection of the two models' vectors.
        vectors[source.label] = build_vectors(
            source=source,
            items=items,
            cells=cells_by_model[source.label],
            item_ids=union_ids,
        )

    not_measured: list[str] = []
    comparison_missing: dict[tuple[str, str], str] = {}
    comparisons: dict[str, dict[str, dict[str, Any]]] = {}
    for set_name, ids in sets.sets.items():
        comparisons[set_name] = {}
        if not ids:
            for baseline in baselines:
                comparison_missing[(set_name, baseline)] = (
                    "the item set is empty (no items match its definition)"
                )
            continue
        for baseline in baselines:
            ref = vectors.get(args.reference)
            base = vectors.get(baseline)
            if ref is None:
                reason = by_label[args.reference].missing_reason or "reference model unusable"
                comparison_missing[(set_name, baseline)] = f"reference model: {reason}"
                continue
            if base is None:
                reason = by_label[baseline].missing_reason or "baseline model unusable"
                comparison_missing[(set_name, baseline)] = f"baseline model: {reason}"
                continue
            common = [i for i in ids if i in ref.index and i in base.index]
            comparisons[set_name][baseline] = paired_comparison(
                set_name=set_name,
                reference=args.reference,
                baseline=baseline,
                set_ids=ids,
                common_ids=common,
                ref=ref,
                base=base,
                seed=args.seed,
                n_resamples=args.resamples,
            )

    verdicts = {
        set_name: overall_verdict(comparisons, baselines, set_name)
        for set_name in ("fresh_seen", "fresh_heldout", "fresh_strict_seen", "fresh_strict_heldout")
    }

    model_stat_rows: dict[str, dict[str, Any]] = {}
    for set_name, ids in sets.sets.items():
        model_stat_rows[set_name] = {}
        for label in labels:
            vectors_for = vectors.get(label)
            source = by_label[label]
            scored_before_d12 = [
                item_id
                for item_id in ids
                if item_id in source.rows
                and item_id in items
                and items[item_id].split == "test"
                and label in vectors
            ]
            if vectors_for is None:
                model_stat_rows[set_name][label] = None
                continue
            present = [i for i in ids if i in vectors_for.index]
            stats = model_stats(
                vectors=vectors_for,
                item_ids=present,
                templates=sorted({vectors_for.template_ids[vectors_for.index[i]] for i in present}),
                n_resamples=args.resamples,
                seed=args.seed,
                alpha=ALPHA,
            )
            if stats is not None:
                stats["n_scored_before_d12"] = len(scored_before_d12)
                stats["coverage"] = (
                    len(scored_before_d12) / len(ids) if ids else 0.0
                )
                stats["n_excluded_by_d12"] = len(scored_before_d12) - stats["n_items"]
            model_stat_rows[set_name][label] = stats

    excluded_cells = [
        cell.to_dict()
        for cells in cells_by_model.values()
        for cell in cells.values()
        if not cell.passed
    ]
    for cell in excluded_cells:
        not_measured.append(
            f"cell excluded from the verdict — {cell['model']} / {cell['template_id']}: "
            f"{cell['status']}"
        )
    for source in sources:
        if source.missing_reason:
            not_measured.append(f"{source.label}: {source.missing_reason}")
    for (set_name, baseline), reason in comparison_missing.items():
        not_measured.append(f"comparison {set_name} / {baseline}: {reason}")

    set_sizes = {
        set_name: (
            len(ids),
            len({items[i].template_id for i in ids}) if ids else 0,
        )
        for set_name, ids in sets.sets.items()
    }
    inputs = {
        "benchmark dir": {"path": str(bench_dir), "state": "present" if bench_dir.is_dir() else "MISSING"},
        "fresh items": {
            "path": str(fresh_dir),
            "state": f"{len(per_file)} jsonl files" if fresh_dir.is_dir() else "MISSING",
        },
        "tier 1 items": {
            "path": str(tier1_dir) if tier1_dir else "(not given; tier-1 set not reported)",
            "state": "present" if tier1_dir and tier1_dir.is_dir() else ("MISSING" if tier1_dir else "not requested"),
        },
        "screen": {
            "path": str(screen_path),
            "state": f"{len(screen)} templates" if screen else "MISSING or empty",
        },
        "acceptance": {
            "path": str(acceptance_path),
            "state": "present" if acceptance_path.is_file() else "MISSING",
        },
        "strict slice start": {
            "path": strict_source,
            "state": strict_start.isoformat(),
        },
    }

    method_notes = [
        "reused: `meddecide.eval.predlog.read_prediction_log` — dedupe by `(run_id, item_id)`, "
        "last row wins; one run per model is analysed (the run with the most distinct items).",
        "reused: `meddecide.eval.metrics.accuracy` (micro accuracy + majority baseline), "
        "`metrics.macro_accuracy` (per-template class recall), `metrics.brier_score`, "
        "`metrics.bootstrap_ci` (plain item bootstrap for the micro CI).",
        "reused: `meddecide.eval.health.evaluate_cell` — all D12 thresholds unchanged. New "
        "here: the wrapper that records the letter-readout-only checks as NOT APPLICABLE for a "
        "pointer head and never as passed; the constant-answer check is applied to every cell.",
        "new in `g1.py`: the template-stratified resample plan (`stratified_resample_indices`) "
        "— items resampled within templates, one shared draw per pair for the paired CI and "
        "independent draws for the unpaired CI; `macro_accuracy_resamples`; "
        "`mean_brier_resamples`; the per-item Brier for items with different option counts "
        "(D16's mean Brier is pooled over items; the template-mean Brier is additional).",
        "new in `g1.py`: the item-set construction (headline fresh = screen-kept templates "
        "minus the superseded ids; seen/held-out by the D14 template list; strict slice by "
        "`record_date >= strict_slice_start`), and the intersection rule for comparisons.",
        f"strict-slice start read from `{strict_source}` as `{strict_start.isoformat()}`.",
    ]
    if per_file:
        method_notes.append(
            "benchmark files indexed: " + ", ".join(f"`{p}` ({n})" for p, n in sorted(per_file.items()))
        )

    provenance = {
        "generated_at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "command": " ".join([Path(sys.argv[0]).name, *sys.argv[1:]]),
        "git_commit": _git_commit(),
        "seed": args.seed,
        "resamples": args.resamples,
        "alpha": ALPHA,
        "reference": args.reference,
        "baselines": baselines,
        "out": str(out_path),
        "strict_slice_start": strict_start.isoformat(),
    }
    dropped: dict[str, dict[str, int]] = {}
    for label in labels:
        counts: Counter[str] = Counter(skipped_by_model.get(label, {}))
        if label in vectors:
            counts.update(vectors[label].dropped)
        dropped[label] = dict(counts)
    return {
        "provenance": provenance,
        "inputs": inputs,
        "sources": [
            {
                "label": source.label,
                "path": source.path,
                "exists": source.exists,
                "run_id": source.run_id,
                "n_runs": len(source.run_ids),
                "run_ids": source.run_ids,
                "n_rows": source.n_raw,
                "n_unique_rows": source.n_unique,
                "n_items": source.n_items,
                "n_without_run_id": source.n_without_run_id,
                "unreadable_lines": source.unreadable_lines,
                "model_ids": source.model_ids,
                "note": source.note,
            }
            for source in sources
        ],
        "models": labels,
        "item_set_notes": sets.notes,
        "set_sizes": set_sizes,
        "verdicts": verdicts,
        "model_stats": model_stat_rows,
        "comparisons": comparisons,
        "comparison_missing": {f"{s}|{b}": r for (s, b), r in comparison_missing.items()},
        "cells": [
            cell.to_dict()
            for label in labels
            for cell in sorted(cells_by_model[label].values(), key=lambda c: (c.tier, c.template_id))
        ],
        "excluded_cells": excluded_cells,
        "dropped": dropped,
        "method_notes": method_notes,
        "not_measured": not_measured,
    }


def _git_commit() -> str:
    try:
        from meddecide.utils.provenance import git_commit

        return git_commit(Path.cwd())
    except Exception:  # pragma: no cover - environment dependent
        return "UNKNOWN"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Gate G1 evaluation machinery (D16): paired bootstrap of MedDecide - baseline in "
            "macro accuracy and mean Brier, seen and held-out templates separately."
        )
    )
    parser.add_argument(
        "--models",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="prediction JSONL for one model; repeatable and comma-separated",
    )
    parser.add_argument("--reference", default="meddecide", help="label of MedDecide itself")
    parser.add_argument(
        "--baselines",
        default=None,
        help="comma list of labels the verdict is computed against (default: every "
        "non-reference model)",
    )
    parser.add_argument("--bench-dir", default="data/bench/v0.2", help="v0.2 benchmark directory")
    parser.add_argument("--fresh", default=None, help="fresh item directory (default <bench-dir>/fresh)")
    parser.add_argument(
        "--tier1",
        default=None,
        help="tier-1 item directory; when given, the tier-1 set is reported",
    )
    parser.add_argument("--screen", default=None, help="screen JSON (default <bench-dir>/screen.json)")
    parser.add_argument(
        "--acceptance",
        default=None,
        help="acceptance JSON with strict_slice_start (default <fresh>/acceptance.json, falling "
        "back to the fresh manifest)",
    )
    parser.add_argument(
        "--strict-slice",
        dest="strict_slice",
        action="store_true",
        default=True,
        help="report the strict-slice sets (default)",
    )
    parser.add_argument(
        "--no-strict-slice", dest="strict_slice", action="store_false", help="skip the strict slice"
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--resamples", type=int, default=DEFAULT_RESAMPLES)
    parser.add_argument("--out", default="loops/student_v0/g1.md", help="markdown report path")
    parser.add_argument("--json", default=None, help="optional machine-readable report path")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run(args)
    text = render_report(report)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    if args.json:
        json_path = Path(args.json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(
            json.dumps(_jsonable(report), indent=1, sort_keys=True), encoding="utf-8"
        )
    for set_name, (verdict, reason) in report["verdicts"].items():
        print(f"{set_name}: {verdict}" + (f" — {reason}" if reason else ""))
    print(f"wrote {out_path}" + (f" and {args.json}" if args.json else ""))
    return 0


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


if __name__ == "__main__":
    raise SystemExit(main())
