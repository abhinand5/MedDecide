#!/usr/bin/env python
"""Build the S6 training mix: pre-window structured-gold items + train/dev + leakage check.

Loop ``student_v0``, task S6. Three stages, each re-runnable on its own:

1. ``--stage prewindow`` — run the **v0.2 fresh builders** on records dated before the
   benchmark window (``record_date < 2026-03-01``), sources ``clinicaltrials`` and
   ``openfda``, plus a **measured** attempt at ``pubmed``:

   * ClinicalTrials.gov: ``fetch_studies(2023-01-01, 2026-02-28)`` fetched in quarterly
     slices (cached raw under the scratch dir) then one ``build_items`` call. The four
     non-held-out templates are kept; ``ct_phase_choice_v1`` is a held-out template (D14)
     and its items are excluded **by template id** with a counted reason.
   * openFDA: ``fetch_labels`` month by month (the API rejects ``skip > 25,000``, and a
     39-month window holds far more than that), deduplicated by ``set_id`` exactly as
     ``fetch_labels`` does across slices, then ``build_items`` (class ``_v1`` + route) and
     ``build_class_v2_items`` (class ``_v2``) with the near-miss rule copied from
     ``scripts/bench/build_v0_2.py::build_openfda_class_v2`` (only ``shared_moa`` /
     ``shared_pe``; hub values carried by more than 20 classes excluded).
     ``fda_boxed_warning_noul_v1`` is held out and never built into training.
   * PubMed: update files no longer exist for 2023-2025 (the FTP ``updatefiles`` listing
     holds only the current 2026 files). The only remaining source of Entrez dates inside
     the window is the annual baseline (1,334 files). This function **measures** the cost
     of one baseline file (download + parse) and records ``NOT MEASURED`` when the
     projection for all of them exceeds the task's 60-minute box, rather than silently
     reading a biased subset.

   Every built template is **balanced per template** (``balance_classes``) and capped at
   ``--cap-per-template`` (20,000) items.

2. ``--stage mix`` — assemble the training mix from the inputs that already exist:

   * ``train.jsonl`` = ``tier1_train`` (S5) + ``prewindow_consistency`` (S2) +
     ``prewindow_structured`` (stage 1). Every row is rewritten to ``split=train`` (the
     split is part of ``item_id``, so the id is recomputed when the split changes) — the
     **input files are never modified**.
   * ``dev.jsonl`` = v0.2 tier-1 dev + v0.2 fresh dev, held-out templates removed.
   * ``manifest.json`` = counts per source x template x qtype for both files, the mixture,
     the seed/caps, every build/drop/removal count, and the checks.

3. The **leakage check** (part of the acceptance) removes and counts four kinds:

   1. a ``source_record_id`` that appears in any v0.2 test or dev item;
   2. a normalised state hash (``stable_hash`` over ``normalize_text(state)``) that appears
      in any v0.2 test or dev item — the S5 variant over ``(state, question)`` is counted
      separately as a second reason;
   3. an item whose ``record_date >= 2026-03-01``. Tier-1 items carry their **dataset's
      release date** (``meddecide.bench.tier1.common``: established test sets are not
      date-filtered, that is the fresh tier's job), which is a corpus date, not a record
      date; every tier-1 source's release date is before the window, so the rule is
      applied uniformly and is satisfied rather than being a window violation;
   4. any item carrying a **held-out** template id, from any window.

   Removed items are counted by reason and by source; the check is then re-run on the
   written file and must report zero.

Usage:
    uv run python scripts/bench/build_training_mix.py --stage all
    uv run python scripts/bench/build_training_mix.py --stage mix      # re-assemble only
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import yaml

from meddecide.bench.fresh import clinicaltrials as ct
from meddecide.bench.fresh import fda_classes, openfda, pubmed
from meddecide.bench.schema import Item, compute_item_id
from meddecide.bench.tier1.common import balance_classes
from meddecide.utils.hashing import normalize_text, stable_hash
from meddecide.utils.io import build_manifest, read_jsonl, write_json, write_jsonl
from meddecide.utils.provenance import Provenance, git_commit, gpu_name, utcnow

# ---------------------------------------------------------------------------
# Constants: the loop's settled window and held-out list (D11, D14)
# ---------------------------------------------------------------------------
WINDOW_START = date(2026, 3, 1)  # v0.2 fresh window start; training must be strictly before
PREWINDOW_START = date(2023, 1, 1)
PREWINDOW_END = date(2026, 2, 28)
STRICT_BEFORE_WINDOW = "record_date < 2026-03-01"

HELD_OUT_TEMPLATES: tuple[str, ...] = (
    "ct_phase_choice_v1",  # D14, held out in ADVISORY section 3
    "fda_boxed_warning_noul_v1",  # D14
    "pubmed_humans_noul_v1",  # D14
    "ct_arm_role_noul_v1",  # D14, chosen in S2 (role structure differs most)
)

CAP_PER_TEMPLATE = 20_000
TRAIN_DIR = Path("data/train/student_v0")
BENCH_DIR = Path("data/bench/v0.2")
CACHE_DIR = Path("/workspace/tmp/s6")
PUBMED_CACHE = Path("/workspace/tmp/pubmed")  # S1's cache; reused, never written here
PUBMED_TIMEBOX_MINUTES = 60.0
PUBMED_UPDATE_BASE = "https://ftp.ncbi.nlm.nih.gov/pubmed/updatefiles/"
PUBMED_BASELINE_BASE = "https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/"

# Leakage reason names (stable strings: tests and the manifest key on them)
R_RECORD_ID = "source_record_id_in_v0_2_test_or_dev"
R_STATE_HASH = "normalised_state_hash_in_v0_2_test_or_dev"
R_STATE_QUESTION_HASH = "normalised_state_question_hash_in_v0_2_test_or_dev"
R_DATE = "record_date_on_or_after_window_start"
R_HELD_OUT = "held_out_template_id"
LEAKAGE_REASONS = (R_RECORD_ID, R_STATE_HASH, R_STATE_QUESTION_HASH, R_DATE, R_HELD_OUT)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def normalize_state(state: str) -> str:
    """The normalisation S6 specifies for the state hash."""
    return normalize_text(state)


def state_hash(item: Item) -> str:
    return stable_hash({"state": normalize_state(item.state)})


def state_question_hash(item: Item) -> str:
    """The S5 leakage hash (state + question), kept as a second, stricter reason."""
    return stable_hash({"state": normalize_state(item.state), "question": normalize_text(item.question)})


def record_key(item: Item) -> str:
    return f"{item.source}|{item.source_record_id}"


def as_train(item: Item) -> tuple[Item, bool]:
    """Return ``(item, changed)`` with ``split=train``; the id is recomputed when it moves.

    ``item_id`` is a hash over the split (``schema.compute_item_id``), so a row whose split
    is rewritten to ``train`` gets its id recomputed — otherwise the id would no longer be
    the deterministic hash of the row it names. Inputs are never modified; this operates on
    rows in memory on their way into ``train.jsonl``.
    """
    if str(item.split) == "train":
        return item, False
    payload = item.model_dump(mode="json")
    payload["split"] = "train"
    payload["item_id"] = compute_item_id(
        item.source, item.source_record_id, item.template_id,
        int(item.meta.get("option_order_seed", 0)), "train",
    )
    return Item.model_validate(payload), True


def counts_by(rows: Sequence[Item], *keys: str) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for row in rows:
        counter["|".join(str(getattr(row, k)) for k in keys)] += 1
    return dict(sorted(counter.items()))


def flat_source_template_qtype(rows: Sequence[Item]) -> dict[str, dict[str, int]]:
    """``source -> template -> qtype -> n`` as nested plain dicts (JSON-friendly)."""
    out: dict[str, dict[str, dict[str, int]]] = {}
    for row in rows:
        node = out.setdefault(row.source, {}).setdefault(row.template_id, {})
        node[str(row.qtype)] = node.get(str(row.qtype), 0) + 1
    return {s: {t: dict(sorted(q.items())) for t, q in sorted(v.items())} for s, v in sorted(out.items())}


def _write_jsonl_gz(path: Path, rows: Iterable[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    return path


def _read_jsonl_gz(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _date_slices(start: date, end: date, *, months: int) -> list[tuple[date, date]]:
    """Split ``[start, end]`` into calendar slices of ``months`` months."""
    slices: list[tuple[date, date]] = []
    cursor = start
    while cursor <= end:
        y, m = cursor.year, cursor.month + months
        y += (m - 1) // 12
        m = (m - 1) % 12 + 1
        last = min(end, date(y, m, 1) - timedelta(days=1))
        slices.append((cursor, last))
        cursor = date(y, m, 1)
    return slices


# ---------------------------------------------------------------------------
# Leakage: index of every v0.2 test/dev item, the remover, and the verifier
# ---------------------------------------------------------------------------
@dataclass
class LeakageIndex:
    """Every v0.2 test/dev item, indexed by record key, state hash and (state, question) hash."""

    record_keys: dict[str, str] = field(default_factory=dict)
    state_hashes: dict[str, str] = field(default_factory=dict)
    content_hashes: dict[str, str] = field(default_factory=dict)
    by_split: Counter[str] = field(default_factory=Counter)
    by_source: Counter[str] = field(default_factory=Counter)
    files: dict[str, int] = field(default_factory=dict)
    n_items: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "definition": "every item whose split is test or dev under data/bench/v0.2 "
                          "(tier1 + fresh + supplementary)",
            "n_items": self.n_items,
            "by_split": dict(sorted(self.by_split.items())),
            "by_source": dict(sorted(self.by_source.items())),
            "files": dict(sorted(self.files.items())),
            "n_distinct_record_keys": len(self.record_keys),
            "n_distinct_state_hashes": len(self.state_hashes),
            "n_distinct_content_hashes": len(self.content_hashes),
        }


def build_leakage_index(bench_dir: Path, *, splits: Sequence[str] = ("test", "dev")) -> LeakageIndex:
    """Index every v0.2 bench JSONL item whose split is test or dev (no network)."""
    index = LeakageIndex()
    paths = (
        sorted((bench_dir / "tier1").glob("*.jsonl"))
        + sorted((bench_dir / "fresh").glob("*.jsonl"))
        + sorted((bench_dir / "supplementary").glob("*.jsonl"))
    )
    if not paths:
        raise FileNotFoundError(f"no v0.2 benchmark JSONL under {bench_dir}")
    for path in paths:
        rows, report = read_jsonl(path, Item)
        if report.n_dropped:
            raise SystemExit(f"{path}: {report.n_dropped} unreadable rows — refusing to index it")
        kept = 0
        for item in rows:
            assert isinstance(item, Item)
            if str(item.split) not in splits:
                continue
            kept += 1
            index.record_keys.setdefault(record_key(item), item.item_id)
            index.state_hashes.setdefault(state_hash(item), item.item_id)
            index.content_hashes.setdefault(state_question_hash(item), item.item_id)
            index.by_split[str(item.split)] += 1
            index.by_source[item.source] += 1
        index.files[str(path.relative_to(bench_dir))] = kept
        index.n_items += kept
    return index


def leakage_reasons(item: Item, index: LeakageIndex, *, window_start: date = WINDOW_START) -> list[str]:
    """The four S6 leakage kinds (the state-hash kind has two variants), per item."""
    reasons: list[str] = []
    if record_key(item) in index.record_keys:
        reasons.append(R_RECORD_ID)
    if state_hash(item) in index.state_hashes:
        reasons.append(R_STATE_HASH)
    if state_question_hash(item) in index.content_hashes:
        reasons.append(R_STATE_QUESTION_HASH)
    if item.record_date >= window_start:
        reasons.append(R_DATE)
    if item.template_id in HELD_OUT_TEMPLATES:
        reasons.append(R_HELD_OUT)
    return reasons


def remove_leaks(
    items: Sequence[Item], index: LeakageIndex, *, window_start: date = WINDOW_START,
    source_origin: dict[str, str] | None = None,
) -> tuple[list[Item], dict[str, Any]]:
    """Remove every item that fails the leakage check; count removals by reason and source.

    ``source_origin`` optionally maps ``item_id -> input component`` so the report can show
    which input each removal came from (the mixture accounting depends on it).
    """
    kept: list[Item] = []
    by_reason: Counter[str] = Counter()
    by_source: Counter[str] = Counter()
    by_source_reason: Counter[str] = Counter()
    by_origin: Counter[str] = Counter()
    by_origin_reason: Counter[str] = Counter()
    examples: list[dict[str, Any]] = []
    for item in items:
        reasons = leakage_reasons(item, index, window_start=window_start)
        if not reasons:
            kept.append(item)
            continue
        origin = (source_origin or {}).get(item.item_id, "unknown")
        by_source[item.source] += 1
        by_origin[origin] += 1
        for reason in reasons:
            by_reason[reason] += 1
            by_source_reason[f"{item.source}|{reason}"] += 1
            by_origin_reason[f"{origin}|{reason}"] += 1
        if len(examples) < 20:
            examples.append(
                {
                    "item_id": item.item_id,
                    "source": item.source,
                    "template_id": item.template_id,
                    "record_date": item.record_date.isoformat(),
                    "origin": origin,
                    "reasons": reasons,
                }
            )
    report = {
        "n_input": len(items),
        "n_kept": len(kept),
        "n_removed_total": len(items) - len(kept),
        "n_removed_source_record_id": by_reason.get(R_RECORD_ID, 0),
        "n_removed_state_hash": by_reason.get(R_STATE_HASH, 0),
        "n_removed_state_question_hash": by_reason.get(R_STATE_QUESTION_HASH, 0),
        "n_removed_date": by_reason.get(R_DATE, 0),
        "n_removed_held_out_template": by_reason.get(R_HELD_OUT, 0),
        "by_reason": dict(sorted(by_reason.items())),
        "by_source": dict(sorted(by_source.items())),
        "by_source_reason": dict(sorted(by_source_reason.items())),
        "by_origin": dict(sorted(by_origin.items())),
        "by_origin_reason": dict(sorted(by_origin_reason.items())),
        "examples": examples,
    }
    return kept, report


def verify_no_leaks(
    items: Sequence[Item], index: LeakageIndex, *, window_start: date = WINDOW_START
) -> dict[str, Any]:
    """Re-run the leakage check on (the written) rows; every count must be zero."""
    by_reason: Counter[str] = Counter()
    by_source: Counter[str] = Counter()
    offending: list[dict[str, Any]] = []
    for item in items:
        reasons = leakage_reasons(item, index, window_start=window_start)
        if not reasons:
            continue
        for reason in reasons:
            by_reason[reason] += 1
        by_source[item.source] += 1
        if len(offending) < 20:
            offending.append(
                {
                    "item_id": item.item_id,
                    "source": item.source,
                    "template_id": item.template_id,
                    "record_date": item.record_date.isoformat(),
                    "reasons": reasons,
                }
            )
    report = {
        "n_items_checked": len(items),
        "n_failing_total": sum(by_source.values()),
        "by_reason": dict(sorted(by_reason.items())),
        "by_source": dict(sorted(by_source.items())),
        "examples": offending,
    }
    report["ok"] = report["n_failing_total"] == 0
    return report


def check_disjointness(left: Sequence[Item], right: Sequence[Item]) -> dict[str, Any]:
    """Count items shared by two row sets, by item id, state hash, content hash and record key."""
    def _keys(rows: Sequence[Item]) -> dict[str, set[str]]:
        return {
            "item_id": {r.item_id for r in rows},
            "state_hash": {state_hash(r) for r in rows},
            "state_question_hash": {state_question_hash(r) for r in rows},
            "record_key": {record_key(r) for r in rows},
        }

    left_keys, right_keys = _keys(left), _keys(right)
    shared = {kind: sorted(left_keys[kind] & right_keys[kind]) for kind in left_keys}
    return {
        "n_left": len(left),
        "n_right": len(right),
        "n_shared_items": len(shared["item_id"]),
        "n_shared_by_kind": {kind: len(v) for kind, v in shared.items()},
        "examples_shared_records": shared["record_key"][:20],
        "ok": not any(shared.values()),
    }


# ---------------------------------------------------------------------------
# Balance + cap per template
# ---------------------------------------------------------------------------
def balance_and_cap(
    items: Sequence[Item], *, cfg: dict[str, Any], salt: str, cap_per_template: int = CAP_PER_TEMPLATE
) -> tuple[list[Item], dict[str, Any]]:
    """Balance every gold class per template, then guarantee <= ``cap_per_template`` items.

    ``balance_classes`` caps each class at ``K = min(target, smallest class)``. Calling it
    with ``target = cap // n_classes`` therefore makes ``n_classes * K <= cap``, so the cap
    is met by construction: no post-hoc uniform subsample is needed (which would undo the
    balance). Templates with a single gold class are dropped by ``balance_classes`` with the
    reason ``single_gold_class_in_split``, which is kept in the report.

    ``cfg`` supplies the seed and the ``fresh_balance_min_class`` reporting floor.
    """
    groups: dict[str, list[Item]] = defaultdict(list)
    for item in items:
        groups[item.template_id].append(item)

    kept: list[Item] = []
    report: dict[str, Any] = {"cap_per_template": cap_per_template, "templates": {}}
    for template_id, rows in sorted(groups.items()):
        by_class = Counter(str(r.gold) for r in rows)
        n_classes = len(by_class)
        smallest = min(by_class.values())
        target = min(cap_per_template // n_classes, smallest) if n_classes else 0
        picked, bal = balance_classes(
            rows, cfg=cfg, salt=f"{salt}:{template_id}", target_per_class=target
        )
        kept.extend(picked)
        report["templates"][template_id] = {
            "n_input": len(rows),
            "n_kept": len(picked),
            "n_classes": n_classes,
            "smallest_class_before": smallest,
            "target_per_class": target,
            "n_over_cap": max(0, len(picked) - cap_per_template),
            "balance": bal,
        }
    report["n_input_total"] = len(items)
    report["n_kept_total"] = len(kept)
    over = {t: v["n_over_cap"] for t, v in report["templates"].items() if v["n_over_cap"]}
    report["templates_over_cap"] = over
    return kept, report


# ---------------------------------------------------------------------------
# Stage 1a: ClinicalTrials.gov, first posted before the window
# ---------------------------------------------------------------------------
def fetch_studies_retrying(
    *, start: date, end: date, attempts: int = 8, base_wait_s: float = 8.0, **kwargs: Any
) -> list[dict[str, Any]]:
    """``clinicaltrials.fetch_studies`` with a backoff around the API's throttling.

    Measured 2026-10-06: when the CT.gov search endpoint is throttled it answers a burst
    with a mix of ``429`` and ``400 Search error``; ``fetch_studies`` only retries ``429``,
    so a bare call dies on the ``400``. The same request succeeds again after 15-45 s, so
    every retryable status (400/429/5xx) is retried here with a growing wait and the last
    error is re-raised if the endpoint never recovers.
    """
    import httpx

    retryable = {400, 408, 429, 500, 502, 503, 504}
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            return ct.fetch_studies(start=start, end=end, **kwargs)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code not in retryable:
                raise
            last = exc
            wait = base_wait_s * (attempt + 1)
            print(
                f"[ct] {start}..{end}: HTTP {exc.response.status_code} "
                f"({exc.response.text[:60]!r}); retry {attempt + 1}/{attempts} in {wait:.0f}s",
                flush=True,
            )
            time.sleep(wait)
    assert last is not None
    raise last


def build_prewindow_clinicaltrials(
    *,
    cfg: dict[str, Any],
    window_start: date = PREWINDOW_START,
    window_end: date = PREWINDOW_END,
    cache_dir: Path = CACHE_DIR,
    cap_per_template: int = CAP_PER_TEMPLATE,
    fetch: Callable[..., list[dict[str, Any]]] = fetch_studies_retrying,
    builder: Callable[..., tuple[list[Any], dict[str, int], list[str]]] = ct.build_items,
    refetch: bool = False,
    slice_months: int = 3,
) -> tuple[list[Item], dict[str, Any]]:
    """Fetch and build CT.gov pre-window items; returns ``(items, report)``.

    The window is fetched in ``slice_months`` slices so a single API query never has to page
    through three years and a failure costs one slice, not the build. ``clinicaltrials.
    build_items`` holds no cross-record state (no distractor pool, no RNG), so slicing the
    *fetch* and then building once over the concatenation is identical to one whole-window
    build.
    """
    t0 = time.perf_counter()
    cache_dir = Path(cache_dir)
    slices = _date_slices(window_start, window_end, months=slice_months)
    fetched: list[dict[str, Any]] = []
    slice_report: list[dict[str, Any]] = []
    n_cached = n_downloaded = 0
    for slice_start, slice_end in slices:
        cache_path = cache_dir / "ct_raw" / f"ct_{slice_start.isoformat()}_{slice_end.isoformat()}.jsonl.gz"
        if cache_path.is_file() and not refetch:
            studies = _read_jsonl_gz(cache_path)
            n_cached += 1
        else:
            studies = fetch(start=slice_start, end=slice_end)
            _write_jsonl_gz(cache_path, studies)
            n_downloaded += 1
        slice_report.append(
            {"start": slice_start.isoformat(), "end": slice_end.isoformat(),
             "n_studies": len(studies), "cached": cache_path.is_file() and not refetch}
        )
        fetched.extend(studies)

    items, drops, notes = builder(fetched, cfg=cfg, window_start=window_start, window_end=window_end)

    by_template_all = Counter(str(i.template_id) for i in items)
    held_out_excluded = Counter(str(i.template_id) for i in items if i.template_id in HELD_OUT_TEMPLATES)
    kept_build = [i for i in items if i.template_id not in HELD_OUT_TEMPLATES]
    normalised = [as_train(i)[0] for i in kept_build]
    n_split_rewritten = sum(1 for i in kept_build if str(i.split) != "train")
    balanced, balance_report = balance_and_cap(
        normalised, cfg=cfg, salt="prewindow_clinicaltrials", cap_per_template=cap_per_template
    )
    report = {
        "source": "clinicaltrials",
        "filter": f"AREA[StudyFirstPostDate]RANGE[{window_start.isoformat()}, {window_end.isoformat()}]",
        "slice_months": slice_months,
        "n_slices": len(slices),
        "n_slices_cached": n_cached,
        "n_slices_downloaded": n_downloaded,
        "slices": slice_report,
        "n_studies_fetched": len(fetched),
        "n_items_built": len(items),
        "items_built_by_template": dict(sorted(by_template_all.items())),
        "n_held_out_items_excluded": sum(held_out_excluded.values()),
        "held_out_items_excluded_by_template": dict(sorted(held_out_excluded.items())),
        "n_split_rewritten_to_train": n_split_rewritten,
        "n_items_after_held_out_exclusion": len(kept_build),
        "build_drops": dict(sorted(drops.items())),
        "build_notes": notes,
        "balance_and_cap": balance_report,
        "wall_clock_s": round(time.perf_counter() - t0, 1),
    }
    return list(balanced), report


# ---------------------------------------------------------------------------
# Stage 1b: openFDA labels first effective before the window
# ---------------------------------------------------------------------------
def hub_class_values(index: dict[str, Any], *, max_df: int) -> set[str]:
    """MoA/PE values carried by more than ``max_df`` classes.

    Exactly the rule of ``scripts/bench/build_v0_2.py::_hub_class_values`` (S1's near-miss
    fix): a value most of the corpus carries cannot be why two classes are near-misses.
    Measured on the full openFDA class index: ``Cell-mediated Immunity [PE]`` occurs on 67
    classes, ``Increased Histamine Release [PE]`` on 65.
    """
    df: dict[str, int] = {}
    for entry in (index.get("classes") or {}).values():
        for value in set(entry.get("moa") or []) | set(entry.get("pe") or []):
            df[value] = df.get(value, 0) + 1
    return {value for value, n in df.items() if n > max_df}


def make_near_miss(index: dict[str, Any], *, max_value_df: int = 20) -> Callable[[str, set[str]], list[dict[str, Any]]]:
    """The near-miss closure of ``build_v0_2.build_openfda_class_v2`` (rule reused verbatim)."""
    hubs = hub_class_values(index, max_df=max_value_df)

    def near_miss(gold_class: str, own: set[str]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for candidate in fda_classes.near_miss_candidates(gold_class, index, exclude=own, max_candidates=200):
            if candidate["rule"] not in ("shared_moa", "shared_pe"):
                continue
            specific = [value for value in candidate["shared"] if value not in hubs]
            if not specific:
                continue
            out.append({**candidate, "shared": specific})
        return out

    return near_miss


def fetch_openfda_months(
    *, window_start: date, window_end: date, cache_dir: Path, refetch: bool = False
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Fetch openFDA labels month by month (cached), deduplicated by ``set_id`` (first wins).

    ``openfda.fetch_labels`` slices long windows by month for the same reason (the API
    rejects ``skip > 25,000``); doing the slicing here additionally caches each month, and
    the ``set_id`` dedupe reproduces what ``fetch_labels`` does across its own slices — a
    label effective in two months is kept once, in its earliest month.
    """
    months = _date_slices(window_start, window_end, months=1)
    labels: list[dict[str, Any]] = []
    seen: set[str] = set()
    n_duplicates = 0
    month_report: list[dict[str, Any]] = []
    n_cached = n_downloaded = 0
    for month_start, month_end in months:
        cache_path = cache_dir / "fda_raw" / f"fda_{month_start.strftime('%Y-%m')}.jsonl.gz"
        if cache_path.is_file() and not refetch:
            batch = _read_jsonl_gz(cache_path)
            n_cached += 1
        else:
            batch = openfda.fetch_labels(start=month_start, end=month_end)
            _write_jsonl_gz(cache_path, batch)
            n_downloaded += 1
        n_new = 0
        for label in batch:
            key = str(label.get("set_id") or label.get("id") or "")
            if key and key in seen:
                n_duplicates += 1
                continue
            seen.add(key)
            labels.append(label)
            n_new += 1
        month_report.append(
            {"month": month_start.strftime("%Y-%m"), "n_labels": len(batch), "n_new_set_ids": n_new}
        )
    return labels, {
        "n_months": len(months),
        "n_months_cached": n_cached,
        "n_months_downloaded": n_downloaded,
        "n_labels_after_set_id_dedupe": len(labels),
        "n_duplicate_set_id_versions_dropped": n_duplicates,
        "months": month_report,
    }


def build_prewindow_openfda(
    *,
    cfg: dict[str, Any],
    window_start: date = PREWINDOW_START,
    window_end: date = PREWINDOW_END,
    cache_dir: Path = CACHE_DIR,
    class_index_path: Path = Path("/workspace/tmp/openfda/class_index_full.json"),
    cap_per_template: int = CAP_PER_TEMPLATE,
    labels: list[dict[str, Any]] | None = None,
    label_fetcher: Callable[..., tuple[list[dict[str, Any]], dict[str, Any]]] | None = None,
    refetch: bool = False,
) -> tuple[list[Item], dict[str, Any]]:
    """Build openFDA pre-window items: class ``_v1``, route ``_v1`` and class ``_v2``.

    ``fda_boxed_warning_noul_v1`` is held out (D14) and is removed **by template id** with a
    counted reason before balancing. The ``_v2`` near-miss rule is the S1 rule, read out of
    ``build_v0_2.build_openfda_class_v2``: only ``shared_moa``/``shared_pe`` candidates, and
    only via a value that is not a corpus hub (> 20 classes).
    """
    t0 = time.perf_counter()
    fetch_report: dict[str, Any]
    if labels is None:
        fetcher = label_fetcher or (
            lambda **kw: fetch_openfda_months(cache_dir=Path(cache_dir), refetch=refetch, **kw)
        )
        labels, fetch_report = fetcher(window_start=window_start, window_end=window_end)
    else:
        fetch_report = {"n_labels_after_set_id_dedupe": len(labels), "provided": True}

    items_v1, drops_v1, notes_v1 = openfda.build_items(
        labels, cfg=cfg, window_start=window_start, window_end=window_end
    )
    index = fda_classes.load_index(Path(class_index_path)) if Path(class_index_path).is_file() else None
    if index is None:
        raise FileNotFoundError(f"openFDA class index missing at {class_index_path}")
    items_v2, drops_v2, notes_v2 = openfda.build_class_v2_items(
        labels,
        cfg=cfg,
        window_start=window_start,
        window_end=window_end,
        near_miss=make_near_miss(index, max_value_df=20),
        class_index_meta=index.get("meta", {}),
    )
    items = [*items_v1, *items_v2]
    by_template_all = Counter(str(i.template_id) for i in items)
    held_out_excluded = Counter(str(i.template_id) for i in items if i.template_id in HELD_OUT_TEMPLATES)
    kept_build = [i for i in items if i.template_id not in HELD_OUT_TEMPLATES]
    normalised = [as_train(i)[0] for i in kept_build]
    n_split_rewritten = sum(1 for i in kept_build if str(i.split) != "train")
    balanced, balance_report = balance_and_cap(
        normalised, cfg=cfg, salt="prewindow_openfda", cap_per_template=cap_per_template
    )
    drops: dict[str, int] = {}
    for reason, n in {**drops_v1}.items():
        drops[f"v1:{reason}"] = drops.get(f"v1:{reason}", 0) + n
    for reason, n in {**drops_v2}.items():
        drops[f"v2:{reason}"] = drops.get(f"v2:{reason}", 0) + n
    report = {
        "source": "openfda",
        "filter": "effective_time in [2023-01-01, 2026-02-28]",
        "fetch": fetch_report,
        "class_index": {"path": str(class_index_path), **index.get("meta", {})},
        "near_miss_rule": {
            "rules_allowed": ["shared_moa", "shared_pe"],
            "max_value_df": 20,
            "source": "scripts/bench/build_v0_2.py::build_openfda_class_v2 (reused verbatim)",
        },
        "n_items_built": len(items),
        "items_built_by_template": dict(sorted(by_template_all.items())),
        "n_held_out_items_excluded": sum(held_out_excluded.values()),
        "held_out_items_excluded_by_template": dict(sorted(held_out_excluded.items())),
        "n_split_rewritten_to_train": n_split_rewritten,
        "n_items_after_held_out_exclusion": len(kept_build),
        "build_drops": dict(sorted(drops.items())),
        "build_notes": [*notes_v1, *notes_v2],
        "balance_and_cap": balance_report,
        "wall_clock_s": round(time.perf_counter() - t0, 1),
    }
    return list(balanced), report


# ---------------------------------------------------------------------------
# Stage 1c: PubMed — measured availability of a pre-window build
# ---------------------------------------------------------------------------
def measure_pubmed_prewindow(
    *,
    window_start: date = PREWINDOW_START,
    window_end: date = PREWINDOW_END,
    cache_dir: Path = CACHE_DIR,
    timebox_minutes: float = PUBMED_TIMEBOX_MINUTES,
    sample_size: int = 1,
) -> dict[str, Any]:
    """Measure whether the prescribed PubMed build fits its box; never silently subsample.

    The ADVISORY's source is "records from update/baseline files dated before 2026-03-01".
    The FTP ``updatefiles`` directory currently holds only the current year's files, so the
    only files that can carry Entrez dates inside the window are the annual baseline files.
    This downloads ``sample_size`` of them, parses them with the builder's own
    ``iter_records``, and projects the full cost. It returns ``NOT MEASURED`` when the
    projection exceeds ``timebox_minutes``; the measured numbers are what justify the call.
    """
    import httpx

    t0 = time.perf_counter()
    report: dict[str, Any] = {
        "source": "pubmed",
        "window_start": window_start.isoformat(),
        "window_end": window_end.isoformat(),
        "timebox_minutes": timebox_minutes,
        "status": "NOT MEASURED",
        "reason": "measurement did not complete",
    }
    cache_dir = Path(cache_dir)
    with httpx.Client(timeout=120.0, follow_redirects=True) as http:
        update_names = _listing(http, PUBMED_UPDATE_BASE)
        baseline_names = _listing(http, PUBMED_BASELINE_BASE)
        report["updatefiles_listing"] = {
            "n_files": len(update_names),
            "years": sorted({n[6:8] for n in update_names}),
            "first": update_names[:2],
            "last": update_names[-2:],
            "url": PUBMED_UPDATE_BASE,
        }
        report["baseline_listing"] = {
            "n_files": len(baseline_names),
            "years": sorted({n[6:8] for n in baseline_names}),
            "first": baseline_names[:2],
            "last": baseline_names[-2:],
            "url": PUBMED_BASELINE_BASE,
        }
        samples = baseline_names[:sample_size]
        report["sample"] = []
        total_bytes = 0
        total_download_s = 0.0
        total_parse_s = 0.0
        total_records = total_in_window = 0
        for name in samples:
            path = cache_dir / "pubmed_baseline" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            dl0 = time.perf_counter()
            if path.is_file():
                n_bytes = path.stat().st_size
            else:
                n_bytes = 0
                with http.stream("GET", PUBMED_BASELINE_BASE + name) as response:
                    response.raise_for_status()
                    with path.open("wb") as fh:
                        for chunk in response.iter_bytes(chunk_size=1 << 20):
                            fh.write(chunk)
                            n_bytes += len(chunk)
            download_s = time.perf_counter() - dl0
            parse0 = time.perf_counter()
            n_records = n_in_window = 0
            for record in pubmed.iter_records([path]):
                n_records += 1
                if window_start <= record.entrez_date <= window_end:
                    n_in_window += 1
            parse_s = time.perf_counter() - parse0
            total_bytes += n_bytes
            total_download_s += download_s
            total_parse_s += parse_s
            total_records += n_records
            total_in_window += n_in_window
            report["sample"].append(
                {
                    "name": name,
                    "bytes": n_bytes,
                    "download_s": round(download_s, 1),
                    "parse_s": round(parse_s, 1),
                    "n_records": n_records,
                    "n_records_in_prewindow": n_in_window,
                }
            )

    n_files = len(baseline_names)
    per_file_s = (total_download_s + total_parse_s) / max(1, len(samples))
    projected_s = per_file_s * n_files
    report["projection"] = {
        "n_baseline_files": n_files,
        "mean_bytes_per_file": round(total_bytes / max(1, len(samples))),
        "projected_total_gb": round(total_bytes / max(1, len(samples)) * n_files / 1e9, 1),
        "mean_download_s_per_file": round(total_download_s / max(1, len(samples)), 1),
        "mean_parse_s_per_file": round(total_parse_s / max(1, len(samples)), 1),
        "projected_download_and_parse_hours": round(projected_s / 3600, 1),
        "measured_in_window_records_per_file": round(total_in_window / max(1, len(samples))),
        "projected_in_window_records": round(total_in_window / max(1, len(samples)) * n_files),
    }
    report["wall_clock_s"] = round(time.perf_counter() - t0, 1)
    if projected_s > timebox_minutes * 60:
        report["reason"] = (
            f"PubMed update files covering 2023-01-01..2026-02-28 no longer exist on the FTP "
            f"(the updatefiles listing holds {len(update_names)} files, all of them the "
            f"current year); "
            f"the only remaining source is the {n_files}-file annual baseline, and a measured "
            f"sample of {len(samples)} file(s) projects "
            f"{report['projection']['projected_download_and_parse_hours']} h "
            f"({report['projection']['projected_total_gb']} GB down + parse) for all of them, "
            f"over the {timebox_minutes:.0f}-minute box. Reading a subset would be a silent "
            f"subsample of the window, so no PubMed training items are built."
        )
    else:
        report["status"] = "WITHIN_BOX"
        report["reason"] = "projection fits the timebox; a full build can be attempted"
    return report


def _listing(http: Any, base: str) -> list[str]:
    response = http.get(base)
    response.raise_for_status()
    names = sorted(
        {
            line.split('href="')[-1].split('"')[0]
            for line in response.text.splitlines()
            if line.split('href="')[-1].split('"')[0].endswith(".xml.gz")
        }
    )
    return names


# ---------------------------------------------------------------------------
# Stage 2: the mix, dev, and the manifest
# ---------------------------------------------------------------------------
def _load_items(path: Path) -> list[Item]:
    rows, report = read_jsonl(path, Item)
    if report.n_dropped:
        raise SystemExit(f"{path}: {report.n_dropped} unreadable rows — refusing to build on it")
    return [row for row in rows if isinstance(row, Item)]


def load_component(path: Path, origin: str, *, normalise_split: bool = True) -> tuple[list[Item], dict[str, Any]]:
    """Load one input component, rewriting every row's split to ``train``."""
    rows = _load_items(path)
    changed = 0
    out: list[Item] = []
    for row in rows:
        item, was = as_train(row) if normalise_split else (row, False)
        changed += int(was)
        out.append(item)
    return out, {
        "path": str(path),
        "origin": origin,
        "n_rows": len(rows),
        "n_split_rewritten_to_train": changed,
        "by_template": dict(sorted(Counter(r.template_id for r in out).items())),
        "by_qtype": dict(sorted(Counter(str(r.qtype) for r in out).items())),
    }


def build_dev(bench_dir: Path = BENCH_DIR) -> tuple[list[Item], dict[str, Any]]:
    """v0.2 tier-1 dev + v0.2 fresh dev, held-out templates removed."""
    rows: list[Item] = []
    sources: dict[str, int] = {}
    for sub in ("tier1", "fresh"):
        for path in sorted((bench_dir / sub).glob("*.jsonl")):
            items = [i for i in _load_items(path) if str(i.split) == "dev"]
            rows.extend(items)
            sources[f"{sub}/{path.name}"] = len(items)
    held_out_removed = Counter(i.template_id for i in rows if i.template_id in HELD_OUT_TEMPLATES)
    dev = [i for i in rows if i.template_id not in HELD_OUT_TEMPLATES]
    return dev, {
        "definition": "every v0.2 tier-1/fresh item with split=dev, held-out templates removed",
        "n_input_dev_items": len(rows),
        "n_held_out_removed": sum(held_out_removed.values()),
        "held_out_removed_by_template": dict(sorted(held_out_removed.items())),
        "n_dev_items": len(dev),
        "by_file": dict(sorted(sources.items())),
    }


def assemble_mix(
    *,
    train_dir: Path = TRAIN_DIR,
    bench_dir: Path = BENCH_DIR,
    cfg: dict[str, Any],
    cap_per_template: int = CAP_PER_TEMPLATE,
    components: Sequence[str] = ("tier1_train", "prewindow_consistency", "prewindow_structured"),
) -> dict[str, Any]:
    """Assemble ``train.jsonl``, ``dev.jsonl`` and ``manifest.json``; run the leakage check."""
    from meddecide.utils.hashing import file_sha256

    t0 = time.perf_counter()
    train_dir = Path(train_dir)
    index = build_leakage_index(Path(bench_dir))

    # ---- inputs -------------------------------------------------------------
    loaded: dict[str, list[Item]] = {}
    component_report: dict[str, Any] = {}
    for name in components:
        rows, report = load_component(train_dir / f"{name}.jsonl", name)
        loaded[name] = rows
        component_report[name] = report

    # ``prewindow_structured`` is the union of the per-source files written by stage 1. If it
    # is absent (stage 1 skipped) it is simply not part of the mix.
    mixed: list[Item] = []
    origin: dict[str, str] = {}
    for name in components:
        for row in loaded[name]:
            mixed.append(row)
            origin[row.item_id] = name

    # ---- balance + cap the reused pre-window consistency component ----------
    # S2 wrote it uncapped (99,822 items of one template). The S6 rule — balanced per
    # template, <= 20,000 items per template — is applied to it here, uniformly with the
    # newly built components, because otherwise a single template would dominate the mix.
    # The input file is untouched; the drops are counted in the manifest.
    consistency_cap_report: dict[str, Any] = {}
    if "prewindow_consistency" in loaded:
        capped, consistency_cap_report = balance_and_cap(
            loaded["prewindow_consistency"], cfg=cfg, salt="prewindow_consistency", cap_per_template=cap_per_template
        )
        mixed = [r for r in mixed if origin.get(r.item_id) != "prewindow_consistency"]
        for row in capped:
            origin[row.item_id] = "prewindow_consistency"
        mixed.extend(capped)

    n_mixed_before_leakage = len(mixed)
    train_rows, leakage_pass1 = remove_leaks(mixed, index, source_origin=origin)

    # ---- write, then re-run the check on the written file -------------------
    train_path = write_jsonl(train_dir / "train.jsonl", train_rows)
    written = _load_items(train_path)
    leakage_written = verify_no_leaks(written, index)

    dev_rows, dev_report = build_dev(Path(bench_dir))
    dev_path = write_jsonl(train_dir / "dev.jsonl", dev_rows)
    dev_written = _load_items(dev_path)
    # The train leakage rule ("no v0.2 test/dev item") cannot be applied to dev.jsonl: dev is
    # made *of* v0.2 dev items, so every row would flag itself. The same four key kinds are
    # therefore checked in the reverse direction — dev against the written train.jsonl — plus
    # the split and held-out-template invariants.
    disjoint = check_disjointness(written, dev_written)
    dev_held_out = Counter(r.template_id for r in dev_written if r.template_id in HELD_OUT_TEMPLATES)
    dev_checks = {
        "definition": "train.jsonl vs dev.jsonl, by the same four key kinds as the leakage check",
        "train_dev_disjoint": disjoint,
        "n_dev_items": len(dev_written),
        "n_dev_test_split_items": sum(1 for r in dev_written if str(r.split) != "dev"),
        "n_dev_held_out_template_items": sum(dev_held_out.values()),
        "dev_held_out_by_template": dict(sorted(dev_held_out.items())),
        "every_dev_row_split_dev": all(str(r.split) == "dev" for r in dev_written),
        "no_held_out_template_in_dev": not dev_held_out,
        "ok": disjoint["ok"] and not dev_held_out
        and all(str(r.split) == "dev" for r in dev_written),
    }

    # ---- manifest -----------------------------------------------------------
    files = {
        "train": train_path,
        "dev": dev_path,
        "prewindow_structured": train_dir / "prewindow_structured.jsonl",
    }
    files = {k: v for k, v in files.items() if v.is_file()}
    rows_by_file = {
        "train": written,
        "dev": dev_written,
        "prewindow_structured": _load_items(train_dir / "prewindow_structured.jsonl")
        if (train_dir / "prewindow_structured.jsonl").is_file()
        else [],
    }
    manifest = build_manifest(files, {k: rows_by_file[k] for k in files})

    mixture = {
        "components": component_report,
        "n_used_from_each_component": dict(
            sorted(Counter(origin[r.item_id] for r in written).items())
        ),
        "n_input_total": n_mixed_before_leakage,
        "n_train_total": len(written),
        "prewindow_consistency_balance_and_cap": consistency_cap_report,
    }
    checks = {
        "leakage_pass1_removals": leakage_pass1,
        "leakage_recheck_on_written_train": leakage_written,
        "dev_checks": dev_checks,
        "counts_close": {
            "train_rows_sum_by_source": sum(manifest["files"]["train"]["count_by_source"].values()),
            "train_n_rows": len(written),
            "mixture_used_total": sum(mixture["n_used_from_each_component"].values()),
            "train_rows_by_template_sum": sum(
                n
                for src in manifest["counts"]["train"].values()
                for sp in src.values()
                for qt in sp.values()
                for n in qt.values()
            ),
        },
        "every_train_row_split_train": all(str(r.split) == "train" for r in written),
        "every_train_row_before_window": all(r.record_date < WINDOW_START for r in written),
        "no_held_out_template_in_train": all(r.template_id not in HELD_OUT_TEMPLATES for r in written),
    }
    checks["counts_close"]["closes"] = (
        checks["counts_close"]["train_n_rows"] == checks["counts_close"]["mixture_used_total"]
        == checks["counts_close"]["train_rows_sum_by_source"]
        == checks["counts_close"]["train_rows_by_template_sum"]
    )
    manifest["run_name"] = "S6_build_training_mix"
    manifest["built_at_utc"] = utcnow()
    manifest["window_rule"] = {
        "window_start": WINDOW_START.isoformat(),
        "rule": STRICT_BEFORE_WINDOW,
        "prewindow": {"start": PREWINDOW_START.isoformat(), "end": PREWINDOW_END.isoformat()},
        "tier1_rule": (
            "tier-1 rows carry their dataset's release date, not a per-record date "
            "(meddecide.bench.tier1.common); established test sets are not date-filtered. "
            "Every tier-1 release date is before 2026-03-01, so the rule holds and is not "
            "a window violation."
        ),
    }
    manifest["held_out_templates"] = list(HELD_OUT_TEMPLATES)
    manifest["seed"] = int(cfg.get("seed", 0))
    manifest["caps"] = {"cap_per_template": cap_per_template, "config": cfg.get("caps", {})}
    manifest["mixture"] = mixture
    manifest["dev"] = dev_report
    manifest["checks"] = checks
    manifest["by_source_template_qtype"] = {
        "train": flat_source_template_qtype(written),
        "dev": flat_source_template_qtype(dev_written),
    }
    manifest["gold_counts"] = {
        "train": dict(sorted(Counter(f"{r.template_id}|{r.gold}" for r in written).items())),
        "dev": dict(sorted(Counter(f"{r.template_id}|{r.gold}" for r in dev_written).items())),
    }
    manifest["leakage_index"] = index.to_dict()
    manifest["files_sha256"] = {k: file_sha256(v) for k, v in files.items()}
    # carry the stage-1 report (source builds, PubMed measurement) into the manifest
    stage1_path = train_dir / "prewindow_build_report.json"
    if stage1_path.is_file():
        manifest["prewindow_build"] = json.loads(stage1_path.read_text())
    manifest["wall_clock_s"] = round(time.perf_counter() - t0, 1)
    manifest["git_commit"] = git_commit()
    write_json(train_dir / "manifest.json", manifest)

    print(
        json.dumps(
            {
                "train_total": len(written),
                "dev_total": len(dev_written),
                "n_removed_pass1": leakage_pass1["n_removed_total"],
                "removed_by_reason": leakage_pass1["by_reason"],
                "recheck_failing": leakage_written["n_failing_total"],
                "train_dev_shared_items": disjoint["n_shared_items"],
                "mixture": mixture["n_used_from_each_component"],
                "wall_clock_s": manifest["wall_clock_s"],
            },
            indent=2,
        )
    )
    return manifest


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_prewindow_stage(args: argparse.Namespace, cfg: dict[str, Any]) -> dict[str, Any]:
    """Run stage 1 for the requested sources and write the component files."""
    t0 = time.perf_counter()
    train_dir = Path(args.train_dir)
    sources = set(args.sources)
    report: dict[str, Any] = {
        "built_at_utc": utcnow(),
        "window": {"start": PREWINDOW_START.isoformat(), "end": PREWINDOW_END.isoformat()},
        "cap_per_template": args.cap_per_template,
        "seed": int(cfg.get("seed", 0)),
        "sources": {},
    }
    components: list[Item] = []
    for source in ("clinicaltrials", "openfda", "pubmed"):
        if source not in sources:
            continue
        if source == "clinicaltrials":
            rows, source_report = build_prewindow_clinicaltrials(
                cfg=cfg, cache_dir=Path(args.cache_dir), cap_per_template=args.cap_per_template,
                refetch=args.refetch, slice_months=args.ct_slice_months,
            )
            write_jsonl(train_dir / "prewindow_clinicaltrials.jsonl", rows)
            report["sources"]["clinicaltrials"] = source_report
            components.extend(rows)
        elif source == "openfda":
            rows, source_report = build_prewindow_openfda(
                cfg=cfg, cache_dir=Path(args.cache_dir), cap_per_template=args.cap_per_template,
                refetch=args.refetch,
            )
            write_jsonl(train_dir / "prewindow_openfda.jsonl", rows)
            report["sources"]["openfda"] = source_report
            components.extend(rows)
        else:
            source_report = measure_pubmed_prewindow(
                cache_dir=Path(args.cache_dir), timebox_minutes=args.pubmed_timebox_minutes
            )
            report["sources"]["pubmed"] = source_report
            if source_report["status"] == "NOT MEASURED":
                report["sources"]["pubmed"]["items_built"] = 0
    if components or any(s in sources for s in ("clinicaltrials", "openfda")):
        write_jsonl(train_dir / "prewindow_structured.jsonl", components)
        report["n_prewindow_structured_items"] = len(components)
        report["by_source"] = dict(sorted(Counter(r.source for r in components).items()))
        report["by_template"] = dict(sorted(Counter(r.template_id for r in components).items()))
        report["by_qtype"] = dict(sorted(Counter(str(r.qtype) for r in components).items()))
        report["checks"] = {
            "every_row_split_train": all(str(r.split) == "train" for r in components),
            "every_row_before_window": all(r.record_date < WINDOW_START for r in components),
            "no_held_out_template": all(r.template_id not in HELD_OUT_TEMPLATES for r in components),
            "no_template_over_cap": all(
                n <= args.cap_per_template for n in Counter(r.template_id for r in components).values()
            ),
        }
    report["wall_clock_s"] = round(time.perf_counter() - t0, 1)
    report["git_commit"] = git_commit()
    write_json(train_dir / "prewindow_build_report.json", report)
    print(json.dumps({"stage": "prewindow", "summary": {k: v for k, v in report.items() if k != "sources"}}, indent=2))
    for source, source_report in report["sources"].items():
        print(
            json.dumps(
                {
                    "source": source,
                    "status": source_report.get("status", "BUILT"),
                    "n_items_built": source_report.get("n_items_built"),
                    "n_prewindow_items": (
                        source_report.get("n_prewindow_structured_items")
                        or source_report.get("balance_and_cap", {}).get("n_kept_total")
                    ),
                    "held_out_excluded": source_report.get("n_held_out_items_excluded"),
                    "reason": source_report.get("reason", ""),
                },
                indent=2,
            )
        )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=["prewindow", "mix", "all"], default="all")
    parser.add_argument("--sources", nargs="*", default=["clinicaltrials", "openfda", "pubmed"])
    parser.add_argument("--config", type=Path, default=Path("configs/bench_v0_2.yaml"))
    parser.add_argument("--train-dir", type=Path, default=TRAIN_DIR)
    parser.add_argument("--bench-dir", type=Path, default=BENCH_DIR)
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    parser.add_argument("--cap-per-template", type=int, default=CAP_PER_TEMPLATE)
    parser.add_argument("--pubmed-timebox-minutes", type=float, default=PUBMED_TIMEBOX_MINUTES)
    parser.add_argument("--ct-slice-months", type=int, default=3,
                        help="ClinicalTrials.gov fetch slice length in months")
    parser.add_argument("--refetch", action="store_true", help="ignore cached raw fetches")
    args = parser.parse_args(argv)

    t0 = time.perf_counter()
    cfg = yaml.safe_load(args.config.read_text())
    Path(args.train_dir).mkdir(parents=True, exist_ok=True)
    if args.stage in ("prewindow", "all"):
        build_prewindow_stage(args, cfg)
    if args.stage in ("mix", "all"):
        assemble_mix(
            train_dir=Path(args.train_dir), bench_dir=Path(args.bench_dir), cfg=cfg,
            cap_per_template=args.cap_per_template,
        )
    prov = Provenance(
        run_name=f"S6_build_training_mix_{args.stage}",
        command=" ".join([sys.executable, *sys.argv]),
        gpu=gpu_name(),
        seed=int(cfg.get("seed", 0)),
        config={
            "stage": args.stage, "sources": sorted(args.sources),
            "cap_per_template": args.cap_per_template, "cache_dir": str(args.cache_dir),
            "train_dir": str(args.train_dir), "bench_dir": str(args.bench_dir),
            "config_file": str(args.config),
        },
    )
    prov.wall_clock_s = round(time.perf_counter() - t0, 1)
    log_dir = Path("outputs/student_v0/S6/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    prov.finish().write(log_dir / f"build_training_mix_{args.stage}_provenance.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
