#!/usr/bin/env python
"""S5 — tier-1 training set: gold-only items from official *train* splits.

Builds the tier-1 half of the loop's training mix (ADVISORY S5 / D13): the official
**train** splits of MedQA, MedMCQA, SciFact (BEIR train qrels), NFCorpus (BEIR train
qrels) and the MedQuAD rows no tier-1 split used. Every item is built with the *same*
construction the v0.2 benchmark uses, so a training item and a benchmark item differ only
by split, and gold comes from a structured source field — never from a model, never from a
test or validation split.

Explicitly excluded (recorded in the manifest): PubMedQA ``pqa_artificial`` (heuristic
labels), MMLU (no medical train split), every test/validation split, ``cais/hle``, the S13
catalog, and any UMLS/SNOMED/MIMIC text.

Two mechanical guards run inside the build (both are acceptance criteria):

* **Leakage** — an item is removed (and counted, by reason and by source) when its
  normalised ``(state + question)`` hash appears in any v0.2 test/dev item, or when its
  ``(source, source_record_id)`` pair does.
* **Gold verification** — an independently written check re-derives each item's gold from
  the raw source row (:mod:`meddecide.bench.verify` is reused where it fits; the source
  mappings live in this file so a loader bug cannot hide itself). Mismatching items are
  dropped with a counted reason.

Usage:
    uv run python scripts/bench/build_tier1_train.py --out data/train/student_v0
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import yaml

from meddecide.bench.schema import Item, QuestionType, Split, make_item
from meddecide.bench.tier1 import mcq, medquad, relevance
from meddecide.bench.tier1.common import make_choice_item, subsample
from meddecide.bench.verify import VerificationReport, normalize_label
from meddecide.utils.hashing import file_sha256, normalize_text, stable_hash
from meddecide.utils.io import read_jsonl, write_json, write_jsonl
from meddecide.utils.provenance import git_commit, utcnow

# ---------------------------------------------------------------------------
# Sources (exactly the five the plan allows)
# ---------------------------------------------------------------------------
MEDQA_ID = mcq.MEDQA_ID
MEDMCQA_ID = mcq.MEDMCQA_ID
SCIFACT_ID = relevance.SCIFACT_ID
SCIFACT_QRELS_ID = "BeIR/scifact-qrels"
NFCORPUS_ID = relevance.NFCORPUS_ID
NFCORPUS_QRELS_ID = "BeIR/nfcorpus-qrels"
MEDQUAD_ID = medquad.MEDQUAD_ID

DATASETS: dict[str, str] = {
    "medqa": MEDQA_ID,
    "medmcqa": MEDMCQA_ID,
    "scifact": SCIFACT_ID,
    "scifact_qrels": SCIFACT_QRELS_ID,
    "nfcorpus": NFCORPUS_ID,
    "nfcorpus_qrels": NFCORPUS_QRELS_ID,
    "medquad": MEDQUAD_ID,
}

# Template ids fixed by S5. The score template is train-specific because the benchmark's
# `_v2` template is built on the *test* qrels (a different pool). The nfcorpus `noul`
# template keeps the loader's default id (`nfcorpus_relevant_noul_v1`) because S5 names no
# train-specific id for it; recorded in the manifest as a judgement call.
SCIFACT_TRAIN_NOUL_TEMPLATE = "scifact_relevant_noul_train_v1"
NFCORPUS_TRAIN_SCORE_TEMPLATE = "nfcorpus_graded_score_train_v1"
NFCORPUS_NOUL_TEMPLATE = "nfcorpus_relevant_noul_v1"

DEFAULT_MEDMCQA_CAP = 60_000
DEFAULT_MEDMCQA_DEV_CAP = 2_000  # the carve v0.1/v0.2 made out of MedMCQA train
LICENSE_FALLBACK = "UNKNOWN"

EXCLUDED_SOURCES: list[dict[str, str]] = [
    {"id": "qiaojin/PubMedQA", "split": "pqa_artificial",
     "reason": "heuristic (model-assisted) labels, not gold; pqa_labeled is test-only (D13)"},
    {"id": "cais/mmlu", "split": "all",
     "reason": "no medical train split; its test/validation splits are evaluation-only"},
    {"id": "cais/hle", "split": "test",
     "reason": "supplementary test set; never trained on (ADVISORY S4)"},
]


# ---------------------------------------------------------------------------
# Hugging Face helpers (pinned revision + licence, R7)
# ---------------------------------------------------------------------------
def _load(dataset_id: str, config: str | None, split: str, revision: str | None = None):
    from datasets import load_dataset

    kwargs: dict[str, Any] = {"split": split}
    if revision:
        kwargs["revision"] = revision
    return load_dataset(dataset_id, config, **kwargs) if config else load_dataset(dataset_id, **kwargs)


def _revision_of(dataset_id: str) -> str:
    from huggingface_hub import HfApi

    return HfApi().dataset_info(dataset_id).sha


def _license_of(dataset_id: str, notes: list[str]) -> str:
    """Read the licence the dataset card declares; UNKNOWN when it declares none."""
    try:
        from huggingface_hub import HfApi

        info = HfApi().dataset_info(dataset_id)
        tags = [t for t in (info.tags or []) if t.startswith("license:")]
        if tags:
            return tags[0].split(":", 1)[1]
        card = info.card_data.to_dict() if info.card_data else {}
        lic = card.get("license")
        if isinstance(lic, list):
            return ",".join(str(x) for x in lic) if lic else LICENSE_FALLBACK
        if lic:
            return str(lic)
    except Exception as exc:  # pragma: no cover - network dependent
        notes.append(f"license lookup for {dataset_id} failed: {type(exc).__name__}")
    return LICENSE_FALLBACK


def _qrels_by_query(qrels: Any) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for row in qrels:
        out.setdefault(str(row["query-id"]), {})[str(row["corpus-id"])] = int(row["score"])
    return out


def _beir_corpus_index(corpus: Any) -> dict[str, str]:
    """Passage text exactly as ``relevance._passage_text`` builds it (independent copy)."""
    return {
        str(r["_id"]): f"{str(r.get('title') or '').strip()}\n{str(r.get('text') or '').strip()}".strip()[
            : relevance.PASSAGE_CHARS
        ]
        for r in corpus
    }


def ids_digest(ids: set[str]) -> str:
    """sha256 over the sorted ids (a reproducibility fingerprint for an exclusion set)."""
    return stable_hash({"ids": "\n".join(sorted(ids))}, length=64)


# ---------------------------------------------------------------------------
# Leakage: normalised (state + question) hash and (source, record id)
# ---------------------------------------------------------------------------
def leakage_content_hash(state: str, question: str) -> str:
    """The exact hash S5 requires: stable_hash over the *normalised* state and question."""
    return stable_hash({"state": normalize_text(state), "question": normalize_text(question)})


def leakage_record_key(source: str, source_record_id: str) -> str:
    return f"{source}|{source_record_id}"


@dataclass
class LeakageIndex:
    """Every v0.2 test/dev item, indexed by content hash and by (source, record id)."""

    content_hashes: dict[str, str] = field(default_factory=dict)
    record_keys: dict[str, str] = field(default_factory=dict)
    by_split: Counter[str] = field(default_factory=Counter)
    by_source: Counter[str] = field(default_factory=Counter)
    files: dict[str, int] = field(default_factory=dict)
    n_items: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "hash_formula": "stable_hash({state: normalize_text(state), "
                            "question: normalize_text(question)})",
            "splits_used": ["test", "dev"],
            "n_items": self.n_items,
            "by_split": dict(sorted(self.by_split.items())),
            "by_source": dict(sorted(self.by_source.items())),
            "files": dict(sorted(self.files.items())),
            "n_distinct_content_hashes": len(self.content_hashes),
            "n_distinct_record_keys": len(self.record_keys),
        }


def build_leakage_index(bench_dir: Path, *, splits: Sequence[str] = ("test", "dev")) -> LeakageIndex:
    """Index every item of every v0.2 tier-1 and fresh JSONL whose split is test or dev."""
    index = LeakageIndex()
    paths = sorted((bench_dir / "tier1").glob("*.jsonl")) + sorted((bench_dir / "fresh").glob("*.jsonl"))
    if not paths:
        raise FileNotFoundError(f"no v0.2 benchmark JSONL under {bench_dir}")
    for path in paths:
        rows, report = read_jsonl(path, Item)
        report.check_closes()
        kept = 0
        for item in rows:
            assert isinstance(item, Item)
            if str(item.split) not in splits:
                continue
            kept += 1
            index.content_hashes.setdefault(leakage_content_hash(item.state, item.question), item.item_id)
            index.record_keys.setdefault(
                leakage_record_key(item.source, item.source_record_id), item.item_id
            )
            index.by_split[str(item.split)] += 1
            index.by_source[item.source] += 1
        index.files[str(path.relative_to(bench_dir))] = kept
        index.n_items += kept
    return index


def apply_leakage_filter(items: Sequence[Item], index: LeakageIndex) -> tuple[list[Item], dict[str, Any]]:
    """Remove items that collide with a v0.2 test/dev item; count every removal.

    Two independent reasons are tested per item and recorded separately:
    ``content_hash_in_v0_2_test_or_dev`` and ``source_record_id_in_v0_2_test_or_dev``.
    """
    kept: list[Item] = []
    by_reason: Counter[str] = Counter()
    by_source: Counter[str] = Counter()
    by_source_reason: Counter[str] = Counter()
    examples: list[dict[str, Any]] = []
    for item in items:
        reasons: list[str] = []
        if leakage_content_hash(item.state, item.question) in index.content_hashes:
            reasons.append("content_hash_in_v0_2_test_or_dev")
        if leakage_record_key(item.source, item.source_record_id) in index.record_keys:
            reasons.append("source_record_id_in_v0_2_test_or_dev")
        if not reasons:
            kept.append(item)
            continue
        for reason in reasons:
            by_reason[reason] += 1
            by_source_reason[f"{item.source}|{reason}"] += 1
        by_source[item.source] += 1
        if len(examples) < 20:
            examples.append({"item_id": item.item_id, "source": item.source, "reasons": reasons})
    report = {
        "n_input": len(items),
        "n_kept": len(kept),
        "n_removed_total": len(items) - len(kept),
        "n_removed_content_hash": by_reason.get("content_hash_in_v0_2_test_or_dev", 0),
        "n_removed_record_id": by_reason.get("source_record_id_in_v0_2_test_or_dev", 0),
        "by_reason": dict(sorted(by_reason.items())),
        "by_source": dict(sorted(by_source.items())),
        "by_source_reason": dict(sorted(by_source_reason.items())),
        "examples": examples,
    }
    return kept, report


# ---------------------------------------------------------------------------
# MedMCQA: reproduce the v0.1/v0.2 dev carve, then cap the rest (seeded)
# ---------------------------------------------------------------------------
def select_medmcqa_train_indices(
    n_rows: int, *, dev_cap: int, cap: int, seed: int
) -> tuple[list[int], list[int], int]:
    """Indices of MedMCQA train rows to use for training, excluding the benchmark's dev carve.

    The dev carve is *bit-identical* to ``split_train_into_dev_train(rows, dev_cap=dev_cap,
    seed=seed, salt="medmcqa")``: the random stream ``f"{seed}:medmcqa:dev"`` and
    ``rng.sample(range(n_rows), dev_cap)`` are the same call the v0.1 benchmark made, so the
    2,000 rows it exported as dev are exactly the rows excluded here. The remaining rows are
    subsampled to ``cap`` with the shared seeded :func:`subsample` (salt ``medmcqa:train``).

    Returns ``(kept_indices, dev_indices, n_dropped_over_cap)``; index order follows the
    source's own row order.
    """
    if n_rows <= dev_cap:
        dev_indices = list(range(n_rows))
        remaining: list[int] = []
    else:
        rng = random.Random(f"{seed}:medmcqa:dev")
        dev_indices = sorted(rng.sample(range(n_rows), dev_cap))
        dev_set = set(dev_indices)
        remaining = [i for i in range(n_rows) if i not in dev_set]
    kept, over = subsample(remaining, cap, seed, salt="medmcqa:train")
    return [int(i) for i in kept], dev_indices, int(over)


# ---------------------------------------------------------------------------
# MedQuAD: same construction as medquad.load_medquad, split forced to TRAIN
# ---------------------------------------------------------------------------
def medquad_used_question_ids(
    rows: Sequence[Any], reference_items: Sequence[Item]
) -> tuple[set[str], dict[str, Any]]:
    """Question ids used by the benchmark's MedQuAD items, matched through the question text.

    A built item does not carry ``question_id``, so the raw row is recovered by its normalised
    question text (the same join ``scripts/bench/verify_gold.py`` uses). Also reports the cost
    of the id rule: MedQuAD reuses a ``question_id`` across *different* questions, so excluding
    by id drops more rows than excluding by text.
    """
    by_text: dict[str, list[Any]] = defaultdict(list)
    for row in rows:
        by_text[normalize_label(str(row.get("question") or ""))].append(row)
    texts = {
        normalize_label(str(item.state).replace("Consumer health question:", ""))
        for item in reference_items
    }
    used_ids: set[str] = set()
    rows_matched_by_text = 0
    for text in texts:
        for row in by_text.get(text, []):
            used_ids.add(str(row.get("question_id")))
            rows_matched_by_text += 1
    rows_excluded_by_id = sum(1 for row in rows if str(row.get("question_id")) in used_ids)
    rows_excluded_by_text = sum(
        1 for row in rows if normalize_label(str(row.get("question") or "")) in texts
    )
    return used_ids, {
        "n_reference_items": len(reference_items),
        "n_reference_distinct_question_texts": len(texts),
        "reference_texts_not_found_in_raw": len(texts - set(by_text)),
        "n_used_question_ids": len(used_ids),
        "rows_matched_by_text": rows_matched_by_text,
        "rows_excluded_by_question_id": rows_excluded_by_id,
        "rows_excluded_by_question_text": rows_excluded_by_text,
        "over_excluded_by_id_rule": rows_excluded_by_id - rows_excluded_by_text,
        "rule": "exclude every question_id that appears in data/bench/v0.2/tier1/medquad.jsonl",
    }


def build_medquad_train_items(
    rows: Sequence[Any],
    *,
    seed: int,
    license_: str,
    used_question_ids: set[str],
) -> tuple[list[Item], dict[str, Any]]:
    """Replicate ``medquad.load_medquad`` item construction with ``split=Split.TRAIN``.

    The loader has no split parameter (it hash-splits a single train file into dev/test), so
    the construction is replicated here. The random stream is consumed in the loader's own
    order — including for rows that are then excluded — so an item built here for a row is
    label-for-label the item the benchmark built for the same row, apart from the split. That
    equality is checked mechanically at build time (``medquad_replication_check``).
    """
    counts = Counter(str(r.get("question_type", "")).strip().lower() for r in rows)
    option_types = sorted(t for t, n in counts.items() if t and n >= medquad.MIN_TYPE_FREQUENCY)
    dropped: Counter[str] = Counter()
    items: list[Item] = []
    if len(option_types) < 2:
        dropped["fewer_than_two_eligible_types"] = len(rows)
        return items, {"dropped_by_reason": dict(dropped), "option_types": option_types}

    usable = [r for r in rows if str(r.get("question_type", "")).strip().lower() in option_types]
    dropped["question_type_below_frequency_threshold"] = len(rows) - len(usable)
    rng = random.Random(f"{seed}:medquad:distractors")
    for row in usable:
        qid = str(row.get("question_id"))
        gold_type = str(row["question_type"]).strip().lower()
        question = str(row.get("question") or "").strip()
        if not question:
            dropped["empty_question"] += 1
            continue
        others = [t for t in option_types if t != gold_type]
        distractors = rng.sample(others, min(3, len(others)))
        labels = [*distractors, gold_type]
        order = list(range(len(labels)))
        rng.shuffle(order)
        gold_index = order.index(len(labels) - 1)
        shuffled = [labels[i] for i in order]
        if qid in used_question_ids:
            dropped["question_id_used_by_v0_2_tier1_split"] += 1
            continue
        record_key = stable_hash(
            {"doc": str(row.get("document_id")), "qid": qid, "q": question}, length=20
        )
        items.append(
            make_choice_item(
                source="medquad",
                record_id=record_key,
                url=str(row.get("document_url") or f"https://huggingface.co/datasets/{MEDQUAD_ID}"),
                license_=license_,
                record_date=date(2017, 1, 1),
                split=Split.TRAIN,
                template_id="medquad_routing_v1",
                skill="routing",
                state=f"Consumer health question: {question}",
                question="Which question type does this question belong to?",
                labels=shuffled,
                gold_index=gold_index,
                option_order_seed=seed,
                meta={"document_source": str(row.get("document_source", "")),
                      "question_focus": str(row.get("question_focus", ""))},
            )
        )
    return items, {
        "dropped_by_reason": dict(sorted(dropped.items())),
        "n_usable_rows": len(usable),
        "option_types": option_types,
        "n_option_types": len(option_types),
    }


def medquad_replication_check(
    shadow_items: Sequence[Item], reference_items: Sequence[Item]
) -> dict[str, Any]:
    """Check the local MedQuAD construction against the benchmark's items.

    Call it on a **shadow build** (the same rows with the v0.2 exclusion switched off). Every
    v0.2 MedQuAD item must be reproduced label-for-label by a shadow item built from a row with
    the same question text — same ordered option labels and same gold label. A reference item
    with no such counterpart means the local replication diverged from ``medquad.load_medquad``.

    Rows that merely *share* a question text with a reference row draw different distractors
    (the loader's random stream is positional), so they are counted separately rather than
    treated as failures: their own items are simply not part of the benchmark.
    """
    by_text: dict[str, list[Item]] = defaultdict(list)
    for item in shadow_items:
        by_text[normalize_label(str(item.state).replace("Consumer health question:", ""))].append(item)
    compared = matched = 0
    examples: list[dict[str, Any]] = []
    for ref in reference_items:
        text = normalize_label(str(ref.state).replace("Consumer health question:", ""))
        candidates = by_text.get(text)
        compared += 1
        if not candidates:
            if len(examples) < 10:
                examples.append({"item_id": ref.item_id, "check": "no_shadow_item_with_this_text"})
            continue
        ref_order = [normalize_label(o.label) for o in ref.options]
        ref_gold = normalize_label(ref.options[ref.gold_index].label)
        if any(
            [normalize_label(o.label) for o in c.options] == ref_order
            and normalize_label(c.options[c.gold_index].label) == ref_gold
            for c in candidates
        ):
            matched += 1
        elif len(examples) < 10:
            examples.append({"item_id": ref.item_id, "check": "labels_or_gold_differ"})
    n_shared_rows = sum(
        len(items) for text, items in by_text.items() if text in {
            normalize_label(str(ref.state).replace("Consumer health question:", ""))
            for ref in reference_items
        }
    )
    ok = compared == matched
    return {
        "n_reference_items_compared": compared,
        "n_reference_items_reproduced": matched,
        "n_shadow_rows_sharing_a_reference_text": n_shared_rows,
        "mismatch_examples": examples,
        "verdict": "PASS" if ok else "MISMATCH",
        "note": "shadow build over the same raw rows with the v0.2 exclusion disabled; shadow "
                "items are not written to the training file",
    }


def medqa_conflicting_stems(ds: Any) -> dict[str, dict[str, Any]]:
    """MedQA stems that occur more than once in the raw split with *different* options/answers.

    MedQA's record id (``stable_hash({question, split})``) is built from the question text, so
    two rows with the same stem are the *same* item: if they disagree on the option set or the
    correct answer, no unambiguous item can be built and the rows are dropped with a counted
    reason (identical repeats are simply deduplicated). No stem text is written out — only
    counts (the repo is public).
    """
    by_stem: dict[str, list[tuple[tuple[tuple[str, str], ...], str]]] = defaultdict(list)
    for row in ds:
        options = row.get("options") or {}
        answer_idx = str(row.get("answer_idx", "")).strip()
        if answer_idx not in options:
            continue
        fingerprint = (
            tuple(sorted((str(k), normalize_label(str(v))) for k, v in options.items())),
            normalize_label(str(options[answer_idx])),
        )
        by_stem[normalize_label(str(row["question"]))].append(fingerprint)
    conflicts: dict[str, dict[str, Any]] = {}
    for stem, rows in by_stem.items():
        option_sets = {r[0] for r in rows}
        answers = {r[1] for r in rows}
        if len(rows) > 1 and (len(option_sets) > 1 or len(answers) > 1):
            conflicts[stem] = {
                "n_rows_with_this_stem": len(rows),
                "n_distinct_option_sets": len(option_sets),
                "n_distinct_correct_answers": len(answers),
            }
    return conflicts


def retemplate(item: Item, template_id: str, *, option_order_seed: int) -> Item:
    """Return the same item under a different template id (the id is part of ``item_id``)."""
    return make_item(
        tier=item.tier,
        source=item.source,
        source_record_id=item.source_record_id,
        source_url=item.source_url,
        source_license=item.source_license,
        record_date=item.record_date,
        split=item.split,
        template_id=template_id,
        skill=item.skill,
        qtype=item.qtype,
        state=item.state,
        question=item.question,
        options=item.options,
        gold=item.gold,
        option_order_seed=option_order_seed,
        meta=item.meta,
    )


# ---------------------------------------------------------------------------
# The "only levels present" score rule (the S1 `_v2` fix)
# ---------------------------------------------------------------------------
def score_levels_present(
    qrels_by_query: dict[str, dict[str, int]],
    query_ids: Sequence[str],
    corpus_ids: set[str],
) -> list[int]:
    """Levels that occur in the sampled pool, using the loader's own grade->level mapping.

    ``grade 0/1/2 -> level 0/1/2`` (``relevance.GRADE_LEVELS``); a negative judgement is the
    lowest level. Only *judged* passages that exist in the corpus count — the same rule the
    fixed ``_v2`` template applies, so every offered level is the gold of at least one item.
    """
    levels: set[int] = set()
    for qid in query_ids:
        for cid, grade in qrels_by_query.get(qid, {}).items():
            if cid in corpus_ids:
                levels.add(min(max(int(grade), 0), len(relevance.GRADE_LEVELS) - 1))
    return sorted(levels)


# ---------------------------------------------------------------------------
# Independent gold verification (raw source row -> gold option, no loader math)
#
# Every verifier returns ``(report, bad_item_ids)``; the caller drops ``bad_item_ids`` with a
# counted reason. ``n_checked + n_unverifiable == n_items`` must hold for each source.
# ---------------------------------------------------------------------------
Verifier = Callable[[Sequence[Item]], tuple[dict[str, Any], set[str]]]


def verify_medqa(items: Sequence[Item], reference: dict[str, dict[str, Any]]) -> tuple[dict[str, Any], set[str]]:
    """Gold must be the option whose *text* the raw record's ``answer_idx`` names."""
    rep = VerificationReport(source="medqa")
    bad: set[str] = set()
    option_set_mismatch = 0
    ambiguous = 0
    for item in items:
        rep.n_items += 1
        rep.gold_class_counts[normalize_label(item.gold)] += 1
        entry = reference.get(normalize_label(item.state))
        if entry is None:
            rep.n_unverifiable += 1
            continue
        rep.n_checked += 1
        gold_label = normalize_label(item.options[item.gold_index].label)
        if len(entry["answers"]) > 1:
            ambiguous += 1
        if gold_label not in entry["answers"]:
            rep.add_mismatch(item.item_id, item.source_record_id,
                             "|".join(sorted(entry["answers"])), gold_label)
            bad.add(item.item_id)
            continue
        if {normalize_label(o.label) for o in item.options} != entry["options"]:
            option_set_mismatch += 1
            rep.add_mismatch(item.item_id, item.source_record_id, "option_set", "option_set")
            bad.add(item.item_id)
    payload = rep.to_dict()
    payload["extra"] = {
        "n_questions_with_several_raw_answer_sets": ambiguous,
        "n_option_set_mismatches": option_set_mismatch,
        "rule": "gold option text == raw options[answer_idx] for the raw row with this question text",
    }
    return payload, bad


def verify_medmcqa(items: Sequence[Item], raw_by_id: dict[str, Any]) -> tuple[dict[str, Any], set[str]]:
    """Gold must be ``opa..opd[cop]`` (0-based), in the A..D order the raw row publishes."""
    rep = VerificationReport(source="medmcqa")
    bad: set[str] = set()
    option_set_mismatch = 0
    for item in items:
        rep.n_items += 1
        rep.gold_class_counts[normalize_label(item.gold)] += 1
        row = raw_by_id.get(str(item.source_record_id))
        if row is None:
            rep.n_unverifiable += 1
            continue
        labels = [row.get("opa"), row.get("opb"), row.get("opc"), row.get("opd")]
        try:
            index = int(row["cop"])
        except (KeyError, TypeError, ValueError):
            rep.n_unverifiable += 1
            continue
        if not 0 <= index < len(labels) or not labels[index]:
            rep.n_unverifiable += 1
            continue
        rep.n_checked += 1
        want = normalize_label(str(labels[index]))
        if normalize_label(item.options[item.gold_index].label) != want:
            rep.add_mismatch(item.item_id, str(item.source_record_id), want,
                             normalize_label(item.options[item.gold_index].label))
            bad.add(item.item_id)
            continue
        if [normalize_label(str(x)) for x in labels] != [normalize_label(o.label) for o in item.options]:
            option_set_mismatch += 1
            rep.add_mismatch(item.item_id, str(item.source_record_id), "option_set", "option_set")
            bad.add(item.item_id)
    payload = rep.to_dict()
    payload["extra"] = {
        "n_option_set_mismatches": option_set_mismatch,
        "rule": "gold option text == [opa,opb,opc,opd][cop] (cop is 0-based)",
    }
    return payload, bad


def verify_medquad(items: Sequence[Item], reference: dict[str, dict[str, Any]]) -> tuple[dict[str, Any], set[str]]:
    """Gold must be the option whose label is the raw row's ``question_type``."""
    rep = VerificationReport(source="medquad")
    bad: set[str] = set()
    option_not_eligible = 0
    state_mismatch = 0
    for item in items:
        rep.n_items += 1
        rep.gold_class_counts[normalize_label(item.gold)] += 1
        text = normalize_label(str(item.state).replace("Consumer health question:", ""))
        entry = reference.get(text)
        if entry is None:
            rep.n_unverifiable += 1
            continue
        rep.n_checked += 1
        if normalize_label(str(item.state)) != normalize_label(f"Consumer health question: {entry['question']}"):
            state_mismatch += 1
            rep.add_mismatch(item.item_id, str(entry["question_id"]), "state_text", "state_text")
            bad.add(item.item_id)
            continue
        want = normalize_label(entry["question_type"])
        if normalize_label(item.options[item.gold_index].label) != want:
            rep.add_mismatch(item.item_id, str(entry["question_id"]), want,
                             normalize_label(item.options[item.gold_index].label))
            bad.add(item.item_id)
            continue
        if any(normalize_label(o.label) not in entry["eligible_types"] for o in item.options):
            option_not_eligible += 1
            rep.add_mismatch(item.item_id, str(entry["question_id"]), "eligible_option", "ineligible_option")
            bad.add(item.item_id)
    payload = rep.to_dict()
    payload["extra"] = {
        "n_state_text_mismatches": state_mismatch,
        "n_items_with_ineligible_option": option_not_eligible,
        "rule": "gold option label == raw question_type; every option is a question type the "
                "source uses at least 20 times",
    }
    return payload, bad


def verify_relevance(
    source: str,
    items: Sequence[Item],
    *,
    qrels_by_query: dict[str, dict[str, int]],
    corpus_index: dict[str, str],
    query_text: dict[str, str],
) -> tuple[dict[str, Any], set[str]]:
    """Gold must be this (query, passage) pair's qrel grade under the fixed relevance rule.

    ``noul``: grade > 0 -> ``yes``; a pair with no positive judgement (grade <= 0, or absent
    from the qrels) -> ``no``. ``score``: the gold option's label must be
    ``GRADE_LEVELS[level_of(grade)]``. The item's recorded ``meta.qrel_grade`` is checked
    against the raw qrels as a third signal, and the state text is recomputed from the corpus
    so a wrong ``(query, passage)`` pairing cannot pass.
    """
    rep = VerificationReport(source=source)
    bad: set[str] = set()
    n_positive = n_negative = 0
    grade_meta_mismatch = state_text_mismatch = 0
    for item in items:
        rep.n_items += 1
        rep.gold_class_counts[normalize_label(item.gold)] += 1
        qid = str(item.meta.get("query_id"))
        cid = str(item.meta.get("corpus_id"))
        judgements = qrels_by_query.get(qid)
        grade = judgements.get(cid) if judgements else None
        doc = corpus_index.get(cid)
        if doc is None or qid not in query_text:
            rep.n_unverifiable += 1
            continue
        expected_state = f"Query: {query_text[qid]}\n\nPassage: {doc}"
        if normalize_text(expected_state) != normalize_text(str(item.state)):
            state_text_mismatch += 1
            rep.add_mismatch(item.item_id, f"{qid}:{cid}", "state_text",
                             normalize_text(str(item.state))[:60])
            bad.add(item.item_id)
            continue
        if grade is not None and int(item.meta.get("qrel_grade", -999)) != int(grade):
            grade_meta_mismatch += 1
            rep.add_mismatch(item.item_id, f"{qid}:{cid}",
                             f"meta_grade={item.meta.get('qrel_grade')}", str(grade))
            bad.add(item.item_id)
            continue
        if item.qtype is QuestionType.NOUL:
            rep.n_checked += 1
            want = "yes" if (grade is not None and grade > 0) else "no"
            if want == "yes":
                n_positive += 1
            else:
                n_negative += 1
            if normalize_label(item.gold) != want:
                rep.add_mismatch(item.item_id, f"{qid}:{cid}", want, item.gold)
                bad.add(item.item_id)
            continue
        if grade is None:
            rep.n_unverifiable += 1
            continue
        level = min(max(int(grade), 0), len(relevance.GRADE_LEVELS) - 1)
        want_label = normalize_label(relevance.GRADE_LEVELS[level])
        want_key = next((o.key for o in item.options if normalize_label(o.label) == want_label), None)
        if want_key is None:
            rep.n_unverifiable += 1
            continue
        rep.n_checked += 1
        if normalize_label(item.gold) != normalize_label(want_key):
            rep.add_mismatch(item.item_id, f"{qid}:{cid}", want_key, item.gold)
            bad.add(item.item_id)
    payload = rep.to_dict()
    payload["extra"] = {
        "n_noul_positives_checked": n_positive,
        "n_noul_negatives_checked": n_negative,
        "n_grade_meta_mismatches": grade_meta_mismatch,
        "n_state_text_mismatches": state_text_mismatch,
        "rule": "noul: qrel grade > 0 -> yes, otherwise no; "
                "score: gold option label == GRADE_LEVELS[grade]",
    }
    return payload, bad


# ---------------------------------------------------------------------------
# Build helpers
# ---------------------------------------------------------------------------
def dedupe_by_item_id(items: Sequence[Item]) -> tuple[list[Item], int]:
    """Drop repeated ``item_id``s (same source record, template, seed and split)."""
    seen: set[str] = set()
    kept: list[Item] = []
    for item in items:
        if item.item_id in seen:
            continue
        seen.add(item.item_id)
        kept.append(item)
    return kept, len(items) - len(kept)


def _drop_by(items: Sequence[Item], predicate: Callable[[Item], bool]) -> tuple[list[Item], int]:
    kept: list[Item] = []
    dropped = 0
    for item in items:
        if predicate(item):
            dropped += 1
            continue
        kept.append(item)
    return kept, dropped


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out", type=Path, default=Path("data/train/student_v0"))
    parser.add_argument("--bench", type=Path, default=Path("data/bench/v0.2"))
    parser.add_argument("--config", type=Path, default=Path("configs/bench_v0_2.yaml"))
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--medmcqa-cap", type=int, default=DEFAULT_MEDMCQA_CAP)
    parser.add_argument("--medmcqa-dev-cap", type=int, default=None)
    parser.add_argument("--sources", nargs="*", default=None)
    args = parser.parse_args()

    t0 = time.perf_counter()
    cfg = yaml.safe_load(args.config.read_text())
    seed = int(args.seed if args.seed is not None else cfg.get("seed", 0))
    caps = cfg.get("caps", {})
    dev_cap = int(
        args.medmcqa_dev_cap
        if args.medmcqa_dev_cap is not None
        else caps.get("tier1_max_dev_items_per_source", DEFAULT_MEDMCQA_DEV_CAP)
    )
    wanted = set(args.sources) if args.sources else None

    def want(name: str) -> bool:
        return wanted is None or name in wanted

    args.out.mkdir(parents=True, exist_ok=True)
    notes: list[str] = []
    sources: dict[str, dict[str, Any]] = {}
    verifiers: dict[str, Verifier] = {}
    dataset_meta: dict[str, dict[str, str]] = {}
    for key, dataset_id in DATASETS.items():
        dataset_meta[key] = {
            "id": dataset_id,
            "revision": _revision_of(dataset_id),
            "license": _license_of(dataset_id, notes),
        }

    items: list[Item] = []

    # ---- MedQA (official train) -------------------------------------------
    if want("medqa"):
        ds = _load(MEDQA_ID, None, "train", dataset_meta["medqa"]["revision"])
        built: list[Item] = []
        unparseable = 0
        for row in ds:
            item = mcq._medqa_item(row, split=Split.TRAIN, seed=seed)
            if item is None:
                unparseable += 1
                continue
            built.append(item)
        built, n_dup = dedupe_by_item_id(built)
        conflicts = medqa_conflicting_stems(ds)
        built, n_conflict = _drop_by(
            built, lambda i: normalize_label(i.state) in conflicts
        )
        reference = _medqa_reference(ds)
        verifiers["medqa"] = lambda subset, ref=reference: verify_medqa(subset, ref)
        items.extend(built)
        sources["medqa"] = {
            "dataset": dataset_meta["medqa"],
            "source_split": "train",
            "template_ids": ["medqa_usmle4_v1"],
            "n_rows_in_split": len(ds),
            "n_items": len(built),
            "dropped_by_reason": {
                "unparseable_options": unparseable,
                "duplicate_item_id": n_dup,
                "duplicate_stem_conflicting_options": n_conflict,
            },
            "conflicting_stems": {
                "n_stems": len(conflicts),
                "details": sorted(conflicts.values(), key=lambda d: -d["n_rows_with_this_stem"]),
                "rule": "same normalised stem, >1 row, different option set or correct answer: "
                        "the stem is the record id, so no unambiguous item exists; dropped",
            },
            "notes": [
                "item construction reused verbatim: mcq._medqa_item(row, split=Split.TRAIN)",
                "the official train split contains 2 placeholder stems ('please refer to the "
                "summary above...') twice, each with a different option set and a different "
                "correct answer; those 2 item identities are dropped as ambiguous, the other "
                "10,174 rows are kept",
            ],
        }

    # ---- MedMCQA (official train minus the benchmark's dev carve, capped) --
    if want("medmcqa"):
        ds = _load(MEDMCQA_ID, None, "train", dataset_meta["medmcqa"]["revision"])
        kept_indices, dev_indices, over_cap = select_medmcqa_train_indices(
            len(ds), dev_cap=dev_cap, cap=args.medmcqa_cap, seed=seed
        )
        built = []
        unparseable = 0
        for i in kept_indices:
            item = mcq._medmcqa_item(ds[i], split=Split.TRAIN, seed=seed)
            if item is None:
                unparseable += 1
                continue
            built.append(item)
        built, n_dup = dedupe_by_item_id(built)
        needed = {i.source_record_id for i in built}
        raw_by_id = {str(row["id"]): row for row in ds if str(row["id"]) in needed}
        verifiers["medmcqa"] = lambda subset, ref=raw_by_id: verify_medmcqa(subset, ref)
        dev_ids = {str(ds[i]["id"]) for i in dev_indices}
        items.extend(built)
        sources["medmcqa"] = {
            "dataset": dataset_meta["medmcqa"],
            "source_split": "train",
            "template_ids": ["medmcqa_4opt_v1"],
            "n_rows_in_split": len(ds),
            "n_items": len(built),
            "dropped_by_reason": {
                "dev_carve_excluded": len(dev_indices),
                "train_over_cap": over_cap,
                "unparseable_record": unparseable,
                "duplicate_item_id": n_dup,
            },
            "dev_carve": {
                "dev_cap": dev_cap,
                "seed": seed,
                "salt": "medmcqa",
                "n_dev_rows_excluded": len(dev_indices),
                "dev_row_ids_digest": ids_digest(dev_ids),
                "note": "identical to split_train_into_dev_train(rows, dev_cap, seed, "
                        "salt='medmcqa'), the carve v0.1/v0.2 exported as dev",
            },
            "cap": {
                "cap": args.medmcqa_cap,
                "n_rows_available_after_carve": len(ds) - len(dev_indices),
                "n_rows_dropped_over_cap": over_cap,
                "salt": "medmcqa:train",
            },
            "notes": [
                "item construction reused verbatim: mcq._medmcqa_item(row, split=Split.TRAIN)",
                "cap applied after the dev carve, so the cap is 60,000 of the rows the dev set "
                "did not use",
            ],
        }

    # ---- SciFact (BEIR train qrels, noul only) ----------------------------
    if want("scifact"):
        queries = list(_load(SCIFACT_ID, "queries", "queries", dataset_meta["scifact"]["revision"]))
        corpus = list(_load(SCIFACT_ID, "corpus", "corpus", dataset_meta["scifact"]["revision"]))
        qrels = list(_load(SCIFACT_QRELS_ID, None, "train", dataset_meta["scifact_qrels"]["revision"]))
        res = relevance.load_beir_relevance(
            source="scifact", dataset_id=SCIFACT_ID, queries=queries, corpus=corpus,
            qrels=qrels, cfg=cfg, revision=dataset_meta["scifact"]["revision"],
            split=Split.TRAIN, record_date=date(2020, 5, 1), graded=False,
        )
        built = [retemplate(i, SCIFACT_TRAIN_NOUL_TEMPLATE, option_order_seed=seed) for i in res.rows]
        built, n_dup = dedupe_by_item_id(built)
        corpus_index = _beir_corpus_index(corpus)
        query_text = {str(q["_id"]): str(q["text"]).strip() for q in queries}
        verifiers["scifact"] = lambda subset, q=qrels, c=corpus_index, t=query_text: verify_relevance(
            "scifact", subset, qrels_by_query=_qrels_by_query(q), corpus_index=c, query_text=t
        )
        items.extend(built)
        sources["scifact"] = {
            "dataset": dataset_meta["scifact"],
            "qrels_dataset": dataset_meta["scifact_qrels"],
            "source_split": "train (official BEIR train qrels)",
            "template_ids": [SCIFACT_TRAIN_NOUL_TEMPLATE],
            "n_items": len(built),
            "dropped_by_reason": {**res.dropped, "duplicate_item_id": n_dup},
            "qrels": {
                "split": "train",
                "n_rows": len(qrels),
                "n_queries_judged": len(_qrels_by_query(qrels)),
                "grade_counts": dict(sorted(Counter(int(r["score"]) for r in qrels).items())),
            },
            "notes": [
                f"template id renamed from the loader default to {SCIFACT_TRAIN_NOUL_TEMPLATE} "
                "(S5); only the id changes, the item construction is the loader's",
                "SciFact train qrels are binary (grade 1); no score template is built",
                *res.notes,
            ],
        }

    # ---- NFCorpus (BEIR train qrels: noul + graded score when possible) ---
    if want("nfcorpus"):
        queries = list(_load(NFCORPUS_ID, "queries", "queries", dataset_meta["nfcorpus"]["revision"]))
        corpus = list(_load(NFCORPUS_ID, "corpus", "corpus", dataset_meta["nfcorpus"]["revision"]))
        qrels = list(_load(NFCORPUS_QRELS_ID, None, "train", dataset_meta["nfcorpus_qrels"]["revision"]))
        qrels_map = _qrels_by_query(qrels)
        corpus_index = _beir_corpus_index(corpus)
        query_text = {str(q["_id"]): str(q["text"]).strip() for q in queries}
        usable = [qid for qid in sorted(query_text) if any(c in corpus_index for c in qrels_map.get(qid, {}))]
        levels_present = score_levels_present(qrels_map, usable, set(corpus_index))
        graded = len(levels_present) >= 2
        res = relevance.load_beir_relevance(
            source="nfcorpus", dataset_id=NFCORPUS_ID, queries=queries, corpus=corpus,
            qrels=qrels, cfg=cfg, revision=dataset_meta["nfcorpus"]["revision"],
            split=Split.TRAIN, record_date=date(2015, 1, 1), graded=graded,
            score_levels=None, score_template_id=NFCORPUS_TRAIN_SCORE_TEMPLATE,
        )
        built, n_dup = dedupe_by_item_id(res.rows)
        score_status = "built" if graded else (
            "NOT MEASURED — the official train qrels are binary (grade 1 only), so the pool "
            "contains a single score level and a graded score question would offer one option. "
            "S5 passes score_levels=None (offer only the levels present in the pool); with one "
            "level present there is no valid score item to build, and inventing a second level "
            "would be inventing gold."
        )
        verifiers["nfcorpus"] = lambda subset, q=qrels_map, c=corpus_index, t=query_text: verify_relevance(
            "nfcorpus", subset, qrels_by_query=q, corpus_index=c, query_text=t
        )
        items.extend(built)
        sources["nfcorpus"] = {
            "dataset": dataset_meta["nfcorpus"],
            "qrels_dataset": dataset_meta["nfcorpus_qrels"],
            "source_split": "train (official BEIR train qrels)",
            "template_ids": [NFCORPUS_NOUL_TEMPLATE, NFCORPUS_TRAIN_SCORE_TEMPLATE],
            "n_items": len(built),
            "dropped_by_reason": {**res.dropped, "duplicate_item_id": n_dup},
            "score_status": score_status,
            "qrels": {
                "split": "train",
                "n_rows": len(qrels),
                "n_queries_judged": len(qrels_map),
                "grade_counts": dict(sorted(Counter(int(r["score"]) for r in qrels).items())),
            },
            "score_levels": {
                "levels_present": levels_present,
                "level_labels_present": [relevance.GRADE_LEVELS[i] for i in levels_present],
                "levels_offered": levels_present if graded else [],
                "rule": "score_levels=None -> offered = levels present in the sampled pool",
            },
            "notes": [
                f"noul template keeps the loader default id {NFCORPUS_NOUL_TEMPLATE} (S5 names "
                f"no train-specific noul id); the score template is {NFCORPUS_TRAIN_SCORE_TEMPLATE}",
                *res.notes,
            ],
        }

    # ---- MedQuAD (rows no tier-1 split used) ------------------------------
    if want("medquad"):
        ds = _load(MEDQUAD_ID, None, "train", dataset_meta["medquad"]["revision"])
        raw_rows = list(ds)
        reference_rows, _ = read_jsonl(args.bench / "tier1" / "medquad.jsonl", Item)
        reference_items = [i for i in reference_rows if isinstance(i, Item)]
        used_ids, used_report = medquad_used_question_ids(raw_rows, reference_items)
        shadow, _ = build_medquad_train_items(
            raw_rows, seed=seed, license_=dataset_meta["medquad"]["license"],
            used_question_ids=set(),
        )
        replication = medquad_replication_check(shadow, reference_items)
        built, mq_report = build_medquad_train_items(
            raw_rows, seed=seed, license_=dataset_meta["medquad"]["license"],
            used_question_ids=used_ids,
        )
        built, n_dup = dedupe_by_item_id(built)
        eligible = set(mq_report.get("option_types", []))
        reference = {
            normalize_label(str(r.get("question") or "")): {
                "question": str(r.get("question") or ""),
                "question_type": str(r.get("question_type", "")).strip().lower(),
                "question_id": str(r.get("question_id")),
                "eligible_types": eligible,
            }
            for r in raw_rows
        }
        verifiers["medquad"] = (
            lambda subset, ref=reference: verify_medquad_ref(subset, ref, replication)
        )
        items.extend(built)
        sources["medquad"] = {
            "dataset": dataset_meta["medquad"],
            "source_split": "train (single official split; rows used by the benchmark excluded)",
            "template_ids": ["medquad_routing_v1"],
            "n_rows_in_split": len(raw_rows),
            "n_items": len(built),
            "dropped_by_reason": {**mq_report["dropped_by_reason"], "duplicate_item_id": n_dup},
            "exclusion_of_v0_2_rows": used_report,
            "replication_check": replication,
            "notes": [
                "medquad.load_medquad hash-splits the single train file into dev/test and has no "
                "split parameter, so its item construction is replicated in this script with "
                "split=Split.TRAIN; the replication is checked against the benchmark's items for "
                "every shared question text using a shadow build over the same rows",
                "option types are computed from the full raw row set (as the loader does), not "
                "from the kept rows, so the option set is the benchmark's",
            ],
        }

    # ---- leakage filter (acceptance) --------------------------------------
    index = build_leakage_index(args.bench)
    n_before_leakage = len(items)
    items, leakage = apply_leakage_filter(items, index)
    assert leakage["n_input"] == n_before_leakage

    # ---- gold verification on exactly what will be written ----------------
    gold_dropped: Counter[str] = Counter()
    gold_reports: dict[str, dict[str, Any]] = {}
    for source_name, verifier in verifiers.items():
        subset = [i for i in items if i.source == source_name]
        payload, bad = verifier(subset)
        gold_reports[source_name] = payload
        sources[source_name]["gold_verification"] = payload
        if bad:
            items = [i for i in items if i.item_id not in bad]
            gold_dropped[f"{source_name}:gold_verification_mismatch"] = len(bad)
    n_checked = sum(r["n_checked"] for r in gold_reports.values())
    n_mismatched = sum(r["n_mismatched"] for r in gold_reports.values())
    n_unverifiable = sum(r["n_unverifiable"] for r in gold_reports.values())
    for name, payload in gold_reports.items():
        if payload["n_checked"] + payload["n_unverifiable"] != payload["n_items"]:
            raise AssertionError(f"gold verification for {name} does not close")

    # ---- checks + counts ---------------------------------------------------
    if any(str(i.split) != "train" for i in items):
        raise AssertionError("every training item must carry Split.TRAIN")
    if any(i.source not in sources for i in items):
        raise AssertionError("an item's source was not built by this run")

    counts = {
        "by_source": dict(sorted(Counter(i.source for i in items).items())),
        "by_template": dict(sorted(Counter(i.template_id for i in items).items())),
        "by_qtype": dict(sorted(Counter(str(i.qtype) for i in items).items())),
        "by_source_template": dict(sorted(Counter(f"{i.source}|{i.template_id}" for i in items).items())),
        "by_source_qtype": dict(sorted(Counter(f"{i.source}|{i.qtype}" for i in items).items())),
        "by_source_template_qtype": dict(
            sorted(Counter(f"{i.source}|{i.template_id}|{i.qtype}" for i in items).items())
        ),
    }

    jsonl_path = write_jsonl(args.out / "tier1_train.jsonl", items)
    manifest = {
        "run_name": "S5_tier1_train",
        "built_at_utc": utcnow(),
        "git_commit": git_commit(),
        "command": " ".join([sys.executable, *sys.argv]),
        "config": str(args.config),
        "seed": seed,
        "bench_reference_dir": str(args.bench),
        "source_allowlist": ["medqa", "medmcqa", "scifact", "nfcorpus", "medquad"],
        "excluded_sources": EXCLUDED_SOURCES,
        "output": {
            "jsonl": str(jsonl_path),
            "n_items": len(items),
            "sha256": file_sha256(jsonl_path),
            "bytes": jsonl_path.stat().st_size,
        },
        "dataset_revisions": dataset_meta,
        "counts": counts,
        "sources": sources,
        "leakage_check": {"reference": index.to_dict(), "result": leakage},
        "gold_verification": {
            "verdict": "PASS — 0 mismatches" if n_mismatched == 0
                       else "MISMATCHES FOUND — items dropped, see per-source reports",
            "n_items_verified": sum(r["n_items"] for r in gold_reports.values()),
            "n_checked": n_checked,
            "n_unverifiable": n_unverifiable,
            "n_mismatched": n_mismatched,
            "dropped_for_mismatch": dict(gold_dropped),
            "per_source": {
                name: {k: payload[k] for k in ("n_items", "n_checked", "n_unverifiable",
                                               "n_mismatched", "ok")}
                for name, payload in sorted(gold_reports.items())
            },
        },
        "caps": {
            "medmcqa_train_items": args.medmcqa_cap,
            "medmcqa_dev_carve_items": dev_cap,
            "medqa_train_items": "uncapped (all parseable official train rows)",
            "beir_query_cap": caps.get("tier1_max_test_items_per_source", 5000),
            "medquad_items": "uncapped (all rows not used by a tier-1 split)",
        },
        "loader_notes": notes,
        "wall_clock_s": round(time.perf_counter() - t0, 1),
    }
    manifest_path = write_json(args.out / "tier1_train_manifest.json", manifest)

    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"\nwrote {jsonl_path} ({len(items)} items) and {manifest_path}", file=sys.stderr)
    return 0


def verify_medquad_ref(
    subset: Sequence[Item], reference: dict[str, dict[str, Any]], replication: dict[str, Any]
) -> tuple[dict[str, Any], set[str]]:
    payload, bad = verify_medquad(subset, reference)
    payload["extra"]["replication_check"] = replication
    return payload, bad


def _medqa_reference(ds: Any) -> dict[str, dict[str, Any]]:
    """normalised question -> {answer label texts, all option label texts} from the raw rows."""
    out: dict[str, dict[str, Any]] = {}
    for row in ds:
        options = row.get("options") or {}
        answer_idx = str(row.get("answer_idx", "")).strip()
        if answer_idx not in options:
            continue
        entry = out.setdefault(
            normalize_label(str(row["question"])), {"answers": set(), "options": set()}
        )
        entry["answers"].add(normalize_label(str(options[answer_idx])))
        entry["options"].update(normalize_label(str(v)) for v in options.values())
    return out


if __name__ == "__main__":
    raise SystemExit(main())
