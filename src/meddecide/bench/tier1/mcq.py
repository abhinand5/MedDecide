"""Tier-1 loaders: MedQA, MedMCQA, PubMedQA, MMLU medical subsets.

Each loader turns an official public split into items, preserving the official split and
recording dataset id + revision + license. Gold is the source's own answer field.
"""

from __future__ import annotations

from datetime import date
from typing import Any

from meddecide.bench.schema import Split, make_item
from meddecide.bench.tier1.common import (
    LoadResult,
    finalize_splits,
    make_choice_item,
    split_train_into_dev_train,
)
from meddecide.utils.hashing import stable_hash

# ---------------------------------------------------------------------------
# MedQA (USMLE, 4-option)
# ---------------------------------------------------------------------------
MEDQA_ID = "GBaker/MedQA-USMLE-4-options"
MEDQA_LICENSE = "UNKNOWN"  # resolved from the dataset card at build time (never hardcode a license claim)


def load_medqa(rows_test: list[Any], rows_train: list[Any], *, cfg: dict[str, Any], revision: str) -> LoadResult:
    """MedQA test = official test; dev = sample of official train."""
    seed = int(cfg.get("seed", 0))
    max_test, max_dev = _caps(cfg)
    result = LoadResult(source="medqa", dataset_id=MEDQA_ID, dataset_revision=revision,
                        license=MEDQA_LICENSE,
                        split_map={"test": MEDQA_ID + ":test", "dev": MEDQA_ID + ":train (sampled)"})

    for rows, split in ((rows_test, Split.TEST), (rows_train, Split.DEV)):
        for row in rows:
            item = _medqa_item(row, split=split, seed=seed)
            if item is None:
                result.drop("unparseable_options")
                continue
            result.rows.append(item)
    result.rows, counts, extra = finalize_splits(
        result.rows, cfg=cfg, source="medqa", salt="medqa", cap_test=max_test, cap_dev=max_dev
    )
    for reason, n in counts.items():
        result.drop(reason, n)
    result.notes.extend(extra)
    return result


def _medqa_item(row: dict[str, Any], *, split: Split, seed: int):
    options = row.get("options") or {}
    answer_idx = str(row.get("answer_idx", "")).strip()
    if not isinstance(options, dict) or len(options) < 2 or answer_idx not in options:
        return None
    keys = sorted(options)  # A, B, C, D as published
    labels = [str(options[k]) for k in keys]
    return make_choice_item(
        source="medqa",
        record_id=stable_hash({"q": str(row.get("question", "")), "split": split.value}, length=20),
        url="https://huggingface.co/datasets/GBaker/MedQA-USMLE-4-options",
        license_=MEDQA_LICENSE,
        record_date=date(2021, 1, 1),
        split=split,
        template_id="medqa_usmle4_v1",
        skill="medical_knowledge",
        state=str(row["question"]).strip(),
        question="Which of the following is the best answer?",
        labels=labels,
        gold_index=keys.index(answer_idx),
        option_order_seed=seed,
        meta={"source_split": str(split), "source_answer_idx": answer_idx,
              "meta_info": str(row.get("meta_info", ""))},
    )


# ---------------------------------------------------------------------------
# MedMCQA
# ---------------------------------------------------------------------------
MEDMCQA_ID = "openlifescienceai/medmcqa"
MEDMCQA_LICENSE = "UNKNOWN"  # resolved from the dataset card at build time


def load_medmcqa(rows_val: list[Any], rows_train: list[Any], *, cfg: dict[str, Any], revision: str) -> LoadResult:
    """MedMCQA: official *validation* is used as tier-1 test (its official test labels are
    not public); dev is carved from official train. Both choices are recorded here."""
    seed = int(cfg.get("seed", 0))
    max_test, max_dev = _caps(cfg)
    result = LoadResult(
        source="medmcqa",
        dataset_id=MEDMCQA_ID,
        dataset_revision=revision,
        license=MEDMCQA_LICENSE,
        split_map={
            "test": MEDMCQA_ID + ":validation (official test labels unreleased)",
            "dev": MEDMCQA_ID + ":train (carved, seeded)",
        },
        notes=[
            "MedMCQA official test split labels are not public; the official validation "
            "split is used as tier-1 test (ADVISORY T3).",
            "dev is carved from the official train split with a fixed seed; the remaining "
            "train rows are not exported in v0.",
        ],
    )
    dev_rows, _, _ = split_train_into_dev_train(rows_train, dev_cap=max_dev, seed=seed, salt="medmcqa")
    for rows, split in ((rows_val, Split.TEST), (dev_rows, Split.DEV)):
        for row in rows:
            item = _medmcqa_item(row, split=split, seed=seed)
            if item is None:
                result.drop("unparseable_record")
                continue
            result.rows.append(item)
    result.rows, counts, extra = finalize_splits(
        result.rows, cfg=cfg, source="medmcqa", salt="medmcqa", cap_test=max_test, cap_dev=max_dev
    )
    for reason, n in counts.items():
        result.drop(reason, n)
    result.notes.extend(extra)
    return result


def _medmcqa_item(row: dict[str, Any], *, split: Split, seed: int):
    labels = [row.get("opa"), row.get("opb"), row.get("opc"), row.get("opd")]
    if any(not isinstance(x, str) or not x.strip() for x in labels):
        return None
    cop = row.get("cop")
    try:
        gold_index = int(cop) - 1
    except (TypeError, ValueError):
        return None
    if not 0 <= gold_index < len(labels):
        return None
    return make_choice_item(
        source="medmcqa",
        record_id=str(row.get("id")),
        url="https://huggingface.co/datasets/openlifescienceai/medmcqa",
        license_=MEDMCQA_LICENSE,
        record_date=date(2022, 4, 1),
        split=split,
        template_id="medmcqa_4opt_v1",
        skill="medical_knowledge",
        state=str(row["question"]).strip(),
        question="Which of the following is the best answer?",
        labels=[str(x) for x in labels],
        gold_index=gold_index,
        option_order_seed=seed,
        meta={"subject_name": str(row.get("subject_name", "")),
              "topic_name": str(row.get("topic_name", "")),
              "choice_type": str(row.get("choice_type", ""))},
    )


# ---------------------------------------------------------------------------
# PubMedQA (expert-labelled)
# ---------------------------------------------------------------------------
PUBMEDQA_ID = "qiaojin/PubMedQA"
PUBMEDQA_LICENSE = "UNKNOWN"  # resolved from the dataset card at build time
PUBMEDQA_LABELS = ["yes", "no", "maybe"]


def load_pubmedqa(rows: list[Any], *, cfg: dict[str, Any], revision: str, dev_fraction: float = 0.5) -> LoadResult:
    """PubMedQA expert-labelled (1,000 items): dev/test split by a deterministic hash of the
    PubMed id, so the same record can never land in both. Gold is ``final_decision``."""
    seed = int(cfg.get("seed", 0))
    result = LoadResult(
        source="pubmedqa",
        dataset_id=PUBMEDQA_ID,
        dataset_revision=revision,
        license=PUBMEDQA_LICENSE,
        split_map={"test": PUBMEDQA_ID + ":pqa_labeled (hash split)",
                   "dev": PUBMEDQA_ID + ":pqa_labeled (hash split)"},
        notes=[
            "pqa_labeled has a single split of 1,000 items; it is divided by a stable hash "
            f"of the PubMed id at dev_fraction={dev_fraction} (no official split exists to preserve).",
        ],
    )
    for row in rows:
        item = _pubmedqa_item(row, seed=seed, dev_fraction=dev_fraction)
        if item is None:
            result.drop("unparseable_record")
            continue
        result.rows.append(item)
    max_test, max_dev = _caps(cfg)
    result.rows, counts, extra = finalize_splits(
        result.rows, cfg=cfg, source="pubmedqa", salt="pubmedqa", cap_test=max_test, cap_dev=max_dev
    )
    for reason, n in counts.items():
        result.drop(reason, n)
    result.notes.extend(extra)
    return result


def _pubmedqa_item(row: dict[str, Any], *, seed: int, dev_fraction: float):
    decision = str(row.get("final_decision", "")).strip().lower()
    if decision not in PUBMEDQA_LABELS:
        return None
    context = row.get("context") or {}
    contexts = context.get("contexts") if isinstance(context, dict) else None
    if not contexts:
        return None
    pubid = str(row.get("pubid"))
    split = _hash_split(pubid, dev_fraction)
    # state = the abstract's context paragraphs only; the answer ("final_decision")
    # never appears in the state, and the question is not duplicated into it.
    state = "\n".join(str(c) for c in contexts).strip()
    if not state:
        return None
    return make_item(
        tier="established",
        source="pubmedqa",
        source_record_id=pubid,
        source_url=f"https://pubmed.ncbi.nlm.nih.gov/{pubid}/",
        source_license=PUBMEDQA_LICENSE,
        record_date=date(2019, 1, 1),
        split=split,
        template_id="pubmedqa_ynm_v1",
        skill="evidence",
        qtype="choice",
        state=state,
        question=f"{str(row['question']).strip()}\nIs the answer yes, no, or maybe?",
        options=[
            {"key": "A", "label": "yes"},
            {"key": "B", "label": "no"},
            {"key": "C", "label": "maybe"},
        ],
        gold={"yes": "A", "no": "B", "maybe": "C"}[decision],
        option_order_seed=seed,
        meta={"source_decision": decision},
    )


def _hash_split(record_id: str, dev_fraction: float) -> Split:
    from meddecide.bench.schema import split_by_record_hash

    return split_by_record_hash(record_id, dev_fraction=dev_fraction, salt="pubmedqa")


# ---------------------------------------------------------------------------
# MMLU medical subsets
# ---------------------------------------------------------------------------
MMLU_ID = "cais/mmlu"
MMLU_LICENSE = "UNKNOWN"  # resolved from the dataset card at build time
MMLU_MEDICAL_SUBJECTS = [
    "anatomy",
    "clinical_knowledge",
    "college_medicine",
    "college_biology",
    "medical_genetics",
    "professional_medicine",
]


def load_mmlu(
    by_subject: dict[str, dict[str, list[Any]]],
    *,
    cfg: dict[str, Any],
    revision: str,
) -> LoadResult:
    """MMLU medical subsets: official ``test`` is tier-1 test, official ``validation`` is dev.

    MMLU's ``dev`` split (5 items/subject) is the few-shot exemplar set and is not used.
    """
    seed = int(cfg.get("seed", 0))
    max_test, max_dev = _caps(cfg)
    result = LoadResult(
        source="mmlu_medical",
        dataset_id=MMLU_ID,
        dataset_revision=revision,
        license=MMLU_LICENSE,
        split_map={"test": MMLU_ID + ":test (6 medical subsets)",
                   "dev": MMLU_ID + ":validation (6 medical subsets)"},
        notes=[f"subjects: {', '.join(MMLU_MEDICAL_SUBJECTS)}",
               "the official `dev` split (5 exemplars/subject) is the few-shot set, not a dev set, and is not used",
               "official test = tier-1 test, official validation = dev; identical question text "
               "found on both sides is forced to test"],
    )
    for subject, splits in sorted(by_subject.items()):
        for split, rows in (
            (Split.TEST, splits.get("test", [])),
            (Split.DEV, splits.get("validation", [])),
        ):
            for row in rows:
                item = _mmlu_item(row, subject=subject, split=split, seed=seed)
                if item is None:
                    result.drop("unparseable_record")
                    continue
                result.rows.append(item)

    # MMLU contains identical stems with different answer sets, and its official
    # validation/test splits are not guaranteed disjoint by content. Identical text is
    # forced onto the test side, then caps are applied (deterministic, seeded).
    result.rows, counts, extra_notes = finalize_splits(
        result.rows, cfg=cfg, source="mmlu_medical", salt="mmlu", cap_test=max_test, cap_dev=max_dev
    )
    for reason, n in counts.items():
        result.drop(reason, n)
    result.notes.extend(extra_notes)
    return result


def _mmlu_item(row: dict[str, Any], *, subject: str, split: Split, seed: int):
    choices = row.get("choices") or []
    answer = row.get("answer")
    try:
        gold_index = int(answer)
    except (TypeError, ValueError):
        return None
    if len(choices) < 2 or not 0 <= gold_index < len(choices):
        return None
    question = str(row["question"]).strip()
    # MMLU rows carry no stable id. The record id hashes subject + stem + option labels:
    # MMLU contains identical stems with different answer sets, and an id built from the
    # stem alone would collide across genuinely different items.
    record_id = stable_hash(
        {"subject": subject, "q": question, "choices": [str(c).strip() for c in choices]}, length=20
    )
    return make_choice_item(
        source="mmlu_medical",
        record_id=record_id,
        url="https://huggingface.co/datasets/cais/mmlu",
        license_=MMLU_LICENSE,
        record_date=date(2020, 9, 1),
        split=split,
        template_id="mmlu_mc_v1",
        skill="medical_knowledge",
        state=question,
        question="Which of the following is the best answer?",
        labels=[str(c).strip() for c in choices],
        gold_index=gold_index,
        option_order_seed=seed,
        meta={"subject": subject, "source_split": split.value},
    )


def _caps(cfg: dict[str, Any]) -> tuple[int, int]:
    from meddecide.bench.tier1.common import caps

    return caps(cfg)
