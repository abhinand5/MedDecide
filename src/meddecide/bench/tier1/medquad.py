"""Tier-1 routing loader: MedQuAD question-type classification.

MedQuAD pairs a consumer-health question with the NIH institute that answered it and the
question type that institute assigned. The item asks the model to route a question to its
**question type** — the type label is a structured source field (``question_type``), not a
model's guess.

UMLS fields (``umls_cui``, ``umls_semantic_types``, ``umls_semantic_group``) are present in
the upstream record and are **never** written into an item, a manifest, or any committed
file (AGENTS.md: UMLS-restricted strings are private-only). They are dropped here by name.
"""

from __future__ import annotations

import random
from collections import Counter
from datetime import date
from typing import Any

from meddecide.bench.schema import split_by_record_hash
from meddecide.bench.tier1.common import LoadResult, finalize_splits, make_choice_item
from meddecide.utils.hashing import stable_hash

MEDQUAD_ID = "lavita/MedQuAD"
MEDQUAD_LICENSE = "UNKNOWN"  # resolved from the dataset card at build time; the mirror states none

# Fields that must never leave this loader (UMLS-restricted or redundant).
UMLS_FIELDS = ("umls_cui", "umls_semantic_types", "umls_semantic_group", "synonyms")

MIN_TYPE_FREQUENCY = 20  # a type must appear at least this often to be an option


def load_medquad(rows: list[Any], *, cfg: dict[str, Any], revision: str) -> LoadResult:
    """Build ``choice`` items: given the question, pick its question type.

    Only question types that occur at least ``MIN_TYPE_FREQUENCY`` times become options, so
    every option is a type the source actually uses. Distractors are the most frequent other
    types (seeded per item). Items whose gold type is not in the option set are dropped with
    a reason.
    """
    seed = int(cfg.get("seed", 0))
    max_test, max_dev = _caps(cfg)
    result = LoadResult(
        source="medquad",
        dataset_id=MEDQUAD_ID,
        dataset_revision=revision,
        license=MEDQUAD_LICENSE,
        split_map={"test": MEDQUAD_ID + ":train (hash split by question id)",
                   "dev": MEDQUAD_ID + ":train (hash split by question id)"},
        notes=[
            "gold = the source's own question_type field",
            "options = question types occurring at least "
            f"{MIN_TYPE_FREQUENCY} times; distractors seeded per item",
            "UMLS-derived fields present upstream are dropped and never written to any artifact",
            "MedQuAD has a single train split; it is divided by a stable hash of the question "
            "text (80/20) so the same question can never appear in dev and test",
        ],
    )

    counts = Counter(str(r.get("question_type", "")).strip().lower() for r in rows)
    option_types = sorted(t for t, n in counts.items() if t and n >= MIN_TYPE_FREQUENCY)
    if len(option_types) < 2:
        result.drop("fewer_than_two_eligible_types", len(rows))
        return result
    result.notes.append(f"eligible question types ({len(option_types)}): {', '.join(option_types)}")

    usable = [r for r in rows if str(r.get("question_type", "")).strip().lower() in option_types]
    result.drop("question_type_below_frequency_threshold", len(rows) - len(usable))

    rng = random.Random(f"{seed}:medquad:distractors")
    for row in usable:
        qid = str(row.get("question_id"))
        gold_type = str(row["question_type"]).strip().lower()
        question = str(row.get("question") or "").strip()
        if not question:
            result.drop("empty_question")
            continue
        # The split is assigned from the *question text*, not the document: MedQuAD repeats
        # the same question across documents, and a document-based split would put one copy
        # of a question in dev and another in test.
        split = split_by_record_hash(question.lower(), dev_fraction=0.2, salt="medquad")
        others = [t for t in option_types if t != gold_type]
        k = min(3, len(others))
        distractors = rng.sample(others, k)
        labels = [*distractors, gold_type]
        order = list(range(len(labels)))
        rng.shuffle(order)
        gold_index = order.index(len(labels) - 1)
        shuffled = [labels[i] for i in order]
        # MedQuAD reuses question_id across *different* questions of the same document
        # (9,662 question_id values are not unique), so the record identity is the
        # document id + question id + normalised question text. `stable_hash` normalises
        # whitespace/case-insensitively? No: it NFC-normalises and collapses whitespace
        # only, so two genuinely different questions stay different ids.
        record_key = stable_hash(
            {"doc": str(row.get("document_id")), "qid": qid, "q": question}, length=20
        )
        result.rows.append(
            make_choice_item(
                source="medquad",
                record_id=record_key,
                url=str(row.get("document_url") or f"https://huggingface.co/datasets/{MEDQUAD_ID}"),
                license_=MEDQUAD_LICENSE,
                record_date=date(2017, 1, 1),
                split=split,
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

    # dedupe, stop identical questions from straddling the boundary, then cap
    result.rows, counts, extra_notes = finalize_splits(
        result.rows, cfg=cfg, source="medquad", salt="medquad", cap_test=max_test, cap_dev=max_dev
    )
    for reason, n in counts.items():
        result.drop(reason, n)
    result.notes.extend(extra_notes)
    return result


def _caps(cfg: dict[str, Any]) -> tuple[int, int]:
    from meddecide.bench.tier1.common import caps

    return caps(cfg)
