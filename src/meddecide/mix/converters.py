"""Converters from replay, catalog and generator records to training rows (osler_v0 O4).

Each converter returns ``(row, reason)``: the row when kept, otherwise ``None`` with the reason (no silent drops).
Labels come from the source (answer keys, human fact-check labels, human relation and answer annotations) or from
the generator code. No language model writes text or labels.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from meddecide.mix.rows import LETTERS, before_window, make_row, parse_pubhealth_date
from meddecide.utils.hashing import stable_hash

MAX_STATE_CHARS = 40_000
GENERAL_QUESTION = "Which option answers the question?"
PUBHEALTH_LABELS = ("true", "false", "unproven", "mixture")  # bigbio/pubhealth ClassLabel order (its script)
CHEMPROT_TYPES = ("Upregulator", "Downregulator", "Agonist", "Antagonist", "Substrate", "Modulator", "Cofactor",
                  "Part_of", "Regulator", "Not")
PUBMED_URL = "https://pubmed.ncbi.nlm.nih.gov/{pmid}/"


def _mc_row(*, source: str, template: str, skill: str, rec_id: str, record_url: str, licence: str, state: str,
            question: str, choices: Iterable[tuple[str, str]], gold: str, record_date: str, tier: str,
            meta: Mapping[str, Any], split: str) -> dict[str, Any]:
    return make_row(source=source, template_id=template, skill=skill, qtype="choice", state=state,
                    question=question, options=list(choices), gold=gold, record_id=rec_id, record_url=record_url,
                    licence=licence, record_date=record_date, split=split, tier=tier, meta=dict(meta))


def multiple_choice_row(rec: Mapping[str, Any], *, split: str, source: str, template: str, skill: str,
                        record_url: str, licence: str, labels: list[str], texts: list[str], answer: str,
                        meta: Mapping[str, Any]) -> tuple[dict[str, Any] | None, str]:
    """Shared path for exam and commonsense multiple choice with an answer key."""
    stem = str(rec.get("question", "")).strip()
    if not stem:
        return None, "blank_state"
    if answer not in labels:
        return None, "answer_not_offered"
    row = _mc_row(source=source, template=template, skill=skill, rec_id=str(rec["id"]), record_url=record_url,
                  licence=licence, state=stem, question=GENERAL_QUESTION, choices=list(zip(labels, texts, strict=True)),
                  gold=answer, record_date="", tier="general_replay", meta=meta, split=split)
    return row, "kept"


def commonsense_row(rec: Mapping[str, Any], split: str) -> tuple[dict[str, Any] | None, str]:
    choices = rec["choices"]
    return multiple_choice_row(
        rec, split=split, source="commonsenseqa", template="csqa_choice_v1", skill="commonsense",
        record_url="https://huggingface.co/datasets/tau/commonsense_qa", licence="mit (Hub card)",
        labels=list(choices["label"]), texts=list(choices["text"]), answer=rec["answerKey"],
        meta={"question_concept": rec.get("question_concept")})


def qasc_row(rec: Mapping[str, Any], split: str) -> tuple[dict[str, Any] | None, str]:
    choices = rec["choices"]
    return multiple_choice_row(
        rec, split=split, source="qasc", template="qasc_science_choice_v1", skill="science",
        record_url="https://huggingface.co/datasets/allenai/qasc", licence="cc-by-4.0 (Hub card)",
        labels=list(choices["label"]), texts=list(choices["text"]), answer=rec["answerKey"],
        meta={"fact_fields_not_read": True})


def pubhealth_row(rec: Mapping[str, Any], split: str) -> tuple[dict[str, Any] | None, str]:
    """Claim verification as a yes/no question: is the fact-check's verdict 'true'? The explanation is not read."""
    try:
        label = PUBHEALTH_LABELS[int(rec["label"])]
    except (KeyError, IndexError, ValueError, TypeError):
        return None, "unknown_label"
    try:
        iso = parse_pubhealth_date(str(rec["date_published"]))
    except ValueError:
        return None, "unparseable_date"
    if not before_window(iso):
        return None, "record_date_in_window"
    claim = str(rec["claim"]).strip()
    main = str(rec.get("main_text") or "").strip()
    state = claim + ("\n\n" + main if main else "")
    if not state.strip():
        return None, "blank_state"
    if len(state) > MAX_STATE_CHARS:
        return None, "over_length_cap"
    row = make_row(source="pubhealth", template_id="pubhealth_claim_true_noul_v1", skill="claim_verification",
                   qtype="noul", state=state, question="Does the fact-check rate this health claim as true?",
                   options=[("yes", "Yes"), ("no", "No")], gold="yes" if label == "true" else "no",
                   record_id=str(rec["claim_id"]), record_url="https://huggingface.co/datasets/bigbio/pubhealth",
                   licence="mit (catalog verdict)", record_date=iso, split=split, tier="catalog",
                   meta={"fact_check_label": label})
    return row, "kept"


def _entity_text(entity: Mapping[str, Any]) -> str:
    text = entity.get("text") or []
    return str(text[0]).strip() if text else ""


def chemprot_rows(rec: Mapping[str, Any], *, pool_date: str, split: str, max_per_doc: int) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Relation-type choice items for one ChemProt abstract. At most ``max_per_doc`` relations per abstract, picked by
    stable hash of the relation id. Returns the rows and a count of every drop reason."""
    reasons: dict[str, int] = {}

    def drop(reason: str) -> None:
        reasons[reason] = reasons.get(reason, 0) + 1

    state = "\n".join(line for p in rec["passages"] for line in p["text"]).strip()
    if not state:
        drop("blank_state")
        return [], reasons
    if len(state) > MAX_STATE_CHARS:
        drop("over_length_cap")
        return [], reasons
    entities = {e["id"]: e for e in rec["entities"]}
    relations = []
    for r in rec["relations"]:
        if r["type"] not in CHEMPROT_TYPES:
            drop("relation_type_not_in_choice_set")
            continue
        relations.append(r)
    relations.sort(key=lambda r: stable_hash({"doc": rec["document_id"], "rel": r["id"]}, length=12))
    rows: list[dict[str, Any]] = []
    for r in relations[:max_per_doc]:
        e1, e2 = entities.get(r["arg1_id"]), entities.get(r["arg2_id"])
        if e1 is None or e2 is None:
            drop("missing_entity")
            continue
        t1, t2 = _entity_text(e1), _entity_text(e2)
        if not t1 or not t2:
            drop("blank_entity_text")
            continue
        choices = [(LETTERS[i], t) for i, t in enumerate(CHEMPROT_TYPES)]
        rows.append(_mc_row(
            source="chemprot", template="chemprot_relation_choice_v1", skill="relation_extraction",
            rec_id=str(rec["document_id"]), record_url=PUBMED_URL.format(pmid=rec["document_id"]),
            licence="public-domain-mark-1.0 (catalog verdict)", state=state,
            question=f"What is the relation between {t1} and {t2} in this abstract?", choices=choices,
            gold=LETTERS[CHEMPROT_TYPES.index(r["type"])], record_date=pool_date, tier="catalog",
            meta={"relation_id": r["id"], "relation_type": r["type"]}, split=split))
    reasons["kept"] = len(rows)
    return rows, reasons


def evidence_row(rec: Mapping[str, Any], *, pool_date: str, split: str) -> tuple[dict[str, Any] | None, str]:
    """Direction-of-effect choice item from an evidence context (human annotated answer)."""
    choices = list(rec["choices"])
    answer = list(rec.get("answer") or [])
    if not answer:
        return None, "no_answer"
    if answer[0] not in choices:
        return None, "answer_not_offered"
    state = str(rec["context"]).strip()
    if not state:
        return None, "blank_state"
    if len(state) > MAX_STATE_CHARS:
        return None, "over_length_cap"
    row = _mc_row(source="evidence_inference", template="evidence_direction_choice_v1",
                  skill="evidence_direction", rec_id=str(rec["document_id"]),
                  record_url=PUBMED_URL.format(pmid=rec["document_id"]), licence="mit (catalog verdict)",
                  state=state, question=str(rec["question"]).strip(),
                  choices=[(LETTERS[i], c) for i, c in enumerate(choices)],
                  gold=LETTERS[choices.index(answer[0])], record_date=pool_date, tier="catalog",
                  meta={"question_id": rec.get("question_id")}, split=split)
    return row, "kept"


def generator_row(rec: Mapping[str, Any]) -> dict[str, Any]:
    """An O3 generator item (seen generators only; the caller filters held-out rows)."""
    return make_row(
        source="generated", template_id=rec["generator"], skill=rec["family"], qtype=rec["qtype"],
        state=rec["state"], question=rec["question"],
        options=[(o["key"], o["label"]) for o in rec["options"]], gold=rec["gold"],
        record_id=rec["patient_id"], record_url="", licence="generated-by-code (this repository)",
        record_date="", split=rec["split"], tier="generated",
        meta={**rec["meta"], "pair_id": rec["pair_id"], "role": rec["role"]})
