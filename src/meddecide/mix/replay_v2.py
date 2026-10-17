"""Share-alike replay for osler_v2 (ADVISORY W2; D36).

D36 allows CC BY-SA sources for **training only**: the training data is never distributed, so the share-alike clause does not
reach the weights. Each source keeps its licence, recorded per source below. Converters return ``(row, reason)`` like the
other mix converters, so every drop is counted (no silent drops). Train splits only.

ARC, BoolQ and SNLI were dropped as "not permissive" in osler_v1 (``replay_v1.CANDIDATES``); D36 changes that rule for
training. OpenBookQA stays dropped: its Hub card states no licence, and the ADVISORY keeps only verifiable licences.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from meddecide.mix.converters import MAX_STATE_CHARS
from meddecide.mix.rows import LETTERS, make_row

ARC_TEMPLATE = "arc_science_choice_v1"
BOOLQ_TEMPLATE = "boolq_passage_noul_v1"
SNLI_TEMPLATE = "snli_nli_choice_v1"
SNLI_LABELS = {0: ("A", "Entailment"), 1: ("B", "Neutral"), 2: ("C", "Contradiction")}

# source -> (decision, licence as stated on the Hub card, where it is stated). Decision applies D36 (training only).
CANDIDATES_V2: dict[str, tuple[str, str, str]] = {
    "ai2_arc": ("kept", "CC BY-SA 4.0", "allenai/ai2_arc Hub card; share-alike; training only (D36)"),
    "boolq": ("kept", "CC BY-SA 3.0", "google/boolq Hub card; share-alike; training only (D36)"),
    "snli": ("kept", "CC BY-SA 4.0", "stanfordnlp/snli Hub card; share-alike; training only (D36)"),
    "openbookqa": ("dropped", "unknown (Hub card states no licence)",
                   "allenai/openbookqa Hub card: no licence stated; the ADVISORY keeps only verifiable licences"),
}


def arc_row(rec: Mapping[str, Any], split: str, config: str) -> tuple[dict[str, Any] | None, str]:
    """ARC: the question is the state; the options are the choices, re-keyed A, B, C ... in their printed order."""
    question = str(rec.get("question", "")).strip()
    if not question:
        return None, "blank_state"
    choices = rec.get("choices") or {}
    labels = [str(x) for x in choices.get("label", [])]
    texts = [str(x).strip() for x in choices.get("text", [])]
    if len(labels) != len(texts) or not 2 <= len(texts) <= len(LETTERS):
        return None, "bad_choices"
    answer_key = str(rec.get("answerKey", "")).strip()
    if answer_key not in labels:
        return None, "answer_missing"
    keys = list(LETTERS[:len(texts)])
    gold = keys[labels.index(answer_key)]
    row = make_row(source=config.lower().replace("-", "_"), template_id=ARC_TEMPLATE, skill="science",
                   qtype="choice", state=question, question="Which option answers the question?",
                   options=list(zip(keys, texts, strict=True)), gold=gold, record_id=str(rec["id"]),
                   record_url="https://huggingface.co/datasets/allenai/ai2_arc",
                   licence=CANDIDATES_V2["ai2_arc"][1] + " (training only, D36)", record_date="", split=split,
                   tier="general_replay", meta={"config": config})
    return row, "kept"


def boolq_row(rec: Mapping[str, Any], split: str, index: int) -> tuple[dict[str, Any] | None, str]:
    """BoolQ: the passage is the state; the question is asked as a yes/no item; gold from the human answer."""
    passage = str(rec.get("passage", "")).strip()
    question = str(rec.get("question", "")).strip().rstrip("?")
    if not passage or not question:
        return None, "blank_state"
    if len(passage) > MAX_STATE_CHARS:
        return None, "over_length_cap"
    answer = rec.get("answer")
    if not isinstance(answer, bool):
        return None, "answer_missing"
    row = make_row(source="boolq", template_id=BOOLQ_TEMPLATE, skill="reading_comprehension", qtype="noul",
                   state=passage, question=f"Based on the passage, is the answer to this question yes: {question}?",
                   options=[("yes", "Yes"), ("no", "No")], gold="yes" if answer else "no",
                   record_id=f"{split}-{index}", record_url="https://huggingface.co/datasets/google/boolq",
                   licence=CANDIDATES_V2["boolq"][1] + " (training only, D36)", record_date="", split=split,
                   tier="general_replay", meta={})
    return row, "kept"


def snli_row(rec: Mapping[str, Any], split: str, index: int) -> tuple[dict[str, Any] | None, str]:
    """SNLI: premise and hypothesis are the state; gold is the human gold label (label -1 means no gold: dropped)."""
    premise = str(rec.get("premise", "")).strip()
    hypothesis = str(rec.get("hypothesis", "")).strip()
    if not premise or not hypothesis:
        return None, "blank_state"
    label = rec.get("label")
    if label not in SNLI_LABELS:
        return None, "no_gold_label"
    options = [SNLI_LABELS[k] for k in (0, 1, 2)]
    row = make_row(source="snli", template_id=SNLI_TEMPLATE, skill="natural_language_inference", qtype="choice",
                   state=f"Premise: {premise}\nHypothesis: {hypothesis}",
                   question="Does the premise entail the hypothesis, contradict it, or neither (neutral)?",
                   options=options, gold=SNLI_LABELS[label][0], record_id=f"{split}-{index}",
                   record_url="https://huggingface.co/datasets/stanfordnlp/snli",
                   licence=CANDIDATES_V2["snli"][1] + " (training only, D36)", record_date="", split=split,
                   tier="general_replay", meta={})
    return row, "kept"
