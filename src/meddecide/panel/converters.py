"""Converters from public evaluation sets to typed decisions (osler_v0 O1, external panel).

Every converter takes one raw row and returns an ``EvalItem``, or raises ``Dropped`` with a named
reason. The builder counts drops by reason, so no row disappears silently (AGENTS.md data rules).

Conventions (the same as the v0.2 tier-1 items): ``state`` holds the stem or vignette,
``question`` holds the instruction, options are keyed A, B, C … for choice and ``yes``/``no`` for
noul. Fields that carry answers or rationales (explanations, RAG passages, full answers) are never
copied into the state.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from meddecide.bench.schema import Option, QuestionType, compute_item_id
from meddecide.panel.schema import EvalItem
from meddecide.utils.hashing import normalize_text

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
EXAM_QUESTION = "Which of the following is the best answer?"
DIAGNOSIS_QUESTION = "Which diagnosis best matches the symptoms?"
PAIR_QUESTION = "Do the two questions have the same meaning?"


class Dropped(Exception):
    """A row the converter cannot turn into a valid decision. ``reason`` is counted, never hidden."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class SourceSpec:
    """Pinned metadata of one public set: where it comes from and under what licence."""

    set_name: str
    repo: str
    revision: str
    licence: str
    source_url: str
    template_id: str
    split: str = "test"
    licence_note: str = ""


MMLU_PRO = SourceSpec(
    set_name="mmlu_pro_health", repo="TIGER-Lab/MMLU-Pro",
    revision="b189ec765aa7ed75c8acfea42df31fdae71f97be", licence="mit",
    source_url="https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro",
    template_id="ext_mmlu_pro_health_choice_v1",
)
MEDXPERTQA_TEXT = SourceSpec(
    set_name="medxpertqa_text", repo="TsinghuaC3I/MedXpertQA",
    revision="7e7c465a68eb2b866926bfa59c8c9d17a8daba65", licence="mit",
    source_url="https://huggingface.co/datasets/TsinghuaC3I/MedXpertQA",
    template_id="ext_medxpertqa_text_choice_v1",
)
MEDEXPQA_EN = SourceSpec(
    set_name="medexpqa_en", repo="HiTZ/MedExpQA",
    revision="78b57699763e00e38acd550f28de24d8a65f1643", licence="cc-by-4.0",
    source_url="https://huggingface.co/datasets/HiTZ/MedExpQA",
    template_id="ext_medexpqa_en_choice_v1",
)
MEDCONCEPTSQA = SourceSpec(
    set_name="medconceptsqa_all", repo="ofir408/MedConceptsQA",
    revision="98c30d83762e51a397c9a7b0eeee6722e751da17", licence="apache-2.0",
    source_url="https://huggingface.co/datasets/ofir408/MedConceptsQA",
    template_id="ext_medconceptsqa_choice_v1",
)
MEDEXQA = SourceSpec(
    set_name="medexqa_test", repo="bluesky333/MedExQA",
    revision="3a9f80d6de2c354956d7e86d323214b92d547d6e", licence="cc-by-nc-sa-4.0",
    source_url="https://huggingface.co/datasets/bluesky333/MedExQA",
    template_id="ext_medexqa_choice_v1",
    licence_note="CC BY-NC-SA 4.0: evaluation only here; release of derived items needs an operator decision",
)
SYMPTOM_DIAGNOSIS = SourceSpec(
    set_name="symptom_to_diagnosis", repo="gretelai/symptom_to_diagnosis",
    revision="722cfb0e11f8ae37339c7f573b5e10429b94df49", licence="apache-2.0",
    source_url="https://huggingface.co/datasets/gretelai/symptom_to_diagnosis",
    template_id="ext_symptom_to_diagnosis_choice_v1",
)
QUESTION_PAIRS = SourceSpec(
    set_name="medical_question_pairs", repo="Lots-of-LoRAs/task1645_medical_question_pair_dataset_text_classification",
    revision="b7ea5fb65be8a32f2fbabe49d070ba22c077782a", licence="apache-2.0",
    source_url="https://huggingface.co/datasets/Lots-of-LoRAs/task1645_medical_question_pair_dataset_text_classification",
    template_id="ext_medical_question_pairs_noul_v1",
    licence_note="Apache-2.0 on the Hub wrapper (Super-NaturalInstructions task 1645); the underlying source licence is not verified",
)

SOURCES = {s.set_name: s for s in [MMLU_PRO, MEDXPERTQA_TEXT, MEDEXPQA_EN, MEDCONCEPTSQA, MEDEXQA,
                                   SYMPTOM_DIAGNOSIS, QUESTION_PAIRS]}


def _build_choice(spec: SourceSpec, record_id: str, state: str, question: str,
                  labels: Sequence[str], gold_index: int, meta: Mapping[str, Any] | None = None) -> EvalItem:
    """Shared shape: lettered options in the given order, gold by index."""
    if not 2 <= len(labels) <= len(LETTERS):
        raise Dropped(f"option_count_{len(labels)}")
    if not 0 <= gold_index < len(labels):
        raise Dropped("gold_index_out_of_range")
    cleaned = [normalize_text(label) for label in labels]
    if any(not label for label in cleaned):
        raise Dropped("blank_option")
    if len(set(cleaned)) != len(cleaned):
        raise Dropped("duplicate_option_text")
    keys = [LETTERS[i] for i in range(len(labels))]
    return EvalItem(
        item_id=compute_item_id(spec.set_name, record_id, spec.template_id, 0, spec.split),
        benchmark="ext_panel",
        set_name=spec.set_name,
        template_id=spec.template_id,
        qtype=QuestionType.CHOICE,
        state=state,
        question=question,
        options=[Option(key=k, label=label) for k, label in zip(keys, cleaned, strict=True)],
        gold=keys[gold_index],
        source_record_id=record_id,
        source_url=spec.source_url,
        licence=spec.licence,
        revision=spec.revision,
        split=spec.split,
        meta=dict(meta or {}),
    )


def mmlu_pro_health(row: Mapping[str, Any]) -> EvalItem:
    """TIGER-Lab/MMLU-Pro (test, category ``health``). Gold is the letter; checked against index."""
    options = [str(o) for o in row["options"]]
    answer = str(row["answer"]).strip()
    index = int(row["answer_index"])
    if not 0 <= index < len(options):
        raise Dropped("answer_index_out_of_range")
    if LETTERS[index] != answer:
        raise Dropped("answer_letter_index_mismatch")
    return _build_choice(MMLU_PRO, str(row["question_id"]), str(row["question"]), EXAM_QUESTION,
                         options, index, {"category": str(row["category"]), "src": str(row.get("src", ""))})


def medxpertqa_text(row: Mapping[str, Any]) -> EvalItem:
    """TsinghuaC3I/MedXpertQA Text (test). Options arrive as a dict keyed A..J."""
    options = row["options"]
    if not isinstance(options, Mapping):
        raise Dropped("options_not_mapping")
    keys = list(options.keys())
    if keys != list(LETTERS[: len(keys)]):
        raise Dropped("option_keys_not_lettered")
    label = str(row["label"]).strip()
    if label not in options:
        raise Dropped("label_not_offered")
    return _build_choice(MEDXPERTQA_TEXT, str(row["id"]), str(row["question"]), EXAM_QUESTION,
                         [str(options[k]) for k in keys], keys.index(label),
                         {"medical_task": str(row.get("medical_task", "")),
                          "body_system": str(row.get("body_system", ""))})


def medexpqa_en(row: Mapping[str, Any]) -> EvalItem:
    """HiTZ/MedExpQA, English test (casimedicos). Options keyed "1".."N"; ``correct_option`` is 1-based.

    Only ``full_question`` and ``options`` are read. ``rag``, ``explanations`` and ``full_answer``
    would carry the answer and are never copied.
    """
    options = row["options"]
    if not isinstance(options, Mapping):
        raise Dropped("options_not_mapping")
    keys = list(options.keys())
    if keys != [str(i + 1) for i in range(len(keys))]:
        raise Dropped("option_keys_not_numbered_from_1")
    correct = int(row["correct_option"])
    if not 1 <= correct <= len(keys):
        raise Dropped("correct_option_out_of_range")
    return _build_choice(MEDEXPQA_EN, str(row["id"]), str(row["full_question"]), EXAM_QUESTION,
                         [str(options[k]) for k in keys], correct - 1,
                         {"subject": str(row.get("type", "")), "year": str(row.get("year", ""))})


def medconceptsqa(row: Mapping[str, Any]) -> EvalItem:
    """ofir408/MedConceptsQA (all test shards). The stem is the first line; options come from option1..4.

    The inline ``A. … B. …`` list inside ``question`` is not used as the state, so the state never
    repeats the options. The gold letter must select the option whose text equals ``answer``.
    """
    stem = str(row["question"]).split("\n", 1)[0].strip()
    if not stem or re.match(r"^[A-D]\.\s", stem):
        raise Dropped("stem_not_found")
    labels = [str(row[f"option{i}"]) for i in range(1, 5)]
    answer_id = str(row["answer_id"]).strip()
    if answer_id not in "ABCD" or len(answer_id) != 1:
        raise Dropped("answer_id_not_a_letter")
    gold_index = "ABCD".index(answer_id)
    if normalize_text(labels[gold_index]) != normalize_text(str(row["answer"])):
        raise Dropped("answer_text_mismatch")
    return _build_choice(MEDCONCEPTSQA, str(row["question_id"]), stem, EXAM_QUESTION, labels, gold_index,
                         {"vocab": str(row.get("vocab", "")), "level": str(row.get("level", ""))})


def medexqa_row(fields: Sequence[str], specialty: str, line_no: int) -> EvalItem:
    """One tab-separated line of a MedExQA test file: question, four options, two explanations, gold letter.

    The explanations are never read. The gold letter is the last field and must be A to D.
    """
    if len(fields) != 8:
        raise Dropped(f"field_count_{len(fields)}")
    question, options, letter = fields[0].strip(), [f.strip() for f in fields[1:5]], fields[7].strip()
    if letter not in "ABCD" or len(letter) != 1:
        raise Dropped("gold_letter_not_A_to_D")
    return _build_choice(MEDEXQA, f"{specialty}:{line_no}", question, EXAM_QUESTION, options,
                         "ABCD".index(letter), {"specialty": specialty})


def symptom_diagnosis(row: Mapping[str, Any], class_order: Sequence[str]) -> EvalItem:
    """gretelai/symptom_to_diagnosis (test). Options are the test split's diagnoses, sorted and lettered.

    ``class_order`` is computed once from the test split, so every item offers the same list in the
    same order (no per-item shuffling; the scoreboard's option-order protocol handles order effects).
    """
    gold = str(row["output_text"]).strip()
    if gold not in class_order:
        raise Dropped("gold_not_in_class_list")
    record_id = hashlib.sha256(str(row["input_text"]).encode("utf-8")).hexdigest()[:16]
    return _build_choice(SYMPTOM_DIAGNOSIS, record_id, str(row["input_text"]), DIAGNOSIS_QUESTION,
                         list(class_order), list(class_order).index(gold))


_PAIR = re.compile(
    r"Input:\s*Sentence1:\s*(?P<q1>.*?)\s*\n\s*Sentence2:\s*(?P<q2>.*?)\s*\nOutput:",
    re.S,
)


def question_pair(row: Mapping[str, Any]) -> EvalItem:
    """task1645 medical question pairs (test). The test pair is the last ``Input:`` block of the prompt.

    The few-shot examples inside the definition are skipped, since only the final block is the item.
    Output ``Similar`` maps to the noul key ``yes``, ``Dissimilar`` to ``no``.
    """
    text = str(row["input"])
    blocks = list(_PAIR.finditer(text))
    if not blocks:
        raise Dropped("pair_block_not_found")
    last = blocks[-1]
    if text[last.end():].count("Input:") > 0:
        raise Dropped("pair_not_last_block")
    q1, q2 = normalize_text(last.group("q1")), normalize_text(last.group("q2"))
    if not q1 or not q2:
        raise Dropped("blank_question")
    output = row["output"]
    label = str(output[0]) if hasattr(output, "__len__") and len(output) == 1 else str(output)
    if label not in ("Similar", "Dissimilar"):
        raise Dropped("label_not_similar_or_dissimilar")
    gold = "yes" if label == "Similar" else "no"
    state = f"Question 1: {q1}\nQuestion 2: {q2}"
    return EvalItem(
        item_id=compute_item_id(QUESTION_PAIRS.set_name, str(row["id"]), QUESTION_PAIRS.template_id, 0,
                                QUESTION_PAIRS.split),
        benchmark="ext_panel",
        set_name=QUESTION_PAIRS.set_name,
        template_id=QUESTION_PAIRS.template_id,
        qtype=QuestionType.NOUL,
        state=state,
        question=PAIR_QUESTION,
        options=[Option(key="yes", label="Yes"), Option(key="no", label="No")],
        gold=gold,
        source_record_id=str(row["id"]),
        source_url=QUESTION_PAIRS.source_url,
        licence=QUESTION_PAIRS.licence,
        revision=QUESTION_PAIRS.revision,
        split=QUESTION_PAIRS.split,
        meta={"label_text": label},
    )
