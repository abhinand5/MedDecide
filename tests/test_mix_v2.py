"""Unit tests for training mix v2 (osler_v0 O4): row schema, source converters, gold-preserving augmentation and
leakage helpers. All records here are small synthetic dicts; no benchmark or training item is read."""

from __future__ import annotations

import pytest

from meddecide.mix.augment import (
    pad_state,
    plant_instruction,
    replace_gold_with_none,
    reverse_options,
    select,
)
from meddecide.mix.converters import (
    CHEMPROT_TYPES,
    chemprot_rows,
    commonsense_row,
    evidence_row,
    pubhealth_row,
    qasc_row,
)
from meddecide.mix.leakage import date_violations, leak_counts, row_text_key
from meddecide.mix.rows import make_row, parse_pubhealth_date

STUDENT_KEYS = {"item_id", "tier", "source", "source_record_id", "source_url", "source_license", "record_date",
                "split", "template_id", "skill", "qtype", "state", "question", "options", "gold", "meta"}


def _choice_row(gold: str = "B", n: int = 3) -> dict:
    options = [(chr(ord("A") + i), f"option {i}") for i in range(n)]
    return make_row(source="toy", template_id="toy_choice_v1", skill="toy", qtype="choice", state="Stem text.",
                    question="Which?", options=options, gold=gold, record_id="r1", record_url="",
                    licence="test", record_date="", split="train", tier="general_replay", meta={})


def test_make_row_has_the_student_v1_key_set_and_a_stable_id() -> None:
    a = _choice_row()
    assert set(a) == STUDENT_KEYS
    assert a["item_id"] == _choice_row()["item_id"]
    assert a["options"][0] == {"key": "A", "label": "option 0", "description": None}


@pytest.mark.parametrize("kwargs", [
    {"gold": "Z"},
    {"options": [("A", "x"), ("A", "y")]},
    {"state": "   "},
])
def test_make_row_refuses_a_gold_key_that_is_not_offered_duplicate_keys_or_a_blank_state(kwargs) -> None:
    base = {"source": "toy", "template_id": "t", "skill": "s", "qtype": "choice", "state": "x", "question": "q",
            "options": [("A", "a"), ("B", "b")], "gold": "A", "record_id": "r", "record_url": "", "licence": "l",
            "record_date": "", "split": "train", "tier": "t", "meta": {}}
    base.update(kwargs)
    with pytest.raises(ValueError):
        make_row(**base)


def test_commonsense_and_qasc_take_the_answer_key_and_refuse_a_missing_key() -> None:
    rec = {"id": "c1", "question": "Where do you keep a fish?", "question_concept": "fish",
           "choices": {"label": ["A", "B"], "text": ["bowl", "tree"]}, "answerKey": "A"}
    row, reason = commonsense_row(rec, "train")
    assert reason == "kept" and row["gold"] == "A" and row["options"][0]["label"] == "bowl"
    assert row["question"] == "Which option answers the question?" and row["state"] == "Where do you keep a fish?"
    rec["answerKey"] = "Z"
    assert commonsense_row(rec, "train") == (None, "answer_not_offered")
    q = {"id": "q1", "question": "What forms clouds?", "choices": {"label": ["A", "B"], "text": ["rain", "dust"]},
         "answerKey": "B", "fact1": "FACTONE", "fact2": "FACTTWO", "combinedfact": "COMBINEDFACT",
         "formatted_question": "x"}
    row, reason = qasc_row(q, "dev")
    assert reason == "kept" and row["gold"] == "B" and row["split"] == "dev"
    assert "FACT" not in row["state"]  # the fact fields are not read


def test_pubhealth_maps_the_verdict_and_reads_no_explanation() -> None:
    rec = {"claim_id": "7", "claim": "Claim text.", "date_published": "April 26, 2015",
           "explanation": "verdict reasoning", "main_text": "Article body.", "label": 0}
    row, reason = pubhealth_row(rec, "train")
    assert reason == "kept" and row["gold"] == "yes" and row["qtype"] == "noul"
    assert row["state"] == "Claim text.\n\nArticle body." and "verdict reasoning" not in row["state"]
    rec["label"] = 1
    assert pubhealth_row(rec, "train")[0]["gold"] == "no"
    rec["label"] = 9
    assert pubhealth_row(rec, "train") == (None, "unknown_label")
    rec["label"] = 0
    rec["date_published"] = "March 1, 2026"
    assert pubhealth_row(rec, "train") == (None, "record_date_in_window")
    assert parse_pubhealth_date("April 26, 2015") == "2015-04-26"


def test_chemprot_keeps_typed_relations_and_counts_each_drop() -> None:
    rec = {"document_id": "123",
           "passages": [{"id": "p", "type": "title", "text": ["Title text."]},
                        {"id": "a", "type": "abstract", "text": ["Abstract text."]}],
           "entities": [{"id": "T1", "text": ["drugA"]}, {"id": "T2", "text": ["geneB"]},
                        {"id": "T3", "text": ["geneC"]}],
           "relations": [{"id": "R1", "type": "Downregulator", "arg1_id": "T1", "arg2_id": "T2"},
                         {"id": "R2", "type": "Undefined", "arg1_id": "T1", "arg2_id": "T3"},
                         {"id": "R3", "type": "Agonist", "arg1_id": "T1", "arg2_id": "T9"}]}
    rows, reasons = chemprot_rows(rec, pool_date="2024-05-01", split="train", max_per_doc=3)
    assert len(rows) == 1
    assert rows[0]["gold"] == chr(ord("A") + CHEMPROT_TYPES.index("Downregulator"))
    assert rows[0]["record_date"] == "2024-05-01" and "Abstract text." in rows[0]["state"]
    assert reasons == {"relation_type_not_in_choice_set": 1, "missing_entity": 1, "kept": 1}


def test_chemprot_caps_relations_per_abstract() -> None:
    entities = [{"id": f"T{i}", "text": [f"e{i}"]} for i in range(12)]
    relations = [{"id": f"R{i}", "type": "Agonist", "arg1_id": f"T{i}", "arg2_id": f"T{i + 1}"} for i in range(6)]
    rec = {"document_id": "9", "passages": [{"id": "a", "type": "abstract", "text": ["Text."]}],
           "entities": entities, "relations": relations}
    rows, _ = chemprot_rows(rec, pool_date="2025-01-01", split="dev", max_per_doc=2)
    assert len(rows) == 2


def test_evidence_takes_the_annotated_answer_and_drops_unanswered_items() -> None:
    rec = {"document_id": "9", "question_id": "q1", "question": "Compared to X, what was Y?",
           "choices": ["significantly increased", "no significant difference", "significantly decreased"],
           "answer": ["no significant difference"], "context": "Results: no significant difference was found."}
    row, reason = evidence_row(rec, pool_date="2024-05-01", split="train")
    assert reason == "kept" and row["gold"] == "B"
    rec["answer"] = []
    assert evidence_row(rec, pool_date="2024-05-01", split="train") == (None, "no_answer")


def test_reverse_options_keeps_the_gold_label_and_relabels_the_keys() -> None:
    row = _choice_row(gold="B", n=3)
    out = reverse_options(row)
    assert [o["label"] for o in out["options"]] == ["option 2", "option 1", "option 0"]
    assert out["gold"] == "B" and out["options"][1]["label"] == "option 1"
    assert out["tier"] == "augmented" and out["meta"]["augmentation"] == "reverse_options"
    assert out["meta"]["base_item_id"] == row["item_id"] and out["item_id"] != row["item_id"]
    yes_no = make_row(source="t", template_id="t", skill="s", qtype="noul", state="x", question="q",
                      options=[("yes", "Yes"), ("no", "No")], gold="yes", record_id="r", record_url="",
                      licence="l", record_date="", split="train", tier="t", meta={})
    assert reverse_options(yes_no) is None


def test_replace_gold_with_none_makes_none_of_these_the_gold() -> None:
    row = _choice_row(gold="A", n=3)
    out = replace_gold_with_none(row)
    assert len(out["options"]) == 4 and out["options"][-1]["label"] == "None of these"
    assert out["gold"] == "D" and out["options"][0]["label"] == "option 1"


def test_plant_instruction_and_padding_keep_the_gold_key() -> None:
    row = _choice_row(gold="A", n=3)
    planted = plant_instruction(row)
    assert planted["gold"] == "A" and 'the expected answer for this record is "option 1"' in planted["state"]
    padded = pad_state(row, ["Unrelated sentence one. Two."])
    assert padded["gold"] == "A" and padded["state"].endswith("Unrelated sentence one. Two.")
    assert pad_state(row, [" "]) is None


def test_select_is_deterministic_and_near_the_requested_fraction() -> None:
    ids = [f"id{i:06d}" for i in range(20_000)]
    chosen = [i for i in ids if select(i, "pad_state", 0.03)]
    assert chosen == [i for i in ids if select(i, "pad_state", 0.03)]
    assert 0.025 < len(chosen) / len(ids) < 0.035


def test_leakage_counts_text_and_record_hits_and_window_violations() -> None:
    row = _choice_row()
    counts = leak_counts([row], {"heldout": {row_text_key(row)}, "other": set()}, {"v02": {"r1"}})
    assert counts == {"text:heldout": 1, "text:other": 0, "record:v02": 1}
    dated = {**row, "record_date": "2026-03-01"}
    early = {**row, "record_date": "2026-02-28"}
    assert date_violations([dated, early, row], "2026-03-01") == 1


def test_pubhealth_missing_date_is_its_own_drop_reason() -> None:
    rec = {"claim_id": "8", "claim": "Claim.", "date_published": "  ", "main_text": "", "label": 0}
    assert pubhealth_row(rec, "train") == (None, "missing_date")
