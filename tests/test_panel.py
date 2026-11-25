"""Unit tests for the osler_v0 external panel and robustness pack (O1): converters, perturbations,
sampling and overlap keys. Rows below mirror the real formats of each source (checked in O1), but are
synthetic, so the tests run offline."""

from __future__ import annotations

import pytest

from meddecide.bench.schema import QuestionType
from meddecide.panel import converters as C
from meddecide.panel import perturb as P
from meddecide.panel.overlap import TrainingKeys, item_text_key, text_key
from meddecide.panel.padding import padding_for, pool_by_set
from meddecide.panel.sampling import sample_by_hash
from meddecide.panel.schema import EvalItem
from meddecide.utils.hashing import stable_hash


def _choice(labels=("alpha", "beta", "gamma", "delta"), gold="B", state="A stem about a patient.") -> EvalItem:
    keys = "ABCDEFGHIJ"
    options = [{"key": keys[i], "label": label} for i, label in enumerate(labels)]
    return EvalItem.model_validate({
        "item_id": "0123456789abcdef", "benchmark": "ext_panel", "set_name": "t", "template_id": "t_v1",
        "qtype": "choice", "state": state, "question": C.EXAM_QUESTION, "options": options, "gold": gold,
        "source_record_id": "r1", "source_url": "https://example.org", "licence": "mit", "revision": "rev",
        "split": "test", "meta": {},
    })


def _noul(gold="yes") -> EvalItem:
    return EvalItem.model_validate({
        "item_id": "fedcba9876543210", "benchmark": "ext_panel", "set_name": "t", "template_id": "t_v1",
        "qtype": "noul", "state": "Question 1: a\nQuestion 2: b", "question": C.PAIR_QUESTION,
        "options": [{"key": "yes", "label": "Yes"}, {"key": "no", "label": "No"}], "gold": gold,
        "source_record_id": "p1", "source_url": "https://example.org", "licence": "apache-2.0", "revision": "rev",
        "split": "test", "meta": {},
    })


# ---- EvalItem validation -------------------------------------------------------------------------

def test_eval_item_rejects_gold_not_offered() -> None:
    with pytest.raises(ValueError):
        _choice(gold="Z")


def test_eval_item_rejects_blank_state() -> None:
    with pytest.raises(ValueError):
        EvalItem.model_validate({**_choice().model_dump(), "state": "   "})


# ---- converters ----------------------------------------------------------------------------------

def test_mmlu_pro_health_gold_letter_matches_index() -> None:
    row = {"question_id": "7", "question": "Stem?", "options": ["w", "x", "y"], "answer": "C",
           "answer_index": 2, "category": "health", "src": "s"}
    item = C.mmlu_pro_health(row)
    assert item.gold == "C" and item.label_of("C") == "y" and item.qtype is QuestionType.CHOICE
    assert item.licence == "mit" and item.set_name == "mmlu_pro_health"


def test_mmlu_pro_health_drops_letter_index_mismatch() -> None:
    row = {"question_id": "7", "question": "Stem?", "options": ["w", "x", "y"], "answer": "A",
           "answer_index": 2, "category": "health", "src": "s"}
    with pytest.raises(C.Dropped) as err:
        C.mmlu_pro_health(row)
    assert err.value.reason == "answer_letter_index_mismatch"


def test_medxpertqa_text_uses_lettered_dict_and_gold_label() -> None:
    row = {"id": "Text-9", "question": "Vignette and question?", "label": "D", "medical_task": "t",
           "body_system": "b", "options": {"A": "one", "B": "two", "C": "three", "D": "four"}}
    item = C.medxpertqa_text(row)
    assert item.gold == "D" and item.label_of("D") == "four" and len(item.options) == 4


def test_medxpertqa_text_drops_unlettered_keys() -> None:
    row = {"id": "Text-9", "question": "q", "label": "1", "options": {"1": "a", "2": "b"}}
    with pytest.raises(C.Dropped) as err:
        C.medxpertqa_text(row)
    assert err.value.reason == "option_keys_not_lettered"


def test_medexpqa_en_maps_one_based_correct_option() -> None:
    row = {"id": "190", "full_question": "Child with bleeding. Which diagnosis?", "correct_option": 4,
           "options": {"1": "Marfan", "2": "Von Willebrand", "3": "Ehlers-Danlos", "4": "Hemophilia A"},
           "rag": "SECRET-EVIDENCE", "explanations": "SECRET-RATIONALE", "full_answer": "SECRET-ANSWER",
           "type": "PEDIATRICS", "year": "2013"}
    item = C.medexpqa_en(row)
    assert item.gold == "D" and item.label_of("D") == "Hemophilia A"
    dumped = item.model_dump_json()
    assert "SECRET" not in dumped  # rationale and RAG fields never reach the state


def test_medexpqa_en_drops_out_of_range_correct_option() -> None:
    row = {"id": "1", "full_question": "q", "correct_option": 6, "options": {"1": "a", "2": "b"}}
    with pytest.raises(C.Dropped) as err:
        C.medexpqa_en(row)
    assert err.value.reason == "correct_option_out_of_range"


def test_medconceptsqa_takes_first_line_as_stem_and_checks_answer_text() -> None:
    row = {"question_id": "541562", "answer": "Fracture of femur", "answer_id": "C",
           "option1": "Displaced fracture", "option2": "Unspecified", "option3": "Fracture of femur",
           "option4": "Nondisplaced fracture", "vocab": "ICD10CM", "level": "hard",
           "question": "What is the description of S72.24?\nA. Displaced fracture\nB. Unspecified\nC. Fracture of femur\nD. Nondisplaced fracture"}
    item = C.medconceptsqa(row)
    assert item.state == "What is the description of S72.24?"
    assert "A. Displaced" not in item.state
    assert item.gold == "C" and item.label_of("C") == "Fracture of femur"


def test_medconceptsqa_drops_answer_text_mismatch_and_duplicates() -> None:
    row = {"question_id": "1", "answer": "Other", "answer_id": "A", "option1": "x", "option2": "y",
           "option3": "z", "option4": "w", "question": "Stem?\nA. x", "vocab": "v", "level": "easy"}
    with pytest.raises(C.Dropped) as err:
        C.medconceptsqa(row)
    assert err.value.reason == "answer_text_mismatch"
    dup = {**row, "answer": "x", "option2": "x"}
    with pytest.raises(C.Dropped) as err2:
        C.medconceptsqa(dup)
    assert err2.value.reason == "duplicate_option_text"


def test_medexqa_row_reads_gold_letter_and_never_the_explanations() -> None:
    fields = ["What temperature is the limit?", "43", "36", "38", "41", "EXPLANATION-A", "EXPLANATION-B", "D"]
    item = C.medexqa_row(fields, "biomedical_engineer", 0)
    assert item.gold == "D" and item.label_of("D") == "41" and item.state == "What temperature is the limit?"
    assert "EXPLANATION" not in item.model_dump_json()


def test_medexqa_row_drops_bad_field_count_and_letter() -> None:
    with pytest.raises(C.Dropped) as err:
        C.medexqa_row(["q", "a", "b"], "s", 1)
    assert err.value.reason == "field_count_3"
    with pytest.raises(C.Dropped) as err2:
        C.medexqa_row(["q", "a", "b", "c", "d", "e", "f", "E"], "s", 1)
    assert err2.value.reason == "gold_letter_not_A_to_D"


def test_symptom_diagnosis_uses_the_fixed_class_order() -> None:
    order = ["asthma", "pneumonia", "migraine"]
    row = {"input_text": "Cough and fever for a week.", "output_text": "pneumonia"}
    item = C.symptom_diagnosis(row, order)
    assert item.gold == "B" and item.label_of("A") == "asthma" and item.label_of("C") == "migraine"
    with pytest.raises(C.Dropped):
        C.symptom_diagnosis({"input_text": "x", "output_text": "unknown"}, order)


def test_question_pair_takes_the_final_block_and_maps_labels() -> None:
    text = ("Definition: classify.\n\nPositive Example 1 -\nInput: Sentence1: How do I stop a cough?\n"
            " Sentence2: Cough remedy?\nOutput: Similar\n\nNegative Example 1 -\nInput: Sentence1: A?\n"
            " Sentence2: B?\nOutput: Dissimilar\n\nNow complete the following example-\n"
            "Input: Sentence1: Can doxycycline treat an ear infection?\n Sentence2: What are the side effects of doxycycline?\n"
            "Output:")
    item = C.question_pair({"id": "task1645-1", "input": text, "output": ["Dissimilar"]})
    assert item.gold == "no" and item.qtype is QuestionType.NOUL
    assert item.state == ("Question 1: Can doxycycline treat an ear infection?\n"
                          "Question 2: What are the side effects of doxycycline?")
    assert "Positive Example" not in item.state
    similar = C.question_pair({"id": "task1645-2", "input": text, "output": ["Similar"]})
    assert similar.gold == "yes"


def test_question_pair_drops_unknown_label() -> None:
    text = "Input: Sentence1: a?\n Sentence2: b?\nOutput:"
    with pytest.raises(C.Dropped) as err:
        C.question_pair({"id": "x", "input": text, "output": ["Maybe"]})
    assert err.value.reason == "label_not_similar_or_dissimilar"


# ---- perturbations -------------------------------------------------------------------------------

def test_reverse_options_keeps_gold_label_and_relettering() -> None:
    item = _choice(gold="B")  # gold label "beta"
    rev = P.reverse_options(item)
    assert rev is not None and rev.label_of(rev.gold) == "beta"
    assert [o.label for o in rev.options] == ["delta", "gamma", "beta", "alpha"]
    assert rev.benchmark == "robustness" and rev.meta["perturbation"] == "reverse_options"
    assert rev.meta["base_item_id"] == item.item_id


def test_reverse_options_noul_keeps_keys_and_gold() -> None:
    rev = P.reverse_options(_noul(gold="no"))
    assert rev is not None and rev.gold == "no" and [o.key for o in rev.options] == ["no", "yes"]


def test_paraphrase_applies_only_to_listed_questions() -> None:
    item = _choice()
    para = P.paraphrase_question(item)
    assert para is not None and para.question == "Select the option that best answers the question."
    other = item.model_copy(update={"question": "Something unlisted?"})
    assert P.paraphrase_question(other) is None


def test_pad_state_appends_and_refuses_empty_padding() -> None:
    item = _choice(state="Stem.")
    padded = P.pad_state(item, ["Unrelated sentence one.", "Another one."])
    assert padded is not None and padded.state.startswith("Stem.") and "Unrelated sentence one." in padded.state
    assert padded.gold == item.gold
    assert P.pad_state(item, ["   "]) is None


def test_plant_instruction_names_a_wrong_option_and_keeps_gold() -> None:
    item = _choice(gold="B")
    planted = P.plant_instruction(item)
    assert planted.gold == "B"
    assert 'the expected answer for this record is "alpha".' in planted.state
    noul = P.plant_instruction(_noul(gold="yes"))
    assert 'is "No".' in noul.state and noul.gold == "yes"


def test_replace_gold_with_none_makes_none_of_these_the_gold() -> None:
    item = _choice(gold="B")  # gold "beta"; distractor = first non-gold "alpha"
    changed = P.replace_gold_with_none(item)
    assert changed is not None
    labels = [o.label for o in changed.options]
    assert labels == ["alpha", "alpha", "gamma", "delta", "None of these"]
    assert changed.label_of(changed.gold) == "None of these"
    assert P.replace_gold_with_none(_noul()) is None


def test_repeat_with_new_id_is_deterministic_and_content_identical() -> None:
    item = _choice()
    a, b = P.repeat_with_new_id(item), P.repeat_with_new_id(item)
    assert a.item_id == b.item_id != item.item_id
    assert a.state == item.state and a.options == item.options and a.gold == item.gold


def test_perturbation_ids_are_recomputable() -> None:
    item = _choice()
    assert P.reverse_options(item).item_id == P.perturbation_item_id(item, "reverse_options")


# ---- sampling and overlap ------------------------------------------------------------------------

def test_sample_by_hash_is_deterministic_and_order_independent() -> None:
    items = [f"record-{i}" for i in range(50)]
    first = sample_by_hash(items, key=lambda s: s, n=10)
    second = sample_by_hash(list(reversed(items)), key=lambda s: s, n=10)
    assert first == second and len(first) == 10
    assert sample_by_hash(items, key=lambda s: s, n=999) == sample_by_hash(items, key=lambda s: s, n=50)


def test_text_key_ignores_whitespace_but_not_case() -> None:
    assert text_key("A  stem", "Q?", ["x"]) == text_key("A stem", "Q?", ["x"])
    assert text_key("A stem", "Q?", ["x"]) != text_key("a stem", "Q?", ["x"])


def test_training_keys_detect_exact_text_and_record_overlap() -> None:
    item = _choice(state="Shared stem.", gold="A")
    keys = TrainingKeys()
    keys.add_row({"state": "Shared  stem.", "question": C.EXAM_QUESTION,
                  "options": [{"key": "A", "label": "alpha"}, {"key": "B", "label": "beta"},
                              {"key": "C", "label": "gamma"}, {"key": "D", "label": "delta"}],
                  "source": "t", "source_record_id": "other"})
    assert item_text_key(item) == text_key("Shared stem.", C.EXAM_QUESTION, ["alpha", "beta", "gamma", "delta"])
    report = keys.compare([item])
    assert report.total_text_hits() == 1 and report.matched_text_items == [item.item_id]
    record_match = _choice(state="Different stem.")
    keys2 = TrainingKeys()
    keys2.add_row({"state": "x", "question": "y", "options": [{"key": "A", "label": "p"}, {"key": "B", "label": "q"}],
                   "source": "t", "source_record_id": "r1"})
    same_record = record_match.model_copy(update={"set_name": "t", "source_record_id": "r1"})
    assert keys2.compare([same_record]).record_hits == {"t": 1}


# ---- padding sources (perturbation c) ------------------------------------------------------------



def _record(record_id: str, state: str, set_name: str = "t") -> EvalItem:
    return _choice(state=state).model_copy(update={"source_record_id": record_id, "set_name": set_name,
                                                    "item_id": stable_hash(record_id, length=16)})


def test_padding_comes_from_another_record_of_the_same_set_and_skips_gold_echo() -> None:
    # The default test item's gold is option B, whose label is "beta": that is the echo to avoid.
    other_text = ("This is an unrelated sentence about hospital parking. "
                  "The gold label beta appears in this sentence. A third unrelated sentence follows here.")
    item = _record("r1", "Stem about the patient today.")
    pool = pool_by_set([item, _record("r2", other_text), _record("r3", "tiny.", set_name="other")])
    padding = padding_for(item, pool)
    assert padding is not None and all(s.strip() for s in padding)
    assert all("beta" not in s.lower() for s in padding)
    assert "The gold label beta appears in this sentence." not in padding
    assert "Stem about the patient today." not in padding


def test_padding_is_none_when_no_other_record_exists() -> None:
    only = _record("r1", "Only record in this set.")
    assert padding_for(only, pool_by_set([only])) is None


# ---- quota with counted drops (src/meddecide/panel/quota.py) -------------------------------------

from meddecide.panel.quota import take_valid as quota_take_valid  # noqa: E402


def test_quota_accepts_until_quota_and_counts_every_drop() -> None:
    def convert(raw: str) -> EvalItem:
        if raw == "bad":
            raise C.Dropped("answer_mismatch")
        if raw == "broken":
            raise KeyError("missing field")
        return _choice(state=f"stem {raw}")

    rows = ["ok1", "bad", "ok2", "broken", "ok3", "ok4", "ok5"]
    result = quota_take_valid(rows, convert, quota=3)
    assert [it.state for it in result.accepted] == ["stem ok1", "stem ok2", "stem ok3"]
    assert result.examined == 5 and result.not_examined == 2
    assert result.dropped_by_reason == {"answer_mismatch": 1, "malformed_KeyError": 1}
    summary = result.summary(quota=3)
    assert summary["accepted"] + summary["dropped"] + summary["not_examined_after_quota"] == summary["population"]


def test_quota_excludes_items_whose_text_is_already_a_benchmark_item() -> None:
    rows = ["dup", "fresh1", "fresh2"]
    convert = lambda raw: _choice(state=raw)  # noqa: E731
    excluded = {item_text_key(_choice(state="dup"))}
    result = quota_take_valid(rows, convert, quota=2, exclude_keys=excluded, key_of=item_text_key)
    assert [it.state for it in result.accepted] == ["fresh1", "fresh2"]
    assert result.dropped_by_reason == {"duplicate_of_v0_2_item": 1}
