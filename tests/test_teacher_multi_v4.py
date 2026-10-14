"""Teacher pages that document several diagnoses (osler_v1 ADVISORY V4, the v2 ICD design): the condition matching and the
verifier that checks every diagnosis on the page, without a network call."""

from __future__ import annotations

from meddecide.mr import icd
from meddecide.mr.prose_build import bucket
from meddecide.mr.teacher_prose import ProseRequest, condition_matches, diagnosis_facts, verify


def _request(*dxs: icd.Diagnosis) -> ProseRequest:
    return ProseRequest(kind="diagnoses", page_class="progress_note",
                        facts={"diagnoses": [diagnosis_facts(d) for d in dxs]}, setting="inpatient")


def _entry(dx: icd.Diagnosis) -> dict:
    fact = diagnosis_facts(dx)
    return {"condition": fact["condition"], "attributes": fact["attributes"], "certainty": fact["certainty"],
            "subject": fact["subject"], "status": fact["status"]}


TARGET = icd.Diagnosis("respiratory_failure", {"acuity": "acute", "gas": "hypoxia"}, certainty="confirmed")
DISTRACTOR = icd.Diagnosis("pneumonia_organism", {"organism": "unspecified"}, certainty="probable")
OTHER = icd.Diagnosis("knee_oa", {"laterality": "right"}, certainty="suspected")


def test_the_british_and_american_spellings_match_and_unrelated_conditions_do_not() -> None:
    assert condition_matches("hyponatremia", "Hyponatraemia (serum sodium 128)")  # family key; condition is British
    assert condition_matches("respiratory_failure", "acute hypoxic respiratory failure")
    assert not condition_matches("respiratory_failure", "pneumonia")
    assert not condition_matches("pneumonia_organism", "acute respiratory failure")


def test_a_page_that_states_every_diagnosis_exactly_is_verified() -> None:
    req = _request(TARGET, DISTRACTOR, OTHER)
    extracted = {"diagnoses": [_entry(TARGET), _entry(DISTRACTOR), _entry(OTHER)]}
    assert verify(req, extracted) == []


def test_a_missing_distractor_is_a_condition_not_reported() -> None:
    req = _request(TARGET, DISTRACTOR)
    problems = verify(req, {"diagnoses": [_entry(TARGET)]})
    assert problems == ["condition not reported: pneumonia"]
    assert [bucket(p) for p in problems] == ["condition_not_reported"]


def test_each_field_mismatch_is_reported_with_its_bucket() -> None:
    req = _request(TARGET, DISTRACTOR)
    wrong = dict(_entry(TARGET), certainty="probable", subject="family", status="history")
    wrong["attributes"] = {**wrong["attributes"], "acuity": "chronic"}
    problems = verify(req, {"diagnoses": [wrong, _entry(DISTRACTOR)]})
    assert {bucket(p) for p in problems} == {"certainty", "subject", "status", "attribute"}


def test_a_paraphrased_condition_name_still_matches_when_its_content_words_are_present() -> None:
    req = _request(TARGET)
    entry = dict(_entry(TARGET), condition="acute hypoxic respiratory failure, documented on admission")
    assert verify(req, {"diagnoses": [entry]}) == []


def test_an_extraction_that_is_not_a_list_gives_no_entries_and_so_fails() -> None:
    req = _request(TARGET)
    assert verify(req, {"diagnoses": None}) == ["condition not reported: respiratory failure"]
