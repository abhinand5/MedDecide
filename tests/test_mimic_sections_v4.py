"""MIMIC discharge-note sections, the section-priority cut and the admission tables (osler_v4 Y1). Synthetic strings and
frames only: no MIMIC text, values or identifiers appear here."""

from __future__ import annotations

import pandas as pd

from meddecide.mr import mimic

NOTE = """Name: ___
Chief Complaint:
chest pain
History of Present Illness:
Synthetic patient reported two days of cough.
Physical Exam:
Afebrile.
Brief Hospital Course:
Treated with fluids.
Discharge Diagnosis:
Principal: pneumonia.
"""


def _words(text: str) -> int:
    return len(text.split())


def test_known_headers_are_recognised_with_inline_text_and_aliases() -> None:
    assert mimic.section_header("Chief Complaint: chest pain") == ("chief_complaint", "chest pain")
    assert mimic.section_header("discharge  DIAGNOSES:") == ("discharge_diagnosis", "")
    assert mimic.section_header("   Pertinent Results:") == ("pertinent_results", "")
    assert mimic.section_header("Discharge Physical Exam: whatever") is None
    assert mimic.section_header("Some ordinary sentence: with a colon") is None


def test_sections_come_in_document_order_and_the_identifier_block_is_kept_apart() -> None:
    secs = mimic.sections(NOTE)
    assert list(secs) == ["identifiers", "chief_complaint", "history_of_present_illness", "physical_exam",
                          "brief_hospital_course", "discharge_diagnosis"]
    assert secs["chief_complaint"] == "chest pain"
    assert secs["discharge_diagnosis"] == "Principal: pneumonia."


def test_a_repeated_header_is_joined_and_an_unknown_header_stays_with_its_section() -> None:
    text = "Discharge Medications:\nA\nUnknown Heading: x\nDischarge Medications:\nB"
    assert mimic.sections(text) == {"discharge_medications": "A\nUnknown Heading: x\nB"}


def test_text_before_the_first_header_is_the_preamble() -> None:
    assert mimic.sections("Opening line.\nChief Complaint:\nx") == {"preamble": "Opening line.", "chief_complaint": "x"}


def test_a_note_that_fits_is_returned_whole() -> None:
    whole, cut = mimic.prioritised_text(mimic.sections(NOTE), 10_000, _words)
    assert cut is False
    assert "Principal: pneumonia." in whole


def test_a_long_note_keeps_priority_order_then_cuts_at_the_budget_with_a_marker() -> None:
    text, cut = mimic.prioritised_text(mimic.sections(NOTE), 16, _words)
    assert cut is True
    assert text.startswith("Chief Complaint:")
    assert text.endswith(mimic.TRUNCATION_MARK)
    assert _words(text) <= 16
    assert "Discharge Diagnosis" not in text
    assert "Name" not in text


def test_every_budget_gives_a_cut_note_within_the_cap_that_ends_with_the_marker() -> None:
    secs = mimic.sections(NOTE)
    full, _ = mimic.prioritised_text(secs, 10_000, _words)
    for budget in range(2, _words(full)):
        text, cut = mimic.prioritised_text(secs, budget, _words)
        assert cut is True
        assert _words(text) <= budget
        assert text.endswith(mimic.TRUNCATION_MARK)


def test_when_the_first_section_alone_is_too_long_it_is_cut_to_the_budget() -> None:
    text, cut = mimic.prioritised_text(mimic.sections(NOTE), 3, _words)
    assert cut is True
    assert text == "Chief\n[... truncated]"
    assert _words(text) <= 3


def test_the_coding_class_follows_the_icd_versions_an_admission_codes() -> None:
    dx = pd.DataFrame({"hadm_id": [1, 1, 2, 3, 3], "icd_version": [10, 10, 9, 10, 9],
                       "seq_num": [1, 2, 1, 1, 2], "icd_code": ["A1", "B2", "C3", "D4", "E5"]})
    out = mimic.coding_summary(dx)
    assert out.loc[1, "coding"] == "icd10"
    assert out.loc[1, "n_icd10"] == 2
    assert out.loc[2, "coding"] == "icd9"
    assert out.loc[3, "coding"] == "mixed"


def test_icd10_codes_follow_seq_num_and_the_principal_code_is_seq_1() -> None:
    dx = pd.DataFrame({"hadm_id": [1, 1, 1], "icd_version": [10, 10, 10], "seq_num": [2, 1, 3],
                       "icd_code": ["B", "A", "C"]})
    out = mimic.icd10_codes(dx)
    assert out.loc[1, "icd10_codes"] == ["A", "B", "C"]
    assert out.loc[1, "principal_icd10"] == "A"


def test_drg_fields_read_hcfa_titles_and_apr_levels_only() -> None:
    drg = pd.DataFrame({"hadm_id": [1, 1, 2], "drg_type": ["HCFA", "APR", "HCFA"], "drg_code": ["291", "2", "872"],
                        "description": ["HEART FAILURE W MCC", "SYNTHETIC APR TEXT", "SEPSIS W/O MCC"],
                        "drg_severity": [None, 3, None], "drg_mortality": [None, 2, None]})
    out = mimic.drg_fields(drg)
    assert out.loc[1, "ms_drg_code"] == "291"
    assert out.loc[1, "ms_drg_title"] == "HEART FAILURE W MCC"
    assert out.loc[1, "apr_severity"] == 3
    assert out.loc[1, "apr_mortality"] == 2
    assert pd.isna(out.loc[2, "apr_severity"])
    assert "SYNTHETIC APR TEXT" not in out.to_string()


def test_every_admission_gets_a_split_a_late_flag_and_empty_lists_when_it_is_uncoded() -> None:
    admissions = pd.DataFrame({"subject_id": [10, 10, 11], "hadm_id": [100, 101, 200]})
    patients = pd.DataFrame({"subject_id": [10, 11], "anchor_year_group": ["2014 - 2016", "2008 - 2010"]})
    dx = pd.DataFrame({"hadm_id": [100, 100], "icd_version": [10, 10], "seq_num": [1, 2], "icd_code": ["A", "B"]})
    procedures = pd.DataFrame({"hadm_id": [100], "icd_version": [10], "seq_num": [1], "icd_code": ["PCS1"]})
    drg = pd.DataFrame({"hadm_id": [100, 100], "drg_type": ["HCFA", "APR"], "drg_code": ["291", "2"],
                        "description": ["T", "APR TEXT"], "drg_severity": [None, 3], "drg_mortality": [None, 2]})
    out = mimic.admission_table(admissions, patients, mimic.coding_summary(dx), mimic.icd10_codes(dx),
                                mimic.pcs_codes(procedures), mimic.drg_fields(drg))
    assert out.loc[100, "coding"] == "icd10"
    assert out.loc[101, "coding"] == "none"
    assert out.loc[101, "icd10_codes"] == []
    assert out.loc[200, "pcs10_codes"] == []
    assert bool(out.loc[100, "late_anchor"]) is True
    assert bool(out.loc[200, "late_anchor"]) is False
    assert out.loc[100, "split"] == out.loc[101, "split"]
    assert out.loc[100, "split"] in {"train", "dev", "test"}
