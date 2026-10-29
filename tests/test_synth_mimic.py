"""Tests for the MIMIC-grounded family builder and the leakage probe helpers.

Synthetic fixtures only: no MIMIC row, note or file is read. The fixtures use invented admissions and stays whose codes
and titles are chosen to exercise the exclusion rules (held-out families, the test split, non-ICD-10 coding).
"""

from __future__ import annotations

import pytest

from meddecide.synth.facts import CaseFacts, Fact
from meddecide.synth.families import STYLES
from meddecide.synth.leakage import (
    ProbeOutcome,
    parse_letter,
    probe_user_prompt,
    sample_items,
    summarize_probe,
)
from meddecide.synth.mimic_families import (
    TEMPLATES,
    GroundedCase,
    apr_cases,
    drg_cc_cases,
    ed_disposition_cases,
    family_of,
    icd_specificity_cases,
    is_held_out_ed,
    is_held_out_icd10,
    msdrg_cases,
    select,
)
from meddecide.synth.render import render_request

TITLES = {
    "E119": "Type 2 diabetes mellitus without complications",
    "E1165": "Type 2 diabetes mellitus with hyperglycemia",
    "E1122": "Type 2 diabetes mellitus with diabetic chronic kidney disease",
    "E785": "Hyperlipidemia, unspecified",
    "E780": "Pure hypercholesterolemia, unspecified",
    "J189": "Pneumonia, unspecified organism",
    "J181": "Lobar pneumonia, unspecified organism",
    "I509": "Heart failure, unspecified",
    "N179": "Acute kidney failure, unspecified",
    "I10": "Essential (primary) hypertension",
    "K219": "Gastro-esophageal reflux disease without esophagitis",
}
BILLABLE = {"E11": ["E119", "E1165", "E1122"], "E78": ["E785", "E780"], "J18": ["J189", "J181"]}
DRG_TITLES = {
    "470": "MAJOR HIP AND KNEE JOINT REPLACEMENT OR REATTACHMENT OF LOWER EXTREMITY WITHOUT MCC",
    "469": "MAJOR HIP AND KNEE JOINT REPLACEMENT OR REATTACHMENT OF LOWER EXTREMITY WITH MCC",
    "313": "CHEST PAIN WITHOUT CC/MCC",
    "291": "HEART FAILURE AND SHOCK WITH MCC",
    "292": "HEART FAILURE AND SHOCK WITH CC",
    "293": "HEART FAILURE AND SHOCK WITHOUT CC/MCC",
    "194": "SIMPLE PNEUMONIA AND PLEURISY WITH CC",
    "195": "SIMPLE PNEUMONIA AND PLEURISY WITHOUT CC/MCC",
}


def _adm(hadm: int, codes: list[str], *, split: str = "train", coding: str = "icd10", principal: str | None = None,
         drg: str | None = "194", apr: float | None = 2.0) -> dict:
    return {
        "hadm_id": hadm,
        "split": split,
        "coding": coding,
        "icd10_codes": codes,
        "principal_icd10": principal if principal is not None else (codes[0] if codes else None),
        "ms_drg_code": drg,
        "ms_drg_title": DRG_TITLES.get(drg) if drg else None,
        "apr_severity": apr,
    }


def _admissions() -> list[dict]:
    return [
        _adm(1, ["E1165", "I10"], drg="194", apr=2.0),
        _adm(2, ["J189", "E785"], drg="195", apr=1.0),
        _adm(3, ["E119", "K219"], drg="313", apr=3.0),
        _adm(4, ["J181", "E780"], drg="292", apr=4.0),
        _adm(5, ["E1122", "K219"], drg="291", apr=2.0),
        _adm(6, ["K219", "I10"], drg="293", apr=1.0),
        _adm(7, ["E1165", "I10"], drg="194", apr=2.0, split="test"),
        _adm(8, ["J189", "E785"], drg="195", apr=2.0, split="dev"),
        _adm(9, ["E1122", "E119"], coding="icd9", drg="470", apr=2.0),
    ]


def _drg_titles(rows: list[dict]) -> dict[str, str]:
    return {r["ms_drg_code"]: r["ms_drg_title"] for r in rows if r["split"] == "train" and r["ms_drg_code"]}


def test_family_names_follow_the_template_and_unknown_templates_are_refused() -> None:
    assert family_of("mimic_msdrg_v1") == "synth_mimic_msdrg_v1"
    with pytest.raises(ValueError):
        family_of("mimic_rad_exam_v1")
    assert "mimic_rad_exam_v1" not in TEMPLATES


def test_heart_failure_and_aki_codes_are_held_out() -> None:
    assert is_held_out_icd10("I509")  # heart failure family and I50 prefix
    assert is_held_out_icd10("N179")  # AKI code (N17 prefix)
    assert not is_held_out_icd10("E119")
    assert is_held_out_ed("I509", 10, "anything")
    assert is_held_out_ed("428", 9, "anything")
    assert is_held_out_ed("X1", 10, "Acute kidney injury, unspecified")
    assert not is_held_out_ed("E119", 10, "Type 2 diabetes mellitus")


def test_icd_specificity_drops_held_out_records_and_test_split_records() -> None:
    rows = [*_admissions(), _adm(10, ["I509", "E785"], drg="292"), _adm(11, ["N179", "K219"], drg="313")]
    built = icd_specificity_cases(rows, TITLES, BILLABLE, seed=1)
    assert built.drops["held_out_condition"] == 2
    assert built.drops["not_train_split"] == 2  # hadm 7 (test) and hadm 8 (dev)
    assert built.drops["non_icd10_coding"] == 1  # hadm 9
    gold = {c.case.label for c in built.cases}
    assert not gold & {"I50.9", "N17.9"}
    for case in built.cases:
        assert case.case.family == "synth_mimic_icd_specificity_v1"
        assert 3 <= len(case.keys) <= 6
        assert case.case.label in case.keys


def test_test_split_records_never_reach_any_builder() -> None:
    test_only = [dict(_adm(20 + i, ["E1165", "I10"], drg="194", apr=2.0), split="test") for i in range(5)]
    titles = TITLES
    drg = _drg_titles(_admissions())
    for built in (
        icd_specificity_cases(test_only, titles, BILLABLE, seed=1),
        msdrg_cases(test_only, titles, drg, seed=1),
        drg_cc_cases(test_only, titles),
        apr_cases(test_only, titles),
    ):
        assert built.cases == []
        assert built.drops["not_train_split"] == 5


def test_ed_stays_outside_training_or_with_held_out_diagnoses_are_dropped() -> None:
    stays = [
        {"stay_id": 1, "split": "test", "disposition": "HOME", "chiefcomplaint": "cough", "temperature": 98.6},
        {"stay_id": 2, "split": "train", "disposition": "HOME", "chiefcomplaint": "cough", "temperature": 98.6},
        {"stay_id": 3, "split": "train", "disposition": "ADMITTED", "chiefcomplaint": "dyspnea", "heartrate": 110},
        {"stay_id": 4, "split": "train", "disposition": "TRANSFER", "chiefcomplaint": "fall"},
    ]
    dx = {
        2: [("E119", 10, "Type 2 diabetes mellitus without complications")],
        3: [("I509", 10, "Heart failure, unspecified")],
    }
    built = ed_disposition_cases(stays, dx)
    assert built.drops["not_train_split"] == 1
    assert built.drops["held_out_condition"] == 1
    labels = {c.case.label for c in built.cases}
    assert labels == {"home", "other"}


def test_msdrg_label_is_the_drg_code_and_titles_are_never_facts() -> None:
    rows = _admissions()
    built = msdrg_cases(rows, TITLES, _drg_titles(rows), seed=3)
    assert built.cases, "fixture should yield cases"
    for case in built.cases:
        assert case.case.label in case.keys
        assert all(f.value != DRG_TITLES.get(case.case.label) for f in case.case.facts)
        assert all("MCC" not in f.value and "CC/MCC" not in f.value for f in case.case.facts)
        assert 4 <= len(case.keys) <= 6
        assert len(case.display) == len(case.keys)


def test_drg_cc_label_is_parsed_from_the_title_and_titles_without_a_marker_are_dropped() -> None:
    rows = [_adm(31, ["E1165", "I10"], drg="194"), _adm(32, ["J189", "E785"], drg="293"),
            _adm(33, ["E119", "K219"], drg="469"), _adm(34, ["K219", "I10"], drg="470")]
    built = drg_cc_cases(rows, TITLES)
    labels = {c.case.label: c.keys for c in built.cases}
    assert set(labels) == {"CC", "neither", "MCC"}
    # DRG 470 ("WITHOUT MCC") has no three-way marker, so it is dropped and counted, never labelled
    assert built.drops["title_without_three_way_marker"] == 1
    assert built.cases and all(c.keys == ("MCC", "CC", "neither") for c in built.cases)


def test_apr_label_is_the_level_and_missing_levels_are_dropped() -> None:
    rows = [_adm(41, ["E1165", "I10"], apr=3.0), _adm(42, ["J189", "E785"], apr=None),
            _adm(43, ["E119", "K219"], apr=5.0)]
    built = apr_cases(rows, TITLES)
    assert [c.case.label for c in built.cases] == ["3"]
    assert built.drops["apr_severity_missing_or_outside_1_to_4"] == 2


def test_rendered_prompt_never_contains_the_label_and_lists_only_facts() -> None:
    rows = _admissions()
    built = icd_specificity_cases(rows, TITLES, BILLABLE, seed=2)
    grounded = built.cases[0]
    request = render_request(grounded.case, target_words=80, style=STYLES[0], spec=grounded.spec)
    assert grounded.case.label not in request.user
    assert "Target length: 80 words" in request.user
    for fact in grounded.case.facts:
        assert fact.value in request.user


def test_selection_is_deterministic_and_balances_labels() -> None:
    cases = []
    for i, label in enumerate(["1", "2", "3", "4"] * 10):
        facts = (Fact("condition", "principal diagnosis", f"Synthetic diagnosis number {i}"),)
        case = CaseFacts(family=family_of("mimic_apr_severity_v1"), facts=facts, label=label)
        cases.append(GroundedCase("mimic_apr_severity_v1", case, ("1", "2", "3", "4"), ("1", "2", "3", "4")))
    first, short_a = select(cases, "mimic_apr_severity_v1", 8, seed=5)
    second, _ = select(cases, "mimic_apr_severity_v1", 8, seed=5)
    assert [c.case.case_id for c in first] == [c.case.case_id for c in second]
    counts = {k: sum(1 for c in first if c.case.label == k) for k in "1234"}
    assert counts == {"1": 2, "2": 2, "3": 2, "4": 2}
    assert short_a["duplicate facts (not selected)"] == 0


def test_grounded_case_rejects_a_label_outside_the_offered_keys() -> None:
    case = CaseFacts(family=family_of("mimic_drg_cc_level_v1"), facts=(Fact("condition", "a", "Synthetic"),), label="MCC")
    with pytest.raises(ValueError):
        GroundedCase("mimic_drg_cc_level_v1", case, ("CC", "neither", "other"), ("CC", "neither", "other"))


def test_parse_letter_takes_the_first_standalone_letter_in_range() -> None:
    assert parse_letter("B", 3) == 1
    assert parse_letter("**C**.", 3) == 2
    assert parse_letter("D", 3) is None  # out of range for three options
    assert parse_letter("I cannot tell", 3) is None  # no standalone letter in range ("I" is out of range here)


def test_probe_prompt_letters_the_options_and_keeps_the_question() -> None:
    prompt = probe_user_prompt("Synthetic text.", ["alpha", "beta"])
    assert "A. alpha" in prompt and "B. beta" in prompt
    assert "Given only this text, which option is the answer?" in prompt
    with pytest.raises(ValueError):
        probe_user_prompt("x", ["only one"])


def test_probe_summary_reports_rate_chance_and_unparsed() -> None:
    outcomes = [ProbeOutcome("t", 4, True), ProbeOutcome("t", 4, False), ProbeOutcome("t", 4, None),
                ProbeOutcome("t", 2, True)]
    summary = summarize_probe(outcomes)
    assert summary["probed"] == 4
    assert summary["unparsed"] == 1
    assert summary["correct"] == 2
    assert summary["answer_recovered_rate"] == pytest.approx(2 / 3)
    assert summary["chance"] == pytest.approx((0.25 + 0.25 + 0.25 + 0.5) / 4)


def test_probe_sample_is_seeded_and_bounded() -> None:
    items = [{"i": i} for i in range(50)]
    assert sample_items(items, 10, seed=1) == sample_items(items, 10, seed=1)
    assert len(sample_items(items, 10, seed=1)) == 10
    assert sample_items(items[:3], 10, seed=1) == items[:3]


def test_icd_specificity_is_excluded_from_the_training_mix_with_a_reason() -> None:
    from meddecide.synth.mimic_families import MIX_EXCLUSIONS, in_training_mix

    assert MIX_EXCLUSIONS["mimic_icd_specificity_v1"] == "answer_in_facts"
    assert not in_training_mix("mimic_icd_specificity_v1")
    assert in_training_mix("mimic_msdrg_v1")
