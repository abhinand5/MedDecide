"""student_v1 training templates: gold mapping, counted drops, held-out and window guards."""

from __future__ import annotations

from datetime import date

import pytest

from meddecide.bench.fresh.pubmed import PubmedRecord
from meddecide.bench.schema import QuestionType, Split
from meddecide.train.templates_v1 import (
    HELD_OUT_TEMPLATES,
    build_ct_items,
    build_openfda_items,
    build_pubmed_items,
)

LONG_TEXT = "A randomised study of an intervention in adults with a documented condition. " * 2


def _study(nct: str, *, masking: str | None = "DOUBLE", sex: str = "ALL", count: int | None = 270,
           study_type: str = "INTERVENTIONAL", first_posted: str = "2024-05-01") -> dict:
    design_info = {"allocation": "RANDOMIZED"}
    if masking is not None:
        design_info["maskingInfo"] = {"masking": masking}
    design = {"studyType": study_type, "designInfo": design_info}
    if count is not None:
        design["enrollmentInfo"] = {"count": count, "type": "ESTIMATED"}
    return {
        "protocolSection": {
            "identificationModule": {"nctId": nct},
            "statusModule": {"studyFirstPostDateStruct": {"date": first_posted}},
            "descriptionModule": {"briefSummary": LONG_TEXT},
            "conditionsModule": {"conditions": ["Hypertension"]},
            "designModule": design,
            "eligibilityModule": {"sex": sex},
        }
    }


def test_ct_masking_and_sex_gold_follow_the_source_fields() -> None:
    items, drops = build_ct_items(
        [_study("NCT00000001", masking="NONE", sex="FEMALE"),
         _study("NCT00000002", masking="QUADRUPLE", sex="MALE")],
        ["ct_masking_choice_v1", "ct_eligibility_sex_choice_v1"],
    )
    masking = {i.source_record_id: i.gold for i in items["ct_masking_choice_v1"]}
    assert masking == {"NCT00000001": "A", "NCT00000002": "E"}
    sex = {i.source_record_id: i.gold for i in items["ct_eligibility_sex_choice_v1"]}
    assert sex == {"NCT00000001": "B", "NCT00000002": "C"}
    assert drops["ct_masking_choice_v1"] == {} and drops["ct_eligibility_sex_choice_v1"] == {}


def test_missing_fields_are_counted_not_silently_dropped() -> None:
    items, drops = build_ct_items(
        [_study("NCT00000003", masking=None), _study("NCT00000004", sex="OTHER")],
        ["ct_masking_choice_v1", "ct_eligibility_sex_choice_v1"],
    )
    assert [i.source_record_id for i in items["ct_masking_choice_v1"]] == ["NCT00000004"]
    assert drops["ct_masking_choice_v1"] == {"masking_missing_or_unknown:None": 1}
    assert drops["ct_eligibility_sex_choice_v1"] == {"sex_missing_or_unknown:OTHER": 1}


@pytest.mark.parametrize(("count", "gold"), [(0, "1"), (49, "1"), (50, "2"), (199, "2"),
                                             (200, "3"), (999, "3"), (1000, "4"), (4999, "4"),
                                             (5000, "5"), (250000, "5")])
def test_enrollment_band_boundaries(count: int, gold: str) -> None:
    items, _ = build_ct_items([_study("NCT00000005", count=count)], ["ct_enrollment_band_score_v1"])
    (item,) = items["ct_enrollment_band_score_v1"]
    assert item.qtype is QuestionType.SCORE
    assert item.gold == gold
    assert item.meta["enrollment_count"] == count


def test_enrollment_metadata_does_not_leak_into_other_templates() -> None:
    items, _ = build_ct_items(
        [_study("NCT00000006", count=12)],
        ["ct_enrollment_band_score_v1", "ct_masking_choice_v1"],
    )
    assert "enrollment_count" in items["ct_enrollment_band_score_v1"][0].meta
    assert "enrollment_count" not in items["ct_masking_choice_v1"][0].meta


def test_interventional_is_yes_and_observational_is_no() -> None:
    items, drops = build_ct_items(
        [_study("NCT00000007", study_type="INTERVENTIONAL"),
         _study("NCT00000008", study_type="OBSERVATIONAL"),
         _study("NCT00000009", study_type="EXPANDED_ACCESS")],
        ["ct_interventional_noul_v1"],
    )
    gold = {i.source_record_id: i.gold for i in items["ct_interventional_noul_v1"]}
    assert gold == {"NCT00000007": "yes", "NCT00000008": "no"}
    assert drops["ct_interventional_noul_v1"] == {"study_type_not_yes_no:EXPANDED_ACCESS": 1}


def test_held_out_templates_cannot_be_built_for_training() -> None:
    for template in HELD_OUT_TEMPLATES:
        with pytest.raises(ValueError, match="held out"):
            build_ct_items([], [template])


def test_records_inside_the_window_are_refused() -> None:
    with pytest.raises(ValueError, match="window"):
        build_ct_items([_study("NCT00000010", first_posted="2026-03-01")], ["ct_masking_choice_v1"])


def test_dev_and_train_split_is_by_record_and_both_occur() -> None:
    studies = [_study(f"NCT{n:08d}") for n in range(1, 201)]
    items, _ = build_ct_items(studies, ["ct_masking_choice_v1"])
    splits = {i.source_record_id: i.split for i in items["ct_masking_choice_v1"]}
    assert set(splits.values()) == {Split.TRAIN, Split.DEV}
    again, _ = build_ct_items(studies, ["ct_masking_choice_v1"])
    assert {i.source_record_id: i.split for i in again["ct_masking_choice_v1"]} == splits


def _label(set_id: str, *, application: str | None, product: str = "HUMAN PRESCRIPTION DRUG") -> dict:
    openfda = {"product_type": [product]}
    if application is not None:
        openfda["application_number"] = [application]
    return {
        "set_id": set_id,
        "effective_time": "20240501",
        "indications_and_usage": LONG_TEXT,
        "openfda": openfda,
    }


def test_openfda_pathway_and_product_type() -> None:
    # openFDA writes the product type without the word LABEL; both spellings are accepted
    items, drops = build_openfda_items(
        [_label("aaaa-1", application="NDA021234"),
         _label("aaaa-2", application="ANDA076543"),
         _label("aaaa-3", application="BLA125000"),
         _label("aaaa-4", application=None),
         _label("aaaa-5", application="NDA000001", product="HUMAN OTC DRUG")],
        ["fda_application_family_choice_v1", "fda_product_type_noul_v1"],
    )
    gold = {i.source_record_id: i.gold for i in items["fda_application_family_choice_v1"]}
    assert gold == {"aaaa-1": "A", "aaaa-2": "B", "aaaa-3": "C", "aaaa-5": "A"}
    assert drops["fda_application_family_choice_v1"] == {"application_family_not_nda_anda_bla": 1}
    prod = {i.source_record_id: i.gold for i in items["fda_product_type_noul_v1"]}
    assert prod["aaaa-1"] == "yes" and prod["aaaa-5"] == "no"


def _pubmed(pmid: str, *, check_tags: list[str], pub_types: list[str]) -> PubmedRecord:
    return PubmedRecord(
        pmid=pmid,
        entrez_date=date(2025, 6, 1),
        title="A trial report",
        abstract=LONG_TEXT,
        pub_types=pub_types,
        mesh_major_topics=[],
        check_tags=check_tags,
        journal="Test journal",
    )


def test_pubmed_check_tags_and_publication_type() -> None:
    records = [
        _pubmed("1001", check_tags=["Female", "Humans"], pub_types=["Randomized Controlled Trial"]),
        _pubmed("1002", check_tags=["Animals"], pub_types=["Journal Article"]),
    ]
    items, drops = build_pubmed_items(
        records,
        ["pubmed_check_female_noul_v1", "pubmed_check_adult_noul_v1", "pubmed_pubtype_choice_v1"],
        age_tags={"1001": ["Adult"]},
    )
    female = {i.source_record_id: i.gold for i in items["pubmed_check_female_noul_v1"]}
    assert female == {"1001": "yes", "1002": "no"}
    adult = {i.source_record_id: i.gold for i in items["pubmed_check_adult_noul_v1"]}
    assert adult == {"1001": "yes", "1002": "no"}
    pubtype = {i.source_record_id: i.gold for i in items["pubmed_pubtype_choice_v1"]}
    assert pubtype == {"1001": "A", "1002": "F"}  # RCT is A; no label mapped gives "Other" = F
    assert drops["pubmed_check_female_noul_v1"] == {}


def test_pubmed_humans_check_is_never_built() -> None:
    with pytest.raises(ValueError, match="held out"):
        build_pubmed_items([], ["pubmed_humans_noul_v1"])
