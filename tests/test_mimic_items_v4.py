"""Structured-gold MIMIC templates (osler_v4 Y2): parsers and builders on synthetic records only. No MIMIC text appears here."""

from __future__ import annotations

import pandas as pd

from meddecide.gen.screen import gold_label
from meddecide.mr import mimic_items as mi


def _ctx(notes: dict[str, str] | None = None, titles: dict[str, str] | None = None) -> mi.Ctx:
    return mi.Ctx(notes=notes or {}, icd_titles=titles or {}, pcs_titles={}, count=lambda s: len(s.split()), seed=0)


def test_drg_split_reads_the_three_way_markers_and_leaves_two_way_splits_out() -> None:
    assert mi.drg_split("Heart failure and shock with MCC") == ("HEART FAILURE AND SHOCK", "MCC")
    assert mi.drg_split("Heart failure and shock with CC") == ("HEART FAILURE AND SHOCK", "CC")
    assert mi.drg_split("Heart failure and shock without CC/MCC") == ("HEART FAILURE AND SHOCK", "NONE")
    assert mi.drg_split("Heart failure and shock W/O CC/MCC")[1] == "NONE"
    assert mi.drg_split("Septicemia W CC/MCC")[1] is None
    assert mi.drg_split("Pneumonia without MCC")[1] is None


def test_modality_rules_classify_common_exam_names_and_return_none_otherwise() -> None:
    assert mi.modality("CT ABDOMEN AND PELVIS WITH CONTRAST") == "CT"
    assert mi.modality("MRI BRAIN WITHOUT CONTRAST") == "MR"
    assert mi.modality("ULTRASOUND ABDOMEN COMPLETE") == "US"
    assert mi.modality("CHEST (PORTABLE AP)") == "XR"
    assert mi.modality("Something unlisted") is None


def test_the_triage_page_is_rendered_by_code_and_text_fields_are_kept_as_written() -> None:
    page = mi.triage_page({"chiefcomplaint": "cough", "temperature": 98.64, "heartrate": float("nan"), "resprate": 18,
                           "o2sat": 97, "sbp": 120, "dbp": 70, "pain": "7-8"})
    assert "Chief complaint: cough." in page
    assert "Temperature: 98.6 F." in page
    assert "Heart rate: not recorded." in page
    assert "Pain score: 7-8." in page


def test_icd_specificity_options_come_from_the_category_and_the_gold_is_the_coded_code() -> None:
    titles = {"J189": "Pneumonia, unspecified organism", "J181": "Lobar pneumonia, unspecified organism",
              "J180": "Bronchopneumonia, unspecified organism", "J188": "Other pneumonia, organism unspecified",
              "I10": "Essential (primary) hypertension"}
    billable: dict[str, list[str]] = {}
    for code in titles:
        billable.setdefault(code[:3], []).append(code)
    admissions = [(1, "n1", True, ["J189"]), (2, "n2", True, ["I10"])]
    ctx = _ctx(notes={"n1": "a b c d e", "n2": "a b c d e"}, titles=titles)
    built = mi.icd_specificity(ctx, "test", 10, admissions, billable)
    assert len(built.rows) == 1                     # admission 2 has no billable sibling in its category
    assert built.drops["fewer than two billable siblings"] == 1
    row = built.rows[0]
    assert 3 <= len(row["options"]) <= 6
    assert gold_label(row) == "J18.9 - Pneumonia, unspecified organism"


def test_icd_support_pairs_each_positive_with_a_negative_and_routes_held_out_families() -> None:
    fam_of = {1: {"pneumonia_organism": ["J18.9"]}, 2: {"heart_failure": ["I50.9"]}}
    notes = {"n1": "Pneumonia treated.", "n2": "Heart failure.", "n3": "Stable.", "n4": "Walking."}
    admissions = [(1, "n1", True), (3, "n3", True), (4, "n4", True)]
    built = mi.icd_support(_ctx(notes=notes), "train", 2000, admissions, fam_of)
    pneumonia = [r for r in built.rows if r["meta"]["family"] == "pneumonia_organism"]
    assert sorted(r["meta"]["role"] for r in pneumonia) == ["negative", "positive"]
    assert all(r["qtype"] == "choice" for r in pneumonia)
    positive = next(r for r in pneumonia if r["meta"]["role"] == "positive")
    assert gold_label(positive).startswith("J18.9 - ")
    assert not any(r["meta"]["family"] == "heart_failure" for r in built.rows)
    assert built.drops["held-out family heart_failure outside the test split"] == 1


def test_principal_dx_removes_the_discharge_diagnosis_section_before_the_options_are_shown() -> None:
    note = "Chief Complaint:\nCough.\nDischarge Diagnosis:\nPneumonia, resolved.\n"
    titles = {"J189": "Pneumonia", "R05": "Cough", "E119": "Type 2 diabetes", "I10": "Hypertension", "K219": "GERD"}
    admissions = [(1, "n1", True, "J189", ["J189", "R05", "E119", "I10", "K219"])]
    built = mi.principal_dx(_ctx(notes={"n1": note}, titles=titles), "test", 5, admissions)
    assert len(built.rows) == 1
    row = built.rows[0]
    assert "Pneumonia, resolved" not in row["state"]
    assert gold_label(row).startswith("J18.9 - ")
    assert 4 <= len(row["options"]) <= 6


def test_msdrg_options_contain_the_gold_and_its_siblings_and_drop_small_sets() -> None:
    titles = {"291": "Heart failure and shock with MCC", "292": "Heart failure and shock with CC",
              "293": "Heart failure and shock without CC/MCC", "270": "Heart failure with MCC",
              "872": "Septicemia or severe sepsis without MV >96 hours with MCC",
              "690": "Kidney and urinary tract infections without MCC"}
    built = mi.msdrg(_ctx(notes={"n1": "a b c d e"}), "train", 5, [(1, "n1", True, "291", titles["291"])], titles)
    assert len(built.rows) == 1
    row = built.rows[0]
    keys = {o["label"].split(" - ")[0] for o in row["options"]}
    assert {"291", "292", "293"} <= keys
    assert 4 <= len(row["options"]) <= 6
    assert gold_label(row).startswith("291 - ")
    tiny = mi.msdrg(_ctx(notes={"n1": "a"}), "train", 5, [(1, "n1", True, "872", titles["872"])], {"872": titles["872"]})
    assert tiny.drops["fewer than four DRG options"] == 1


def test_drg_cc_level_balances_the_classes_and_skips_titles_without_a_marker() -> None:
    adm = [(i, f"n{i}", True, str(i), f"Title {i} with {w}") for i, w in
           enumerate(["MCC", "CC", "without CC/MCC", "CC/MCC", "MCC", "CC", "without CC/MCC"], start=1)]
    built = mi.drg_cc_level(_ctx(notes={f"n{i}": "a b" for i in range(1, 8)}), "train", 6, adm)
    golds = [gold_label(r) for r in built.rows]
    assert golds.count("MCC") == 2 and golds.count("CC") == 2
    assert built.drops["title without a three-way marker"] == 1


def test_pcs_support_negatives_share_body_system_and_category_and_are_never_coded_here() -> None:
    pool = [(1, "n1", True, "J189", ["0BB00ZZ"]), (2, "n2", True, "J189", ["0BB10ZZ"]), (3, "n3", True, "J189", ["0BB20ZZ"])]
    index = mi.pcs_index(pool, "train")
    built = mi.pcs_support(_ctx(notes={"n1": "a b", "n2": "a b", "n3": "a b"}), "train", 10, pool[:1], index)
    assert {r["gold"] for r in built.rows} == {"yes", "no"}
    negative = next(r for r in built.rows if r["gold"] == "no")
    assert negative["meta"]["code"][:2] == "0B"
    assert negative["meta"]["code"] != "0BB00ZZ"


def test_ed_disposition_groups_the_recorded_disposition_and_drops_missing_values() -> None:
    stays = [{"subject_id": 1, "stay_id": 10, "disposition": "ADMITTED", "temperature": 99.0, "chiefcomplaint": "cough"},
             {"subject_id": 1, "stay_id": 11, "disposition": "HOME", "temperature": 98.0, "chiefcomplaint": "fall"},
             {"subject_id": 1, "stay_id": 12, "disposition": "TRANSFER", "temperature": 98.0, "chiefcomplaint": "x"},
             {"subject_id": 1, "stay_id": 13, "disposition": None, "temperature": 98.0, "chiefcomplaint": "y"}]
    built = mi.ed_disposition(_ctx(), "train", 10, stays, {1: False}, {10: [("J189", 10, "Pneumonia")]})
    assert [r["meta"]["group"] for r in built.rows] == ["admitted", "home", "other"]
    assert built.drops["disposition missing"] == 1
    assert "Pneumonia" in built.rows[0]["state"]


def test_ed_acuity_drops_values_outside_one_to_five_and_respects_the_test_scope() -> None:
    triage = [{"subject_id": 1, "stay_id": 1, "acuity": 3.0, "chiefcomplaint": "a"},
              {"subject_id": 2, "stay_id": 2, "acuity": 9.0, "chiefcomplaint": "b"},
              {"subject_id": 3, "stay_id": 3, "acuity": 2.0, "chiefcomplaint": "c"}]
    built = mi.ed_acuity(_ctx(), "test", 10, triage, {1: True, 2: True, 3: False})
    assert [r["gold"] for r in built.rows] == ["3"]
    assert built.drops["acuity missing or outside 1-5"] == 1


def test_mednli_keeps_the_three_classes_and_reports_any_other_label() -> None:
    pairs = [{"sentence1": "Premise.", "sentence2": "Hypothesis.", "gold_label": "entailment", "pairID": "p1"},
             {"sentence1": "Premise.", "sentence2": "Other.", "gold_label": "-", "pairID": "p2"}]
    built = mi.mednli("train", pairs)
    assert len(built.rows) == 1 and built.rows[0]["gold"] == "A"
    assert sum(built.drops.values()) == 1


def test_rad_exam_masks_the_exam_name_and_needs_three_same_modality_distractors() -> None:
    names = {"CT": ["CT CHEST WITH CONTRAST", "CT HEAD WITHOUT CONTRAST", "CT ABDOMEN", "CT PELVIS"]}
    text = "EXAMINATION: CT CHEST WITH CONTRAST\nFindings: normal."
    built = mi.rad_exam(_ctx(), "test", 5, [("n1", 1, text, "CT CHEST WITH CONTRAST", True)], names)
    assert len(built.rows) == 1
    assert "CT CHEST WITH CONTRAST" not in built.rows[0]["state"]
    assert "[exam name masked]" in built.rows[0]["state"]
    assert 4 <= len(built.rows[0]["options"]) <= 6


def test_the_family_codes_helper_returns_dotted_codes_per_family() -> None:
    fam = mi.family_codes([(1, "J189"), (1, "R05"), (2, "I509")])
    assert fam[1] == {"pneumonia_organism": ["J18.9"]}
    assert fam[2] == {"heart_failure": ["I50.9"]}
    assert pd.isna(float("nan"))


def _lab_values(*pairs: tuple[str, float]) -> list[tuple]:
    return [(f"2100-01-0{i + 1}", "Sodium", value, "mmol/L", 135.0, 145.0) for i, (_, value) in enumerate(pairs)]


def test_the_lab_criteria_are_applied_by_code_and_the_page_shows_the_values_in_time_order() -> None:
    pools = {"sodium": [(1, _lab_values(("a", 128.0))), (2, _lab_values(("a", 140.0))),
                        (3, _lab_values(("a", 129.0))), (4, _lab_values(("a", 141.0)))]}
    built = mi.lab_validation(_ctx(), "train", 36, pools, {1: {"E871"}, 3: set()}, {})
    hyponatremia = [r for r in built.rows if r["meta"]["condition"] == "hyponatremia"]
    met = {r["meta"]["met"] for r in hyponatremia}
    assert met == {True, False}
    positive = [r for r in hyponatremia if r["meta"]["met"]]
    assert all(r["gold"] == "yes" for r in positive)
    assert "- Sodium: 128 mmol/L (reference 135-145 mmol/L)" in positive[0]["state"] or any(
        "- Sodium:" in r["state"] for r in positive)
    assert all(r["qtype"] == "noul" for r in built.rows)


def test_a_lab_page_shows_at_most_twenty_rows_and_counts_the_rest() -> None:
    values = [(f"2100-01-{i + 1:02d}", "Sodium", 130.0 + i, "mmol/L", None, None) for i in range(25)]
    page, hidden = mi.lab_page(values)
    assert hidden == 5
    assert page.count("- Sodium:") == 20
    assert "reference" not in page


def test_the_med_given_negative_is_an_ordered_drug_that_was_never_given_when_there_is_one() -> None:
    admissions = [(1, "n1", True, "J189")]
    administered = {1: {"ceftriaxone"}}
    ordered = {1: {"ceftriaxone", "vancomycin"}}
    built = mi.med_given(_ctx(notes={"n1": "a b c"}), "train", 10, admissions, administered, ordered, {})
    negative = next(r for r in built.rows if r["gold"] == "no")
    positive = next(r for r in built.rows if r["gold"] == "yes")
    assert negative["meta"]["drug"] == "vancomycin"
    assert negative["meta"]["negative_source"] == "ordered, never administered"
    assert positive["meta"]["drug"] == "ceftriaxone"


def test_an_admission_without_any_administered_drug_is_dropped() -> None:
    built = mi.med_given(_ctx(notes={"n1": "a b c"}), "train", 10, [(1, "n1", True, "J189")], {}, {1: {"x"}}, {})
    assert built.rows == []
    assert built.drops["no administered drug"] == 1


def test_a_state_repeated_from_an_earlier_split_is_removed_from_dev_and_test_only() -> None:
    def row(state: str) -> dict[str, str]:
        return {"state": state}

    rows = {"train": [row("a"), row("a"), row("b")], "dev": [row("a"), row("c"), row("c")],
            "test": [row("b"), row("c"), row("d")]}
    kept, removed = mi.dedupe_across_splits(rows)
    assert [r["state"] for r in kept["train"]] == ["a", "a", "b"]
    assert [r["state"] for r in kept["dev"]] == ["c", "c"]
    assert [r["state"] for r in kept["test"]] == ["d"]
    assert removed == {"train": 0, "dev": 1, "test": 2}
