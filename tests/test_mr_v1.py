"""osler_v1 medical-record audit code: indicator rules, page rendering, OCR noise, generators, ICD-10-CM rules, teacher
prose verification and the MIMIC readers (CPU, no data needed)."""

from __future__ import annotations

import gzip
import json
import random
import re

import pytest

from meddecide.mr import criteria, icd, mimic
from meddecide.mr.criteria import ChunkFacts, MedAdmin, Observation
from meddecide.mr.generators import GENERATORS, mr_icd_support_template_v1, mr_indicator_evidence_v1
from meddecide.mr.noise import dress_page, ocr_noise
from meddecide.mr.pages import PAGE_CLASSES, make_context, render_flowsheet_page, render_lab_page
from meddecide.mr.teacher_prose import ProseRequest, chunk_facts, diagnosis_facts, generate, verify

# ---- indicator rules ------------------------------------------------------------------------------


def _f(*obs: Observation, meds: tuple[MedAdmin, ...] = ()) -> ChunkFacts:
    return ChunkFacts(observations=tuple(obs), meds=meds)


@pytest.mark.parametrize(("key", "name", "yes", "no"), [
    ("lactate_gt_2", "lactate", 2.1, 2.0), ("map_lt_65", "map", 64, 65), ("platelets_lt_150", "platelets", 149, 150),
    ("bilirubin_ge_1_2", "bilirubin", 1.2, 1.19), ("pao2_lt_60", "pao2", 59, 60), ("sodium_lt_135", "sodium", 134, 135),
    ("sirs_heart_rate", "heart_rate", 91, 90), ("sirs_temperature", "temperature", 38.1, 38.0),
])
def test_thresholds_are_strict_where_published(key, name, yes, no) -> None:
    assert criteria.supports(key, _f(Observation(name, yes)))
    assert not criteria.supports(key, _f(Observation(name, no)))


def test_prior_encounter_and_baseline_values_do_not_count() -> None:
    assert not criteria.supports("lactate_gt_2", _f(Observation("lactate", 4.0, encounter="prior")))
    assert not criteria.supports("spo2_lt_91_room_air", _f(Observation("spo2", 86, baseline=True)))
    assert not criteria.supports("spo2_lt_91_room_air", _f(Observation("spo2", 86, on_room_air=False)))
    assert criteria.supports("spo2_lt_91_room_air", _f(Observation("spo2", 86)))


def test_vasopressor_counts_only_when_given_in_this_encounter() -> None:
    given = MedAdmin("norepinephrine", "0.1 mcg/kg/min")
    assert criteria.supports("vasopressor_given", _f(meds=(given,)))
    for status in ("held", "refused", "not_given"):
        assert not criteria.supports("vasopressor_given", _f(meds=(MedAdmin("norepinephrine", "x", status=status),)))
    assert not criteria.supports("vasopressor_given", _f(meds=(MedAdmin("midodrine", "10 mg"),)))
    assert not criteria.supports("vasopressor_given", _f(meds=(MedAdmin("vasopressin", "x", encounter="prior"),)))


def test_blood_gas_pairs_and_pf_ratio() -> None:
    assert criteria.supports("hypercapnic_acidosis", _f(Observation("paco2", 60, time_h=3), Observation("ph", 7.28, time_h=3)))
    # compensated: high PaCO2 with a normal pH is not acidosis
    assert not criteria.supports("hypercapnic_acidosis", _f(Observation("paco2", 60), Observation("ph", 7.38)))
    # values from different blood gases (hours apart) do not pair
    assert not criteria.supports("hypercapnic_acidosis",
                                 _f(Observation("paco2", 60, time_h=1), Observation("ph", 7.28, time_h=9)))
    assert criteria.supports("pf_ratio_lt_300", _f(Observation("pao2", 80), Observation("fio2", 40)))   # 200
    assert not criteria.supports("pf_ratio_lt_300", _f(Observation("pao2", 95), Observation("fio2", 21)))  # 452


def test_kdigo_creatinine_rise() -> None:
    rise = _f(Observation("creatinine", 1.0, time_h=2), Observation("creatinine", 1.35, time_h=30))
    assert criteria.supports("aki_creatinine_rise", rise)
    slow = _f(Observation("creatinine", 1.0, time_h=2), Observation("creatinine", 1.35, time_h=100))
    assert not criteria.supports("aki_creatinine_rise", slow)  # +0.35 but over 48 h, and below 1.5x
    vs_baseline = _f(Observation("creatinine", 0.8, baseline=True, encounter="prior"), Observation("creatinine", 1.3))
    assert criteria.supports("aki_creatinine_rise", vs_baseline)


def test_alternative_units_round_trip() -> None:
    o = Observation("lactate", 2.5, display_unit="mg/dL")
    value, unit = o.displayed()
    assert unit == "mg/dL" and value == pytest.approx(22.52, abs=0.01)
    t = Observation("temperature", 38.5, display_unit="F")
    assert t.displayed() == (pytest.approx(101.3), "F")
    with pytest.raises(ValueError):
        Observation("sodium", 130, display_unit="mg/dL")


# ---- pages and noise --------------------------------------------------------------------------------

def test_twelve_page_classes_with_other_and_no_vitals_class() -> None:
    keys = [k for k, _ in PAGE_CLASSES]
    assert len(keys) == 12 and keys[-1] == "other" and "vitals" not in keys and "flowsheet" in keys


def test_rendered_values_appear_on_the_page() -> None:
    rng = random.Random(0)
    facts = _f(Observation("lactate", 3.4, time_h=5), Observation("sodium", 131, time_h=5))
    page = render_lab_page(make_context(rng), facts, rng)
    assert "3.4" in page and "131" in page
    fl = render_flowsheet_page(make_context(rng), _f(Observation("spo2", 88, on_room_air=False, time_h=3)), rng)
    assert "88" in fl and "Room air" not in fl


def test_ocr_noise_never_touches_digits() -> None:
    # letters may turn into digits (l -> 1, O -> 0), as in real OCR, but every number on the page survives verbatim
    text = "Lactate 2.1 mmol/L; Sodium 133; Platelets 149 on 05/28/2025 10:21 at Mercy Valley Hospital " * 20
    noisy = ocr_noise(text, random.Random(1), rate=0.3)
    numbers = [tok for tok in text.split() if any(ch.isdigit() for ch in tok)]
    noisy_tokens = noisy.split()
    assert all(tok in noisy_tokens for tok in numbers)
    assert noisy != text


def test_dress_page_keeps_the_body() -> None:
    body = "LAB RESULTS\nLactate 2.4 mmol/L"
    dressed = dress_page(body, random.Random(3), noise_rate=0.0)
    assert body in dressed


# ---- generators -----------------------------------------------------------------------------------

@pytest.mark.parametrize("gen", GENERATORS, ids=lambda g: g.name)
def test_generators_are_deterministic_and_well_formed(gen) -> None:
    for i in range(30):
        a = gen.make(f"{gen.name}-{i}")
        b = gen.make(f"{gen.name}-{i}")
        assert [x.as_dict() for x in a] == [x.as_dict() for x in b]
        for item in a:
            keys = [k for k, _ in item.options]
            assert item.gold in keys and len(set(keys)) == len(keys)
            assert all(re.fullmatch(r"[A-Za-z0-9]{1,8}", k) for k in keys)


def test_indicator_twins_flip_and_labels_match_the_rule() -> None:
    golds = []
    traps = set()
    for i in range(200):
        base, twin = mr_indicator_evidence_v1(f"p{i}")
        assert base.gold != twin.gold and base.question == twin.question
        golds.append(base.gold)
        traps.add(base.meta["trap"])
    share_yes = golds.count("yes") / len(golds)
    assert 0.35 < share_yes < 0.65
    assert {"prior_encounter", "value_cross", "baseline", "on_oxygen", "mar_status"} <= traps


def test_icd_template_twins_flip() -> None:
    nones = 0
    for i in range(200):
        base, twin = mr_icd_support_template_v1(f"p{i}")
        assert base.gold != twin.gold and base.options == twin.options
        nones += (base.meta["code"] is None) + (twin.meta["code"] is None)
    assert 0.2 < nones / 400 < 0.6


# ---- ICD-10-CM rules ------------------------------------------------------------------------------

def test_icd_resolution_and_guideline_rules() -> None:
    dx = icd.Diagnosis("respiratory_failure", {"acuity": "acute_on_chronic", "gas": "hypoxia"})
    assert icd.supported_code(dx, "inpatient") == "J96.21"
    suspected = icd.Diagnosis("sepsis_organism", {"organism": "e_coli"}, certainty="suspected")
    assert icd.supported_code(suspected, "inpatient") == "A41.51"   # inpatient: uncertain diagnosis coded (II.H)
    assert icd.supported_code(suspected, "outpatient") is None      # outpatient: not coded (IV.H)
    assert icd.supported_code(icd.Diagnosis("aki", {}, certainty="ruled_out"), "inpatient") is None
    assert icd.supported_code(icd.Diagnosis("aki", {}, status="history"), "inpatient") is None
    assert icd.supported_code(icd.Diagnosis("aki", {}, subject="family"), "inpatient") is None
    assert icd.supported_code(icd.Diagnosis("heart_failure", {"type": "systolic", "acuity": "unspecified"}),
                              "inpatient") == "I50.20"
    with pytest.raises(ValueError):
        icd.Diagnosis("ckd", {"stage": "6"})


def test_every_family_resolves_every_attribute_combination() -> None:
    import itertools

    for fam in icd.FAMILIES:
        keys = list(fam.attributes)
        for values in itertools.product(*(fam.attributes[k] for k in keys)):
            assert fam.resolve(dict(zip(keys, values, strict=True))) in fam.codes


def test_order_file_parser_and_verification(tmp_path) -> None:
    def line(order: int, code: str, billable: str, short: str, long: str) -> str:
        return f"{order:05d} {code:<7} {billable} {short:<60} {long}"
    rows = [line(1, "J96", "0", "Respiratory failure, NEC", "Respiratory failure, not elsewhere classified"),
            line(2, "J9601", "1", "Acute respiratory failure with hypoxia", "Acute respiratory failure with hypoxia")]
    path = tmp_path / "order.txt"
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    order = icd.load_order_file(path)
    assert order["J96.01"].billable and not order["J96"].billable
    report = icd.verify_codes(order)
    assert "J96.01" not in report["missing"] and "J96.00" in report["missing"]


# ---- teacher prose ----------------------------------------------------------------------------------

class FakeTeacher:
    def __init__(self, page: str, extracted: dict, local: bool = False) -> None:
        self.page, self.extracted, self.local, self.calls = page, extracted, local, 0

    def chat(self, messages, *, temperature, max_tokens):
        self.calls += 1
        return self.page if self.calls % 2 == 1 else "```json\n" + json.dumps(self.extracted) + "\n```"


def _dx_request(source: str = "generated") -> ProseRequest:
    dx = icd.Diagnosis("respiratory_failure", {"acuity": "acute", "gas": "hypoxia"}, certainty="probable")
    return ProseRequest("diagnosis", "discharge_summary", diagnosis_facts(dx), source=source)


def test_verified_diagnosis_page_is_kept_and_a_changed_fact_is_dropped() -> None:
    req = _dx_request()
    good = {"condition": "respiratory failure", "attributes": {"acuity": "acute", "gas": "hypoxia"},
            "certainty": "probable", "subject": "patient", "status": "current"}
    assert generate(FakeTeacher("page text", good), req).ok
    bad = {**good, "certainty": "confirmed"}
    result = generate(FakeTeacher("page text", bad), req)
    assert not result.ok and any("certainty" in r for r in result.reasons)


def test_chunk_verification_checks_values_and_context() -> None:
    facts = _f(Observation("spo2", 88, on_room_air=True, time_h=3), meds=(MedAdmin("norepinephrine", "x"),))
    req = ProseRequest("chunk", "progress_note", chunk_facts(facts))
    ok = {"observations": [{"name": "spo2", "value": 88, "encounter": "current", "baseline": False, "on_room_air": True}],
          "medications": [{"drug": "Norepinephrine", "status": "given", "encounter": "current"}]}
    assert verify(req, ok) == []
    on_o2 = json.loads(json.dumps(ok))
    on_o2["observations"][0]["on_room_air"] = False
    assert verify(req, on_o2)


def test_credentialed_facts_never_reach_a_hosted_teacher() -> None:
    with pytest.raises(PermissionError):
        generate(FakeTeacher("x", {}), _dx_request(source="mimiciv"))
    local = FakeTeacher("x", {}, local=True)
    generate(local, _dx_request(source="mimiciv"))  # allowed on a local teacher
    assert local.calls == 2


# ---- MIMIC readers ----------------------------------------------------------------------------------

def test_mimic3_page_class_mapping() -> None:
    assert mimic.mimic3_page_class("Discharge summary", "Report") == ("discharge_summary", "category Discharge summary")
    assert mimic.mimic3_page_class("Physician ", "Physician Resident Admission Note")[0] == "admission"
    assert mimic.mimic3_page_class("Physician ", "Physician Attending Progress Note")[0] == "progress_note"
    assert mimic.mimic3_page_class("Nursing", "Nursing Progress Note")[0] == "progress_note"
    assert mimic.mimic3_page_class("Nursing/other", "Report")[0] is None
    assert mimic.mimic3_page_class("Echo", "Report")[0] == "other"


def test_chunk_pages_keeps_all_text() -> None:
    text = "\n".join(f"line {i} " + "x" * 50 for i in range(300))
    pages = mimic.chunk_pages(text)
    assert len(pages) > 3 and all(len(p) <= 3300 for p in pages)
    assert "\n".join(pages) == text


def test_mimic3_reader_and_admission_codes(tmp_path) -> None:
    notes = tmp_path / "NOTEEVENTS.csv.gz"
    with gzip.open(notes, "wt", encoding="utf-8") as fh:
        fh.write("ROW_ID,SUBJECT_ID,HADM_ID,CHARTDATE,CATEGORY,DESCRIPTION,ISERROR,TEXT\n")
        fh.write('1,10,100,2150-01-01,Discharge summary,Report,,"Admission Date: x\nDischarge Date: y"\n')
        fh.write('2,10,100,2150-01-01,Nutrition,Nutrition,1,"error note"\n')
        fh.write('3,11,,2150-01-01,Social Work,Social Work,,"sw note"\n')
        fh.write('4,12,101,2150-01-01,Physician ,Unknown type,,"x"\n')
    counts: dict[str, int] = {}
    pages = list(mimic.iter_mimic3_pages(notes, counts=counts))
    assert [p.page_class for p in pages] == ["discharge_summary", "other"]
    assert counts["dropped: ISERROR=1"] == 1 and sum(v for k, v in counts.items() if k.startswith("dropped")) == 2
    dx = tmp_path / "diagnoses_icd.csv.gz"
    with gzip.open(dx, "wt", encoding="utf-8") as fh:
        fh.write("subject_id,hadm_id,seq_num,icd_code,icd_version\n")
        fh.write("1,500,1,J9601,10\n1,500,2,N179,10\n1,500,3,5849,9\n2,501,1,I5021,10\n2,501,2,I5023,10\n")
    coded = mimic.admission_family_codes(dx)
    assert coded[500] == {"respiratory_failure": ["J96.01"], "aki": ["N17.9"]}
    assert mimic.family_label("heart_failure", coded[501]) == ("", "ambiguous: I50.21,I50.23")
    assert mimic.family_label("ckd", coded[500]) == (None, "not coded")
    assert mimic.late_anchor("2014 - 2016") and not mimic.late_anchor("2011 - 2013")


def test_held_out_lists_mark_items() -> None:
    from meddecide.mr.generators import HELD_OUT_ICD_FAMILIES, HELD_OUT_INDICATORS, is_held_out

    assert HELD_OUT_INDICATORS <= set(criteria.INDICATOR_BY_KEY)
    assert HELD_OUT_ICD_FAMILIES <= set(icd.FAMILY_BY_KEY)
    assert is_held_out({"indicator": "aki_creatinine_rise"}) and is_held_out({"family": "heart_failure"})
    assert not is_held_out({"indicator": "lactate_gt_2"}) and not is_held_out({"page_class": "bill"})
