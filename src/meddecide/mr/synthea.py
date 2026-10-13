"""Synthea encounters as a source of record facts (osler_v1; operator decision 2026-10-13: synthetic-first before MIMIC).

Synthea (MIT; `richardyoung/synthea-575k-patients` on the Hub, CSV/parquet tables) gives whole synthetic patient
histories: encounters, LOINC-coded observations, medications, procedures and costs. One emergency or inpatient encounter
becomes a ``SyntheaEncounter``: its measurements as ``criteria.Observation`` (canonical units, hours from the encounter
start), its medications as MAR lines, and charge lines from its procedures, medications and base cost. The renderers in
``pages`` draw these as lab, flowsheet, MAR, bill and ED pages; ``criteria`` computes the indicator labels; and the same
facts go to the teacher (synthetic data, so a hosted teacher is allowed, D30) for narrative pages.

Only explicitly mapped LOINC codes are used, each with its expected UCUM unit; anything else is counted and skipped
(no silent drops). Arterial oxygen saturation (2708-6) is not mapped to SpO2: its oxygen context is unknown.
"""

from __future__ import annotations

import random
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime

from meddecide.gen.core import GeneratedItem, make_item, rng_for
from meddecide.mr import criteria
from meddecide.mr.criteria import ChunkFacts, MedAdmin, Observation
from meddecide.mr.noise import dress_page
from meddecide.mr.pages import (
    PAGE_CLASSES,
    Charge,
    make_context,
    render_bill_page,
    render_ed_page,
    render_flowsheet_page,
    render_lab_page,
    render_mar_page,
)

# LOINC code -> (our observation name, expected UCUM unit)
LOINC: dict[str, tuple[str, str]] = {
    "2524-7": ("lactate", "mmol/L"), "32693-4": ("lactate", "mmol/L"), "2518-9": ("lactate", "mmol/L"),
    "6690-2": ("wbc", "10*3/uL"), "777-3": ("platelets", "10*3/uL"), "1975-2": ("bilirubin", "mg/dL"),
    "38483-4": ("creatinine", "mg/dL"), "2160-0": ("creatinine", "mg/dL"),
    "2947-0": ("sodium", "mmol/L"), "2951-2": ("sodium", "mmol/L"),
    "8310-5": ("temperature", "Cel"), "8867-4": ("heart_rate", "/min"), "9279-1": ("respiratory_rate", "/min"),
    "59408-5": ("spo2", "%"), "2703-7": ("pao2", "mm[Hg]"), "2019-8": ("paco2", "mm[Hg]"),
    "2744-1": ("ph", "[pH]"), "3150-0": ("fio2", "%"), "19996-8": ("fio2", "%"),
}
BP = {"8480-6": "sbp", "8462-4": "dbp"}
UNIT_ALIASES = {"pH": "[pH]", "mmHg": "mm[Hg]"}


def _ts(value: str) -> datetime:
    return datetime.fromisoformat(str(value).replace("Z", "+00:00")).replace(tzinfo=None)


def _hours(when: str, start: datetime) -> float:
    return round((_ts(when) - start).total_seconds() / 3600.0, 2)


def observations_to_facts(rows: Iterable[Mapping[str, object]], start: datetime,
                          counts: dict[str, int] | None = None) -> list[Observation]:
    """Synthea observation rows (DATE, CODE, VALUE, UNITS) of one encounter -> Observations. Blood-pressure pairs taken
    at the same time become a mean arterial pressure (SBP + 2 DBP) / 3."""
    counts = counts if counts is not None else {}
    out: list[Observation] = []
    bp: dict[str, dict[str, float]] = {}
    for r in rows:
        code = str(r["CODE"])
        unit = UNIT_ALIASES.get(str(r.get("UNITS") or ""), str(r.get("UNITS") or ""))
        try:
            value = float(str(r["VALUE"]))
        except ValueError:
            counts["skipped: non-numeric value"] = counts.get("skipped: non-numeric value", 0) + 1
            continue
        if code in BP:
            bp.setdefault(str(r["DATE"]), {})[BP[code]] = value
            continue
        if code not in LOINC:
            continue
        name, expected = LOINC[code]
        if unit != expected:
            key = f"skipped: {name} unit {unit!r}"
            counts[key] = counts.get(key, 0) + 1
            continue
        out.append(Observation(name, value, time_h=_hours(str(r["DATE"]), start)))
        counts[f"kept: {name}"] = counts.get(f"kept: {name}", 0) + 1
    for when, pair in bp.items():
        if "sbp" in pair and "dbp" in pair:
            out.append(Observation("map", (pair["sbp"] + 2 * pair["dbp"]) / 3, time_h=_hours(when, start)))
            counts["kept: map"] = counts.get("kept: map", 0) + 1
    return sorted(out, key=lambda o: (o.time_h, o.name))


_DOSE = re.compile(r"\s(\d[\d.,]*\s*(MG|MCG|UNT|ML|MEQ|%|G)\b.*)$", re.I)


def split_drug(description: str) -> tuple[str, str]:
    """'norepinephrine 1 MG/ML Injectable Solution' -> ('norepinephrine', '1 MG/ML Injectable Solution')."""
    m = _DOSE.search(description)
    if not m:
        return description.strip(), ""
    return description[:m.start()].strip(), m.group(1).strip()


@dataclass(frozen=True)
class SyntheaEncounter:
    encounter_id: str
    patient_id: str
    encounter_class: str                 # emergency, inpatient, ...
    description: str
    reason: str
    age: int
    sex: str
    observations: tuple[Observation, ...]
    meds: tuple[MedAdmin, ...]
    charges: tuple[Charge, ...]
    conditions: tuple[str, ...] = ()     # condition descriptions recorded at this encounter
    meta: dict[str, object] = field(default_factory=dict)

    @property
    def facts(self) -> ChunkFacts:
        return ChunkFacts(observations=self.observations, meds=self.meds)


def build_encounter(enc: Mapping[str, object], patient: Mapping[str, object], obs: Iterable[Mapping[str, object]],
                    meds: Iterable[Mapping[str, object]], procs: Iterable[Mapping[str, object]],
                    conds: Iterable[Mapping[str, object]], counts: dict[str, int] | None = None) -> SyntheaEncounter:
    start = _ts(str(enc["START"]))
    birth = _ts(str(patient["BIRTHDATE"]))
    age = start.year - birth.year - ((start.month, start.day) < (birth.month, birth.day))
    mar = []
    for m in meds:
        drug, dose = split_drug(str(m["DESCRIPTION"]))
        mar.append(MedAdmin(drug, dose or "as ordered", time_h=max(0.0, _hours(str(m["START"]), start))))
    charges = [Charge(0.0, "RB-1000", f"{str(enc['ENCOUNTERCLASS']).upper()} ENCOUNTER - BASE", 1,
                      round(float(enc.get("BASE_ENCOUNTER_COST") or 0.0), 2))]
    for p in procs:
        charges.append(Charge(max(0.0, _hours(str(p["START"]), start)), f"PRC-{str(p['CODE'])[-5:]}",
                              re.sub(r"\s*\(procedure\)\s*$", "", str(p["DESCRIPTION"])).upper()[:43], 1,
                              round(float(p.get("BASE_COST") or 0.0), 2)))
    for m in meds:
        charges.append(Charge(max(0.0, _hours(str(m["START"]), start)), f"RX-{str(m['CODE'])[-5:]}",
                              str(m["DESCRIPTION"]).upper()[:43], int(m.get("DISPENSES") or 1),
                              round(float(m.get("BASE_COST") or 0.0), 2)))
    return SyntheaEncounter(
        encounter_id=str(enc["Id"]), patient_id=str(enc["PATIENT"]), encounter_class=str(enc["ENCOUNTERCLASS"]),
        description=re.sub(r"\s*\((procedure|encounter|regime/therapy)\)\s*$", "", str(enc["DESCRIPTION"])),
        reason=str(enc.get("REASONDESCRIPTION") or "") if str(enc.get("REASONDESCRIPTION") or "") != "nan" else "",
        age=age, sex="male" if str(patient["GENDER"]).upper().startswith("M") else "female",
        observations=tuple(observations_to_facts(obs, start, counts)), meds=tuple(mar), charges=tuple(charges),
        conditions=tuple(sorted({re.sub(r"\s*\((disorder|finding|situation)\)\s*$", "", str(c["DESCRIPTION"]))
                                 for c in conds})))


PAGE_LETTER = {k: "ABCDEFGHIJKL"[i] for i, (k, _) in enumerate(PAGE_CLASSES)}
LAB_NAMES = {"lactate", "wbc", "platelets", "bilirubin", "creatinine", "sodium", "pao2", "paco2", "ph", "fio2"}
VITAL_NAMES = {"temperature", "heart_rate", "respiratory_rate", "map", "spo2", "fio2"}


def renderable_pages(enc: SyntheaEncounter) -> list[str]:
    pages = []
    if any(o.name in LAB_NAMES for o in enc.observations):
        pages.append("laboratory")
    if sum(o.name in VITAL_NAMES for o in enc.observations) >= 2:
        pages.append("flowsheet")
    if enc.meds:
        pages.append("mar")
    if len(enc.charges) >= 3:
        pages.append("bill")
    if enc.encounter_class == "emergency" and any(o.name in VITAL_NAMES for o in enc.observations):
        pages.append("emergency_department")
    return pages


def page_facts(enc: SyntheaEncounter, page: str) -> ChunkFacts:
    """Exactly the facts a page of this type shows; indicator labels are computed on these, never on the whole
    encounter."""
    if page == "laboratory":
        return ChunkFacts(tuple(o for o in enc.observations if o.name in LAB_NAMES))
    if page in ("flowsheet", "emergency_department"):
        return ChunkFacts(tuple(o for o in enc.observations if o.name in VITAL_NAMES))
    if page == "mar":
        return ChunkFacts(meds=enc.meds)
    if page == "bill":
        return ChunkFacts()
    raise ValueError(page)


def render_page(enc: SyntheaEncounter, page: str, rng: random.Random) -> str:
    facts = page_facts(enc, page)
    ctx = make_context(rng)
    if page == "laboratory":
        body = render_lab_page(ctx, facts, rng)
    elif page == "flowsheet":
        body = render_flowsheet_page(ctx, facts, rng)
    elif page == "mar":
        body = render_mar_page(ctx, facts, rng)
    elif page == "bill":
        body = render_bill_page(ctx, enc.charges, rng)
    else:
        body = render_ed_page(ctx, facts, enc.reason or enc.description.lower(), rng)
    return dress_page(body, rng)


def page_category_items(enc: SyntheaEncounter) -> list[GeneratedItem]:
    """One page-category item per renderable page type of the encounter (template ``synthea_page_category_v1``)."""
    out = []
    options = [(PAGE_LETTER[k], label) for k, label in PAGE_CLASSES]
    for page in renderable_pages(enc):
        rng = rng_for("synthea_page_category_v1", enc.encounter_id, page)
        state = render_page(enc, page, rng)
        out.append(make_item("synthea_page_category_v1", "MR_page_category", f"{enc.patient_id}:{enc.encounter_id}:{page}",
                             "base", "choice", state, rng.choice(("Which type of medical-record page is this?",
                                                                  "Classify this page of the medical record.")),
                             options, PAGE_LETTER[page], {"page_class": page, "source": "synthea"}))
    return out


INDICATOR_NAMES: dict[str, set[str]] = {
    "lactate_gt_2": {"lactate"}, "map_lt_65": {"map"}, "sirs_temperature": {"temperature"},
    "sirs_heart_rate": {"heart_rate"}, "sirs_respiratory": {"respiratory_rate", "paco2"}, "sirs_wbc": {"wbc"},
    "platelets_lt_150": {"platelets"}, "bilirubin_ge_1_2": {"bilirubin"}, "pao2_lt_60": {"pao2"},
    "spo2_lt_91_room_air": {"spo2"}, "hypercapnic_acidosis": {"paco2", "ph"}, "pf_ratio_lt_300": {"pao2", "fio2"},
    "aki_creatinine_rise": {"creatinine"}, "sodium_lt_135": {"sodium"},
}


def indicator_items(enc: SyntheaEncounter, held_out: frozenset[str] = frozenset()) -> list[GeneratedItem]:
    """Indicator questions answerable from the encounter's own values (template ``synthea_indicator_evidence_v1``);
    the label is ``criteria.supports`` on the values shown. The builder balances yes/no by sampling."""
    out = []
    candidates = [(k, "laboratory" if need & (LAB_NAMES - {"fio2"}) else "flowsheet") for k, need in INDICATOR_NAMES.items()]
    candidates.append(("vasopressor_given", "mar"))
    for key, page in candidates:
        if key in held_out:
            continue
        facts = page_facts(enc, page)
        shown = {o.name for o in facts.observations}
        need = INDICATOR_NAMES.get(key, set())
        if key == "vasopressor_given":
            if not facts.meds:
                continue
        elif not (need <= shown or (key == "sirs_respiratory" and need & shown)):
            continue
        rng = rng_for("synthea_indicator_evidence_v1", enc.encounter_id, key)
        gold = "yes" if criteria.supports(key, facts) else "no"
        ind = criteria.INDICATOR_BY_KEY[key]
        out.append(make_item("synthea_indicator_evidence_v1", "MR_indicator_evidence",
                             f"{enc.patient_id}:{enc.encounter_id}:{key}", "base", "noul", render_page(enc, page, rng),
                             f"Does this page show {ind.description} during the current admission?",
                             [("yes", "Yes"), ("no", "No")], gold, {"indicator": key, "source": "synthea", "page": page}))
    return out


NARRATIVE_PAGES = ("discharge_summary", "admission", "progress_note", "consult_note", "emergency_department",
                   "nutrition", "radiology")


def narrative_facts(enc: SyntheaEncounter) -> dict[str, object]:
    """The facts a teacher may use to write a narrative page about this encounter (all synthetic)."""
    return {
        "age": enc.age, "sex": enc.sex, "encounter": enc.description, "reason": enc.reason,
        "conditions": list(enc.conditions)[:8],
        "medications": sorted({m.drug for m in enc.meds})[:8],
        "measurements": [{"name": o.name, "value": round(o.displayed()[0], 2), "unit": o.displayed()[1],
                          "hours_from_start": o.time_h} for o in enc.observations[:20]],
    }
