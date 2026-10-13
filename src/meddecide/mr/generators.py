"""Generated medical-record audit decisions (osler_v1; D20 labels by code, D27 text sources).

Three generators that need no MIMIC and no teacher:

  mr_page_category_tabular_v1   choice over the twelve page classes; the page is a rendered laboratory, flowsheet,
                                MAR, bill or ED-triage page with fax/OCR dressing (narrative classes come from MIMIC
                                notes and teacher prose, not from here)
  mr_indicator_evidence_v1      noul: does this page show supporting evidence of one clinical indicator in the current
                                encounter? The twin changes one fact (a value across the threshold, a prior admission,
                                a documented baseline, oxygen instead of room air, a held MAR dose, a non-vasopressor
                                drug) and the label is recomputed by ``meddecide.mr.criteria``
  mr_icd_support_template_v1    choice over one ICD-10-CM code family plus "none": which code does a one-diagnosis
                                assessment support? Template text, so it teaches the coding rules (certainty by setting,
                                history, family, specificity), not language; the teacher variant replaces it at scale

Every label is recomputed from the parts the page was drawn from; a generator that fails to flip its twin retries
with the next variant seed and raises if it cannot.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass, replace

from meddecide.gen.core import GeneratedItem, make_item, rng_for
from meddecide.mr import criteria, icd
from meddecide.mr.criteria import ALT_UNITS, ChunkFacts, MedAdmin, Observation
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

YN = [("yes", "Yes"), ("no", "No")]
# held out of every training source in osler_v1 (generated, teacher prose and MIMIC alike); evaluation only
HELD_OUT_INDICATORS = frozenset({"aki_creatinine_rise"})
HELD_OUT_ICD_FAMILIES = frozenset({"heart_failure"})


def is_held_out(meta: dict) -> bool:
    return meta.get("indicator") in HELD_OUT_INDICATORS or meta.get("family") in HELD_OUT_ICD_FAMILIES
LETTERS = "ABCDEFGHIJKLMNOP"
MAX_TRIES = 12


@dataclass(frozen=True)
class MRGenerator:
    name: str
    family: str
    held_out: bool
    make: Callable[[str], list[GeneratedItem]]


# ---- value ranges: each indicator's "meets" and "does not meet" ranges keep a margin from the threshold, so display
# rounding and unit conversion cannot move a value across it ----------------------------------------------------------

def _u(rng: random.Random, lo: float, hi: float) -> float:
    return rng.uniform(lo, hi)


DISTRACTOR_RANGES: dict[str, tuple[float, float]] = {
    "lactate": (0.6, 1.9), "wbc": (4.5, 11.0), "platelets": (160, 380), "bilirubin": (0.3, 1.0), "creatinine": (0.6, 1.1),
    "sodium": (136, 144), "temperature": (36.4, 37.6), "heart_rate": (60, 88), "respiratory_rate": (12, 19),
    "map": (68, 98), "spo2": (93, 99),
}
LAB_NAMES = ("lactate", "wbc", "platelets", "bilirubin", "creatinine", "sodium")
VITAL_NAMES = ("temperature", "heart_rate", "respiratory_rate", "map", "spo2")


def _distractors(rng: random.Random, names: tuple[str, ...], exclude: set[str], k: int, times: list[float]) -> list[Observation]:
    pool = [n for n in names if n not in exclude]
    out = []
    for n in rng.sample(pool, min(k, len(pool))):
        lo, hi = DISTRACTOR_RANGES[n]
        for t in times:
            out.append(Observation(n, _u(rng, lo, hi), time_h=t))
    return out


@dataclass(frozen=True)
class IndicatorCase:
    page: str                                    # "lab", "flowsheet" or "mar"
    primary: Callable[[random.Random, bool], tuple[list[Observation], list[MedAdmin]]]
    traps: tuple[str, ...]                       # traps that turn a supporting fact into a non-supporting one


def _single(name: str, yes: tuple[tuple[float, float], ...], no: tuple[tuple[float, float], ...], **kw):
    def build(rng: random.Random, meets: bool) -> tuple[list[Observation], list[MedAdmin]]:
        lo, hi = rng.choice(yes if meets else no)
        return [Observation(name, _u(rng, lo, hi), time_h=rng.uniform(2, 40), **kw)], []
    return build


def _vaso(rng: random.Random, meets: bool) -> tuple[list[Observation], list[MedAdmin]]:
    if meets:
        drug, dose = rng.choice((("norepinephrine", "0.08 mcg/kg/min IV infusion"), ("vasopressin", "0.03 units/min IV"),
                                 ("phenylephrine", "100 mcg/min IV infusion"), ("epinephrine", "0.05 mcg/kg/min IV")))
    else:
        drug, dose = rng.choice((("midodrine", "10 mg PO TID"), ("nicardipine", "5 mg/h IV infusion"),
                                 ("metoprolol tartrate", "25 mg PO BID"), ("sodium chloride 0.9%", "1000 mL IV bolus")))
    return [], [MedAdmin(drug, dose, time_h=rng.uniform(2, 40))]


def _hypercapnic(rng: random.Random, meets: bool) -> tuple[list[Observation], list[MedAdmin]]:
    t = rng.uniform(2, 40)
    if meets:
        paco2, ph = _u(rng, 53, 90), _u(rng, 7.10, 7.32)
    elif rng.random() < 0.5:
        paco2, ph = _u(rng, 36, 48), _u(rng, 7.10, 7.45)
    else:
        paco2, ph = _u(rng, 53, 80), _u(rng, 7.37, 7.45)   # compensated: high PaCO2 with a normal pH
    return [Observation("paco2", paco2, time_h=t), Observation("ph", ph, time_h=t)], []


def _pf(rng: random.Random, meets: bool) -> tuple[list[Observation], list[MedAdmin]]:
    t = rng.uniform(2, 40)
    fio2 = rng.choice((21, 28, 32, 36, 40, 50, 60, 80))
    ratio = _u(rng, 90, 280) if meets else _u(rng, 320, 480)
    return [Observation("pao2", ratio * fio2 / 100, time_h=t), Observation("fio2", fio2, time_h=t)], []


def _aki(rng: random.Random, meets: bool) -> tuple[list[Observation], list[MedAdmin]]:
    c0 = _u(rng, 0.7, 1.5)
    t0 = rng.uniform(1, 8)
    t1 = t0 + rng.uniform(8, 40)
    c1 = c0 + _u(rng, 0.4, 2.0) if meets else c0 + _u(rng, -0.2, 0.15)
    return [Observation("creatinine", c0, time_h=t0), Observation("creatinine", c1, time_h=t1)], []


CASES: dict[str, IndicatorCase] = {
    "lactate_gt_2": IndicatorCase("lab", _single("lactate", ((2.2, 8.0),), ((0.6, 1.9),)),
                                  ("prior_encounter", "value_cross")),
    "map_lt_65": IndicatorCase("flowsheet", _single("map", ((45, 62),), ((68, 100),)), ("prior_encounter", "value_cross")),
    "vasopressor_given": IndicatorCase("mar", _vaso, ("mar_status", "prior_encounter", "drug_swap")),
    "sirs_temperature": IndicatorCase("flowsheet", _single("temperature", ((38.3, 40.5), (34.0, 35.7)), ((36.3, 37.8),)),
                                      ("prior_encounter", "value_cross")),
    "sirs_heart_rate": IndicatorCase("flowsheet", _single("heart_rate", ((94, 150),), ((55, 87),)),
                                     ("prior_encounter", "value_cross")),
    "sirs_respiratory": IndicatorCase("flowsheet", _single("respiratory_rate", ((23, 36),), ((12, 19),)),
                                      ("prior_encounter", "value_cross")),
    "sirs_wbc": IndicatorCase("lab", _single("wbc", ((12.6, 30.0), (1.0, 3.6)), ((4.4, 11.4),)),
                              ("prior_encounter", "value_cross")),
    "platelets_lt_150": IndicatorCase("lab", _single("platelets", ((20, 142),), ((158, 400),)),
                                      ("prior_encounter", "baseline", "value_cross")),
    "bilirubin_ge_1_2": IndicatorCase("lab", _single("bilirubin", ((1.3, 8.0),), ((0.2, 1.1),)),
                                      ("prior_encounter", "baseline", "value_cross")),
    "pao2_lt_60": IndicatorCase("lab", _single("pao2", ((38, 57),), ((64, 110),)),
                                ("prior_encounter", "baseline", "value_cross")),
    "spo2_lt_91_room_air": IndicatorCase("flowsheet", _single("spo2", ((78, 89),), ((93, 100),)),
                                         ("on_oxygen", "baseline", "prior_encounter", "value_cross")),
    "hypercapnic_acidosis": IndicatorCase("lab", _hypercapnic, ("prior_encounter", "baseline", "value_cross")),
    "pf_ratio_lt_300": IndicatorCase("lab", _pf, ("prior_encounter", "value_cross")),
    "aki_creatinine_rise": IndicatorCase("lab", _aki, ("prior_encounter", "value_cross")),
    "sodium_lt_135": IndicatorCase("lab", _single("sodium", ((118, 133),), ((136, 146),)),
                                   ("prior_encounter", "baseline", "value_cross")),
}
assert set(CASES) == set(criteria.INDICATOR_BY_KEY), "every indicator needs a case"


def _apply_trap(trap: str, obs: list[Observation], meds: list[MedAdmin], rng: random.Random
                ) -> tuple[list[Observation], list[MedAdmin]]:
    if trap == "prior_encounter":
        shift = -24 * rng.randint(20, 400)
        return ([replace(o, encounter="prior", time_h=o.time_h + shift) for o in obs],
                [replace(m, encounter="prior", time_h=m.time_h + shift) for m in meds])
    if trap == "baseline":
        return [replace(o, baseline=True) for o in obs], meds
    if trap == "on_oxygen":
        return [replace(o, on_room_air=False) if o.name == "spo2" else o for o in obs], meds
    if trap == "mar_status":
        status = rng.choice(("held", "refused", "not_given"))
        return obs, [replace(m, status=status) for m in meds]
    raise ValueError(trap)


def _with_units(obs: list[Observation], rng: random.Random, p: float = 0.25) -> list[Observation]:
    out = []
    for o in obs:
        if o.name in ALT_UNITS and rng.random() < p:
            out.append(replace(o, display_unit=ALT_UNITS[o.name][0]))
        elif o.name == "temperature" and rng.random() < 0.5:
            out.append(replace(o, display_unit="F"))
        else:
            out.append(o)
    return out


def _render(page: str, facts: ChunkFacts, rng: random.Random) -> str:
    ctx = make_context(rng)
    if page == "lab":
        body = render_lab_page(ctx, facts, rng)
    elif page == "flowsheet":
        body = render_flowsheet_page(ctx, facts, rng)
    elif page == "mar":
        body = render_mar_page(ctx, facts, rng)
    else:
        raise ValueError(page)
    return dress_page(body, rng)


def _indicator_facts(key: str, case: IndicatorCase, rng: random.Random, meets: bool, trap: str | None
                     ) -> ChunkFacts:
    # draw order keeps the base and its twin identical except for the trap: primary facts and their units first, then
    # the distractors, and the trap's own randomness last
    obs, meds = case.primary(rng, meets)
    obs = _with_units(obs, rng)
    times = sorted({round(rng.uniform(1, 44), 1) for _ in range(rng.randint(1, 3))})
    primary_names = {o.name for o in obs}
    extra_obs: list[Observation] = []
    extra_meds: list[MedAdmin] = []
    if case.page == "lab":
        extra_obs = _distractors(rng, LAB_NAMES, primary_names, rng.randint(2, 4), times)
    elif case.page == "flowsheet":
        extra_obs = _distractors(rng, VITAL_NAMES, primary_names, rng.randint(2, 4), times)
    else:
        extra_meds = [MedAdmin(d, dose, time_h=rng.uniform(1, 44), status=rng.choice(("given", "given", "held")))
                      for d, dose in rng.sample((("ceftriaxone", "1 g IV q24h"), ("heparin", "5,000 units SC q8h"),
                                                 ("pantoprazole", "40 mg IV daily"),
                                                 ("acetaminophen", "650 mg PO q6h PRN"),
                                                 ("insulin lispro", "4 units SC with meals"),
                                                 ("piperacillin-tazobactam", "4.5 g IV q6h")), rng.randint(2, 4))]
    if trap is not None and trap not in ("value_cross", "drug_swap"):
        obs, meds = _apply_trap(trap, obs, meds, rng)
    obs = obs + extra_obs
    meds = meds + extra_meds
    return ChunkFacts(observations=tuple(obs), meds=tuple(meds), meta={"indicator": key, "trap": trap})


def mr_indicator_evidence_v1(pid: str) -> list[GeneratedItem]:
    """Base and twin differ by one trap (or one value across the threshold); labels come from the criteria rule."""
    for attempt in range(MAX_TRIES):
        rng = rng_for("mr_indicator_evidence_v1", pid, str(attempt))
        key = rng.choice(sorted(CASES))
        case = CASES[key]
        trap = rng.choice(case.traps)
        seed = rng.randrange(10**9)
        # the "supporting" version and its trapped (or crossed) counterpart share the seed, so only the trap differs
        sup = _indicator_facts(key, case, random.Random(seed), True, None)
        if trap in ("value_cross", "drug_swap"):
            non = _indicator_facts(key, case, random.Random(seed), False, None)
        else:
            non = _indicator_facts(key, case, random.Random(seed), True, trap)
        if criteria.supports(key, sup) == criteria.supports(key, non):
            continue
        render_seed = rng.randrange(10**9)
        order = [("base", sup), ("twin", non)] if rng.random() < 0.5 else [("base", non), ("twin", sup)]
        ind = criteria.INDICATOR_BY_KEY[key]
        question = rng.choice((
            f"Does this page show {ind.description} during the current admission?",
            f"Does this page contain supporting evidence of {ind.description} in this encounter?",
            f"For clinical validation of {ind.condition}: does this page document {ind.description} during the current "
            f"admission?",
        ))
        out = []
        for role, facts in order:
            state = _render(case.page, facts, random.Random(render_seed))
            gold = "yes" if criteria.supports(key, facts) else "no"
            out.append(make_item("mr_indicator_evidence_v1", "MR_indicator_evidence", pid, role, "noul", state,
                                 question, YN, gold, {"indicator": key, "trap": trap, "page": case.page}))
        return out
    raise RuntimeError(f"mr_indicator_evidence_v1/{pid}: no flipping twin in {MAX_TRIES} tries")


# ---- page category ------------------------------------------------------------------------------

BILL_LINES: tuple[tuple[str, str, float], ...] = (
    ("RB-1110", "ROOM & BOARD SEMI-PRIVATE", 2450.0), ("RB-2000", "ICU ROOM & BOARD", 6890.0),
    ("J0696", "CEFTRIAXONE SODIUM INJ 250 MG", 38.5), ("J2543", "PIPERACILLIN/TAZOBACTAM INJ 1 G", 41.2),
    ("J3370", "VANCOMYCIN HCL INJ 500 MG", 22.8), ("J1644", "HEPARIN SODIUM INJ PER 1000 U", 9.7),
    ("J2405", "ONDANSETRON HCL INJ PER 1 MG", 14.1), ("J1940", "FUROSEMIDE INJ UP TO 20 MG", 11.3),
    ("J1815", "INSULIN INJ PER 5 UNITS", 6.4), ("A4217", "STERILE WATER/SALINE 500 ML", 27.0),
    ("LAB-3021", "LACTIC ACID", 96.0), ("LAB-3100", "COMPREHENSIVE METABOLIC PANEL", 212.0),
    ("LAB-3200", "CBC W/ AUTO DIFF", 141.0), ("LAB-3350", "BLOOD GAS ARTERIAL", 318.0),
    ("RAD-7110", "CHEST X-RAY 1 VIEW", 412.0), ("PHY-0900", "RESPIRATORY THERAPY TREATMENT", 188.0),
    ("SUP-4410", "IV START KIT", 64.0), ("PHR-0001", "PHARMACY - GENERAL", 18.0),
)
CHIEF_COMPLAINTS = ("shortness of breath", "fever and confusion", "chest pain", "abdominal pain", "fall at home",
                    "weakness", "cough and fever", "syncope", "altered mental status", "leg swelling")
PAGE_LETTER = {k: LETTERS[i] for i, (k, _) in enumerate(PAGE_CLASSES)}


def _random_facts(rng: random.Random, names: tuple[str, ...]) -> ChunkFacts:
    times = sorted({round(rng.uniform(1, 70), 1) for _ in range(rng.randint(1, 5))})
    return ChunkFacts(observations=tuple(_with_units(_distractors(rng, names, set(), rng.randint(2, 5), times), rng)))


def mr_page_category_tabular_v1(pid: str) -> list[GeneratedItem]:
    rng = rng_for("mr_page_category_tabular_v1", pid)
    page = rng.choice(("laboratory", "flowsheet", "mar", "bill", "emergency_department"))
    ctx = make_context(rng)
    if page == "laboratory":
        body = render_lab_page(ctx, _random_facts(rng, LAB_NAMES), rng)
    elif page == "flowsheet":
        body = render_flowsheet_page(ctx, _random_facts(rng, VITAL_NAMES), rng)
    elif page == "mar":
        meds = tuple(MedAdmin(d, dose, time_h=rng.uniform(1, 70), status=rng.choice(("given", "given", "held")))
                     for d, dose in rng.sample((("ceftriaxone", "1 g IV q24h"), ("heparin", "5,000 units SC q8h"),
                                                ("pantoprazole", "40 mg IV daily"), ("metoprolol", "25 mg PO BID"),
                                                ("furosemide", "40 mg IV BID"), ("vancomycin", "1,250 mg IV q12h"),
                                                ("acetaminophen", "650 mg PO q6h PRN")), rng.randint(3, 6)))
        body = render_mar_page(ctx, ChunkFacts(meds=meds), rng)
    elif page == "bill":
        charges = [Charge(rng.uniform(0, 96), code, desc, units := rng.randint(1, 4), round(price * units, 2))
                   for code, desc, price in rng.sample(BILL_LINES, rng.randint(5, 12))]
        body = render_bill_page(ctx, charges, rng)
    else:
        body = render_ed_page(ctx, _random_facts(rng, VITAL_NAMES), rng.choice(CHIEF_COMPLAINTS), rng)
    state = dress_page(body, rng)
    options = [(PAGE_LETTER[k], label) for k, label in PAGE_CLASSES]
    question = rng.choice(("Which type of medical-record page is this?", "Classify this page of the medical record.",
                           "What category does this record page belong to?"))
    return [make_item("mr_page_category_tabular_v1", "MR_page_category", pid, "base", "choice", state, question,
                      options, PAGE_LETTER[page], {"page_class": page})]


# ---- ICD-10-CM code support (template text) -----------------------------------------------------

ATTR_TEXT: dict[str, dict[str, str]] = {
    "acuity": {"acute": "acute", "chronic": "chronic", "acute_on_chronic": "acute on chronic", "unspecified": ""},
    "gas": {"hypoxia": "hypoxic", "hypercapnia": "hypercapnic", "unspecified": ""},
    "stage": {"1": "stage 1", "2": "stage 2", "3": "stage 3", "3a": "stage 3a", "3b": "stage 3b", "4": "stage 4",
              "5": "stage 5", "esrd": "end-stage (on dialysis)"},
    "complication": {"none": "without complications", "hyperglycemia": "with hyperglycemia",
                     "diabetic_ckd": "with diabetic chronic kidney disease", "hypoglycemia": "with hypoglycemia"},
    "organism": {"unspecified": "", "e_coli": "due to E. coli", "mssa": "due to MSSA", "mrsa": "due to MRSA",
                 "pseudomonas": "due to Pseudomonas"},
    "shock": {"no": "without shock", "yes": "with septic shock"},
    "type": {"systolic": "systolic", "diastolic": "diastolic (HFpEF)", "unspecified": ""},
    "severity": {"mild": "mild", "moderate": "moderate", "severe": "severe"},
    "laterality": {"right": "of the right knee", "left": "of the left knee", "bilateral": "of both knees",
                   "unspecified": ""},
}
CERTAINTY_TEXT = {"confirmed": ("", "Diagnosis: "), "probable": ("probable ", "Likely "),
                  "suspected": ("suspected ", "Possible "), "ruled_out": ("", "Ruled out: ")}


def _dx_phrase(dx: icd.Diagnosis) -> str:
    fam = icd.FAMILY_BY_KEY[dx.family]
    a = dx.attributes
    pre = " ".join(ATTR_TEXT[k][a[k]] for k in ("acuity", "severity", "type", "gas") if k in a and ATTR_TEXT[k][a[k]])
    post = " ".join(ATTR_TEXT[k][a[k]] for k in ("stage", "complication", "organism", "shock", "laterality")
                    if k in a and ATTR_TEXT[k][a[k]])
    return " ".join(x for x in (pre, fam.condition, post) if x)


def render_dx_template(dx: icd.Diagnosis, setting: icd.Setting, rng: random.Random) -> str:
    """A one-diagnosis assessment section. Template text: it carries the coding facts, not clinical language."""
    phrase = _dx_phrase(dx)
    head = rng.choice(("Inpatient discharge summary - assessment", "Hospital course / final diagnoses",
                       "Discharge diagnoses")) if setting == "inpatient" else rng.choice(
        ("Outpatient clinic note - assessment and plan", "Office visit - assessment", "Clinic follow-up - impression"))
    if dx.subject == "family":
        line = rng.choice((f"Family history: mother with {phrase}.", f"Father had {phrase}."))
    elif dx.status == "history":
        line = rng.choice((f"History of {phrase}, resolved.", f"Past {phrase}; no longer active."))
    elif dx.certainty == "ruled_out":
        line = rng.choice((f"{_cap(phrase)} was ruled out.", f"Ruled out: {phrase}."))
    elif dx.certainty == "confirmed":
        line = rng.choice((f"{_cap(phrase)}.", f"Diagnosis: {phrase}.", f"Assessment: {phrase}."))
    else:
        word = rng.choice(("probable", "likely")) if dx.certainty == "probable" else rng.choice(("suspected", "possible"))
        line = rng.choice((f"{_cap(word)} {phrase}.", f"Assessment: {word} {phrase}."))
    return f"{head}\n{line}\nPlan: continue current management; follow up as scheduled."


def _cap(s: str) -> str:
    return s[:1].upper() + s[1:]


def _random_dx(rng: random.Random, family: str) -> icd.Diagnosis:
    fam = icd.FAMILY_BY_KEY[family]
    attrs = {k: rng.choice(v) for k, v in fam.attributes.items()}
    certainty = rng.choice(("confirmed", "confirmed", "probable", "suspected", "ruled_out"))
    subject = "family" if rng.random() < 0.08 else "patient"
    status = "history" if rng.random() < 0.08 else "current"
    return icd.Diagnosis(family, attrs, certainty, subject, status)


def _one_fact_changes(dx: icd.Diagnosis) -> list[icd.Diagnosis]:
    """Every diagnosis that differs from ``dx`` in exactly one fact."""
    fam = icd.FAMILY_BY_KEY[dx.family]
    out = [replace(dx, certainty=c) for c in ("confirmed", "probable", "suspected", "ruled_out") if c != dx.certainty]
    out.append(replace(dx, subject="patient" if dx.subject == "family" else "family"))
    out.append(replace(dx, status="current" if dx.status == "history" else "history"))
    for k, values in fam.attributes.items():
        out += [replace(dx, attributes={**dx.attributes, k: v}) for v in values if v != dx.attributes[k]]
    return out


def mr_icd_support_template_v1(pid: str) -> list[GeneratedItem]:
    for attempt in range(MAX_TRIES):
        rng = rng_for("mr_icd_support_template_v1", pid, str(attempt))
        family = rng.choice([f.key for f in icd.FAMILIES])
        setting = rng.choice(("inpatient", "outpatient"))
        base = _random_dx(rng, family)
        g_base = icd.supported_code(base, setting)
        flips = [t for t in _one_fact_changes(base) if icd.supported_code(t, setting) != g_base]
        if not flips:
            continue
        twin = rng.choice(flips)
        g_twin = icd.supported_code(twin, setting)
        fam = icd.FAMILY_BY_KEY[family]
        codes = sorted(fam.codes)
        options = [(LETTERS[i], f"{c} - {fam.codes[c]}") for i, c in enumerate(codes)]
        none_key = LETTERS[len(codes)]
        options.append((none_key, "None of these codes is supported by the documentation"))
        render_seed = rng.randrange(10**9)
        question = rng.choice(("Which ICD-10-CM code does this documentation support, as the most specific code?",
                               "Select the correct ICD-10-CM code for the documented diagnosis, if any."))
        pair = [(base, g_base), (twin, g_twin)]
        if rng.random() < 0.5:  # the drawn diagnosis is biased towards a codeable one; either may be the base
            pair.reverse()
        out = []
        for role, (dx, gold_code) in zip(("base", "twin"), pair, strict=True):
            state = render_dx_template(dx, setting, random.Random(render_seed))
            gold = none_key if gold_code is None else LETTERS[codes.index(gold_code)]
            out.append(make_item("mr_icd_support_template_v1", "MR_icd_support", pid, role, "choice", state, question,
                                 options, gold, {"family": family, "setting": setting, "code": gold_code,
                                                 "certainty": dx.certainty, "subject": dx.subject,
                                                 "status": dx.status, "attributes": dict(dx.attributes)}))
        return out
    raise RuntimeError(f"mr_icd_support_template_v1/{pid}: no flipping twin in {MAX_TRIES} tries")


GENERATORS: tuple[MRGenerator, ...] = (
    MRGenerator("mr_page_category_tabular_v1", "MR_page_category", False, mr_page_category_tabular_v1),
    MRGenerator("mr_indicator_evidence_v1", "MR_indicator_evidence", False, mr_indicator_evidence_v1),
    MRGenerator("mr_icd_support_template_v1", "MR_icd_support", False, mr_icd_support_template_v1),
)
