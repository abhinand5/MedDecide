"""Generator families for osler_v0 O3 (D20). Each generator returns a base decision and a minimal-pair twin.

The twin changes exactly one structured fact, and the gold label is recomputed from the parts, so it flips.
Every label below is a plain function of the parts; nothing is written by a language model.

Families (nine generators):
  A note facts        note_negation_v1, note_subject_v1, note_timing_v2, note_allergy_med_v1, note_lab_range_v1 (HELD OUT)
  B eligibility       criterion_eligibility_v1
  C triage policy     policy_triage_v1 (HELD OUT)
  D coding            code_assignment_v1, code_family_v1
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass

from meddecide.gen.core import (
    ALLERGENS,
    CONDITIONS,
    LABS,
    MEDICATIONS,
    Allergy,
    GeneratedItem,
    Lab,
    Medication,
    Patient,
    Problem,
    make_item,
    render_note,
    rng_for,
)

YN = [("yes", "Yes"), ("no", "No")]
CHOICE_LETTERS = "ABCDEFGHIJ"
EXAM = "Which of the following is the best answer?"


@dataclass(frozen=True)
class Family:
    name: str
    held_out: bool
    make: Callable[[str], list[GeneratedItem]]


def _other_conditions(rng: random.Random, exclude: set[str], k: int) -> list[str]:
    pool = [c for c, _ in CONDITIONS if c not in exclude]
    return rng.sample(pool, k)


def _patient(pid: str, rng: random.Random, target: Problem, extra: tuple[Problem, ...] = ()) -> Patient:
    meds = tuple(Medication(name, dose) for name, dose in rng.sample(MEDICATIONS, 2))
    allergies = tuple(Allergy(s, r) for s, r in rng.sample(ALLERGENS, 1))
    labs = tuple(Lab(n, u, lo, hi, None) for n, u, lo, hi, _ in LABS[:2])
    return Patient(pid=pid, age=rng.randint(18, 88), sex=rng.choice(["male", "female"]),
                   problems=(target, *extra), medications=meds, allergies=allergies, labs=labs)


def _condition_code(name: str) -> str:
    return next(code for c, code in CONDITIONS if c == name)


SYL_PRE = ("ara", "bel", "cor", "dex", "eli", "fen", "gal", "hyd", "ixo", "jul", "kal", "lor", "mir", "nov", "oxa",
           "pri", "quin", "rel", "sol", "tra")
SYL_MID = ("ta", "zo", "mi", "vo", "ka")
SYL_SUF = ("pril", "mab", "olol", "statin", "azole", "cillin", "mycin", "dipine", "gliptin", "zepam")
DRUG_POOL = tuple(f"{a}{b}{c}" for a in SYL_PRE for b in SYL_MID for c in SYL_SUF)
DOSES = ("5 mg daily", "10 mg twice daily", "20 mg at night", "50 mg as needed", "100 mg daily", "2 puffs as needed")
REACTIONS = ("rash", "urticaria", "nausea", "contact dermatitis", "anaphylaxis", "itching")


# ---- A: note facts with minimal-pair twins -------------------------------------------------------

def note_negation_v1(pid: str) -> list[GeneratedItem]:
    """Does the note document that the patient currently has <condition>? Every note also carries one affirmed and one
    negated distractor, so negation cues appear in both answers. Twin flips the target's affirmed/negated status."""
    rng = rng_for("note_negation_v1", pid)
    cond, _ = rng.choice(CONDITIONS)
    d_aff, d_neg = _other_conditions(rng, {cond}, 2)
    affirmed = rng.random() < 0.5
    out = []
    for role, aff in (("base", affirmed), ("twin", not affirmed)):
        target = Problem(cond, _condition_code(cond), affirmed=aff)
        extra = (Problem(d_aff, _condition_code(d_aff), affirmed=True),
                 Problem(d_neg, _condition_code(d_neg), affirmed=False))
        p = _patient(pid, rng_for("note_negation_v1", pid, "fixed"), target, extra)
        state = render_note(p, rng_for("note_negation_v1", pid, "render"))
        gold = "yes" if aff else "no"
        out.append(make_item("note_negation_v1", "A_note_facts", pid, role, "noul", state,
                             f"Does the note document that the patient currently has {cond}?", YN, gold,
                             {"condition": cond, "affirmed": aff}))
    return out


def note_subject_v1(pid: str) -> list[GeneratedItem]:
    """Is <condition> documented as the patient's own diagnosis (not a family member's)? Every note carries one
    patient-own and one family-history distractor. Twin flips the target's subject."""
    rng = rng_for("note_subject_v1", pid)
    cond, _ = rng.choice(CONDITIONS)
    d_own, d_fam = _other_conditions(rng, {cond}, 2)
    patient_subject = rng.random() < 0.5
    out = []
    for role, subj in (("base", patient_subject), ("twin", not patient_subject)):
        target = Problem(cond, _condition_code(cond), subject="patient" if subj else "family")
        extra = (Problem(d_own, _condition_code(d_own), subject="patient"),
                 Problem(d_fam, _condition_code(d_fam), subject="family"))
        p = _patient(pid, rng_for("note_subject_v1", pid, "fixed"), target, extra)
        state = render_note(p, rng_for("note_subject_v1", pid, "render"))
        gold = "yes" if subj else "no"
        out.append(make_item("note_subject_v1", "A_note_facts", pid, role, "noul", state,
                             f"Is {cond} documented as the patient's own diagnosis (not a family member's)?",
                             YN, gold, {"condition": cond, "subject": "patient" if subj else "family"}))
    return out


def _first_cap(s: str) -> str:
    return s[:1].upper() + s[1:]


def note_timing_v2(pid: str) -> list[GeneratedItem]:
    """Is <condition> an active problem at the visit? Active means the most recent episode is within two years of the
    visit. The note states years, not the words active or resolved, so the answer needs a comparison of numbers.
    Twin moves the most recent episode across the two-year threshold (past <-> current). Shared facts (onset, the
    distractor's dates, the visit date, the age) are drawn once, so the twin differs only in the episode year."""
    rng = rng_for("note_timing_v2", pid)
    cond, _ = rng.choice(CONDITIONS)
    (d_cond,) = _other_conditions(rng, {cond}, 1)
    visit = rng.randint(2021, 2025)
    age = rng.randint(18, 88)
    base_active = rng.random() < 0.5
    e_active = visit - rng.randint(0, 2)
    e_inactive = visit - rng.randint(3, 8)
    onset = rng.randint(max(1990, min(e_active, e_inactive) - 10), min(e_active, e_inactive))
    other_episode = visit - rng.randint(0, 8)
    other_onset = rng.randint(max(1990, other_episode - 10), other_episode)
    out = []
    for role, active in (("base", base_active), ("twin", not base_active)):
        episode = e_active if active else e_inactive
        state = "\n".join([
            f"HPI: {age}-year-old patient seen in clinic. Visit date: {visit}.",
            f"Past medical history: {_first_cap(cond)} (diagnosed {onset}; most recent episode {episode}). "
            f"{_first_cap(d_cond)} (diagnosed {other_onset}; most recent episode {other_episode}).",
        ])
        gold = "yes" if active else "no"
        out.append(make_item("note_timing_v2", "A_note_facts", pid, role, "noul", state,
                             f"Is {cond} an active problem at the visit on {visit}?", YN, gold,
                             {"condition": cond, "visit": visit, "episode": episode, "active": active}))
    return out



def note_allergy_med_v1(pid: str) -> list[GeneratedItem]:
    """Is the patient currently taking <drug>? The drug is on the medication list (yes) or on the allergy list (no).
    Names come from a 1,000-name synthetic pool (no real product names), so no name recurs often enough to act as a
    cue. Both answers carry two medication lines and two allergy lines. The twin moves the target drug to the other
    list (the gold flips); distractor lines are drawn again for the twin and do not bear on the answer."""
    rng = rng_for("note_allergy_med_v1", pid)
    drug = rng.choice(DRUG_POOL)
    dose = rng.choice(DOSES)
    reaction = rng.choice(REACTIONS)
    on_medications = rng.random() < 0.5
    out = []
    for role, on_med in (("base", on_medications), ("twin", not on_medications)):
        drng = rng_for("note_allergy_med_v1", pid, role, "distractors")
        others = [d for d in DRUG_POOL if d != drug]
        d1, d2, d3 = drng.sample(others, 3)
        if on_med:
            meds = (Medication(drug, dose), Medication(d1, drng.choice(DOSES)))
            allergies = (Allergy(d2, drng.choice(REACTIONS)), Allergy(d3, drng.choice(REACTIONS)))
        else:
            meds = (Medication(d1, drng.choice(DOSES)), Medication(d2, drng.choice(DOSES)))
            allergies = (Allergy(drug, reaction), Allergy(d3, drng.choice(REACTIONS)))
        target = Problem(CONDITIONS[0][0], CONDITIONS[0][1])
        p = Patient(pid=pid, age=rng.randint(18, 88), sex="female", problems=(target,), medications=meds,
                    allergies=allergies, labs=())
        state = render_note(p, rng_for("note_allergy_med_v1", pid, "render"))
        gold = "yes" if on_med else "no"
        out.append(make_item("note_allergy_med_v1", "A_note_facts", pid, role, "noul", state,
                             f"Is the patient currently taking {drug}?", YN, gold,
                             {"drug": drug, "on_medication_list": on_med}))
    return out




def note_lab_range_v1(pid: str) -> list[GeneratedItem]:
    """Is the patient's <lab> within the reference range? Choice: within / below / above. Twin moves the value
    to the other side of the range (inside <-> outside). HELD OUT (not used for training, selection or screening
    decisions)."""
    rng = rng_for("note_lab_range_v1", pid)
    name, unit, lo, hi, (vlo, vhi) = rng.choice(LABS)
    options = [("A", "within the reference range"), ("B", "below the reference range"),
               ("C", "above the reference range")]

    def side(v: float) -> str:
        return "within" if lo <= v <= hi else ("below" if v < lo else "above")

    base_value = round(rng.uniform(vlo, vhi), 1)
    if side(base_value) != "within":
        twin_value = round(rng.uniform(lo + 0.1, hi - 0.1), 1)
    elif rng.random() < 0.5:
        twin_value = round(rng.uniform(lo - min(1.0, lo * 0.5), lo - 0.05), 1)
    else:
        twin_value = round(rng.uniform(hi + 0.05, hi + min(1.0, hi * 0.3)), 1)
    out = []
    for role, value in (("base", base_value), ("twin", twin_value)):
        measured = Lab(name, unit, lo, hi, value)
        p = Patient(pid=pid, age=rng.randint(18, 88), sex="male", problems=(Problem(CONDITIONS[0][0], CONDITIONS[0][1]),),
                    labs=(measured,))
        state = render_note(p, rng_for("note_lab_range_v1", pid, "render"))
        key = {"within": "A", "below": "B", "above": "C"}[side(value)]
        out.append(make_item("note_lab_range_v1", "A_note_facts", pid, role, "choice", state,
                             f"Is the patient's {name} within the reference range?", options, key,
                             {"lab": name, "value": value, "side": side(value)}))
    return out


# ---- B: eligibility -------------------------------------------------------------------------------

def criterion_eligibility_v1(pid: str) -> list[GeneratedItem]:
    """Criterion: '<lab> at or above <threshold>'. The note gives the value, or omits it (not enough information).
    Twin: moves the value across the threshold (meets <-> does not meet), or adds the missing value on one side."""
    rng = rng_for("criterion_eligibility_v1", pid)
    name, unit, lo, hi, (vlo, vhi) = rng.choice(LABS)
    thr = round((lo + hi) / 2, 1)
    options = [("A", "Meets the criterion"), ("B", "Does not meet the criterion"),
               ("C", "Not enough information in the note")]
    criterion = f"Trial criterion: \"{name} at or above {thr:g} {unit}.\" Does the patient meet this criterion?"
    base_measured = rng.random() < 0.75
    base_val = round(rng.uniform(vlo, vhi), 1) if base_measured else None
    if base_measured:
        twin_val = round(rng.uniform(thr + 0.2, vhi), 1) if base_val < thr else round(rng.uniform(vlo, thr - 0.2), 1)
    else:
        twin_val = round(rng.uniform(vlo, vhi), 1)
    out = []
    for role, val in (("base", base_val), ("twin", twin_val)):
        labs = (Lab(name, unit, lo, hi, val),)
        p = Patient(pid=pid, age=rng.randint(18, 88), sex="female", problems=(Problem(CONDITIONS[1][0], CONDITIONS[1][1]),),
                    labs=labs)
        state = render_note(p, rng_for("criterion_eligibility_v1", pid, "render"))
        state = f"{state}\n\n{criterion}" if state else criterion
        key = "C" if val is None else "A" if val >= thr else "B"
        out.append(make_item("criterion_eligibility_v1", "B_eligibility", pid, role, "choice", state,
                             "Does the patient meet this criterion?", options, key,
                             {"lab": name, "threshold": thr, "value": val}))
    return out


# ---- C: triage policy (HELD OUT) ------------------------------------------------------------------

POLICY = ("Triage policy: level 3 if systolic BP is below 90 mmHg, or if chest pain comes with sweating; "
          "level 2 if systolic BP is 90 to 139 mmHg, or 180 mmHg or above; otherwise level 1.")


def _triage_level(sbp: int, chest_pain: bool, sweating: bool) -> int:
    if sbp < 90 or (chest_pain and sweating):
        return 3
    if 90 <= sbp <= 139 or sbp >= 180:
        return 2
    return 1


def policy_triage_v1(pid: str) -> list[GeneratedItem]:
    """The level the written policy assigns to a presentation. Twin changes one fact (systolic BP across a threshold,
    or sweating), so the level changes. HELD OUT (not used for training, selection or threshold decisions)."""
    rng = rng_for("policy_triage_v1", pid)
    sbp = rng.randint(80, 200)
    chest_pain = rng.random() < 0.5
    sweating = rng.random() < 0.5
    if chest_pain and sweating and sbp < 90:
        # keep one reason per level: with both rules true, the sweating twin must still change the level
        sbp = rng.randint(90, 200)
    base = (sbp, chest_pain, sweating)
    level = _triage_level(*base)
    candidates = [(sbp, chest_pain, not sweating), (rng.randint(80, 89), chest_pain, sweating),
                  (rng.randint(140, 179), chest_pain, sweating), (rng.randint(90, 139), chest_pain, sweating)]
    twin = next(c for c in candidates if _triage_level(*c) != level)
    keys = ["1", "2", "3"]
    options = [(k, f"level {k}") for k in keys]
    out = []
    for role, (s, cp, sw) in (("base", base), ("twin", twin)):
        presentation = (f"Presentation: systolic BP {s} mmHg; chest pain {'yes' if cp else 'no'}; "
                        f"sweating {'yes' if sw else 'no'}.")
        state = f"{POLICY}\n\n{presentation}"
        lvl = _triage_level(s, cp, sw)
        out.append(make_item("policy_triage_v1", "C_triage_policy", pid, role, "score", state,
                             "Which triage level does the policy assign?", options, str(lvl),
                             {"systolic_bp": s, "chest_pain": cp, "sweating": sw}))
    return out


# ---- D: coding ------------------------------------------------------------------------------------

def _code_list(rng: random.Random, target: str, others: list[str]) -> tuple[list[str], str]:
    names = [target, *others]
    rng.shuffle(names)
    letters = CHOICE_LETTERS
    items = [f"{letters[i]}) {_condition_code(n)} {n}" for i, n in enumerate(names)]
    return items, letters[names.index(target)]


def code_assignment_v1(pid: str) -> list[GeneratedItem]:
    """Which listed code applies to the note's documented condition? E = none of these. Every note carries one negated
    and one family-history distractor outside the code list. Twin flips the target's affirmed status."""
    rng = rng_for("code_assignment_v1", pid)
    target, _ = rng.choice(CONDITIONS)
    others = _other_conditions(rng, {target}, 3)
    code_items, target_letter = _code_list(rng, target, others)
    shown = [item.split(") ", 1)[1].split(" ", 1)[0] for item in code_items]
    d_neg, d_fam = _other_conditions(rng, {target, *others}, 2)
    affirmed = rng.random() < 0.5
    out = []
    for role, aff in (("base", affirmed), ("twin", not affirmed)):
        prob = Problem(target, _condition_code(target), affirmed=aff)
        extra = (Problem(d_neg, _condition_code(d_neg), affirmed=False),
                 Problem(d_fam, _condition_code(d_fam), subject="family"))
        p = _patient(pid, rng_for("code_assignment_v1", pid, "fixed"), prob, extra)
        state = render_note(p, rng_for("code_assignment_v1", pid, "render"))
        state = "Code list: " + "; ".join(code_items) + ".\n\n" + state
        options = [(CHOICE_LETTERS[i], code) for i, code in enumerate(shown)] + [("E", "none of these")]
        gold = target_letter if aff else "E"
        out.append(make_item("code_assignment_v1", "D_coding", pid, role, "choice", state,
                             "Which code applies to the condition documented as current for the patient? "
                             "Choose none of these if no listed condition is a current diagnosis of the patient.",
                             options, gold, {"target": target, "affirmed": aff}))
    return out


def code_family_v1(pid: str) -> list[GeneratedItem]:
    """As code_assignment_v1, but the twin flips the target's subject (the patient's own diagnosis or a family
    member's). Every note carries one negated and one family-history distractor outside the code list."""
    rng = rng_for("code_family_v1", pid)
    target, _ = rng.choice(CONDITIONS)
    others = _other_conditions(rng, {target}, 3)
    code_items, target_letter = _code_list(rng, target, others)
    shown = [item.split(") ", 1)[1].split(" ", 1)[0] for item in code_items]
    d_neg, d_fam = _other_conditions(rng, {target, *others}, 2)
    own = rng.random() < 0.5
    out = []
    for role, subj in (("base", own), ("twin", not own)):
        prob = Problem(target, _condition_code(target), subject="patient" if subj else "family")
        extra = (Problem(d_neg, _condition_code(d_neg), affirmed=False),
                 Problem(d_fam, _condition_code(d_fam), subject="family"))
        p = _patient(pid, rng_for("code_family_v1", pid, "fixed"), prob, extra)
        state = render_note(p, rng_for("code_family_v1", pid, "render"))
        state = "Code list: " + "; ".join(code_items) + ".\n\n" + state
        options = [(CHOICE_LETTERS[i], code) for i, code in enumerate(shown)] + [("E", "none of these")]
        gold = target_letter if subj else "E"
        out.append(make_item("code_family_v1", "D_coding", pid, role, "choice", state,
                             "Which code applies to a condition the patient has? Choose none of these if the "
                             "listed condition is only a family member's history.",
                             options, gold, {"target": target, "subject": "patient" if subj else "family"}))
    return out


FAMILIES: tuple[Family, ...] = (
    Family("note_negation_v1", False, note_negation_v1),
    Family("note_subject_v1", False, note_subject_v1),
    Family("note_timing_v2", False, note_timing_v2),
    Family("note_allergy_med_v1", False, note_allergy_med_v1),
    Family("note_lab_range_v1", True, note_lab_range_v1),
    Family("criterion_eligibility_v1", False, criterion_eligibility_v1),
    Family("policy_triage_v1", True, policy_triage_v1),
    Family("code_assignment_v1", False, code_assignment_v1),
    Family("code_family_v1", False, code_family_v1),
)
GENERATORS: dict[str, Family] = {f.name: f for f in FAMILIES}
HELD_OUT = tuple(sorted(f.name for f in FAMILIES if f.held_out))
