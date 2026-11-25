"""Gold-by-construction clinical decision generators: structured parts, rendering, splits (osler_v0 O3, D20).

Every state is assembled by code from structured parts (problems, medications, allergies, labs with units and
reference ranges). Every label is computed by code from the same parts. No language model writes any text or
label. Surface variation (section order, phrasing, abbreviations, filler) comes from fixed template tables and a
seeded random generator, so the same patient id always yields the same text.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Literal

from meddecide.utils.hashing import normalize_text, stable_hash

Split = Literal["train", "dev", "test"]

# ---- structured parts ---------------------------------------------------------------------------

CONDITIONS: tuple[tuple[str, str], ...] = (
    ("type 2 diabetes mellitus", "E11"),
    ("essential hypertension", "I10"),
    ("asthma", "J45"),
    ("chronic kidney disease", "N18"),
    ("hypothyroidism", "E03"),
    ("atrial fibrillation", "I48"),
    ("iron deficiency anaemia", "D50"),
    ("migraine", "G43"),
)
MEDICATIONS: tuple[tuple[str, str], ...] = (
    ("metformin", "500 mg twice daily"),
    ("lisinopril", "10 mg daily"),
    ("salbutamol inhaler", "2 puffs as needed"),
    ("levothyroxine", "50 mcg daily"),
    ("apixaban", "5 mg twice daily"),
    ("atorvastatin", "20 mg at night"),
    ("ferrous sulfate", "200 mg daily"),
    ("sumatriptan", "50 mg as needed"),
)
ALLERGENS: tuple[tuple[str, str], ...] = (
    ("penicillin", "rash"),
    ("sulfonamides", "urticaria"),
    ("latex", "contact dermatitis"),
    ("codeine", "nausea"),
)
# name, unit, reference low, reference high, plausible value range (low, high) for generation
LABS: tuple[tuple[str, str, float, float, tuple[float, float]], ...] = (
    ("haemoglobin", "g/dL", 12.0, 16.0, (7.0, 18.0)),
    ("creatinine", "mg/dL", 0.6, 1.2, (0.3, 3.0)),
    ("HbA1c", "%", 4.0, 5.6, (4.5, 11.0)),
    ("potassium", "mmol/L", 3.5, 5.0, (2.5, 6.5)),
)


@dataclass(frozen=True)
class Problem:
    condition: str
    code: str
    affirmed: bool = True          # False = negated ("never diagnosed with", "denies")
    subject: Literal["patient", "family"] = "patient"
    status: Literal["active", "resolved"] = "active"


@dataclass(frozen=True)
class Medication:
    name: str
    dose: str
    status: Literal["current", "discontinued"] = "current"


@dataclass(frozen=True)
class Allergy:
    substance: str
    reaction: str


@dataclass(frozen=True)
class Lab:
    name: str
    unit: str
    ref_low: float
    ref_high: float
    value: float | None            # None = not measured in this note


@dataclass(frozen=True)
class Patient:
    pid: str
    age: int
    sex: str
    problems: tuple[Problem, ...] = ()
    medications: tuple[Medication, ...] = ()
    allergies: tuple[Allergy, ...] = ()
    labs: tuple[Lab, ...] = ()
    extra: dict[str, object] = field(default_factory=dict)


# ---- rendering (template tables; no generated language) ----------------------------------------

AFFIRM = ["Active problems include {c}.", "{C} is documented.", "The patient has {c}.", "Known {c}."]
NEGATE = ["Never diagnosed with {c}.", "The patient denies {c}.", "No history of {c}.",
          "{C} was ruled out."]
FAMILY = ["Family history: mother with {c}.", "Mother has {c}.", "Family history of {c} in a sister."]
RESOLVED = ["History of {c}, now resolved.", "Past {c}, resolved.", "{C} in the past, no longer active."]
MED_LINE = ["{m} {d}.", "Takes {m} {d}.", "Maintained on {m} {d}."]
ALLERGY_LINE = ["Reports a {r} to {s}.", "Allergic to {s} ({r}).", "{S}: {r}."]
LAB_LINE = ["{n} {v} {u} (ref {lo}-{hi} {u}).", "{N}: {v} {u} [ref {lo}-{hi}].", "{n} measured at {v} {u}; reference {lo}-{hi} {u}."]
FILLER = ["Otherwise stable.", "Seen in clinic for review.", "No acute complaints today.",
          "Discussed the plan with the patient."]
SECTION_ORDERS = [("hpi", "pmh", "meds", "allergy", "results"), ("hpi", "meds", "allergy", "pmh", "results"),
                  ("hpi", "results", "pmh", "meds", "allergy")]


def _cap(s: str) -> str:
    return s[:1].upper() + s[1:]


def render_note(p: Patient, rng: random.Random) -> str:
    """A clinical-style note from the parts. Deterministic given ``rng``."""
    order = SECTION_ORDERS[rng.randrange(len(SECTION_ORDERS))]
    sections: dict[str, list[str]] = {k: [] for k in ("hpi", "pmh", "meds", "allergy", "results")}
    sections["hpi"].append(f"{p.age}-year-old {p.sex} seen in clinic. {rng.choice(FILLER)}")
    for pr in p.problems:
        if pr.subject == "family":
            sections["pmh"].append(rng.choice(FAMILY).format(c=pr.condition))
        elif not pr.affirmed:
            sections["pmh"].append(rng.choice(NEGATE).format(c=pr.condition, C=_cap(pr.condition)))
        elif pr.status == "resolved":
            sections["pmh"].append(rng.choice(RESOLVED).format(c=pr.condition, C=_cap(pr.condition)))
        else:
            sections["pmh"].append(rng.choice(AFFIRM).format(c=pr.condition, C=_cap(pr.condition)))
    for m in p.medications:
        if m.status == "current":
            sections["meds"].append(rng.choice(MED_LINE).format(m=m.name, d=m.dose))
    for a in p.allergies:
        sections["allergy"].append(rng.choice(ALLERGY_LINE).format(s=a.substance, r=a.reaction, S=_cap(a.substance)))
    for lab in p.labs:
        if lab.value is None:
            continue
        sections["results"].append(rng.choice(LAB_LINE).format(
            n=lab.name, N=_cap(lab.name), v=f"{lab.value:g}", u=lab.unit,
            lo=f"{lab.ref_low:g}", hi=f"{lab.ref_high:g}"))
    labels = {"hpi": "HPI", "pmh": "Past medical history", "meds": "Medications",
              "allergy": "Allergies", "results": "Results"}
    out = []
    for key in order:
        body = " ".join(sections[key]) if sections[key] else ("none" if key != "hpi" else "")
        if key == "allergy" and not sections[key]:
            body = "none known"
        if key == "meds" and not sections[key]:
            body = "none"
        if key == "results" and not sections[key]:
            body = "none available"
        if key == "pmh" and not sections[key]:
            body = "none"
        if body:
            out.append(f"{labels[key]}: {body}")
    return "\n".join(out)


# ---- ids, splits, items -------------------------------------------------------------------------

def split_of(pid: str) -> Split:
    """Patient-level split by stable hash: 80% train, 10% dev, 10% test. Twins share a patient, so they share a split."""
    bucket = int(stable_hash(pid, length=8), 16) % 100
    if bucket < 80:
        return "train"
    return "dev" if bucket < 90 else "test"


def rng_for(*parts: str) -> random.Random:
    return random.Random(int(stable_hash({"parts": list(parts)}, length=12), 16))


@dataclass(frozen=True)
class GeneratedItem:
    """One decision in the benchmark item shape (EvalItem-compatible fields)."""

    item_id: str
    generator: str
    family: str
    pair_id: str
    role: Literal["base", "twin"]
    split: str
    qtype: str
    state: str
    question: str
    options: tuple[tuple[str, str], ...]   # (key, label)
    gold: str                              # option key
    meta: dict[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "item_id": self.item_id, "generator": self.generator, "family": self.family,
            "pair_id": self.pair_id, "role": self.role, "split": self.split, "qtype": self.qtype,
            "state": self.state, "question": self.question,
            "options": [{"key": k, "label": lab} for k, lab in self.options], "gold": self.gold,
            "meta": self.meta,
        }


def make_item(generator: str, family: str, pid: str, role: str, qtype: str, state: str, question: str,
              options: list[tuple[str, str]], gold: str, meta: dict[str, object] | None = None) -> GeneratedItem:
    keys = [k for k, _ in options]
    if gold not in keys:
        raise ValueError(f"{generator}/{pid}: gold {gold!r} is not an option")
    if len(set(keys)) != len(keys):
        raise ValueError(f"{generator}/{pid}: duplicate keys")
    if not normalize_text(state):
        raise ValueError(f"{generator}/{pid}: blank state")
    item_id = stable_hash({"generator": generator, "pid": pid, "role": role}, length=16)
    return GeneratedItem(item_id=item_id, generator=generator, family=family, pair_id=f"{generator}:{pid}",
                         role=role, split=split_of(pid), qtype=qtype, state=state, question=question,
                         options=tuple(options), gold=gold, meta=dict(meta or {}))
