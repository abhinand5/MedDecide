"""ICD-10-CM code support: which code (if any) the documentation of one diagnosis supports, computed by code.

A small curated set of code families whose members differ by one documented attribute (acuity, gas exchange, stage,
organism, type, severity, laterality). Within a family the codes are mutually exclusive, so exactly one is correct for
a documented diagnosis, or none is. The rule follows the ICD-10-CM Official Guidelines for Coding and Reporting:

- inpatient: an uncertain diagnosis ("probable", "suspected", "likely") documented at discharge is coded as if it existed
  (Section II.H; Section III.C);
- outpatient: an uncertain diagnosis is not coded (Section IV.H);
- a ruled-out diagnosis, a resolved condition (history of) and a family member's condition are not coded as the
  condition itself;
- the most specific code the documentation supports is the correct one (an "unspecified" code is wrong when the
  attribute is documented).

The code titles are paraphrased from the public CMS ICD-10-CM tabular list. ``load_order_file`` reads the CMS order file
so a loop can verify that every curated code exists and is billable in the year it uses (``verify_codes``).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

Certainty = Literal["confirmed", "probable", "suspected", "ruled_out"]
Setting = Literal["inpatient", "outpatient"]


@dataclass(frozen=True)
class CodeFamily:
    key: str
    condition: str                                  # how the condition is named in questions
    attributes: dict[str, tuple[str, ...]]           # attribute -> allowed values ("unspecified" = not documented)
    codes: dict[str, str]                           # code -> title
    resolve: Callable[[Mapping[str, str]], str]     # documented attributes -> the one correct code


def _rf(a: Mapping[str, str]) -> str:
    acuity = {"acute": "0", "chronic": "1", "acute_on_chronic": "2"}[a["acuity"]]
    gas = {"unspecified": "0", "hypoxia": "1", "hypercapnia": "2"}[a["gas"]]
    return f"J96.{acuity}{gas}"


def _hf(a: Mapping[str, str]) -> str:
    t, acuity = a["type"], a["acuity"]
    if t == "unspecified":
        return "I50.9"
    base = {"systolic": "I50.2", "diastolic": "I50.3"}[t]
    return base + {"unspecified": "0", "acute": "1", "chronic": "2", "acute_on_chronic": "3"}[acuity]


FAMILIES: tuple[CodeFamily, ...] = (
    CodeFamily(
        "respiratory_failure", "respiratory failure",
        {"acuity": ("acute", "chronic", "acute_on_chronic"), "gas": ("unspecified", "hypoxia", "hypercapnia")},
        {"J96.00": "Acute respiratory failure, unspecified whether with hypoxia or hypercapnia",
         "J96.01": "Acute respiratory failure with hypoxia", "J96.02": "Acute respiratory failure with hypercapnia",
         "J96.10": "Chronic respiratory failure, unspecified whether with hypoxia or hypercapnia",
         "J96.11": "Chronic respiratory failure with hypoxia", "J96.12": "Chronic respiratory failure with hypercapnia",
         "J96.20": "Acute and chronic respiratory failure, unspecified whether with hypoxia or hypercapnia",
         "J96.21": "Acute and chronic respiratory failure with hypoxia",
         "J96.22": "Acute and chronic respiratory failure with hypercapnia"},
        _rf),
    CodeFamily(
        "ckd", "chronic kidney disease",
        {"stage": ("1", "2", "3", "3a", "3b", "4", "5", "esrd")},
        {"N18.1": "Chronic kidney disease, stage 1", "N18.2": "Chronic kidney disease, stage 2 (mild)",
         "N18.30": "Chronic kidney disease, stage 3 unspecified", "N18.31": "Chronic kidney disease, stage 3a",
         "N18.32": "Chronic kidney disease, stage 3b", "N18.4": "Chronic kidney disease, stage 4 (severe)",
         "N18.5": "Chronic kidney disease, stage 5", "N18.6": "End stage renal disease"},
        lambda a: {"1": "N18.1", "2": "N18.2", "3": "N18.30", "3a": "N18.31", "3b": "N18.32", "4": "N18.4",
                   "5": "N18.5", "esrd": "N18.6"}[a["stage"]]),
    CodeFamily(
        "type2_diabetes", "type 2 diabetes mellitus",
        {"complication": ("none", "hyperglycemia", "diabetic_ckd", "hypoglycemia")},
        {"E11.9": "Type 2 diabetes mellitus without complications",
         "E11.65": "Type 2 diabetes mellitus with hyperglycemia",
         "E11.22": "Type 2 diabetes mellitus with diabetic chronic kidney disease",
         "E11.649": "Type 2 diabetes mellitus with hypoglycemia without coma"},
        lambda a: {"none": "E11.9", "hyperglycemia": "E11.65", "diabetic_ckd": "E11.22",
                   "hypoglycemia": "E11.649"}[a["complication"]]),
    CodeFamily(
        "sepsis_organism", "sepsis",
        {"organism": ("unspecified", "e_coli", "mssa", "mrsa")},
        {"A41.9": "Sepsis, unspecified organism", "A41.51": "Sepsis due to Escherichia coli [E. coli]",
         "A41.01": "Sepsis due to methicillin susceptible Staphylococcus aureus",
         "A41.02": "Sepsis due to methicillin resistant Staphylococcus aureus"},
        lambda a: {"unspecified": "A41.9", "e_coli": "A41.51", "mssa": "A41.01", "mrsa": "A41.02"}[a["organism"]]),
    CodeFamily(
        "severe_sepsis", "severe sepsis",
        {"shock": ("no", "yes")},
        {"R65.20": "Severe sepsis without septic shock", "R65.21": "Severe sepsis with septic shock"},
        lambda a: {"no": "R65.20", "yes": "R65.21"}[a["shock"]]),
    CodeFamily(
        "heart_failure", "heart failure",
        {"type": ("unspecified", "systolic", "diastolic"),
         "acuity": ("unspecified", "acute", "chronic", "acute_on_chronic")},
        {"I50.9": "Heart failure, unspecified", "I50.20": "Unspecified systolic (congestive) heart failure",
         "I50.21": "Acute systolic (congestive) heart failure", "I50.22": "Chronic systolic (congestive) heart failure",
         "I50.23": "Acute on chronic systolic (congestive) heart failure",
         "I50.30": "Unspecified diastolic (congestive) heart failure",
         "I50.31": "Acute diastolic (congestive) heart failure", "I50.32": "Chronic diastolic (congestive) heart failure",
         "I50.33": "Acute on chronic diastolic (congestive) heart failure"},
        _hf),
    CodeFamily(
        "malnutrition", "protein-calorie malnutrition",
        {"severity": ("mild", "moderate", "severe")},
        {"E44.1": "Mild protein-calorie malnutrition", "E44.0": "Moderate protein-calorie malnutrition",
         "E43": "Unspecified severe protein-calorie malnutrition"},
        lambda a: {"mild": "E44.1", "moderate": "E44.0", "severe": "E43"}[a["severity"]]),
    CodeFamily(
        "pneumonia_organism", "pneumonia",
        {"organism": ("unspecified", "pseudomonas")},
        {"J18.9": "Pneumonia, unspecified organism", "J15.1": "Pneumonia due to Pseudomonas"},
        lambda a: {"unspecified": "J18.9", "pseudomonas": "J15.1"}[a["organism"]]),
    CodeFamily(
        "knee_oa", "primary osteoarthritis of the knee",
        {"laterality": ("unspecified", "right", "left", "bilateral")},
        {"M17.10": "Unilateral primary osteoarthritis, unspecified knee",
         "M17.11": "Unilateral primary osteoarthritis, right knee",
         "M17.12": "Unilateral primary osteoarthritis, left knee", "M17.0": "Bilateral primary osteoarthritis of knee"},
        lambda a: {"unspecified": "M17.10", "right": "M17.11", "left": "M17.12", "bilateral": "M17.0"}[a["laterality"]]),
    CodeFamily("hyponatremia", "hyponatraemia", {}, {"E87.1": "Hypo-osmolality and hyponatremia"},
               lambda a: "E87.1"),
    CodeFamily("metabolic_encephalopathy", "metabolic encephalopathy", {}, {"G93.41": "Metabolic encephalopathy"},
               lambda a: "G93.41"),
    CodeFamily("aki", "acute kidney failure", {}, {"N17.9": "Acute kidney failure, unspecified"}, lambda a: "N17.9"),
)
FAMILY_BY_KEY = {f.key: f for f in FAMILIES}


@dataclass(frozen=True)
class Diagnosis:
    """One diagnosis as the documentation states it."""

    family: str
    attributes: dict[str, str] = field(default_factory=dict)
    certainty: Certainty = "confirmed"
    subject: Literal["patient", "family"] = "patient"
    status: Literal["current", "history"] = "current"

    def __post_init__(self) -> None:
        fam = FAMILY_BY_KEY[self.family]
        if set(self.attributes) != set(fam.attributes):
            raise ValueError(f"{self.family}: attributes {sorted(self.attributes)} != {sorted(fam.attributes)}")
        for k, v in self.attributes.items():
            if v not in fam.attributes[k]:
                raise ValueError(f"{self.family}: {k}={v!r} not in {fam.attributes[k]}")


def codeable(dx: Diagnosis, setting: Setting) -> bool:
    """Whether the documentation supports coding the condition at all (before choosing the code)."""
    if dx.subject != "patient" or dx.status != "current" or dx.certainty == "ruled_out":
        return False
    if dx.certainty in ("probable", "suspected"):
        return setting == "inpatient"
    return True


def supported_code(dx: Diagnosis, setting: Setting) -> str | None:
    """The one code in the diagnosis's family that the documentation supports, or None."""
    if not codeable(dx, setting):
        return None
    fam = FAMILY_BY_KEY[dx.family]
    code = fam.resolve(dx.attributes)
    if code not in fam.codes:
        raise AssertionError(f"{fam.key}: resolver returned {code!r}, not a family code")
    return code


# ---- the CMS order file (verification) ----------------------------------------------------------

@dataclass(frozen=True)
class OrderRow:
    code: str            # with the dot inserted after the third character
    billable: bool
    short: str
    long: str


def _dotted(raw: str) -> str:
    raw = raw.strip()
    return raw if len(raw) <= 3 else f"{raw[:3]}.{raw[3:]}"


def load_order_file(path: str | Path) -> dict[str, OrderRow]:
    """Parse the CMS ``icd10cm_order_<year>.txt`` fixed-width file: order (1-5), code (7-13), billable flag (15),
    short title (17-76), long title (78-)."""
    rows: dict[str, OrderRow] = {}
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        if len(line) < 17 or not line[:5].strip().isdigit():
            continue
        code = _dotted(line[6:13])
        rows[code] = OrderRow(code=code, billable=line[14] == "1", short=line[16:76].strip(), long=line[77:].strip())
    return rows


def verify_codes(order: Mapping[str, OrderRow]) -> dict[str, list[str]]:
    """Every curated code must exist and be billable in the given order file. Returns {"missing": [...],
    "not_billable": [...]} (both empty when the curated table is valid for that year)."""
    missing, not_billable = [], []
    for fam in FAMILIES:
        for code in fam.codes:
            row = order.get(code)
            if row is None:
                missing.append(code)
            elif not row.billable:
                not_billable.append(code)
    return {"missing": missing, "not_billable": not_billable}
