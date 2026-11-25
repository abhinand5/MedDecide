"""Three synthetic families defined in code: lab-value interpretation, medication indication, triage.

Every value here is invented by this module. No family reads a record, a file or a model. A case's gold label is
computed by a rule in this module from the same values that appear in the facts, so the label never comes from
generated text. Facts avoid the label words (checked by ``render_request`` and by ``verify_text``).
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass

from meddecide.synth.facts import CaseFacts, Fact, TemplateSpec
from meddecide.synth.render import StyleSpec

LAB_FAMILY = "synth_lab_interp_v1"
MED_FAMILY = "synth_med_indication_v1"
TRIAGE_FAMILY = "synth_triage_v1"
FAMILIES: tuple[str, ...] = (LAB_FAMILY, MED_FAMILY, TRIAGE_FAMILY)

LAB_LABELS = ("normal", "low", "high", "critical")
MED_LABELS = ("rate control", "pain control", "infection treatment", "blood pressure control")
TRIAGE_LABELS = ("emergent", "urgent", "routine", "self-care")

STYLES: tuple[StyleSpec, ...] = (
    StyleSpec("ward", "ward progress note", ("Terse clinical shorthand is acceptable.",)),
    StyleSpec("ed", "emergency department note", ()),
    StyleSpec("clinic", "outpatient clinic letter", ("Write in full sentences.",)),
)

# ---- lab-value interpretation -------------------------------------------------------------------------------

LAB_SYMPTOMS: tuple[str, ...] = (
    "fatigue for three days",
    "ankle swelling since last week",
    "poor oral intake this week",
    "muscle cramps in both legs",
    "no symptoms reported",
)
LAB_HOME_MEDS: tuple[str, ...] = (
    "lisinopril 10 mg daily",
    "spironolactone 25 mg daily",
    "furosemide 40 mg daily",
    "losartan 50 mg daily",
)


def potassium_label(value: float) -> str:
    """Gold label for a potassium value in mmol/L (applied to the one-decimal value written in the facts)."""
    if value >= 6.5:
        return "critical"
    if value >= 5.1:
        return "high"
    if value < 3.5:
        return "low"
    return "normal"


_LAB_RANGES: dict[str, tuple[float, float]] = {
    "critical": (6.5, 7.4),
    "high": (5.1, 6.4),
    "low": (2.8, 3.4),
    "normal": (3.5, 5.0),
}


def _lab_case(rng: random.Random, target: str) -> CaseFacts:
    low, high = _LAB_RANGES[target]
    value = round(rng.uniform(low, high), 1)
    facts = (
        Fact("demographic", "age", f"{rng.randrange(18, 91)} years"),
        Fact("demographic", "sex", rng.choice(("male", "female"))),
        Fact("lab", "potassium", f"K {value:.1f} mmol/L"),
        Fact("finding", "symptom", rng.choice(LAB_SYMPTOMS)),
        Fact("medication", "home medicine", rng.choice(LAB_HOME_MEDS)),
    )
    label = potassium_label(value)
    if label != target:  # the rule is the gold; resampling keeps the target mix balanced
        raise _Resample
    return CaseFacts(family=LAB_FAMILY, facts=facts, label=label)


# ---- medication indication ----------------------------------------------------------------------------------

# (prescription text, gold indication from the drug class, presentation texts that fit the class)
MED_TABLE: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    (
        "metoprolol tartrate 25 mg by mouth twice daily",
        "rate control",
        ("resting heart rate 112 bpm with an irregularly irregular pulse", "resting heart rate 98 bpm and palpitations"),
    ),
    (
        "ibuprofen 400 mg by mouth every six hours",
        "pain control",
        ("knee pain rated 7 out of 10", "back pain rated 6 out of 10 after lifting"),
    ),
    (
        "amoxicillin 500 mg by mouth three times daily",
        "infection treatment",
        ("temperature 38.4 C with a productive cough", "urinary frequency and temperature 38.1 C"),
    ),
    (
        "amlodipine 5 mg by mouth once daily",
        "blood pressure control",
        ("blood pressure 168/96 mmHg on repeat check", "seated blood pressure 152/94 mmHg at follow-up"),
    ),
)


def _med_case(rng: random.Random, target: str) -> CaseFacts:
    rows = [row for row in MED_TABLE if row[1] == target]
    prescription, label, presentations = rng.choice(rows)
    facts = (
        Fact("demographic", "age", f"{rng.randrange(18, 91)} years"),
        Fact("medication", "prescription", prescription),
        Fact("finding", "presentation", rng.choice(presentations)),
    )
    return CaseFacts(family=MED_FAMILY, facts=facts, label=label)


# ---- triage -------------------------------------------------------------------------------------------------

ACUTE_COMPLAINTS: tuple[str, ...] = ("chest pain for two hours", "shortness of breath on waking", "confusion noticed by family")
ROUTINE_COMPLAINTS: tuple[str, ...] = ("ankle sprain after a fall two days ago", "right knee pain for three weeks")
SELF_COMPLAINTS: tuple[str, ...] = ("mild sore throat for two days", "runny nose and sneezing since yesterday")


def triage_label(spo2: int, heart_rate: int, complaint_group: str) -> str:
    """Gold triage label from SpO2 (percent), heart rate (bpm) and the complaint group of the case."""
    if spo2 < 90:
        return "emergent"
    if spo2 < 94 or heart_rate >= 120:
        return "urgent"
    if complaint_group == "self":
        return "self-care"
    return "routine"


_TRIAGE_VITALS: dict[str, tuple[tuple[int, int], tuple[int, int]]] = {
    # tier: ((spo2 low, high), (heart rate low, high)); the rule above decides the final label
    "emergent": ((82, 89), (100, 130)),
    "urgent": ((90, 93), (80, 115)),
    "routine": ((95, 99), (70, 100)),
    "self-care": ((96, 100), (60, 90)),
}


def _triage_case(rng: random.Random, target: str) -> CaseFacts:
    (spo2_lo, spo2_hi), (hr_lo, hr_hi) = _TRIAGE_VITALS[target]
    if target in ("emergent", "urgent"):
        group = "acute"
        complaint = rng.choice(ACUTE_COMPLAINTS)
    elif target == "routine":
        group = rng.choice(("acute", "routine"))
        complaint = rng.choice(ROUTINE_COMPLAINTS if group == "routine" else ACUTE_COMPLAINTS)
    else:
        group = "self"
        complaint = rng.choice(SELF_COMPLAINTS)
    spo2 = rng.randrange(spo2_lo, spo2_hi + 1)
    heart_rate = rng.randrange(hr_lo, hr_hi + 1)
    label = triage_label(spo2, heart_rate, group)
    if label != target:
        raise _Resample
    facts = (
        Fact("demographic", "age", f"{rng.randrange(18, 91)} years"),
        Fact("finding", "chief complaint", complaint),
        Fact("finding", "spo2", f"SpO2 {spo2}% on room air"),
        Fact("finding", "heart rate", f"heart rate {heart_rate} bpm"),
    )
    return CaseFacts(family=TRIAGE_FAMILY, facts=facts, label=label)


# ---- specs and plan -----------------------------------------------------------------------------------------


def _options(labels: tuple[str, ...]) -> Callable[[tuple[Fact, ...]], tuple[str, ...]]:
    def build(_facts: tuple[Fact, ...]) -> tuple[str, ...]:
        return labels

    return build


SPECS: dict[str, TemplateSpec] = {
    LAB_FAMILY: TemplateSpec(LAB_FAMILY, LAB_LABELS, _options(LAB_LABELS), label_match="token"),
    MED_FAMILY: TemplateSpec(MED_FAMILY, MED_LABELS, _options(MED_LABELS), label_match="token"),
    TRIAGE_FAMILY: TemplateSpec(TRIAGE_FAMILY, TRIAGE_LABELS, _options(TRIAGE_LABELS), label_match="token"),
}

_BUILDERS: dict[str, Callable[[random.Random, str], CaseFacts]] = {
    LAB_FAMILY: _lab_case,
    MED_FAMILY: _med_case,
    TRIAGE_FAMILY: _triage_case,
}
_TARGETS: dict[str, tuple[str, ...]] = {
    LAB_FAMILY: LAB_LABELS,
    MED_FAMILY: MED_LABELS,
    TRIAGE_FAMILY: TRIAGE_LABELS,
}
MAX_ATTEMPTS = 1000


class _Resample(Exception):
    """Internal: a sampled value fell in another label's range, so the builder draws again."""


@dataclass(frozen=True)
class PlannedCase:
    """One case of the run: facts, gold label, template spec, style, target length and a global index."""

    index: int
    case: CaseFacts
    spec: TemplateSpec
    style: StyleSpec
    target_words: int

    @property
    def word_range(self) -> tuple[int, int]:
        return (self.target_words // 2, self.target_words * 2)


def build_plan(n_cases: int, *, seed: int) -> list[PlannedCase]:
    """Deterministic plan: ``n_cases`` split across the three families in turn, each family with its own RNG.

    Cases are drawn so the gold labels are balanced within a family, and duplicate fact lists are redrawn.
    """
    if n_cases < 1:
        raise ValueError("n_cases must be at least 1")
    per_family = [len(range(f, n_cases, len(FAMILIES))) for f in range(len(FAMILIES))]
    planned: dict[int, PlannedCase] = {}
    for f_index, family in enumerate(FAMILIES):
        rng = random.Random(f"{seed}:{family}")
        targets = _TARGETS[family]
        seen: set[str] = set()
        for slot in range(per_family[f_index]):
            index = slot * len(FAMILIES) + f_index
            case = _draw_case(rng, family, targets[slot % len(targets)], seen)
            style = STYLES[rng.randrange(len(STYLES))]
            target_words = rng.randrange(60, 161)
            planned[index] = PlannedCase(index, case, SPECS[family], style, target_words)
    return [planned[i] for i in sorted(planned)]


def _draw_case(rng: random.Random, family: str, target: str, seen: set[str]) -> CaseFacts:
    builder = _BUILDERS[family]
    for _ in range(MAX_ATTEMPTS):
        try:
            case = builder(rng, target)
        except _Resample:
            continue
        if case.case_id not in seen:
            seen.add(case.case_id)
            return case
    raise RuntimeError(f"could not draw a distinct case for {family} / {target}")
