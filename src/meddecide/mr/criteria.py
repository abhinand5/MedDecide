"""Clinical-indicator evidence: does a record chunk support one indicator of a condition, computed from structured facts.

An indicator is a single published threshold that clinical-validation reviewers look for when they check a coded
condition (sepsis, acute respiratory failure, acute kidney injury, hyponatraemia). The rule here decides whether the
facts in one chunk count as supporting evidence **for the current encounter**:

- the observation belongs to the current encounter (a prior admission's value does not count);
- it is not documented as the patient's chronic baseline (a COPD patient's usual SpO2 of 88 % is not acute evidence);
- the value meets the threshold after unit conversion (values may be rendered in alternative units);
- an oxygen-saturation threshold applies only on room air;
- a medication counts only when the MAR shows it given (held, refused or ordered-not-given doesn't count).

Thresholds and their sources (each is a single published cut-off; the reviewer's full judgement needs more than one):
  Sepsis-3 (Singer et al., JAMA 2016): lactate > 2 mmol/L and vasopressors for septic shock; MAP < 65 mmHg.
  SIRS (Bone et al., Chest 1992): temperature > 38 or < 36 C; heart rate > 90; respiratory rate > 20 or PaCO2 < 32 mmHg;
      WBC > 12 or < 4 x10^9/L.
  SOFA, score >= 1 (Vincent et al., 1996): platelets < 150 x10^3/uL; bilirubin >= 1.2 mg/dL.
  Acute respiratory failure (widely used clinical-validation criteria): PaO2 < 60 mmHg; SpO2 < 91 % on room air;
      PaCO2 > 50 mmHg with pH < 7.35; P/F ratio < 300.
  KDIGO AKI (2012): creatinine rise >= 0.3 mg/dL within 48 h, or >= 1.5x baseline within 7 days.
  Hyponatraemia: sodium < 135 mmol/L.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

Encounter = Literal["current", "prior"]
MarStatus = Literal["given", "held", "refused", "not_given"]

# canonical units for each measured quantity
CANONICAL_UNITS: dict[str, str] = {
    "lactate": "mmol/L", "map": "mmHg", "temperature": "C", "heart_rate": "/min", "respiratory_rate": "/min",
    "wbc": "x10^9/L", "platelets": "x10^3/uL", "bilirubin": "mg/dL", "pao2": "mmHg", "spo2": "%", "paco2": "mmHg",
    "ph": "", "fio2": "%", "creatinine": "mg/dL", "sodium": "mmol/L",
}

# alternative display units: (unit, factor) where canonical = displayed * factor (temperature handled separately)
ALT_UNITS: dict[str, tuple[str, float]] = {
    "lactate": ("mg/dL", 1 / 9.008),      # lactate 1 mmol/L = 9.008 mg/dL
    "creatinine": ("umol/L", 1 / 88.42),  # creatinine 1 mg/dL = 88.42 umol/L
    "bilirubin": ("umol/L", 1 / 17.1),    # bilirubin 1 mg/dL = 17.1 umol/L
    "pao2": ("kPa", 7.50062),             # 1 kPa = 7.50062 mmHg
    "paco2": ("kPa", 7.50062),
}

VASOPRESSORS = ("norepinephrine", "epinephrine", "vasopressin", "phenylephrine", "dopamine")


@dataclass(frozen=True)
class Observation:
    """One measured value. ``value`` is in the canonical unit; ``display_unit`` is how a renderer should show it."""

    name: str
    value: float
    time_h: float = 0.0                   # hours from admission of the encounter it belongs to
    encounter: Encounter = "current"
    baseline: bool = False                # documented as the patient's chronic baseline
    on_room_air: bool = True              # oxygen saturation context
    display_unit: str | None = None       # None = canonical

    def __post_init__(self) -> None:
        if self.name not in CANONICAL_UNITS:
            raise ValueError(f"unknown observation {self.name!r}")
        if self.display_unit not in (None, CANONICAL_UNITS[self.name]):
            allowed = ALT_UNITS.get(self.name, ("",))[0]
            if not (self.name == "temperature" and self.display_unit == "F") and self.display_unit != allowed:
                raise ValueError(f"{self.name}: unsupported display unit {self.display_unit!r}")

    def displayed(self) -> tuple[float, str]:
        """(value, unit) as rendered."""
        unit = self.display_unit or CANONICAL_UNITS[self.name]
        if unit == CANONICAL_UNITS[self.name]:
            return self.value, unit
        if self.name == "temperature":
            return self.value * 9 / 5 + 32, "F"
        return self.value / ALT_UNITS[self.name][1], unit


@dataclass(frozen=True)
class MedAdmin:
    """One MAR line."""

    drug: str
    dose: str
    time_h: float = 0.0
    status: MarStatus = "given"
    encounter: Encounter = "current"


@dataclass(frozen=True)
class ChunkFacts:
    """Everything a chunk shows. Generators build these; renderers draw them; the rule below reads them."""

    observations: tuple[Observation, ...] = ()
    meds: tuple[MedAdmin, ...] = ()
    meta: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class Indicator:
    key: str
    condition: str
    description: str            # used in the question text


INDICATORS: tuple[Indicator, ...] = (
    Indicator("lactate_gt_2", "sepsis", "a lactate above 2 mmol/L"),
    Indicator("map_lt_65", "sepsis", "a mean arterial pressure below 65 mmHg"),
    Indicator("vasopressor_given", "sepsis", "a vasopressor administered"),
    Indicator("sirs_temperature", "sepsis", "a temperature above 38 C or below 36 C"),
    Indicator("sirs_heart_rate", "sepsis", "a heart rate above 90/min"),
    Indicator("sirs_respiratory", "sepsis", "a respiratory rate above 20/min or a PaCO2 below 32 mmHg"),
    Indicator("sirs_wbc", "sepsis", "a white cell count above 12 or below 4 x10^9/L"),
    Indicator("platelets_lt_150", "sepsis", "a platelet count below 150 x10^3/uL"),
    Indicator("bilirubin_ge_1_2", "sepsis", "a total bilirubin of 1.2 mg/dL or more"),
    Indicator("pao2_lt_60", "acute respiratory failure", "a PaO2 below 60 mmHg"),
    Indicator("spo2_lt_91_room_air", "acute respiratory failure", "an SpO2 below 91% on room air"),
    Indicator("hypercapnic_acidosis", "acute respiratory failure", "a PaCO2 above 50 mmHg with a pH below 7.35"),
    Indicator("pf_ratio_lt_300", "acute respiratory failure", "a PaO2/FiO2 ratio below 300"),
    Indicator("aki_creatinine_rise", "acute kidney injury",
              "a creatinine rise of at least 0.3 mg/dL within 48 hours or to 1.5 times baseline within 7 days"),
    Indicator("sodium_lt_135", "hyponatraemia", "a sodium below 135 mmol/L"),
)
INDICATOR_BY_KEY = {i.key: i for i in INDICATORS}


def _acute(facts: ChunkFacts, name: str) -> list[Observation]:
    """Current-encounter, non-baseline observations of ``name``, in time order."""
    obs = [o for o in facts.observations if o.name == name and o.encounter == "current" and not o.baseline]
    return sorted(obs, key=lambda o: o.time_h)


def _any(facts: ChunkFacts, name: str, pred) -> list[Observation]:
    return [o for o in _acute(facts, name) if pred(o.value)]


def _paired(facts: ChunkFacts, a: str, b: str, window_h: float = 1.0) -> list[tuple[Observation, Observation]]:
    """Current, non-baseline pairs of ``a`` and ``b`` measured within ``window_h`` of each other (one blood gas)."""
    return [(x, y) for x in _acute(facts, a) for y in _acute(facts, b) if abs(x.time_h - y.time_h) <= window_h]


def evidence(indicator: str, facts: ChunkFacts) -> list[object]:
    """The facts in this chunk that support ``indicator`` for the current encounter (empty = not supported)."""
    if indicator not in INDICATOR_BY_KEY:
        raise ValueError(f"unknown indicator {indicator!r}")
    if indicator == "lactate_gt_2":
        return _any(facts, "lactate", lambda v: v > 2.0)
    if indicator == "map_lt_65":
        return _any(facts, "map", lambda v: v < 65)
    if indicator == "vasopressor_given":
        return [m for m in facts.meds if m.encounter == "current" and m.status == "given"
                and any(m.drug.lower().startswith(v) for v in VASOPRESSORS)]
    if indicator == "sirs_temperature":
        return _any(facts, "temperature", lambda v: v > 38.0 or v < 36.0)
    if indicator == "sirs_heart_rate":
        return _any(facts, "heart_rate", lambda v: v > 90)
    if indicator == "sirs_respiratory":
        return _any(facts, "respiratory_rate", lambda v: v > 20) + _any(facts, "paco2", lambda v: v < 32)
    if indicator == "sirs_wbc":
        return _any(facts, "wbc", lambda v: v > 12 or v < 4)
    if indicator == "platelets_lt_150":
        return _any(facts, "platelets", lambda v: v < 150)
    if indicator == "bilirubin_ge_1_2":
        return _any(facts, "bilirubin", lambda v: v >= 1.2)
    if indicator == "pao2_lt_60":
        return _any(facts, "pao2", lambda v: v < 60)
    if indicator == "spo2_lt_91_room_air":
        return [o for o in _acute(facts, "spo2") if o.value < 91 and o.on_room_air]
    if indicator == "hypercapnic_acidosis":
        return [pair for pair in _paired(facts, "paco2", "ph") if pair[0].value > 50 and pair[1].value < 7.35]
    if indicator == "pf_ratio_lt_300":
        return [pair for pair in _paired(facts, "pao2", "fio2") if pair[1].value > 0
                and pair[0].value / (pair[1].value / 100.0) < 300]
    if indicator == "aki_creatinine_rise":
        return _creatinine_rise(facts)
    if indicator == "sodium_lt_135":
        return _any(facts, "sodium", lambda v: v < 135)
    raise AssertionError(indicator)  # pragma: no cover


def _creatinine_rise(facts: ChunkFacts) -> list[tuple[Observation, Observation]]:
    """KDIGO: a rise of >= 0.3 mg/dL within 48 h, or to >= 1.5x a reference value within 7 days. A documented baseline
    (``baseline=True``, any encounter) or an earlier current-encounter value can be the reference."""
    current = _acute(facts, "creatinine")
    refs = [o for o in facts.observations if o.name == "creatinine" and o.baseline]
    out = []
    for later in current:
        for ref in refs:
            if later.value >= 1.5 * ref.value:
                out.append((ref, later))
        for earlier in current:
            dt = later.time_h - earlier.time_h
            if dt <= 0:
                continue
            if (dt <= 48 and later.value - earlier.value >= 0.3) or (dt <= 168 and later.value >= 1.5 * earlier.value):
                out.append((earlier, later))
    return out


def supports(indicator: str, facts: ChunkFacts) -> bool:
    return bool(evidence(indicator, facts))


def supported_indicators(facts: ChunkFacts, keys: Sequence[str] | None = None) -> list[str]:
    return [k for k in (keys or [i.key for i in INDICATORS]) if supports(k, facts)]
