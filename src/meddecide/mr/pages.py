"""Medical-record page categories and code renderers for the tabular page types.

The taxonomy is the operator's (2026-10-13): eleven page classes plus ``other``. Vitals pages are not a class of their
own; a page of vital signs is a flowsheet.

The renderers draw labs, flowsheet, MAR, bill and ED-triage pages from structured facts in several layouts. Real
pages of these types are machine-printed by an EHR, so a code renderer over real (MIMIC) or simulated values is a
faithful source of text; the layouts differ in column order, separators, date formats, abnormal flags and headers so
that no single layout is learnt as the class. Narrative classes (discharge summary, admission, progress, consult,
radiology, nutrition) come from real notes or teacher prose, not from here.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass

from meddecide.mr.criteria import CANONICAL_UNITS, ChunkFacts, MedAdmin, Observation

PAGE_CLASSES: tuple[tuple[str, str], ...] = (
    ("discharge_summary", "Discharge summary"),
    ("admission", "Admission note (history and physical, admission type and reason)"),
    ("progress_note", "Progress note"),
    ("consult_note", "Consultation note"),
    ("radiology", "Radiology report"),
    ("emergency_department", "Emergency department record"),
    ("laboratory", "Laboratory results"),
    ("nutrition", "Nutritional assessment"),
    ("flowsheet", "Flowsheet (vital signs, intake and output, nursing measurements)"),
    ("bill", "Bill or itemised charges"),
    ("mar", "Medication administration record (MAR)"),
    ("other", "Other"),
)
PAGE_KEYS = tuple(k for k, _ in PAGE_CLASSES)

LAB_LABELS: dict[str, tuple[str, ...]] = {
    "lactate": ("Lactate", "LACTIC ACID", "Lactic Acid, Plasma", "LACT"),
    "wbc": ("WBC", "White Blood Cells", "WBC Count", "Leukocytes"),
    "platelets": ("Platelets", "PLT", "Platelet Count"),
    "bilirubin": ("Bilirubin, Total", "T BILI", "Total Bilirubin", "TBIL"),
    "pao2": ("pO2", "PaO2", "PO2 (ART)", "Arterial pO2"),
    "paco2": ("pCO2", "PaCO2", "PCO2 (ART)", "Arterial pCO2"),
    "ph": ("pH", "PH (ART)", "Arterial pH"),
    "creatinine": ("Creatinine", "CREAT", "Creatinine, Serum", "CRE"),
    "sodium": ("Sodium", "NA", "Sodium, Serum", "Na+"),
    "fio2": ("FiO2", "FIO2 (ABG)", "Inspired O2 (ABG)"),
}
VITAL_LABELS: dict[str, tuple[str, ...]] = {
    "temperature": ("Temp", "Temperature", "T", "TEMP"),
    "heart_rate": ("HR", "Heart Rate", "Pulse", "P"),
    "respiratory_rate": ("RR", "Resp Rate", "Respirations", "R"),
    "map": ("MAP", "Mean Arterial Pressure", "ABP Mean", "NBP Mean"),
    "spo2": ("SpO2", "O2 Sat", "Pulse Ox", "SaO2 (pulse ox)"),
    "fio2": ("FiO2", "FiO2 (%)", "Inspired O2 Fraction"),
}
# reference ranges used only for the H/L flag column (the flag is a distractor: a value can be flagged high without
# meeting the indicator's threshold)
REF_RANGES: dict[str, tuple[float, float]] = {
    "lactate": (0.5, 1.6), "wbc": (4.0, 11.0), "platelets": (150, 400), "bilirubin": (0.1, 1.0), "pao2": (80, 100),
    "paco2": (35, 45), "ph": (7.35, 7.45), "creatinine": (0.6, 1.2), "sodium": (136, 145),
}
FACILITIES = ("Riverside General Hospital", "St. Brendan Medical Center", "Lakeview Regional Health",
              "Northgate Community Hospital", "Mercy Valley Hospital", "Harbor Point Medical Center")
O2_DEVICES = ("2 L nasal cannula", "4 L nasal cannula", "non-rebreather", "high-flow nasal cannula 40 L", "BiPAP")


@dataclass(frozen=True)
class PageContext:
    """Header facts shared by every page of one synthetic record."""

    facility: str
    mrn: str
    patient: str
    dob: str
    admit_date: str            # ISO date of admission
    encounter_id: str


def make_context(rng: random.Random) -> PageContext:
    first = rng.choice(("J.", "M.", "A.", "R.", "S.", "L.", "D.", "K."))
    last = rng.choice(("Doe", "Roe", "Smith", "Patel", "Nguyen", "Garcia", "Okafor", "Kowalski", "Haddad", "Lindqvist"))
    year = rng.randint(1935, 2000)
    return PageContext(facility=rng.choice(FACILITIES), mrn=f"{rng.randint(10**7, 10**8 - 1)}",
                       patient=f"{last}, {first}", dob=f"{year}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}",
                       admit_date=f"2025-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}",
                       encounter_id=f"{rng.randint(10**8, 10**9 - 1)}")


def _stamp(ctx: PageContext, time_h: float, style: int) -> str:
    """A date-time for ``time_h`` hours after admission (prior-encounter times are rendered before admission)."""
    from datetime import datetime, timedelta

    base = datetime.fromisoformat(ctx.admit_date) + timedelta(hours=8)
    t = base + timedelta(hours=time_h)
    return (t.strftime("%m/%d/%Y %H:%M"), t.strftime("%Y-%m-%d %H%M"), t.strftime("%d-%b-%y %H:%M").upper())[style % 3]


def _fmt(value: float, name: str, unit: str | None = None) -> str:
    if unit is not None and unit != CANONICAL_UNITS[name]:
        return f"{round(value):d}" if unit == "umol/L" else f"{value:.1f}"
    if name in ("ph",):
        return f"{value:.2f}"
    if name in ("platelets", "map", "heart_rate", "respiratory_rate", "spo2", "fio2", "sodium", "pao2", "paco2"):
        return f"{round(value):d}"
    return f"{value:.1f}"


def _flag(o: Observation) -> str:
    lo, hi = REF_RANGES.get(o.name, (float("-inf"), float("inf")))
    return "H" if o.value > hi else "L" if o.value < lo else ""


def _header(ctx: PageContext, title: str, rng: random.Random) -> list[str]:
    """A page header; about one page in four is a continuation page that carries no title (as in a long record)."""
    if rng.random() < 0.25:
        return [rng.choice(("(continued)", f"{ctx.patient}  MRN {ctx.mrn}  (cont.)", "", "--- continued ---"))]
    style = rng.randrange(3)
    if style == 0:
        return [ctx.facility.upper(), title.upper(), f"Patient: {ctx.patient}   MRN: {ctx.mrn}   DOB: {ctx.dob}",
                f"Encounter: {ctx.encounter_id}   Admit: {ctx.admit_date}", "-" * 72]
    if style == 1:
        return [f"{ctx.facility} | {title}", f"{ctx.patient} | MRN {ctx.mrn} | DOB {ctx.dob} | Enc {ctx.encounter_id}", ""]
    return [title, f"Name: {ctx.patient}", f"Med Rec #: {ctx.mrn}    Acct #: {ctx.encounter_id}",
            f"Facility: {ctx.facility}", "=" * 60]


def _obs_label(o: Observation, rng: random.Random) -> str:
    table = LAB_LABELS if o.name in LAB_LABELS else VITAL_LABELS
    return rng.choice(table[o.name])


BASELINE_NOTES = (" (patient's baseline)", " (chronic - baseline per PCP)", " [baseline]", " (usual for patient)")
PRIOR_NOTES = (" (prior admission)", " [previous encounter]", " (outside this admission)", " - prior visit",
               " (from earlier encounter)")


def _prior_note(o: Observation | MedAdmin) -> str:
    """Marks a value from an earlier encounter; the wording varies so that no single string carries the label."""
    if o.encounter != "prior":
        return ""
    return PRIOR_NOTES[int(abs(o.time_h) * 13) % len(PRIOR_NOTES)]


def render_lab_page(ctx: PageContext, facts: ChunkFacts, rng: random.Random) -> str:
    """Cumulative laboratory report: one row per result."""
    title = rng.choice(("Laboratory Results", "Cumulative Lab Report", "LAB RESULTS - ALL", "Chemistry / Hematology"))
    lines = _header(ctx, title, rng)
    layout = rng.randrange(3)
    stamp_style = rng.randrange(3)
    if layout == 0:
        lines.append(f"{'Collected':<18}{'Test':<22}{'Result':>10} {'Flag':<5}{'Units':<10}{'Ref Range'}")
    elif layout == 1:
        lines.append("Test | Result | Units | Reference | Collected")
    obs = sorted((o for o in facts.observations if o.name in LAB_LABELS), key=lambda o: o.time_h)
    for o in obs:
        value, unit = o.displayed()
        rr = REF_RANGES.get(o.name)
        ref = f"{rr[0]:g}-{rr[1]:g}" if rr and unit == CANONICAL_UNITS[o.name] else ""
        when = _stamp(ctx, o.time_h, stamp_style)
        label = _obs_label(o, rng)
        flag = _flag(o)
        note = _prior_note(o) + (BASELINE_NOTES[int(abs(o.time_h) * 7) % len(BASELINE_NOTES)] if o.baseline else "")
        if layout == 0:
            lines.append(f"{when:<18}{label:<22}{_fmt(value, o.name, unit):>10} {flag:<5}{unit:<10}{ref}{note}")
        elif layout == 1:
            lines.append(f"{label} | {_fmt(value, o.name, unit)}{(' ' + flag) if flag else ''} | {unit} | {ref} | {when}{note}")
        else:
            lines.append(f"{when}  {label}: {_fmt(value, o.name, unit)} {unit} {('[' + flag + ']') if flag else ''}"
                         f"{(' (ref ' + ref + ')') if ref else ''}{note}".rstrip())
    if not obs:
        lines.append("No results in this period.")
    return "\n".join(lines)


def render_flowsheet_page(ctx: PageContext, facts: ChunkFacts, rng: random.Random) -> str:
    """Vital-signs flowsheet: one row per charting time, with the oxygen device column."""
    title = rng.choice(("Flowsheet - Vital Signs", "Nursing Flowsheet", "VITALS / I&O FLOWSHEET", "ICU Flowsheet"))
    lines = _header(ctx, title, rng)
    stamp_style = rng.randrange(3)
    names = [n for n in ("temperature", "heart_rate", "respiratory_rate", "map", "spo2", "fio2")
             if any(o.name == n for o in facts.observations)]
    labels = {n: rng.choice(VITAL_LABELS[n]) for n in names}
    times = sorted({(o.time_h, o.encounter, o.baseline) for o in facts.observations if o.name in VITAL_LABELS})
    layout = rng.randrange(2)
    if layout == 0:
        lines.append(f"{'Time':<18}" + "".join(f"{labels[n]:>10}" for n in names) + "  O2 Device")
    for t, enc, base in times:
        row = {o.name: o for o in facts.observations
               if o.name in VITAL_LABELS and o.time_h == t and o.encounter == enc and o.baseline == base}
        device = "Room air"
        spo2 = row.get("spo2")
        if spo2 is not None and not spo2.on_room_air:
            device = O2_DEVICES[int(abs(t) * 7) % len(O2_DEVICES)]
        cells = {}
        for n in names:
            if n in row:
                v, u = row[n].displayed()
                cells[n] = _fmt(v, n, u) + ("F" if u == "F" else "")
            else:
                cells[n] = "--"
        note = (PRIOR_NOTES[int(abs(t) * 13) % len(PRIOR_NOTES)] if enc == "prior" else "") + (
            BASELINE_NOTES[int(abs(t) * 7) % len(BASELINE_NOTES)] if base else "")
        when = _stamp(ctx, t, stamp_style)
        if layout == 0:
            lines.append(f"{when:<18}" + "".join(f"{cells[n]:>10}" for n in names) + f"  {device}{note}")
        else:
            parts = ", ".join(f"{labels[n]} {cells[n]}" for n in names if cells[n] != "--")
            lines.append(f"{when}: {parts}; O2: {device}{note}")
    return "\n".join(lines)


MAR_STATUS_TEXT: dict[str, tuple[str, ...]] = {
    "given": ("Given", "GIVEN", "Administered", "Admin"),
    "held": ("Held", "HELD - parameters", "Not given - held"),
    "refused": ("Refused", "Pt refused", "REFUSED"),
    "not_given": ("Not Given", "Ordered - not administered", "Due - not given", "Missed"),
}


def render_mar_page(ctx: PageContext, facts: ChunkFacts, rng: random.Random) -> str:
    title = rng.choice(("Medication Administration Record", "MAR", "eMAR - Administration History",
                        "Medication Administration Summary"))
    lines = _header(ctx, title, rng)
    stamp_style = rng.randrange(3)
    layout = rng.randrange(2)
    if layout == 0:
        lines.append(f"{'Scheduled/Given':<18}{'Medication':<28}{'Dose':<26}{'Status':<26}By")
    for m in sorted(facts.meds, key=lambda m: m.time_h):
        status = rng.choice(MAR_STATUS_TEXT[m.status])
        nurse = rng.choice(("RN AB", "RN KT", "RN JM", "RN SL", "RN PW"))
        when = _stamp(ctx, m.time_h, stamp_style)
        note = _prior_note(m)
        if layout == 0:
            lines.append(f"{when:<18}{m.drug:<28}{m.dose:<26}{status:<26}{nurse}{note}")
        else:
            lines.append(f"{when} - {m.drug} {m.dose} - {status} ({nurse}){note}")
    if not facts.meds:
        lines.append("No administrations recorded for this period.")
    return "\n".join(lines)


@dataclass(frozen=True)
class Charge:
    service_date_h: float
    code: str                 # HCPCS Level II or a facility charge-master code (no CPT descriptors)
    description: str
    units: int
    amount: float


def render_bill_page(ctx: PageContext, charges: Sequence[Charge], rng: random.Random) -> str:
    title = rng.choice(("Itemized Statement", "DETAILED PATIENT BILL", "Itemized Charges", "Patient Account Detail"))
    lines = _header(ctx, title, rng)
    stamp_style = rng.randrange(3)
    lines.append(f"{'Date':<18}{'Code':<10}{'Description':<44}{'Units':>6}{'Charge':>12}")
    total = 0.0
    for c in sorted(charges, key=lambda c: c.service_date_h):
        total += c.amount
        lines.append(f"{_stamp(ctx, c.service_date_h, stamp_style)[:10]:<18}{c.code:<10}{c.description[:43]:<44}"
                     f"{c.units:>6}{c.amount:>12,.2f}")
    lines.append(f"{'':<78}{'-' * 12}")
    lines.append(f"{'TOTAL CHARGES':<78}{total:>12,.2f}")
    lines.append(rng.choice(("Insurance payments and adjustments are not shown.", "Balance due upon receipt.",
                             "This is not a bill for services covered by your plan until adjudicated.")))
    return "\n".join(lines)


ED_ARRIVAL = ("Ambulance", "Walk-in", "Private vehicle", "Transfer from outside facility")
ACUITY = ("ESI 1", "ESI 2", "ESI 3", "ESI 4", "ESI 5")


def render_ed_page(ctx: PageContext, facts: ChunkFacts, chief_complaint: str, rng: random.Random) -> str:
    """ED triage record: arrival, acuity, chief complaint and triage vital signs (structured, as EHRs print it)."""
    title = rng.choice(("Emergency Department Record", "ED TRIAGE", "Emergency Dept - Triage Assessment",
                        "ED Visit Summary"))
    lines = _header(ctx, title, rng)
    lines.append(f"Arrival: {_stamp(ctx, -2, rng.randrange(3))}   Mode: {rng.choice(ED_ARRIVAL)}   "
                 f"Acuity: {rng.choice(ACUITY[:4])}")
    lines.append(f"Chief complaint: {chief_complaint}")
    vit = []
    for o in sorted(facts.observations, key=lambda o: o.name):
        if o.name in VITAL_LABELS and o.encounter == "current":
            v, u = o.displayed()
            vit.append(f"{_obs_label(o, rng)} {_fmt(v, o.name, u)}{u if u in ('%', 'F', 'C') else ''}")
    lines.append("Triage vitals: " + (", ".join(vit) if vit else "not recorded"))
    lines.append(rng.choice(("Disposition: admitted to inpatient.", "Disposition: admit - medicine.",
                             "Disposition: ICU admission.", "Disposition: observation.")))
    return "\n".join(lines)
