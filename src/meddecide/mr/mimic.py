"""MIMIC readers for the medical-record audit tasks (osler_v1). Credentialed data: local processing only.

- MIMIC-III ``NOTEEVENTS``: real notes with a ``CATEGORY`` and ``DESCRIPTION`` field -> page classes.
- MIMIC-IV-Note ``discharge`` / ``radiology``: real discharge summaries and radiology reports.
- MIMIC-IV ``hosp/diagnoses_icd``: the coded diagnoses of each admission (ICD-9 and ICD-10; only ICD-10 is used) ->
  which curated ICD-10-CM family code the discharge summary carries.

Nothing here writes MIMIC text anywhere but under ``data/`` (gitignored). Splits are by patient (``subject_id``), with the
same 80/10/10 hash as the generators. MIMIC-III (2001-2012) and MIMIC-IV (2008-2019) overlap in time with unlinked
patient ids, so an evaluation set that must be disjoint from MIMIC-III training text uses MIMIC-IV admissions whose
``anchor_year_group`` starts in 2014 or later (``late_anchor``).
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from meddecide.gen.core import split_of
from meddecide.mr import icd

# the dataset key of MIMIC-IV in subject_split (ADVISORY osler_v4 §3.3); MIMIC-III uses "mimiciii"
MIMIC4_DATASET = "mimic4"

# MIMIC-III NOTEEVENTS CATEGORY (stripped) -> page class; "Physician" and "Nursing" are split by DESCRIPTION below.
CATEGORY_TO_PAGE: dict[str, str] = {
    "Discharge summary": "discharge_summary",
    "Radiology": "radiology",
    "Nutrition": "nutrition",
    "Consult": "consult_note",
    "ECG": "other", "Echo": "other", "Respiratory": "other", "Rehab Services": "other", "Social Work": "other",
    "Case Management": "other", "Pharmacy": "other", "General": "other",
}


def mimic3_page_class(category: str, description: str) -> tuple[str | None, str]:
    """(page class or None, reason). ``None`` drops the note with the reason recorded (no silent drops)."""
    cat = (category or "").strip()
    desc = (description or "").strip().lower()
    if cat in CATEGORY_TO_PAGE:
        return CATEGORY_TO_PAGE[cat], f"category {cat}"
    if cat == "Physician":
        if "admission" in desc or "h&p" in desc or "history and physical" in desc:
            return "admission", "physician admission note"
        if "progress" in desc or "intensivist" in desc or "icu note" in desc:
            return "progress_note", "physician progress note"
        if "consult" in desc:
            return "consult_note", "physician consult note"
        return None, f"physician note type not mapped: {desc[:40]}"
    if cat in ("Nursing", "Nursing/other"):
        if "progress" in desc:
            return "progress_note", "nursing progress note"
        return None, f"nursing note type not mapped: {desc[:40]}"
    return None, f"category not mapped: {cat}"


def subject_split(subject_id: int | str, dataset: str) -> str:
    """Patient-level split; the dataset is part of the key because MIMIC-III and MIMIC-IV ids are unrelated."""
    return split_of(f"{dataset}-subject-{int(subject_id)}")


def chunk_pages(text: str, max_chars: int = 3200, min_chars: int = 400) -> list[str]:
    """Cut a note into page-sized chunks at line boundaries (a printed page is roughly 3,000 characters)."""
    lines = text.splitlines()
    pages, cur, size = [], [], 0
    for line in lines:
        if size + len(line) + 1 > max_chars and size >= min_chars:
            pages.append("\n".join(cur))
            cur, size = [], 0
        cur.append(line)
        size += len(line) + 1
    if cur:
        tail = "\n".join(cur)
        if pages and len(tail) < min_chars:
            pages[-1] = pages[-1] + "\n" + tail
        else:
            pages.append(tail)
    return [p for p in pages if p.strip()]


@dataclass(frozen=True)
class NotePage:
    dataset: str
    note_id: str
    subject_id: int
    hadm_id: int | None
    page_class: str
    page_index: int
    n_pages: int
    text: str
    split: str


def iter_mimic3_pages(noteevents_csv: str | Path, *, counts: dict[str, int] | None = None,
                      chunksize: int = 50_000) -> Iterator[NotePage]:
    """Every page of every mapped, non-error MIMIC-III note. ``counts`` collects kept/dropped reasons."""
    counts = counts if counts is not None else {}
    cols = ["ROW_ID", "SUBJECT_ID", "HADM_ID", "CATEGORY", "DESCRIPTION", "ISERROR", "TEXT"]
    for frame in pd.read_csv(noteevents_csv, usecols=cols, chunksize=chunksize, dtype={"HADM_ID": "Int64"},
                             low_memory=False):
        for row in frame.itertuples(index=False):
            if row.ISERROR == 1:
                counts["dropped: ISERROR=1"] = counts.get("dropped: ISERROR=1", 0) + 1
                continue
            page_class, reason = mimic3_page_class(row.CATEGORY, row.DESCRIPTION)
            if page_class is None:
                counts[f"dropped: {reason}"] = counts.get(f"dropped: {reason}", 0) + 1
                continue
            counts[f"kept: {reason}"] = counts.get(f"kept: {reason}", 0) + 1
            pages = chunk_pages(str(row.TEXT))
            hadm = None if pd.isna(row.HADM_ID) else int(row.HADM_ID)
            for i, page in enumerate(pages):
                yield NotePage("mimiciii", str(row.ROW_ID), int(row.SUBJECT_ID), hadm, page_class, i, len(pages), page,
                               subject_split(row.SUBJECT_ID, "mimiciii"))


def iter_mimic4_notes(note_csv: str | Path, page_class: str, *, chunksize: int = 20_000) -> Iterator[NotePage]:
    """MIMIC-IV-Note ``discharge.csv.gz`` (page_class="discharge_summary") or ``radiology.csv.gz`` ("radiology"),
    one NotePage per page."""
    cols = ["note_id", "subject_id", "hadm_id", "text"]
    for frame in pd.read_csv(note_csv, usecols=cols, chunksize=chunksize, dtype={"hadm_id": "Int64"}):
        for row in frame.itertuples(index=False):
            pages = chunk_pages(str(row.text))
            hadm = None if pd.isna(row.hadm_id) else int(row.hadm_id)
            for i, page in enumerate(pages):
                yield NotePage("mimiciv", str(row.note_id), int(row.subject_id), hadm, page_class, i, len(pages), page,
                               subject_split(row.subject_id, MIMIC4_DATASET))


# ---- ICD-10 family codes per admission ----------------------------------------------------------

CODE_TO_FAMILY: dict[str, str] = {code: fam.key for fam in icd.FAMILIES for code in fam.codes}


def undotted(code: str) -> str:
    return code.replace(".", "")


UNDOTTED_TO_CODE = {undotted(c): c for c in CODE_TO_FAMILY}


def admission_family_codes(diagnoses_icd_csv: str | Path) -> dict[int, dict[str, list[str]]]:
    """{hadm_id: {family: [curated codes coded on the admission]}} from MIMIC-IV ``diagnoses_icd`` (ICD-10 rows only)."""
    df = pd.read_csv(diagnoses_icd_csv, usecols=["hadm_id", "icd_code", "icd_version"], dtype={"icd_code": str})
    df = df[df["icd_version"] == 10]
    out: dict[int, dict[str, list[str]]] = {}
    for hadm, code in zip(df["hadm_id"], df["icd_code"].str.strip(), strict=True):
        dotted = UNDOTTED_TO_CODE.get(code)
        if dotted is None:
            continue
        out.setdefault(int(hadm), {}).setdefault(CODE_TO_FAMILY[dotted], []).append(dotted)
    return out


def icd10_admissions(diagnoses_icd_csv: str | Path) -> set[int]:
    df = pd.read_csv(diagnoses_icd_csv, usecols=["hadm_id", "icd_version"])
    return {int(h) for h in df.loc[df["icd_version"] == 10, "hadm_id"].unique()}


def family_label(family: str, coded: Mapping[str, list[str]]) -> tuple[str | None, str]:
    """The admission's label for one family: ("<code>", "coded") when exactly one family code is coded, (None, "not coded")
    when none is, and ("", "ambiguous") when several are (dropped by the caller with that reason)."""
    codes = sorted(set(coded.get(family, [])))
    if not codes:
        return None, "not coded"
    if len(codes) > 1:
        return "", "ambiguous: " + ",".join(codes)
    return codes[0], "coded"


def late_anchor(anchor_year_group: str) -> bool:
    """MIMIC-IV ``patients.anchor_year_group`` starting in 2014 or later (no overlap with MIMIC-III's 2001-2012)."""
    return int(str(anchor_year_group)[:4]) >= 2014


# ---- discharge-note sections (osler_v4 Y1; ADVISORY §3.8) -------------------------------------------

# normalised header text -> canonical section key (the MIMIC-IV discharge-note template; identifiers kept apart)
SECTION_KEYS: dict[str, str] = {
    "name": "identifiers", "unit no": "identifiers", "date of birth": "identifiers", "facility": "identifiers",
    "admission date": "dates", "discharge date": "dates", "sex": "demographics", "service": "service",
    "allergies": "allergies", "attending": "attending",
    "chief complaint": "chief_complaint",
    "major surgical or invasive procedure": "procedures",
    "history of present illness": "history_of_present_illness",
    "past medical history": "past_medical_history", "social history": "social_history",
    "family history": "family_history",
    "physical exam": "physical_exam", "physical exam on admission": "physical_exam",
    "pertinent results": "pertinent_results",
    "brief hospital course": "brief_hospital_course",
    "medications on admission": "medications_on_admission",
    "discharge medications": "discharge_medications",
    "discharge disposition": "discharge_disposition",
    "discharge diagnosis": "discharge_diagnosis", "discharge diagnoses": "discharge_diagnosis",
    "discharge condition": "discharge_condition", "discharge instructions": "discharge_instructions",
    "followup instructions": "followup_instructions",
}
# the printed header of each priority section (rendered in a prompt; other sections use a generic title)
SECTION_TITLES: dict[str, str] = {
    "chief_complaint": "Chief Complaint", "history_of_present_illness": "History of Present Illness",
    "brief_hospital_course": "Brief Hospital Course", "procedures": "Major Surgical or Invasive Procedure",
    "discharge_diagnosis": "Discharge Diagnosis", "discharge_medications": "Discharge Medications",
    "pertinent_results": "Pertinent Results",
}
# section priority when a note must be cut (ADVISORY §3.8); the other sections follow in document order
PRIORITY_SECTIONS: tuple[str, ...] = (
    "chief_complaint", "history_of_present_illness", "brief_hospital_course", "procedures", "discharge_diagnosis",
    "discharge_medications", "pertinent_results",
)
TRUNCATION_MARK = "[... truncated]"

_HEADER = re.compile(r"^\s*([A-Za-z][A-Za-z ]{1,60}?)\s*:(.*)$")


def section_header(line: str) -> tuple[str, str] | None:
    """(canonical key, text after the colon) when the line opens a known section; None for any other line."""
    match = _HEADER.match(line)
    if match is None:
        return None
    key = SECTION_KEYS.get(" ".join(match.group(1).lower().split()))
    return None if key is None else (key, match.group(2).strip())


def sections(text: str) -> dict[str, str]:
    """A discharge note split into its known sections, in document order. Text before the first known header is kept as
    ``preamble``. A repeated header is joined under its key. A header-like line that is not a known section stays with the
    section above it. Empty sections are dropped."""
    lines_by_key: dict[str, list[str]] = {"preamble": []}
    current = "preamble"
    for line in text.splitlines():
        found = section_header(line)
        if found is None:
            lines_by_key[current].append(line)
            continue
        current, rest = found
        lines_by_key.setdefault(current, [])
        if rest:
            lines_by_key[current].append(rest)
    bodies = {key: "\n".join(lines).strip() for key, lines in lines_by_key.items()}
    return {key: body for key, body in bodies.items() if body}


def _render(key: str, body: str) -> str:
    if key == "preamble":
        return body
    return f"{SECTION_TITLES.get(key, key.replace('_', ' ').capitalize())}:\n{body}"


def _fit_prefix(text: str, room: int, count: Callable[[str], int]) -> str:
    """The longest character prefix of ``text`` whose count fits in ``room``."""
    low, high = 0, len(text)
    while low < high:
        mid = (low + high + 1) // 2
        if count(text[:mid].rstrip()) <= room:
            low = mid
        else:
            high = mid - 1
    return text[:low]


def _fill(order: list[str], budget: int, secs: Mapping[str, str], count: Callable[[str], int]) -> str:
    """Whole sections in ``order`` while they fit with the marker's tokens reserved; the section that does not fit is cut
    to the remaining room and followed by the marker, so a cut note always ends with the marker and fits the budget."""
    reserve = count(TRUNCATION_MARK)
    parts: list[str] = []
    used = 0
    for key in order:
        block = _render(key, secs[key])
        size = count(block)
        if used + size <= budget - reserve:
            parts.append(block)
            used += size
            continue
        cut = _fit_prefix(block, budget - reserve - used, count).rstrip()
        parts.append((cut + "\n" if cut else "") + TRUNCATION_MARK)
        break
    return "\n\n".join(parts)


def prioritised_text(secs: Mapping[str, str], max_tokens: int, count: Callable[[str], int]) -> tuple[str, bool]:
    """The note text for a prompt capped at ``max_tokens`` (``count`` gives a token length). A note that fits is returned
    whole with cut=False. Otherwise the priority sections come first, in ADVISORY order, then the other sections in
    document order, until the budget is spent; the section that does not fit is cut at the budget with a marker, and
    later sections are dropped. Returns (text, cut)."""
    document_order = list(secs)
    full = "\n\n".join(_render(key, secs[key]) for key in document_order)
    if count(full) <= max_tokens:
        return full, False
    order = [key for key in PRIORITY_SECTIONS if key in secs] + [key for key in document_order
                                                               if key not in PRIORITY_SECTIONS]
    budget = max_tokens
    for _ in range(8):
        text = _fill(order, budget, secs, count)
        excess = count(text) - max_tokens
        if excess <= 0:
            return text, True
        budget -= excess   # joining blocks can add a token at a boundary; shrink the budget and try again
    raise RuntimeError("prioritised_text: the note did not fit the prompt cap after 8 tries")


# ---- admission tables (osler_v4 Y1) -----------------------------------------------------------------

def coding_summary(diagnoses: pd.DataFrame) -> pd.DataFrame:
    """Per admission (index hadm_id): ICD-10 and ICD-9 row counts and the coding class icd10, icd9, mixed or none. ICD-10
    admissions are the coding substrate (ADVISORY §3.3); the others are recorded and not used."""
    is10 = diagnoses["icd_version"].eq(10)
    by_hadm = diagnoses["hadm_id"]
    out = pd.DataFrame({"n_icd10": is10.groupby(by_hadm).sum(), "n_icd9": (~is10).groupby(by_hadm).sum()}).astype(int)
    pure10 = (out["n_icd10"] > 0) & (out["n_icd9"] == 0)
    pure9 = (out["n_icd9"] > 0) & (out["n_icd10"] == 0)
    out["coding"] = np.select([pure10, pure9, (out["n_icd10"] > 0) & (out["n_icd9"] > 0)],
                              ["icd10", "icd9", "mixed"], default="none")
    return out


def icd10_codes(diagnoses: pd.DataFrame) -> pd.DataFrame:
    """Per admission: the ICD-10-CM codes in seq_num order (icd10_codes) and the code of seq_num 1 (principal_icd10)."""
    rows = diagnoses[diagnoses["icd_version"].eq(10)].sort_values(["hadm_id", "seq_num"], kind="stable")
    codes = rows.groupby("hadm_id")["icd_code"].agg(list).rename("icd10_codes")
    principal = rows[rows["seq_num"].eq(1)].groupby("hadm_id")["icd_code"].first().rename("principal_icd10")
    return pd.concat([codes, principal], axis=1)


def pcs_codes(procedures: pd.DataFrame) -> pd.DataFrame:
    """Per admission: the ICD-10-PCS codes in seq_num order (codes only; titles are kept in the dictionary table)."""
    rows = procedures[procedures["icd_version"].eq(10)].sort_values(["hadm_id", "seq_num"], kind="stable")
    return rows.groupby("hadm_id")["icd_code"].agg(list).rename("pcs10_codes").to_frame()


def drg_fields(drg: pd.DataFrame) -> pd.DataFrame:
    """Per admission: the MS-DRG (HCFA rows: code and CMS title, public domain) and the APR-DRG severity and
    risk-of-mortality levels (APR rows: levels only; the APR description is never read, D46)."""
    hcfa = drg[drg["drg_type"].eq("HCFA")].sort_values(["hadm_id", "drg_code"], kind="stable")
    ms = hcfa.drop_duplicates("hadm_id").set_index("hadm_id")[["drg_code", "description"]]
    ms = ms.rename(columns={"drg_code": "ms_drg_code", "description": "ms_drg_title"})
    apr = drg[drg["drg_type"].eq("APR")].drop_duplicates("hadm_id").set_index("hadm_id")[["drg_severity", "drg_mortality"]]
    apr = apr.rename(columns={"drg_severity": "apr_severity", "drg_mortality": "apr_mortality"})
    rows = drg.groupby(["hadm_id", "drg_type"]).size().unstack(fill_value=0)
    counts = rows.rename(columns=lambda t: f"n_drg_{str(t).lower()}")
    return ms.join(apr, how="outer").join(counts, how="outer")


def admission_table(admissions: pd.DataFrame, patients: pd.DataFrame, coding: pd.DataFrame, codes: pd.DataFrame,
                    pcs: pd.DataFrame, drg: pd.DataFrame) -> pd.DataFrame:
    """One row per admission: patient split (``subject_split``, patient-level), late-anchor flag, coding class, ICD-10 codes,
    principal code, PCS codes and DRG fields. Admissions without a row in a side table get empty values, counted by the
    caller."""
    base = admissions[["hadm_id", "subject_id"]].drop_duplicates("hadm_id").set_index("hadm_id")
    groups = patients.set_index("subject_id")["anchor_year_group"]
    base = base.join(groups, on="subject_id")
    splits = {sid: subject_split(sid, MIMIC4_DATASET) for sid in base["subject_id"].unique()}
    base["split"] = base["subject_id"].map(splits)
    base["late_anchor"] = base["anchor_year_group"].map(lambda g: late_anchor(g) if isinstance(g, str) else False)
    out = base.join(coding).join(codes).join(pcs).join(drg)
    for col in ("n_icd10", "n_icd9"):
        out[col] = out[col].fillna(0).astype(int)
    out["coding"] = out["coding"].fillna("none")
    for col in ("icd10_codes", "pcs10_codes"):
        out[col] = out[col].apply(lambda v: v if isinstance(v, list) else [])
    for col in ("n_drg_hcfa", "n_drg_apr"):
        if col in out.columns:
            out[col] = out[col].fillna(0).astype(int)
    return out
