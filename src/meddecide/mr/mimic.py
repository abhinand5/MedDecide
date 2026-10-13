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

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from meddecide.gen.core import split_of
from meddecide.mr import icd

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
                               subject_split(row.subject_id, "mimiciv"))


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
