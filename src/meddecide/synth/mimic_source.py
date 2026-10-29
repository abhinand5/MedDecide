"""Reads the structured MIMIC tables the grounded families need, as plain records. Local files only, no network.

Only structured fields are read: the admissions table (split, coding, coded ICD-10 codes, principal code, MS-DRG,
APR-DRG severity), the ICD-10-CM title table, and the MIMIC-IV-ED tables (stay, triage vitals, ED diagnoses). No
discharge note and no radiology report is loaded. The caller decides what leaves the machine: nothing does, except
aggregate counts.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from meddecide.mr import mimic

BUILD = Path("data/mimic_build")
PHYSIONET = Path("data/physionet")
ED_DIR = PHYSIONET / "mimic-iv-ed" / "2.2" / "ed"
TRIAGE_COLUMNS = ("chiefcomplaint", "temperature", "heartrate", "resprate", "o2sat", "sbp", "dbp", "pain")


def load_admissions(build: Path = BUILD) -> list[dict[str, Any]]:
    """Admission records (all splits; the families drop non-training rows and count them)."""
    frame = pd.read_parquet(
        build / "admissions.parquet",
        columns=["hadm_id", "split", "coding", "icd10_codes", "principal_icd10", "ms_drg_code", "ms_drg_title",
                 "apr_severity"],
    )
    records: list[dict[str, Any]] = []
    for row in frame.to_dict("records"):
        codes = row["icd10_codes"]
        records.append({
            "hadm_id": int(row["hadm_id"]),
            "split": str(row["split"]),
            "coding": str(row["coding"]),
            "icd10_codes": [str(c) for c in codes] if codes is not None else [],
            "principal_icd10": None if _missing(row["principal_icd10"]) else str(row["principal_icd10"]),
            "ms_drg_code": None if _missing(row["ms_drg_code"]) else str(row["ms_drg_code"]),
            "ms_drg_title": None if _missing(row["ms_drg_title"]) else str(row["ms_drg_title"]),
            "apr_severity": None if _missing(row["apr_severity"]) else float(row["apr_severity"]),
        })
    return records


def load_icd10_titles(build: Path = BUILD) -> dict[str, str]:
    """Undotted ICD-10-CM code -> long title (version 10 only)."""
    frame = pd.read_parquet(build / "icd10cm_titles.parquet", columns=["icd_code", "icd_version", "long_title"])
    frame = frame[frame["icd_version"].eq(10)]
    return {str(c): str(t) for c, t in zip(frame["icd_code"], frame["long_title"], strict=True)}


def load_ed_stays(split_of_subject: dict[int, str] | None = None) -> tuple[list[dict[str, Any]], dict[int, list[tuple]]]:
    """ED stays with triage fields merged, plus ED diagnoses by stay. ``split_of_subject`` maps a subject to its
    patient-level split; when None the split is computed with the osler_v4 rule (``mimic.subject_split``)."""
    stays = pd.read_csv(ED_DIR / "edstays.csv.gz", dtype={"disposition": str})
    triage = pd.read_csv(ED_DIR / "triage.csv.gz", dtype={"pain": str, "chiefcomplaint": str})
    dx = pd.read_csv(ED_DIR / "diagnosis.csv.gz", dtype={"icd_code": str, "icd_title": str})
    triage_by_stay = {int(r["stay_id"]): r for r in triage.to_dict("records")}
    subjects = {int(s) for s in stays["subject_id"]}
    if split_of_subject is None:
        split_of_subject = {s: mimic.subject_split(s, mimic.MIMIC4_DATASET) for s in subjects}
    records: list[dict[str, Any]] = []
    for row in stays.to_dict("records"):
        stay_id = int(row["stay_id"])
        tri = triage_by_stay.get(stay_id)
        if tri is None:
            continue
        merged: dict[str, Any] = {
            "stay_id": stay_id,
            "split": split_of_subject.get(int(row["subject_id"]), "unassigned"),
            "disposition": None if _missing(row.get("disposition")) else str(row.get("disposition")),
        }
        for column in TRIAGE_COLUMNS:
            merged[column] = None if _missing(tri.get(column)) else tri.get(column)
        records.append(merged)
    dx_by_stay: dict[int, list[tuple]] = defaultdict(list)
    for sid, code, version, title in zip(dx["stay_id"], dx["icd_code"], dx["icd_version"], dx["icd_title"], strict=True):
        dx_by_stay[int(sid)].append((str(code), int(version), "" if _missing(title) else str(title)))
    return records, dict(dx_by_stay)


def _missing(value: Any) -> bool:
    return value is None or (isinstance(value, float) and pd.isna(value)) or (not isinstance(value, str) and pd.isna(value))
