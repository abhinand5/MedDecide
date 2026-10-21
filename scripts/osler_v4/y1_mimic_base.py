"""Y1: the MIMIC-IV foundation (osler_v4 ADVISORY Y1). Credentialed data: local processing only (D28, D37).

Reads the official MIMIC-IV 3.1 ``hosp`` tables and MIMIC-IV-Note 2.2 discharge and radiology notes from
``data/physionet/``. Writes private parquet under ``data/mimic_build/`` (gitignored). Writes aggregate counts only to
``outputs/osler_v4/Y1/aggregates.json`` and to ``loops/osler_v4/mimic.md`` (generated here; no text, values or identifiers).

The APR-DRG description is dropped as soon as it is read (D46: no 3M text is kept or written).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from meddecide.mr import mimic

PHYS = Path("data/physionet")
BUILD = Path("data/mimic_build")
OUT = Path("outputs/osler_v4/Y1")
REPORT = Path("loops/osler_v4/mimic.md")
TOKENIZER = "perplexity-ai/pplx-decider-flash-v1.2"   # the prompt cap of ADVISORY §3.8 is counted with this tokenizer
PROMPT_CAP = 8192
SPLITS = ("train", "dev", "test")


def _log(message: str) -> None:
    print(time.strftime("%H:%M:%S"), message, flush=True)


def _token_lengths(texts: pd.Series, tokenizer: Any, batch: int = 4096) -> np.ndarray:
    """Token counts with the pplx tokenizer, no special tokens, in batches (lengths only; no ids are kept)."""
    values = texts.fillna("").astype(str).tolist()
    out = np.zeros(len(values), dtype=np.int32)
    for start in range(0, len(values), batch):
        enc = tokenizer(values[start:start + batch], add_special_tokens=False, return_length=True)
        out[start:start + batch] = np.asarray(enc["length"], dtype=np.int32)
    return out


def _percentiles(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {"n": 0}
    p50, p95, p99 = np.percentile(values, [50, 95, 99])
    return {"n": int(values.size), "p50": round(float(p50), 1), "p95": round(float(p95), 1), "p99": round(float(p99), 1),
            "max": int(values.max())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--physionet", type=Path, default=PHYS)
    parser.add_argument("--build", type=Path, default=BUILD)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--report", type=Path, default=REPORT)
    args = parser.parse_args()
    args.build.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    os.environ["HF_HUB_OFFLINE"] = "1"   # the tokenizer comes from the local cache; no network call during this step
    hosp = args.physionet / "mimiciv" / "3.1" / "hosp"
    note = args.physionet / "mimic-iv-note" / "2.2" / "note"
    agg: dict[str, Any] = {"prompt_cap_tokens": PROMPT_CAP, "tokenizer": TOKENIZER}

    _log("read hosp tables")
    patients = pd.read_csv(hosp / "patients.csv.gz", usecols=["subject_id", "anchor_year_group"])
    admissions = pd.read_csv(hosp / "admissions.csv.gz", usecols=["subject_id", "hadm_id"])
    dx = pd.read_csv(hosp / "diagnoses_icd.csv.gz", usecols=["hadm_id", "seq_num", "icd_code", "icd_version"],
                     dtype={"icd_code": str})
    px = pd.read_csv(hosp / "procedures_icd.csv.gz", usecols=["hadm_id", "seq_num", "icd_code", "icd_version"],
                     dtype={"icd_code": str})
    drg = pd.read_csv(hosp / "drgcodes.csv.gz", usecols=["hadm_id", "drg_type", "drg_code", "description",
                                                        "drg_severity", "drg_mortality"], dtype={"drg_code": str})
    drg.loc[drg["drg_type"].ne("HCFA"), "description"] = None   # APR descriptions are never kept (D46)
    d_dx = pd.read_csv(hosp / "d_icd_diagnoses.csv.gz", usecols=["icd_code", "icd_version", "long_title"],
                       dtype={"icd_code": str})
    d_px = pd.read_csv(hosp / "d_icd_procedures.csv.gz", usecols=["icd_code", "icd_version", "long_title"],
                       dtype={"icd_code": str})

    agg["patients_total"] = int(patients["subject_id"].nunique())
    agg["admissions_total"] = int(admissions["hadm_id"].nunique())
    agg["dx_rows_by_icd_version"] = {str(k): int(v) for k, v in dx["icd_version"].value_counts().items()}
    agg["pcs_rows_by_icd_version"] = {str(k): int(v) for k, v in px["icd_version"].value_counts().items()}
    agg["drg_rows_by_type"] = {str(k): int(v) for k, v in drg["drg_type"].value_counts(dropna=False).items()}

    _log("coding tables")
    coding = mimic.coding_summary(dx)
    codes = mimic.icd10_codes(dx)
    pcs = mimic.pcs_codes(px)
    drg_fields = mimic.drg_fields(drg)
    adm = mimic.admission_table(admissions, patients, coding, codes, pcs, drg_fields)
    agg["admissions_without_patient_row"] = int(adm["anchor_year_group"].isna().sum())
    adm = adm.reset_index()

    dx10 = dx[dx["icd_version"].eq(10)].sort_values(["hadm_id", "seq_num"], kind="stable")
    dx10 = dx10.merge(admissions.drop_duplicates("hadm_id")[["hadm_id", "subject_id"]], on="hadm_id", how="left")
    dx10[["hadm_id", "subject_id", "seq_num", "icd_code"]].to_parquet(args.build / "dx_icd10.parquet", index=False)
    px10 = px[px["icd_version"].eq(10)].sort_values(["hadm_id", "seq_num"], kind="stable")
    px10[["hadm_id", "seq_num", "icd_code"]].to_parquet(args.build / "pcs_icd10.parquet", index=False)
    d_dx[d_dx["icd_version"].eq(10)].to_parquet(args.build / "icd10cm_titles.parquet", index=False)
    d_px[d_px["icd_version"].eq(10)].to_parquet(args.build / "icd10pcs_titles.parquet", index=False)

    _log("tokenizer")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)

    # every split is the patient-level key of the subject (never a lookup through admissions), so a note, an admission and
    # an ED stay of one patient always share a split
    split_by_subject = {sid: mimic.subject_split(sid, mimic.MIMIC4_DATASET) for sid in patients["subject_id"].unique()}
    adm_subject = adm.drop_duplicates("hadm_id").set_index("hadm_id")["subject_id"]

    _log("discharge notes")
    dn = pd.read_csv(note / "discharge.csv.gz", usecols=["note_id", "subject_id", "hadm_id", "note_seq", "text"],
                     dtype={"hadm_id": "Int64"})
    agg["discharge_notes_total"] = len(dn)
    agg["discharge_notes_without_hadm"] = int(dn["hadm_id"].isna().sum())
    dn["tokens"] = _token_lengths(dn["text"], tokenizer)
    dn["split"] = dn["subject_id"].map(split_by_subject)
    agg["discharge_notes_subject_not_in_patients"] = int(dn["split"].isna().sum())
    dn["in_admissions"] = dn["hadm_id"].notna() & dn["hadm_id"].isin(adm_subject.index)
    dn["adm_subject"] = dn["hadm_id"].map(adm_subject)
    agg["discharge_notes_hadm_not_in_admissions"] = int((dn["hadm_id"].notna() & ~dn["in_admissions"]).sum())
    agg["discharge_notes_hadm_subject_mismatch"] = int((dn["in_admissions"] & dn["adm_subject"].ne(dn["subject_id"])).sum())
    dn["linked"] = dn["in_admissions"] & dn["adm_subject"].eq(dn["subject_id"])
    agg["discharge_notes_linked"] = int(dn["linked"].sum())
    linked = dn[dn["linked"]]
    agg["discharge_notes_split_mismatch"] = int((linked["split"] != linked["adm_subject"].map(split_by_subject)).sum())
    dn.drop(columns=["note_seq", "adm_subject"]).to_parquet(args.build / "discharge_notes.parquet", index=False)
    first = linked.sort_values(["hadm_id", "note_seq"], kind="stable")
    per_hadm = first.groupby("hadm_id").size()
    agg["admissions_with_2plus_discharge_notes"] = int((per_hadm > 1).sum())
    first = first.drop_duplicates("hadm_id").set_index("hadm_id")[["note_id", "tokens"]]
    first = first.rename(columns={"note_id": "discharge_note_id", "tokens": "discharge_tokens"})
    adm = adm.merge(first, left_on="hadm_id", right_index=True, how="left")
    adm["has_discharge_note"] = adm["discharge_note_id"].notna()
    agg["discharge_notes_in_admissions"] = int(adm["has_discharge_note"].sum())
    tokens = dn["tokens"].to_numpy()
    agg["discharge_tokens_all"] = _percentiles(tokens)
    agg["discharge_tokens_by_split"] = {s: _percentiles(dn.loc[dn["split"].eq(s), "tokens"].to_numpy()) for s in SPLITS}
    agg["discharge_notes_over_prompt_cap"] = int((tokens > PROMPT_CAP).sum())
    del dn, first

    _log("radiology notes")
    rn = pd.read_csv(note / "radiology.csv.gz", usecols=["note_id", "subject_id", "hadm_id", "text"],
                     dtype={"hadm_id": "Int64"})
    agg["radiology_notes_total"] = len(rn)
    rn["tokens"] = _token_lengths(rn["text"], tokenizer)
    rn["split"] = rn["subject_id"].map(split_by_subject)
    agg["radiology_notes_subject_not_in_patients"] = int(rn["split"].isna().sum())
    agg["radiology_notes_subject_without_admission"] = int((~rn["subject_id"].isin(adm["subject_id"])).sum())
    rn.to_parquet(args.build / "radiology_notes.parquet", index=False)
    agg["radiology_tokens_all"] = _percentiles(rn["tokens"].to_numpy())
    agg["radiology_notes_by_split"] = {s: int(rn["split"].eq(s).sum()) for s in SPLITS}
    del rn

    rd = pd.read_csv(note / "radiology_detail.csv.gz")
    rd.to_parquet(args.build / "radiology_detail.parquet", index=False)
    agg["radiology_detail_rows_by_field"] = {str(k): int(v) for k, v in rd["field_name"].value_counts().items()}
    del rd

    _log("split and coding summaries")
    adm.to_parquet(args.build / "admissions.parquet", index=False)
    patient_splits = patients["subject_id"].drop_duplicates().map(
        lambda sid: mimic.subject_split(sid, mimic.MIMIC4_DATASET)).value_counts()
    agg["patients_by_split"] = {s: int(patient_splits.get(s, 0)) for s in SPLITS}
    agg["admissions_by_split"] = {s: int(adm["split"].eq(s).sum()) for s in SPLITS}
    agg["admissions_late_anchor_by_split"] = {s: int((adm["split"].eq(s) & adm["late_anchor"]).sum()) for s in SPLITS}
    agg["admissions_with_discharge_note_by_split"] = {s: int((adm["split"].eq(s) & adm["has_discharge_note"]).sum())
                                                      for s in SPLITS}
    agg["coding_class_admissions"] = {str(k): int(v) for k, v in adm["coding"].value_counts().items()}
    coded = adm["coding"].ne("none")
    agg["icd10_share_of_coded_admissions"] = float((adm["coding"].eq("icd10")).sum() / max(coded.sum(), 1))
    agg["icd10_share_of_all_admissions"] = float(adm["coding"].eq("icd10").mean())
    agg["icd10_substrate_admissions_by_split"] = {s: int((adm["split"].eq(s) & adm["coding"].eq("icd10")
                                                          & adm["has_discharge_note"]).sum()) for s in SPLITS}
    agg["ms_drg_admissions"] = int(adm["ms_drg_code"].notna().sum())
    agg["apr_admissions"] = int(adm["apr_severity"].notna().sum())
    agg["apr_severity_counts"] = {str(int(k)): int(v) for k, v in adm["apr_severity"].value_counts().items()}
    agg["apr_mortality_counts"] = {str(int(k)): int(v) for k, v in adm["apr_mortality"].value_counts().items()}

    (args.out / "aggregates.json").write_text(json.dumps(agg, indent=2, sort_keys=True) + "\n")
    _log(f"wrote {args.out / 'aggregates.json'}")
    write_report(agg, args.report)
    _log(f"wrote {args.report}")


def write_report(agg: dict[str, Any], path: Path) -> None:
    """The committed Y1 report: aggregate counts and percentiles only, generated from aggregates.json."""
    def row(label: str, values: dict[str, Any]) -> str:
        return f"| {label} | " + " | ".join(str(values.get(s, "")) for s in SPLITS) + f" | {sum(values.values())} |"

    tok = agg["discharge_tokens_all"]
    by = agg["discharge_tokens_by_split"]
    lines = [
        "# MIMIC foundation — osler_v4 Y1 (aggregates only)",
        "",
        "Generated by `scripts/osler_v4/y1_mimic_base.py` from `outputs/osler_v4/Y1/aggregates.json`. Counts and percentiles only:",
        "no MIMIC text, values or identifiers. The private tables are under `data/mimic_build/` (gitignored).",
        "",
        "Caveat (ADVISORY §7.1): a code in `diagnoses_icd` is the coder's assignment. It records what was coded, including",
        "coding errors and omissions; it does not prove the documentation supports the code.",
        "",
        "## 1. Patients, admissions and notes per split (patient-level split, `subject_split` key `mimic4`)",
        "",
        "| count | train | dev | test | total |",
        "|---|---|---|---|---|",
        row("patients", agg["patients_by_split"]),
        row("admissions", agg["admissions_by_split"]),
        row("admissions, late-anchor patients", agg["admissions_late_anchor_by_split"]),
        row("admissions with a discharge note", agg["admissions_with_discharge_note_by_split"]),
        row("radiology notes", agg["radiology_notes_by_split"]),
        "",
        f"Patients in hosp: {agg['patients_total']}; admissions: {agg['admissions_total']}; admissions without a patient row: "
        f"{agg['admissions_without_patient_row']}.",
        "",
        "## 2. Discharge notes",
        "",
        f"Notes: {agg['discharge_notes_total']}; without an admission id: {agg['discharge_notes_without_hadm']}; admission "
        f"id not in hosp: {agg['discharge_notes_hadm_not_in_admissions']}; admission id of another patient: "
        f"{agg['discharge_notes_hadm_subject_mismatch']}; subject not in patients: "
        f"{agg['discharge_notes_subject_not_in_patients']}. Linked to an admission: {agg['discharge_notes_linked']} "
        f"(split mismatch among linked notes: {agg['discharge_notes_split_mismatch']}); admissions with two or more notes: "
        f"{agg['admissions_with_2plus_discharge_notes']} (the first note by note_seq is used).",
        "",
        f"Token length with the pplx tokenizer (all notes): p50 {tok.get('p50')}, p95 {tok.get('p95')}, p99 {tok.get('p99')}, "
        f"max {tok.get('max')}. Notes over the {agg['prompt_cap_tokens']}-token prompt cap: "
        f"{agg['discharge_notes_over_prompt_cap']}.",
        "",
        "| split | notes | p50 | p95 | p99 |",
        "|---|---|---|---|---|",
    ]
    for s in SPLITS:
        d = by[s]
        lines.append(f"| {s} | {d.get('n', 0)} | {d.get('p50', '')} | {d.get('p95', '')} | {d.get('p99', '')} |")
    rad = agg["radiology_tokens_all"]
    lines += [
        "",
        f"Radiology notes: {agg['radiology_notes_total']}; token p50 {rad.get('p50')}, p95 {rad.get('p95')}, max "
        f"{rad.get('max')}; subject not in patients: {agg['radiology_notes_subject_not_in_patients']}; subject without "
        f"any hospital admission: {agg['radiology_notes_subject_without_admission']}.",
        "",
        "## 3. Coding and DRG",
        "",
        "| diagnosis coding class per admission | count |",
        "|---|---|",
    ]
    for k, v in sorted(agg["coding_class_admissions"].items()):
        lines.append(f"| {k} | {v} |")
    lines += [
        "",
        f"ICD-10 share of admissions that code any diagnosis: {agg['icd10_share_of_coded_admissions']:.4f}; of all "
        f"admissions: {agg['icd10_share_of_all_admissions']:.4f}.",
        "",
        f"ICD-10 diagnosis rows: {agg['dx_rows_by_icd_version'].get('10', 0)}; ICD-9 rows (recorded, not used): "
        f"{agg['dx_rows_by_icd_version'].get('9', 0)}. ICD-10-PCS procedure rows: "
        f"{agg['pcs_rows_by_icd_version'].get('10', 0)}.",
        "",
        "DRG rows by type: " + ", ".join(f"{k} {v}" for k, v in sorted(agg["drg_rows_by_type"].items())) + ".",
        f"Admissions with an MS-DRG (HCFA): {agg['ms_drg_admissions']}; with APR-DRG levels: {agg['apr_admissions']}. "
        "APR descriptions are never read into the tables (D46).",
        "",
        "## 4. Radiology detail fields (counts of rows by field)",
        "",
        "| field | rows |",
        "|---|---|",
    ]
    for k, v in sorted(agg["radiology_detail_rows_by_field"].items()):
        lines.append(f"| {k} | {v} |")
    lines.append("")
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
