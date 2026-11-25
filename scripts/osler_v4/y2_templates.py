"""Y2: MIMIC structured-gold templates, the ED templates and MedNLI (osler_v4 ADVISORY Y2 and the operator's data steer of
2026-10-21). Credentialed data: local processing only (D28, D37). Labels come from database fields or from code.

Reads the Y1 tables (data/mimic_build/) and the official MIMIC-IV-ED 2.2 and MedNLI 1.0.0 files (data/physionet/). Writes
private JSONL items under data/mimic_build/items/ (gitignored) and counts only to outputs/osler_v4/Y2/. The committed report
(loops/osler_v4/mimic.md §2) holds counts, screens and majority rates only.

Usage: uv run python scripts/osler_v4/y2_templates.py [--only TEMPLATE ...]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
from transformers import AutoTokenizer

from meddecide.gen.screen import NaiveBayes, gold_label, macro_accuracy
from meddecide.mr import mimic
from meddecide.mr import mimic_items as mi
from meddecide.utils.io import write_json

BUILD = Path("data/mimic_build")
ITEMS = BUILD / "items"
PHYS = Path("data/physionet")
OUT = Path("outputs/osler_v4/Y2")
TOKENIZER = "perplexity-ai/pplx-decider-flash-v1.2"
QUOTA = {"train": 40_000, "dev": 500, "test": 500}
SPLITS = ("train", "dev", "test")
TEMPLATES = ("mimic_icd_support_v1", "mimic_icd_specificity_v1", "mimic_principal_dx_v1", "mimic_msdrg_v1",
             "mimic_drg_cc_level_v1", "mimic_apr_severity_v1", "mimic_apr_mortality_v1", "mimic_pcs_support_v1",
             "mimic_rad_exam_v1", "mimic_ed_acuity_v1", "mimic_ed_disposition_v1", "mednli_v1")


def _log(message: str) -> None:
    print(time.strftime("%H:%M:%S"), message, flush=True)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load(physio: Path) -> dict[str, Any]:
    adm = pd.read_parquet(BUILD / "admissions.parquet").set_index("hadm_id")
    notes = pd.read_parquet(BUILD / "discharge_notes.parquet", columns=["note_id", "text"])
    icd_titles = pd.read_parquet(BUILD / "icd10cm_titles.parquet")
    pcs_titles = pd.read_parquet(BUILD / "icd10pcs_titles.parquet")
    dx = pd.read_parquet(BUILD / "dx_icd10.parquet", columns=["hadm_id", "icd_code"])
    rad = pd.read_parquet(BUILD / "radiology_notes.parquet", columns=["note_id", "hadm_id", "subject_id", "split", "text"])
    detail = pd.read_parquet(BUILD / "radiology_detail.parquet")
    patients = pd.read_csv(physio / "mimiciv" / "3.1" / "hosp" / "patients.csv.gz", usecols=["subject_id", "anchor_year_group"])
    ed = physio / "mimic-iv-ed" / "2.2" / "ed"
    late_of = {int(s): mimic.late_anchor(g) for s, g in zip(patients["subject_id"], patients["anchor_year_group"], strict=True)}
    return {
        "adm": adm,
        "notes": dict(zip(notes["note_id"], notes["text"], strict=True)),
        "icd_titles": {str(c): str(t) for c, t in zip(icd_titles["icd_code"], icd_titles["long_title"], strict=True)},
        "pcs_titles": {str(c): str(t) for c, t in zip(pcs_titles["icd_code"], pcs_titles["long_title"], strict=True)},
        "dx": dx, "rad": rad, "detail": detail, "late_of": late_of,
        "edstays": pd.read_csv(ed / "edstays.csv.gz", dtype={"disposition": str}),
        "triage": pd.read_csv(ed / "triage.csv.gz", dtype={"pain": str, "chiefcomplaint": str}),
        "ed_dx": pd.read_csv(ed / "diagnosis.csv.gz", dtype={"icd_code": str, "icd_title": str}),
        "mednli": {s: [json.loads(line) for line in (physio / "mednli" / "1.0.0" / f"mli_{s}_v1.jsonl").read_text().splitlines()
                       if line.strip()] for s in ("train", "dev", "test")},
    }


def split_records(adm: pd.DataFrame, split: str, seed: int) -> list[dict[str, Any]]:
    """The admissions of one split with a discharge note, in a stable order, as plain records (hadm_id is a field)."""
    frame = adm[adm["has_discharge_note"] & adm["split"].eq(split)].reset_index()
    return mi.stable_order(frame.to_dict("records"), seed, f"pool:{split}", lambda r: r["hadm_id"])


def _principal(record: dict[str, Any]) -> str:
    value = record["principal_icd10"]
    return "" if value is None or pd.isna(value) else str(value)


def _codes(record: dict[str, Any]) -> list[str]:
    return [str(c) for c in record["icd10_codes"]] if record["coding"] == "icd10" else []


def build_all(data: dict[str, Any], only: set[str], seed: int, count) -> dict[str, dict[str, mi.Built]]:
    adm = data["adm"]
    has_note = adm[adm["has_discharge_note"]]
    token_of = {str(n): int(t) for n, t in zip(has_note["discharge_note_id"], has_note["discharge_tokens"], strict=True)}
    ctx = mi.Ctx(notes=data["notes"], icd_titles=data["icd_titles"], pcs_titles=data["pcs_titles"], count=count, seed=seed,
                 token_of=token_of)
    results: dict[str, dict[str, mi.Built]] = defaultdict(dict)
    fam_rows = list(zip(data["dx"]["hadm_id"].astype(int), data["dx"]["icd_code"].astype(str), strict=True))
    fam_of = mi.family_codes(fam_rows)
    billable: dict[str, list[str]] = defaultdict(list)
    for code in data["icd_titles"]:
        billable[code[:3]].append(code)
    drg_titles = {str(c): str(t) for c, t in adm[["ms_drg_code", "ms_drg_title"]].dropna().drop_duplicates("ms_drg_code")
                  .itertuples(index=False)}
    names: dict[str, set[str]] = defaultdict(set)
    for name in data["detail"].loc[data["detail"]["field_name"].eq("exam_name"), "field_value"].dropna().unique():
        mod = mi.modality(str(name))
        if mod is not None:
            names[mod].add(str(name))
    names_by_mod = {k: sorted(v) for k, v in names.items()}
    rad_by_note = data["detail"][data["detail"]["field_name"].eq("exam_name") & data["detail"]["field_ordinal"].eq(1)]
    exam_of = dict(zip(rad_by_note["note_id"], rad_by_note["field_value"], strict=True))
    for split in SPLITS:
        quota = QUOTA[split]
        recs = split_records(adm, split, seed)
        pool = [(int(r["hadm_id"]), str(r["discharge_note_id"]), bool(r["late_anchor"]), _principal(r), _codes(r))
                for r in recs]
        codes_pool = [x for x in pool if x[4]]
        icd_rows = [(h, n, late) for h, n, late, _, _ in codes_pool]
        drg_pool = [(int(r["hadm_id"]), str(r["discharge_note_id"]), bool(r["late_anchor"]), str(r["ms_drg_code"]),
                     str(r["ms_drg_title"])) for r in recs if isinstance(r["ms_drg_code"], str)]
        apr_sev = [(int(r["hadm_id"]), str(r["discharge_note_id"]), bool(r["late_anchor"]), "", int(r["apr_severity"]))
                   for r in recs if pd.notna(r["apr_severity"])]
        apr_mort = [(int(r["hadm_id"]), str(r["discharge_note_id"]), bool(r["late_anchor"]), "", int(r["apr_mortality"]))
                    for r in recs if pd.notna(r["apr_mortality"])]
        pcs_pool = [(int(r["hadm_id"]), str(r["discharge_note_id"]), bool(r["late_anchor"]), _principal(r),
                     [str(c) for c in r["pcs10_codes"]]) for r in recs
                    if _principal(r) and r["pcs10_codes"] is not None and len(r["pcs10_codes"]) > 0]
        spec_pool = [(h, n, late, codes) for h, n, late, _, codes in codes_pool]
        principal_pool = [(h, n, late, p, codes) for h, n, late, p, codes in codes_pool if p and len(set(codes)) >= 4]
        if "mimic_icd_support_v1" in only:
            _log(f"{split}: icd_support")
            results["mimic_icd_support_v1"][split] = mi.icd_support(ctx, split, quota, icd_rows, fam_of)
        if "mimic_icd_specificity_v1" in only:
            _log(f"{split}: icd_specificity")
            results["mimic_icd_specificity_v1"][split] = mi.icd_specificity(ctx, split, quota, spec_pool, billable)
        if "mimic_principal_dx_v1" in only:
            _log(f"{split}: principal_dx")
            results["mimic_principal_dx_v1"][split] = mi.principal_dx(ctx, split, quota, principal_pool)
        if "mimic_msdrg_v1" in only:
            _log(f"{split}: msdrg")
            results["mimic_msdrg_v1"][split] = mi.msdrg(ctx, split, quota, drg_pool, drg_titles)
        if "mimic_drg_cc_level_v1" in only:
            _log(f"{split}: drg_cc_level")
            results["mimic_drg_cc_level_v1"][split] = mi.drg_cc_level(ctx, split, quota, drg_pool)
        if "mimic_apr_severity_v1" in only:
            results["mimic_apr_severity_v1"][split] = mi.apr_level(ctx, split, quota, apr_sev, "mimic_apr_severity",
                                                                   "mimic_apr_severity_v1")
        if "mimic_apr_mortality_v1" in only:
            results["mimic_apr_mortality_v1"][split] = mi.apr_level(ctx, split, quota, apr_mort, "mimic_apr_mortality",
                                                                    "mimic_apr_mortality_v1")
        if "mimic_pcs_support_v1" in only:
            _log(f"{split}: pcs_support")
            index = mi.pcs_index(pcs_pool, split)
            results["mimic_pcs_support_v1"][split] = mi.pcs_support(ctx, split, quota, pcs_pool, index)
        if "mimic_rad_exam_v1" in only:
            _log(f"{split}: rad_exam")
            rad = data["rad"][data["rad"]["split"].eq(split) & data["rad"]["hadm_id"].notna()]
            reports = []
            for note_id, hadm, text in zip(rad["note_id"], rad["hadm_id"], rad["text"], strict=True):
                if str(note_id) not in exam_of:
                    continue
                exam = exam_of.get(note_id)
                if exam is None or int(hadm) not in adm.index:
                    continue
                reports.append((str(note_id), int(hadm), str(text), str(exam), bool(adm.at[int(hadm), "late_anchor"])))
            reports = mi.stable_order(reports, seed, f"rad:{split}", lambda r: r[0])
            results["mimic_rad_exam_v1"][split] = mi.rad_exam(ctx, split, quota, reports, names_by_mod)
        if "mimic_ed_acuity_v1" in only or "mimic_ed_disposition_v1" in only:
            triage = data["triage"]
            stays = data["edstays"]
            split_of_subject = {int(s): mimic.subject_split(int(s), mimic.MIMIC4_DATASET) for s in triage["subject_id"].unique()}
            t_split = triage[triage["subject_id"].map(split_of_subject).eq(split)]
            if "mimic_ed_acuity_v1" in only:
                _log(f"{split}: ed_acuity")
                rows = mi.stable_order(t_split.to_dict("records"), seed, f"ed_acuity:{split}", lambda r: r["stay_id"])
                results["mimic_ed_acuity_v1"][split] = mi.ed_acuity(ctx, split, quota, rows, data["late_of"])
            if "mimic_ed_disposition_v1" in only:
                _log(f"{split}: ed_disposition")
                s_split = stays[stays["subject_id"].map(split_of_subject).eq(split)]
                tri = t_split.set_index("stay_id")
                merged = []
                for stay in s_split.to_dict("records"):
                    if int(stay["stay_id"]) not in tri.index:
                        continue
                    row = {**stay, **tri.loc[int(stay["stay_id"])].to_dict()}
                    merged.append(row)
                merged = mi.stable_order(merged, seed, f"ed_disp:{split}", lambda r: r["stay_id"])
                dx_by_stay: dict[int, list[tuple[str, int, str]]] = defaultdict(list)
                for sid, code, version, title in zip(data["ed_dx"]["stay_id"], data["ed_dx"]["icd_code"],
                                                     data["ed_dx"]["icd_version"], data["ed_dx"]["icd_title"], strict=True):
                    dx_by_stay[int(sid)].append((str(code), int(version), str(title)))
                results["mimic_ed_disposition_v1"][split] = mi.ed_disposition(ctx, split, quota, merged, data["late_of"],
                                                                              dx_by_stay)
    if "mednli_v1" in only:
        for split in ("train", "dev", "test"):
            _log(f"mednli {split}")
            results["mednli_v1"][split] = mi.mednli(split, data["mednli"][split])
    return results


def screens(rows_by_split: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """The D35 screens on dev: Naive Bayes trained on train (macro accuracy) and the majority-label rate."""
    train, dev = rows_by_split.get("train", []), rows_by_split.get("dev", [])
    if not train or not dev:
        return {"status": "NOT MEASURED — no train or dev rows"}
    nb = NaiveBayes().fit(train)
    pairs = [(gold_label(r), nb.predict(r)) for r in dev]
    golds = Counter(gold_label(r) for r in dev)
    return {"nb_macro_accuracy_dev": round(macro_accuracy(pairs) or 0.0, 4),
            "majority_rate_dev": round(golds.most_common(1)[0][1] / len(dev), 4), "dev_items": len(dev),
            "majority_label": golds.most_common(1)[0][0]}


def write_rows(path: Path, rows: list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with path.open("w") as handle:
        for row in rows:
            line = json.dumps(row) + "\n"
            digest.update(line.encode("utf-8"))
            handle.write(line)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", nargs="*", default=list(TEMPLATES))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--physionet", type=Path, default=PHYS)
    args = parser.parse_args()
    os.environ["HF_HUB_OFFLINE"] = "1"
    OUT.mkdir(parents=True, exist_ok=True)
    only = set(args.only)
    _log("load")
    data = load(args.physionet)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)

    def count(text: str) -> int:
        return len(tokenizer(text, add_special_tokens=False)["input_ids"])

    results = build_all(data, only, args.seed, count)
    summary: dict[str, Any] = {"seed": args.seed, "quota": QUOTA, "templates": {}}
    for template, by_split in results.items():
        rows_by_split: dict[str, list[dict[str, Any]]] = {}
        entry: dict[str, Any] = {"splits": {}}
        kept, removed = mi.dedupe_across_splits({split: built.rows for split, built in by_split.items()})
        for split, built in by_split.items():
            built.rows = kept[split]
        entry["cross_split_duplicates_removed"] = removed
        for split, built in by_split.items():
            rows_by_split[split] = built.rows
            digest = write_rows(ITEMS / f"{template}__{split}.jsonl", built.rows)
            heldout_digest = None
            if built.heldout:
                heldout_digest = write_rows(ITEMS / f"{template}__heldout.jsonl", built.heldout)
            entry["splits"][split] = {
                "items": len(built.rows), "heldout_items": len(built.heldout),
                "gold_counts": dict(Counter(gold_label(r) for r in built.rows)),
                "drops": dict(built.drops), "shortage": dict(built.shortage), "truncated_states": built.truncated,
                "sha256": digest, "heldout_sha256": heldout_digest,
            }
        entry["screens"] = screens(rows_by_split)
        summary["templates"][template] = entry
        _log(f"{template}: {json.dumps(entry['screens'])}")
    write_json(OUT / "counts.json", summary)
    _log(f"wrote {OUT / 'counts.json'}")


if __name__ == "__main__":
    main()
