"""Y2 stage 2: mimic_lab_validation_v1 (labevents) and mimic_med_given_v1 (eMAR and prescriptions), osler_v4 ADVISORY Y2.
Credentialed data: local only (D28, D37). Labels come from labevents values under fixed criteria and from eMAR events.

Writes private items under data/mimic_build/items/ and counts only to outputs/osler_v4/Y2/counts_lab_med.json.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
from transformers import AutoTokenizer

from meddecide.gen.screen import NaiveBayes, gold_label, macro_accuracy
from meddecide.mr import mimic_items as mi
from meddecide.utils.io import write_json

BUILD = Path("data/mimic_build")
ITEMS = BUILD / "items"
PHYS = Path("data/physionet")
OUT = Path("outputs/osler_v4/Y2")
TOKENIZER = "perplexity-ai/pplx-decider-flash-v1.2"
QUOTA = {"train": 40_000, "dev": 500, "test": 500}
SPLITS = ("train", "dev", "test")


def _log(message: str) -> None:
    print(time.strftime("%H:%M:%S"), message, flush=True)


def labels_to_items(dictionary: pd.DataFrame) -> dict[str, list[int]]:
    """analyte -> the itemids whose dictionary label is one of the accepted labels (each match logged)."""
    out: dict[str, list[int]] = {}
    for analyte, (labels, _, _) in mi.ANALYTES.items():
        named = dictionary[dictionary["label"].isin(labels)]
        hits = named[named["fluid"].eq("Blood")]          # the criteria are for blood values only (urine, fluid: excluded)
        out[analyte] = sorted(int(i) for i in hits["itemid"])
        _log(f"dictionary {analyte}: {len(hits)} blood itemids of {len(named)} named for labels {list(labels)}")
    return out


def read_labs(hosp: Path, item_ids: set[int], windows: pd.DataFrame) -> pd.DataFrame:
    """labevents rows for the wanted itemids whose charttime is inside the admission window (checked row by row)."""
    keep = []
    usecols = ["hadm_id", "itemid", "charttime", "valuenum", "valueuom", "ref_range_lower", "ref_range_upper"]
    for chunk in pd.read_csv(hosp / "labevents.csv.gz", usecols=usecols, chunksize=5_000_000,
                             dtype={"hadm_id": "Int64", "valueuom": str}):
        chunk = chunk[chunk["itemid"].isin(item_ids) & chunk["hadm_id"].notna() & chunk["valuenum"].notna()]
        if len(chunk):
            keep.append(chunk)
    labs = pd.concat(keep, ignore_index=True)
    labs["hadm_id"] = labs["hadm_id"].astype("int64")
    labs = labs.merge(windows, on="hadm_id", how="inner")
    inside = labs["charttime"].ge(labs["admittime"]) & labs["charttime"].le(labs["dischtime"])
    return labs[inside]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--physionet", type=Path, default=PHYS)
    parser.add_argument("--skip-labs", action="store_true")
    parser.add_argument("--skip-med", action="store_true")
    args = parser.parse_args()
    os.environ["HF_HUB_OFFLINE"] = "1"
    OUT.mkdir(parents=True, exist_ok=True)
    hosp = args.physionet / "mimiciv" / "3.1" / "hosp"
    adm = pd.read_parquet(BUILD / "admissions.parquet").set_index("hadm_id")
    windows = pd.read_csv(hosp / "admissions.csv.gz", usecols=["hadm_id", "admittime", "dischtime"])
    windows["hadm_id"] = windows["hadm_id"].astype("int64")
    patients = pd.read_csv(hosp / "patients.csv.gz", usecols=["subject_id", "gender"])
    sex_of_subject = dict(zip(patients["subject_id"], patients["gender"], strict=True))
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)

    def count(text: str) -> int:
        return len(tokenizer(text, add_special_tokens=False)["input_ids"])

    notes = pd.read_parquet(BUILD / "discharge_notes.parquet", columns=["note_id", "text"])
    dx = pd.read_parquet(BUILD / "dx_icd10.parquet", columns=["hadm_id", "icd_code"])
    coded: dict[int, set[str]] = defaultdict(set)
    for h, c in zip(dx["hadm_id"], dx["icd_code"].astype(str), strict=True):
        coded[int(h)].add(c)
    has_note = adm[adm["has_discharge_note"]]
    token_of = {str(n): int(t) for n, t in zip(has_note["discharge_note_id"], has_note["discharge_tokens"], strict=True)}
    ctx = mi.Ctx(notes=dict(zip(notes["note_id"], notes["text"], strict=True)), icd_titles={}, pcs_titles={}, count=count,
                 seed=0, token_of=token_of)
    sex = {int(h): str(sex_of_subject.get(int(s), "")) for h, s in zip(adm.index, adm["subject_id"], strict=True)}
    results: dict[str, dict[str, mi.Built]] = defaultdict(dict)
    counts: dict[str, Any] = {"quota": QUOTA, "templates": {}}

    if not args.skip_labs:
        dictionary = pd.read_csv(hosp / "d_labitems.csv.gz", usecols=["itemid", "label", "fluid"])
        analyte_items = labels_to_items(dictionary)
        item_to_analyte = {i: a for a, ids in analyte_items.items() for i in ids}
        wanted = set(item_to_analyte)
        _log(f"labevents: {len(wanted)} itemids")
        raw = read_labs(hosp, wanted, windows)
        _log(f"labevents inside the admission window: {len(raw)} rows")
        raw["analyte"] = raw["itemid"].map(item_to_analyte)
        unit_drop: Counter[str] = Counter()
        keep_rows = []
        for analyte, (_, _unit, accepted) in mi.ANALYTES.items():
            part = raw[raw["analyte"].eq(analyte)]
            ok = part["valueuom"].isin(accepted)
            unit_drop[analyte] = int((~ok).sum())
            keep_rows.append(part[ok])
        labs = pd.concat(keep_rows, ignore_index=True)
        labs["label"] = labs["itemid"].map(dict(zip(dictionary["itemid"], dictionary["label"], strict=True)))
        counts["lab_rows_inside_window"] = len(raw)
        counts["lab_unit_mismatch_dropped"] = dict(unit_drop)
        counts["lab_rows_kept"] = len(labs)
        pools_all: dict[str, dict[str, list]] = {s: defaultdict(list) for s in SPLITS}
        grouped = labs.sort_values(["hadm_id", "charttime"]).groupby(["analyte", "hadm_id"], sort=False)
        for (analyte, hadm), part in grouped:
            hadm = int(hadm)
            if hadm not in adm.index:
                continue
            split = str(adm.at[hadm, "split"])
            if split == "test" and not bool(adm.at[hadm, "late_anchor"]):
                continue
            values = [(str(t), str(lab), float(v), str(u), (None if pd.isna(lo) else float(lo)),
                       (None if pd.isna(hi) else float(hi)))
                      for t, lab, v, u, lo, hi in zip(part["charttime"], part["label"], part["valuenum"],
                                                      part["valueuom"], part["ref_range_lower"],
                                                      part["ref_range_upper"], strict=True)]
            pools_all[split][analyte].append((hadm, values))
        for split in SPLITS:
            _log(f"lab_validation {split}")
            results["mimic_lab_validation_v1"][split] = mi.lab_validation(ctx, split, QUOTA[split], pools_all[split],
                                                                          coded, sex)
    if not args.skip_med:
        emar = pd.read_csv(hosp / "emar.csv.gz", usecols=["hadm_id", "medication", "event_txt"],
                           dtype={"hadm_id": "Int64"})
        rx = pd.read_csv(hosp / "prescriptions.csv.gz", usecols=["hadm_id", "drug"], dtype={"hadm_id": "Int64"})
        given = emar[emar["hadm_id"].notna() & emar["event_txt"].eq("Administered") & emar["medication"].notna()]
        administered: dict[int, set[str]] = defaultdict(set)
        for h, m in zip(given["hadm_id"], given["medication"].astype(str), strict=True):
            administered[int(h)].add(m.strip().lower())
        ordered: dict[int, set[str]] = defaultdict(set)
        rx = rx[rx["hadm_id"].notna() & rx["drug"].notna()]
        for h, d in zip(rx["hadm_id"], rx["drug"].astype(str), strict=True):
            ordered[int(h)].add(d.strip().lower())
        counts["emar_administered_rows"] = len(given)
        counts["prescription_rows"] = len(rx)
        for split in SPLITS:
            pool = adm[adm["has_discharge_note"] & adm["split"].eq(split)].reset_index()
            recs = mi.stable_order(pool.to_dict("records"), 0, f"medpool:{split}", lambda r: r["hadm_id"])
            admissions = [(int(r["hadm_id"]), str(r["discharge_note_id"]), bool(r["late_anchor"]),
                           "" if pd.isna(r["principal_icd10"]) else str(r["principal_icd10"])) for r in recs]
            category_drugs: dict[str, set[str]] = defaultdict(set)
            for h, _, _, principal in admissions:
                if principal and h in administered:
                    category_drugs[principal[:3]].update(administered[h])
            _log(f"med_given {split}")
            results["mimic_med_given_v1"][split] = mi.med_given(
                ctx, split, QUOTA[split], admissions, administered, ordered,
                {k: sorted(v) for k, v in category_drugs.items()})
    for template, by_split in results.items():
        entry: dict[str, Any] = {"splits": {}}
        rows_by_split: dict[str, list[dict[str, Any]]] = {}
        kept, removed = mi.dedupe_across_splits({split: built.rows for split, built in by_split.items()})
        for split, built in by_split.items():
            built.rows = kept[split]
        entry["cross_split_duplicates_removed"] = removed
        for split, built in by_split.items():
            rows_by_split[split] = built.rows
            digest = _write(ITEMS / f"{template}__{split}.jsonl", built.rows)
            entry["splits"][split] = {"items": len(built.rows), "gold_counts": dict(Counter(gold_label(r) for r in built.rows)),
                                      "drops": dict(built.drops), "shortage": dict(built.shortage),
                                      "truncated_states": built.truncated, "sha256": digest}
        entry["screens"] = _screens(rows_by_split)
        counts["templates"][template] = entry
        _log(f"{template}: {json.dumps(entry['screens'])}")
    write_json(OUT / "counts_lab_med.json", counts)
    _log(f"wrote {OUT / 'counts_lab_med.json'}")


def _write(path: Path, rows: list[dict[str, Any]]) -> str:
    import hashlib

    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with path.open("w") as handle:
        for row in rows:
            line = json.dumps(row) + "\n"
            digest.update(line.encode("utf-8"))
            handle.write(line)
    return digest.hexdigest()


def _screens(rows_by_split: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    train, dev = rows_by_split.get("train", []), rows_by_split.get("dev", [])
    if not train or not dev:
        return {"status": "NOT MEASURED — no train or dev rows"}
    nb = NaiveBayes().fit(train)
    golds = Counter(gold_label(r) for r in dev)
    return {"nb_macro_accuracy_dev": round(macro_accuracy([(gold_label(r), nb.predict(r)) for r in dev]) or 0.0, 4),
            "majority_rate_dev": round(golds.most_common(1)[0][1] / len(dev), 4), "dev_items": len(dev)}


if __name__ == "__main__":
    main()
