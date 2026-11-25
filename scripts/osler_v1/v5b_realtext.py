"""V5b: real clinical text with structured labels (osler_v1 ADVISORY V5b; D27, D32).

Subcommands:
  fetch   download the public sources into the Hugging Face cache (never into the repo)
  build   write the items, the splits, the drop counts, the licence record and the manifest

Sources and uses (ADVISORY V5b):
- MTSamples (`harishnair04/mtsamples`; licence uncertain, accepted by the operator under D32): page-category items.
  Split by document hash (`split_of`).
- CodiEsp (`Biomedical-TeMU/CodiEsp_corpus`, CC BY 4.0; machine-translated English text): ICD-10-CM support items over the
  FY2026 order file. Corpus train -> our train; corpus dev -> our test (its own test set has no codes and is unused).
- PMC-Patients (`zhengyun21/PMC-Patients`; CC BY-NC-SA 4.0 overall, the operator chose all of it; the licence subset is kept
  per row): similar-patient decisions over at most 40,000 anchor patients (the first 40,000 rows). Split by the anchor. Train
  pairs whose other patient is a dev or test anchor, or appears in a dev or test pair, are dropped (counted).
- Evaluation only, never trained on: mkurman's commercial clinical-case set (`mkurman/clinical-case-icd10-diagnosis`, CC BY
  4.0 rows; the three CC BY-ND rows are dropped, since no-derivatives does not allow building items from them) and the NHS
  synthetic notes (`NHSEDataScience/synthetic_clinical_notes`): NOT MEASURED unless the note type maps cleanly to the page
  classes (timebox 1 h; it does not, see the manifest).

Outputs (gitignored): data/gen/osler_v1/realtext/*.jsonl and manifest.json; outputs/osler_v1/V5b/report.json.

Usage:
    source scripts/pod_env.sh && uv run --frozen python scripts/osler_v1/v5b_realtext.py fetch
    source scripts/pod_env.sh && uv run --frozen python scripts/osler_v1/v5b_realtext.py build
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from huggingface_hub import snapshot_download

from meddecide.gen.core import split_of
from meddecide.mr import icd
from meddecide.mr.realtext import (
    codiesp_items,
    page_item,
    pmc_similar_items,
    read_codiesp_split,
    read_mtsamples,
    read_pmc_patients,
)
from meddecide.utils.hashing import file_sha256
from meddecide.utils.io import write_json
from meddecide.utils.provenance import git_commit, utcnow

REPO = Path(__file__).resolve().parents[2]
ORDER_FILE = REPO / "data" / "cms" / "icd10cm" / "2026" / "icd10cm_order_2026.txt"
OUT_DATA = REPO / "data" / "gen" / "osler_v1" / "realtext"
OUT_REPORT = REPO / "outputs" / "osler_v1" / "V5b" / "report.json"
PMC_ANCHORS = 40_000
MKURMAN_BYND = "CC BY-ND 4.0"


def fetch() -> dict[str, str]:
    paths = {
        "mtsamples": snapshot_download("harishnair04/mtsamples", repo_type="dataset", allow_patterns=["mtsamples.csv"]),
        "codiesp": snapshot_download("Biomedical-TeMU/CodiEsp_corpus", repo_type="dataset",
                                     allow_patterns=["train/trainD.tsv", "train/text_files_en/*", "dev/devD.tsv",
                                                     "dev/text_files_en/*"]),
        "pmc": snapshot_download("zhengyun21/PMC-Patients", repo_type="dataset",
                                 allow_patterns=["PMC-Patients.csv"]),
        "mkurman": snapshot_download("mkurman/clinical-case-icd10-diagnosis", repo_type="dataset",
                                     allow_patterns=["data/commercial/train-*.parquet"]),
        "nhs": snapshot_download("NHSEDataScience/synthetic_clinical_notes", repo_type="dataset",
                                 allow_patterns=["silver/synthetic_clinical_notes.csv"]),
    }
    return {k: str(v) for k, v in paths.items()}


def write_split_rows(rows_by_key: dict[str, list[dict[str, Any]]]) -> dict[str, dict[str, int]]:
    OUT_DATA.mkdir(parents=True, exist_ok=True)
    counts: dict[str, dict[str, int]] = {}
    for key, rows in rows_by_key.items():
        path = OUT_DATA / f"{key}.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        counts[key] = dict(Counter(r["split"] for r in rows))
    return counts


def build() -> None:
    paths = fetch()
    order = icd.load_order_file(ORDER_FILE)
    drops: dict[str, Counter[str]] = {k: Counter() for k in ("mtsamples", "codiesp", "pmc", "mkurman", "nhs")}
    groups: dict[str, list[dict[str, Any]]] = {}

    # MTSamples: page category, split by document hash
    pages = read_mtsamples(Path(paths["mtsamples"]) / "mtsamples.csv", counts=drops["mtsamples"])
    mts = [page_item(p, "mtsamples_page_category_v1").as_dict() for p in pages]
    for row in mts:
        row["split"] = split_of(row["pair_id"].split(":", 1)[1])
    groups["mtsamples_page_category_v1"] = mts
    per_class = Counter(p.page_class for p in pages)

    # CodiEsp: train -> train, dev -> test (the corpus test set has no codes)
    codi: list[dict[str, Any]] = []
    for corpus_split, our_split in (("train", "train"), ("dev", "test")):
        for case_id, text, codes in read_codiesp_split(paths["codiesp"], corpus_split):
            for item in codiesp_items(case_id, text, codes, order, counts=drops["codiesp"]):
                row = item.as_dict()
                row["split"] = our_split
                codi.append(row)
    groups["codiesp_icd_support_v1"] = codi

    # PMC-Patients: first 40,000 anchors; split by anchor; train pairs touching a dev or test patient are dropped
    everyone = list(read_pmc_patients(Path(paths["pmc"]) / "PMC-Patients.csv"))
    known = {p.uid for p in everyone}
    anchor_uids: set[str] = set()
    for p in everyone:  # the first PMC_ANCHORS patients (file order) that link to another patient in the file
        if len(anchor_uids) >= PMC_ANCHORS:
            break
        if any(u in known for u in p.similar):
            anchor_uids.add(p.uid)
    pmc_items = [it.as_dict() for it in pmc_similar_items(everyone, seed=0, counts=drops["pmc"],
                                                          anchor_uids=anchor_uids)]
    for row in pmc_items:
        row["split"] = split_of(row["pair_id"].split(":", 1)[1])
    dev_test_anchors = {r["meta"]["pair_uids"][0] for r in pmc_items if r["split"] in ("dev", "test")}
    dev_test_pairs = {u for r in pmc_items if r["split"] in ("dev", "test") for u in r["meta"]["pair_uids"]}
    kept_pmc: list[dict[str, Any]] = []
    for row in pmc_items:
        if row["split"] == "train":
            other = row["meta"]["pair_uids"][1]
            if other in dev_test_anchors or other in dev_test_pairs:
                drops["pmc"]["dropped: train pair touches a dev or test patient"] += 1
                continue
        kept_pmc.append(row)
    groups["pmc_similar_patient_v1"] = kept_pmc
    drops["pmc"]["patients read"] = len(everyone)
    drops["pmc"]["anchors selected"] = len(anchor_uids)

    # evaluation only: mkurman (CC BY 4.0 rows; CC BY-ND rows dropped) -> test split; NHS -> NOT MEASURED
    mku_rows = pq.read_table(Path(paths["mkurman"]) / "data" / "commercial" / "train-00000-of-00001.parquet").to_pylist()
    mku: list[dict[str, Any]] = []
    for r in mku_rows:
        if r["license"] == MKURMAN_BYND:
            drops["mkurman"]["dropped: CC BY-ND 4.0 row"] += 1
            continue
        for item in codiesp_items(str(r["case_id"]), str(r["input"]), [str(r["icd10_code"]).upper()], order,
                                  counts=drops["mkurman"], template="mkurman_icd_support_v1", source="mkurman"):
            row = item.as_dict()
            row["split"] = "test"
            mku.append(row)
    groups["mkurman_icd_support_v1"] = mku
    nhs_note_types = Counter()
    with open(Path(paths["nhs"]) / "silver" / "synthetic_clinical_notes.csv", encoding="utf-8-sig", newline="") as fh:
        for r in csv.DictReader(fh):
            nhs_note_types[r["note_type"]] += 1
    nhs_status = {
        "status": "NOT MEASURED",
        "reason": ("note_type is a department or care-setting label (for example 'Medicine Inpatients', 'Physiotherapy "
                   "Documentation', 'Pre-op Consent'); it does not map cleanly to the twelve page classes"),
        "note_types": len(nhs_note_types),
        "rows": sum(nhs_note_types.values()),
    }

    counts = write_split_rows(groups)
    manifest = {
        "written_utc": utcnow(), "git_commit": git_commit(REPO),
        "sources": {
            "mtsamples": {"repo": "harishnair04/mtsamples", "licence": "uncertain; accepted by the operator (D32)",
                          "file_sha256": file_sha256(Path(paths["mtsamples"]) / "mtsamples.csv")},
            "codiesp": {"repo": "Biomedical-TeMU/CodiEsp_corpus", "licence": "CC BY 4.0",
                        "text": "machine-translated English (text_files_en)"},
            "pmc": {"repo": "zhengyun21/PMC-Patients", "licence": "CC BY-NC-SA 4.0 overall (operator choice D32); subset per row",
                    "anchors": PMC_ANCHORS, "file_sha256": file_sha256(Path(paths["pmc"]) / "PMC-Patients.csv")},
            "mkurman": {"repo": "mkurman/clinical-case-icd10-diagnosis", "licence": "CC BY 4.0 rows; CC BY-ND 4.0 rows dropped",
                        "use": "evaluation only (test split)"},
            "nhs": nhs_status,
        },
        "drops": {k: dict(v) for k, v in drops.items()},
        "rows": counts,
        "pages_per_class_mtsamples": dict(sorted(per_class.items())),
        "file_sha256": {f"{k}.jsonl": file_sha256(OUT_DATA / f"{k}.jsonl") for k in groups},
        "order_file_sha256": file_sha256(ORDER_FILE),
    }
    write_json(OUT_DATA / "manifest.json", manifest)
    write_json(OUT_REPORT, manifest)
    print(json.dumps({"rows": counts, "drops": {k: dict(v) for k, v in drops.items()}}, indent=1)[:3000], flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("fetch", "build"))
    args = parser.parse_args()
    if args.command == "fetch":
        print(json.dumps(fetch(), indent=1), flush=True)
    else:
        build()


if __name__ == "__main__":
    main()
