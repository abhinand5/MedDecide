"""osler_v3 X3: the real-text substrate for the claim and restyle generators (ADVISORY X3; D43).

Documents from four real sources, each with a split that every item made from it carries:
- MultiCaRe case texts (CC BY 4.0; cases.parquet at the pinned revision of scripts/osler_v3/x1_fetch.py), split by article;
- MTSamples transcriptions (`harishnair04/mtsamples`; the licence uncertainty the operator accepted, D32), split by document;
- CodiEsp TRAIN cases (CC BY 4.0) that osler_v2 trained on (its block C1; the 15 % carve-out of osler_v2 is its dev and has no
  training rows, so it is excluded by construction; counts reported);
- PMC-Patients patient summaries (CC BY-NC-SA 4.0 overall; the subset is recorded per row), split by patient.

Writes data/gen/osler_v3/x3_substrate.jsonl (gitignored) and outputs/osler_v3/X3/substrate.json (counts only; no text).

Usage:  source scripts/pod_env.sh && uv run --frozen python scripts/osler_v3/x3_substrate.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download, snapshot_download

from meddecide.mix.protected import PARAGRAPH_MIN_CHARS, paragraph_key, protected_paragraph_keys
from meddecide.mix.substrate import paragraphs, passage, split_of
from meddecide.mr.realtext import read_codiesp_split, read_mtsamples, read_pmc_patients
from meddecide.utils.io import read_json, write_json, write_jsonl
from meddecide.utils.provenance import git_commit, utcnow

REPO = Path(__file__).resolve().parents[2]
FETCH = REPO / "outputs" / "osler_v3" / "X1" / "fetch.json"
OUT_ROWS = REPO / "data" / "gen" / "osler_v3" / "x3_substrate.jsonl"
OUT = REPO / "outputs" / "osler_v3" / "X3" / "substrate.json"
MULTICARE_MIN_CHARS = 400


def multicare_rows() -> tuple[list[dict[str, Any]], Counter[str]]:
    counts: Counter[str] = Counter()
    fetched = next(r for r in read_json(FETCH)["files"] if r["source"] == "MultiCaRe")
    path = hf_hub_download(fetched["repo"], fetched["file"], repo_type="dataset", revision=fetched["revision"])
    table = pq.read_table(path, columns=["cases", "article_id"])
    rows = []
    for article_id, cases in zip(table.column("article_id").to_pylist(), table.column("cases").to_pylist(), strict=True):
        for case in cases:
            counts["cases_read"] += 1
            text = str(case.get("case_text") or "").strip()
            if len(text) < MULTICARE_MIN_CHARS:
                counts["dropped_text_under_400_chars"] += 1
                continue
            rows.append({"key": f"multicare:{case['case_id']}", "source": "multicare", "doc_id": str(case["case_id"]),
                         "split": split_of(str(article_id)), "text": text, "licence": "CC BY 4.0",
                         "record_url": f"https://www.ncbi.nlm.nih.gov/pmc/articles/{article_id}/"})
            counts[f"kept_{rows[-1]['split']}"] += 1
    return rows, counts


def mtsamples_rows() -> tuple[list[dict[str, Any]], Counter[str]]:
    counts: Counter[str] = Counter()
    folder = snapshot_download("harishnair04/mtsamples", repo_type="dataset", allow_patterns=["mtsamples.csv"])
    pages = read_mtsamples(Path(folder) / "mtsamples.csv", counts=counts)
    rows = []
    for page in pages:
        key = f"mtsamples:{page.doc_id}"
        rows.append({"key": key, "source": "mtsamples", "doc_id": page.doc_id, "split": split_of(key), "text": page.text,
                     "licence": "unrecorded (MTSamples; operator-accepted, D32)",
                     "record_url": "https://www.mtsamples.com/"})
        counts[f"kept_{rows[-1]['split']}"] += 1
    return rows, counts


def fastdev_v2_codiesp_cases() -> set[str]:
    """The CodiEsp case ids of osler_v2's fast dev v2 (its dev carve-out), read from the item ids (codiesp:<case>:<code>)."""
    cases: set[str] = set()
    for line in (REPO / "data" / "bench" / "osler_v2" / "fastdev_v2.jsonl").read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        parts = str(rec.get("source_record_id", "")).split(":")
        if len(parts) >= 3 and parts[1] == "codiesp":
            cases.add(parts[2])
    return cases


def c1_codiesp_cases() -> set[str]:
    """The CodiEsp case ids that osler_v2's training block C1 holds (its mr_codiesp rows). osler_v2's carve-out cases have no
    training rows, so the cases of C1 are exactly the 493 cases less the carve-out (419 cases; checked)."""
    cases: set[str] = set()
    with (REPO / "data" / "train" / "osler_v2" / "block_C1.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            if row["source"] == "mr_codiesp":
                parts = str(row["source_record_id"]).split(":")
                cases.add(parts[2] if len(parts) >= 3 else str(row["source_record_id"]))
    return cases


def codiesp_rows() -> tuple[list[dict[str, Any]], Counter[str]]:
    counts: Counter[str] = Counter()
    folder = snapshot_download("Biomedical-TeMU/CodiEsp_corpus", repo_type="dataset",
                               allow_patterns=["train/trainD.tsv", "train/text_files_en/*"])
    cases = read_codiesp_split(folder, "train")
    trained = c1_codiesp_cases()
    dev_cases = fastdev_v2_codiesp_cases()
    rows = []
    for case_id, text, _codes in cases:
        if case_id in dev_cases:
            counts["excluded_in_osler_v2_fast_dev_v2"] += 1
            continue
        if case_id not in trained:
            counts["excluded_not_in_osler_v2_training"] += 1
            continue
        rows.append({"key": f"codiesp:{case_id}", "source": "codiesp", "doc_id": str(case_id), "split": "train",
                     "text": text.strip(), "licence": "CC BY 4.0", "record_url": "https://temu.bsc.es/codiesp/"})
        counts["kept_train"] += 1
    counts["osler_v2_training_cases"] = len(trained)
    return rows, counts


def pmc_rows() -> tuple[list[dict[str, Any]], Counter[str]]:
    counts: Counter[str] = Counter()
    folder = snapshot_download("zhengyun21/PMC-Patients", repo_type="dataset", allow_patterns=["PMC-Patients.csv"])
    rows = []
    for patient in read_pmc_patients(Path(folder) / "PMC-Patients.csv"):
        counts["read"] += 1
        key = f"pmc:{patient.uid}"
        split = split_of(key)
        rows.append({"key": key, "source": "pmc_patients", "doc_id": str(patient.uid), "split": split,
                     "text": patient.text.strip(), "licence": f"CC BY-NC-SA 4.0 (subset {patient.licence_subset})",
                     "record_url": "https://pmc-patients.github.io/"})
        counts[f"kept_{split}"] += 1
    return rows, counts


def main() -> int:
    report: dict[str, Any] = {"built_utc": utcnow(), "git_commit": git_commit(REPO), "command": "scripts/osler_v3/x3_substrate.py",
                              "sources": {}}
    protected = protected_paragraph_keys(REPO)
    report["protected_paragraph_keys"] = len(protected)
    rows: list[dict[str, Any]] = []
    for name, build in (("multicare", multicare_rows), ("mtsamples", mtsamples_rows), ("codiesp", codiesp_rows),
                        ("pmc_patients", pmc_rows)):
        part, counts = build()
        clean = []
        for r in part:
            if any(paragraph_key(para) in protected for para in paragraphs(r["text"])
                   if len(para) >= PARAGRAPH_MIN_CHARS):
                counts["excluded_paragraph_in_protected_set"] += 1
                continue
            clean.append(r)
        usable = [r for r in clean if passage(r["text"], r["key"]) is not None]
        report["sources"][name] = {"documents": len(part), "usable_passage": len(usable), "counts": dict(counts),
                                   "by_split": dict(Counter(r["split"] for r in usable))}
        rows.extend(usable)
        print(f"{name}: {len(part)} documents, {len(usable)} with a usable passage, splits {report['sources'][name]['by_split']}",
              flush=True)
    write_jsonl(OUT_ROWS, rows)
    report["documents_total"] = len(rows)
    report["by_split"] = dict(Counter(r["split"] for r in rows))
    report["by_source_and_split"] = {f"{s}/{sp}": n for (s, sp), n in sorted(Counter((r["source"], r["split"]) for r in rows).items())}
    report["rows_file"] = str(OUT_ROWS.relative_to(REPO))
    write_json(OUT, report)
    print(f"substrate: {len(rows)} documents with a passage; splits {report['by_split']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
