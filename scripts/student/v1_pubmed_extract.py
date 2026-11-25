#!/usr/bin/env python
"""V4: pre-window PubMed records from a recorded subset of 2026 baseline files (train-only).

Parses each file in the subset with the fresh builder's own parser
(``meddecide.bench.fresh.pubmed.parse_pubmed_article``), keeps the records whose record date
(Entrez date, the builder's definition) falls in the pre-window ``[2023-01-01, 2026-02-28]``, and
writes one gzipped JSONL with the fields the V4 PubMed templates need.

The fresh builder's check-tag set covers only humans/animals/sex/pregnancy/rodents, so the
age-group check tags are read here from the same MeSH headings. Every drop is counted.

Usage: ``python scripts/student/v1_pubmed_extract.py --files-dir /workspace/tmp/pubmed_v1 \
          --out data/train/student_v1/sources/pubmed_prewindow.jsonl.gz``
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from meddecide.bench.fresh.pubmed import parse_pubmed_article  # noqa: E402

PRE_WINDOW_START = date(2023, 1, 1)
PRE_WINDOW_END = date(2026, 2, 28)  # strictly before the v0.2 window start (2026-03-01)
AGE_TAGS = {
    "Infant", "Child", "Child, Preschool", "Infant, Newborn", "Adolescent", "Adult",
    "Young Adult", "Middle Aged", "Aged", "Aged, 80 and over",
}


def process_file(path_str: str) -> dict:
    """Parse one baseline file; returns the kept records plus the counts for the report."""
    path = Path(path_str)
    counts: Counter = Counter()
    kept: list[dict] = []
    with gzip.open(path, "rb") as fh:
        for _event, element in ET.iterparse(fh, events=("end",)):
            if element.tag != "PubmedArticle":
                continue
            counts["articles_seen"] += 1
            citation = element.find("MedlineCitation")
            age_tags = []
            if citation is not None:
                for node in citation.findall("MeshHeadingList/MeshHeading/DescriptorName"):
                    if node.text and node.text.strip() in AGE_TAGS:
                        age_tags.append(node.text.strip())
            record = parse_pubmed_article(element)
            element.clear()
            if record is None:
                counts["unparsable_or_no_record_date_or_no_pubtype"] += 1
                continue
            if record.entrez_date < PRE_WINDOW_START:
                counts["before_pre_window"] += 1
                continue
            if record.entrez_date > PRE_WINDOW_END:
                counts["after_pre_window_excluded"] += 1
                continue
            counts["prewindow_kept"] += 1
            kept.append({
                "pmid": record.pmid,
                "record_date": record.entrez_date.isoformat(),
                "title": record.title,
                "abstract": record.abstract,
                "pub_types": record.pub_types,
                "mesh_major_topics": record.mesh_major_topics,
                "check_tags": record.check_tags,
                "age_tags": sorted(set(age_tags)),
                "journal": record.journal,
                "doi": record.doi,
            })
    return {
        "file": path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "bytes": path.stat().st_size,
        "counts": dict(counts),
        "records": kept,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--files-dir", type=Path, required=True)
    parser.add_argument("--file-glob", default="pubmed26n*.xml.gz")
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    files = sorted(args.files_dir.glob(args.file_glob))
    if not files:
        raise SystemExit(f"no files matching {args.file_glob} in {args.files_dir}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    manifest_files = []
    totals: Counter = Counter()
    year_counts: Counter = Counter()
    tag_counts: Counter = Counter()
    with gzip.open(args.out, "wt", encoding="utf-8") as out, ProcessPoolExecutor(args.workers) as pool:
        for result in pool.map(process_file, [str(p) for p in files]):
            totals.update(result["counts"])
            manifest_files.append({"name": result["file"], "sha256": result["sha256"],
                                   "bytes": result["bytes"], "counts": result["counts"]})
            for record in result["records"]:
                year_counts[record["record_date"][:4]] += 1
                for tag in record["check_tags"] + record["age_tags"]:
                    tag_counts[tag] += 1
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
    report = {
        "kind": "pubmed_prewindow_extract",
        "pre_window": [PRE_WINDOW_START.isoformat(), PRE_WINDOW_END.isoformat()],
        "date_rule": "entrez_date (meddecide.bench.fresh.pubmed.parse_pubmed_article)",
        "n_files": len(files),
        "files": manifest_files,
        "totals": dict(totals),
        "kept_by_record_year": dict(sorted(year_counts.items())),
        "tag_counts_in_kept": dict(tag_counts.most_common()),
        "output": str(args.out),
    }
    report_path = args.out.with_name(args.out.name.replace(".jsonl.gz", "_report.json"))
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"files": len(files), **dict(totals)}, sort_keys=True))


if __name__ == "__main__":
    main()
