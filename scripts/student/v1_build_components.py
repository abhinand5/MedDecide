#!/usr/bin/env python
"""V4: build the new pre-window training components (ClinicalTrials.gov, openFDA, PubMed).

Inputs are cached raw records, never fetched here:
  ClinicalTrials.gov  /workspace/tmp/s6/ct_raw/*.jsonl.gz     (monthly slices, first-posted dates)
  openFDA labels      /workspace/tmp/s6/fda_raw/*.jsonl.gz    (monthly slices, effective dates)
  PubMed              data/train/student_v1/sources/pubmed_prewindow.jsonl.gz (V4 extraction)

Each record is checked against the pre-window cut (2023-01-01 .. 2026-02-28) before any template
sees it. Records outside it are counted as ``outside_prewindow`` and never built.

Writes ``data/train/student_v1/components/<source>_v1.jsonl`` (items, train and dev) and
``outputs/student_v1/V4/components_report.json`` (counts and drops by template and reason).
"""

from __future__ import annotations

import argparse
import gzip
import json
import subprocess
import sys
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from meddecide.bench.fresh.clinicaltrials import _record_date  # noqa: E402
from meddecide.bench.fresh.openfda import _effective_date  # noqa: E402
from meddecide.bench.fresh.pubmed import PubmedRecord  # noqa: E402
from meddecide.train.templates_v1 import (  # noqa: E402
    build_ct_items,
    build_openfda_items,
    build_pubmed_items,
)
from meddecide.utils.hashing import stable_hash  # noqa: E402

PRE_START = date(2023, 1, 1)
PRE_END = date(2026, 2, 28)
CT_RAW = Path("/workspace/tmp/s6/ct_raw")
FDA_RAW = Path("/workspace/tmp/s6/fda_raw")
PUBMED_EXTRACT = ROOT / "data" / "train" / "student_v1" / "sources" / "pubmed_prewindow.jsonl.gz"
COMPONENTS = ROOT / "data" / "train" / "student_v1" / "components"
REPORT = ROOT / "outputs" / "student_v1" / "V4" / "components_report.json"

CT_TEMPLATES = ["ct_masking_choice_v1", "ct_eligibility_sex_choice_v1",
                "ct_enrollment_band_score_v1", "ct_interventional_noul_v1"]
FDA_TEMPLATES = ["fda_application_family_choice_v1", "fda_product_type_noul_v1"]
PUBMED_TEMPLATES = ["pubmed_check_female_noul_v1", "pubmed_check_male_noul_v1",
                    "pubmed_check_adult_noul_v1", "pubmed_check_child_noul_v1",
                    "pubmed_pubtype_choice_v1"]


def read_gz_jsonl(paths: list[Path]):
    for path in paths:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)


def in_prewindow(day: date | None) -> bool:
    return day is not None and PRE_START <= day <= PRE_END


def write_items(path: Path, items_by_template: dict[str, list[Any]]) -> dict[str, dict[str, int]]:
    """Write all items to one JSONL (deterministic order) and return train/dev counts."""
    rows = [item for tid in sorted(items_by_template) for item in items_by_template[tid]]
    rows.sort(key=lambda i: i.item_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for item in rows:
            handle.write(item.model_dump_json() + "\n")
    counts: dict[str, dict[str, int]] = {}
    for tid, items in items_by_template.items():
        per = Counter(str(i.split) for i in items)
        counts[tid] = {"train": per.get("train", 0), "dev": per.get("dev", 0)}
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ct-raw", type=Path, default=CT_RAW)
    parser.add_argument("--fda-raw", type=Path, default=FDA_RAW)
    parser.add_argument("--pubmed", type=Path, default=PUBMED_EXTRACT)
    parser.add_argument("--pubmed-sample-percent", type=int, default=2,
                        help="keep this percentage of PubMed records, chosen by a stable hash of the PMID")
    args = parser.parse_args()

    report: dict[str, Any] = {"kind": "v1_components", "window": [PRE_START.isoformat(), PRE_END.isoformat()]}

    # ClinicalTrials.gov
    ct_files = sorted(args.ct_raw.glob("ct_*.jsonl.gz"))
    outside = Counter()

    def ct_records():
        for study in read_gz_jsonl(ct_files):
            if not in_prewindow(_record_date(study)):
                outside["ct_outside_prewindow"] += 1
                continue
            yield study

    ct_items, ct_drops = build_ct_items(ct_records(), CT_TEMPLATES)
    ct_counts = write_items(COMPONENTS / "clinicaltrials_v1.jsonl", ct_items)
    report["clinicaltrials"] = {"files": [p.name for p in ct_files], "outside": dict(outside),
                                "counts": ct_counts, "drops": ct_drops}

    # openFDA
    fda_files = sorted(args.fda_raw.glob("fda_*.jsonl.gz"))
    outside = Counter()

    def fda_records():
        for label in read_gz_jsonl(fda_files):
            if not in_prewindow(_effective_date(label)):
                outside["fda_outside_prewindow"] += 1
                continue
            yield label

    fda_items, fda_drops = build_openfda_items(fda_records(), FDA_TEMPLATES)
    fda_counts = write_items(COMPONENTS / "openfda_v1.jsonl", fda_items)
    report["openfda"] = {"files": [p.name for p in fda_files], "outside": dict(outside),
                         "counts": fda_counts, "drops": fda_drops}

    # PubMed (extraction already applied the pre-window cut and counted it)
    age_tags: dict[str, list[str]] = {}

    sampled = Counter()

    def pubmed_records():
        for row in read_gz_jsonl([args.pubmed]):
            day = date.fromisoformat(row["record_date"])
            if not in_prewindow(day):
                raise ValueError(f"PubMed record {row['pmid']} dated {day} is outside the pre-window")
            sampled["read"] += 1
            bucket = int(stable_hash({"pmid": row["pmid"], "salt": "v1_pubmed_subsample"}, length=8), 16)
            if bucket / 16**8 * 100 >= args.pubmed_sample_percent:
                continue
            sampled["kept_by_subsample"] += 1
            age_tags[row["pmid"]] = row.get("age_tags", [])
            yield PubmedRecord(
                pmid=row["pmid"], entrez_date=day, title=row["title"], abstract=row["abstract"],
                pub_types=row["pub_types"], mesh_major_topics=row["mesh_major_topics"],
                check_tags=row["check_tags"], journal=row["journal"], doi=row.get("doi"),
            )

    pm_items, pm_drops = build_pubmed_items(pubmed_records(), PUBMED_TEMPLATES, age_tags=age_tags)
    pm_counts = write_items(COMPONENTS / "pubmed_v1.jsonl", pm_items)
    report["pubmed"] = {"source": str(args.pubmed.relative_to(ROOT)), "counts": pm_counts,
                        "drops": pm_drops, "subsample": dict(sampled),
                        "subsample_percent": args.pubmed_sample_percent,
                        "subsample_rule": "stable_hash({pmid, salt: v1_pubmed_subsample}) bucket < percent"}

    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, OSError):
        head = "unknown"
    report["git_commit"] = head
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    totals = {tid: sum(c.values()) for source in (ct_counts, fda_counts, pm_counts) for tid, c in source.items()}
    print(json.dumps(totals, sort_keys=True))


if __name__ == "__main__":
    main()
