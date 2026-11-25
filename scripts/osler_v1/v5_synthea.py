"""V5a: Synthea emergency and inpatient encounters -> page-category and indicator items (osler_v1 ADVISORY V5a).

The Hub copy (`richardyoung/synthea-575k-patients`, MIT, about 140 GB) is never downloaded whole. ``extract`` streams
the encounters table from the start and keeps emergency and inpatient encounters until ``--encounters`` are found,
then streams the observations, medications, procedures and conditions tables from the start, keeping rows of those
encounters. Synthea writes each patient's records together, so matches cluster near the start of every table; a table
is stopped after ``--patience`` consecutive row groups without a match once matches have started (the coverage of the
chosen encounters is recorded, never assumed).

``build`` turns the extracted tables into items: ``synthea_page_category_v1`` and ``synthea_indicator_evidence_v1``,
split by patient (``split_of``), held-out indicators moved to heldout_test.jsonl, indicator labels balanced per
indicator by down-sampling the majority answer in train (counts recorded).

Usage:
    source scripts/pod_env.sh && uv run --frozen python scripts/osler_v1/v5_synthea.py extract --encounters 40000
    source scripts/pod_env.sh && uv run --frozen python scripts/osler_v1/v5_synthea.py build
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[2]
HUB = "datasets/richardyoung/synthea-575k-patients/data"
TABLES = {
    "observations": ["DATE", "PATIENT", "ENCOUNTER", "CODE", "VALUE", "UNITS"],
    "medications": ["START", "PATIENT", "ENCOUNTER", "CODE", "DESCRIPTION", "BASE_COST", "DISPENSES"],
    "procedures": ["START", "PATIENT", "ENCOUNTER", "CODE", "DESCRIPTION", "BASE_COST"],
    "conditions": ["START", "PATIENT", "ENCOUNTER", "CODE", "DESCRIPTION"],
}


def extract(args: argparse.Namespace) -> None:
    from huggingface_hub import HfFileSystem

    out = REPO / args.out
    out.mkdir(parents=True, exist_ok=True)
    fs = HfFileSystem()
    report: dict[str, object] = {"source": HUB, "target_encounters": args.encounters}
    with fs.open(f"{HUB}/patients.parquet", "rb") as fh:
        pq.write_table(pq.read_table(fh, columns=["Id", "BIRTHDATE", "GENDER"]), out / "patients.parquet")
    keep: list[pd.DataFrame] = []
    n = 0
    with fs.open(f"{HUB}/encounters.parquet", "rb") as fh:
        pf = pq.ParquetFile(fh)
        groups = 0
        for i in range(pf.metadata.num_row_groups):
            df = pf.read_row_group(i).to_pandas()
            groups += 1
            df = df[df["ENCOUNTERCLASS"].isin(["emergency", "inpatient"])]
            keep.append(df)
            n += len(df)
            if n >= args.encounters:
                break
    enc = pd.concat(keep).head(args.encounters)
    enc.to_parquet(out / "encounters.parquet")
    ids = set(enc["Id"])
    report["encounters"] = {"kept": len(enc), "row_groups_read": groups,
                            "by_class": dict(Counter(enc["ENCOUNTERCLASS"]))}
    for table, cols in TABLES.items():
        parts: list[pd.DataFrame] = []
        seen: set[str] = set()
        started, quiet, read = False, 0, 0
        with fs.open(f"{HUB}/{table}.parquet", "rb") as fh:
            pf = pq.ParquetFile(fh)
            for i in range(pf.metadata.num_row_groups):
                df = pf.read_row_group(i, columns=cols).to_pandas()
                read += 1
                df = df[df["ENCOUNTER"].isin(ids)]
                if len(df):
                    started, quiet = True, 0
                    parts.append(df)
                    seen |= set(df["ENCOUNTER"])
                elif started:
                    quiet += 1
                    if quiet >= args.patience:
                        break
                if read >= args.max_row_groups:
                    break
        frame = pd.concat(parts) if parts else pd.DataFrame(columns=cols)
        frame.to_parquet(out / f"{table}.parquet")
        report[table] = {"rows": len(frame), "row_groups_read": read, "row_groups_total": pf.metadata.num_row_groups,
                         "encounters_with_rows": len(seen), "coverage": round(len(seen) / max(1, len(ids)), 4)}
        print(table, report[table], flush=True)
    (out / "extract_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def build(args: argparse.Namespace) -> None:
    from meddecide.gen.core import split_of
    from meddecide.mr.generators import HELD_OUT_INDICATORS
    from meddecide.mr.synthea import build_encounter, indicator_items, page_category_items

    src = REPO / args.out
    out = REPO / args.items
    out.mkdir(parents=True, exist_ok=True)
    patients = {r["Id"]: r for r in pd.read_parquet(src / "patients.parquet").to_dict("records")}
    enc = pd.read_parquet(src / "encounters.parquet")
    grouped = {t: defaultdict(list) for t in TABLES}
    for t in TABLES:
        for r in pd.read_parquet(src / f"{t}.parquet").to_dict("records"):
            grouped[t][r["ENCOUNTER"]].append(r)
    counts: dict[str, int] = {}
    splits: dict[str, list[dict]] = {"train": [], "dev": [], "test": [], "heldout_test": []}
    for e in enc.to_dict("records"):
        p = patients.get(e["PATIENT"])
        if p is None:
            counts["dropped: patient missing"] = counts.get("dropped: patient missing", 0) + 1
            continue
        se = build_encounter(e, p, grouped["observations"][e["Id"]], grouped["medications"][e["Id"]],
                             grouped["procedures"][e["Id"]], grouped["conditions"][e["Id"]], counts)
        split = split_of(f"synthea-{se.patient_id}")
        items = page_category_items(se) + indicator_items(se)
        for it in items:
            row = it.as_dict()
            row["split"] = split
            row["patient_id"] = f"synthea-{se.patient_id}"
            held = row["meta"].get("indicator") in HELD_OUT_INDICATORS
            row["held_out"] = held
            if held:
                if split != "train":
                    splits["heldout_test"].append(row)
                else:
                    counts["dropped: held-out indicator in train"] = counts.get(
                        "dropped: held-out indicator in train", 0) + 1
            else:
                splits[split].append(row)
    rng = random.Random(0)
    balance: dict[str, dict[str, int]] = {}
    for s in ("train", "dev", "test"):
        rows = splits[s]
        by_ind: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
        other = []
        for r in rows:
            if r["generator"] == "synthea_indicator_evidence_v1":
                by_ind[r["meta"]["indicator"]][r["gold"]].append(r)
            else:
                other.append(r)
        kept = list(other)
        for ind, by_gold in sorted(by_ind.items()):
            yes, no = by_gold.get("yes", []), by_gold.get("no", [])
            m = min(len(yes), len(no))
            if m == 0:
                balance[f"{s}:{ind}"] = {"yes": len(yes), "no": len(no), "kept_each": 0}
                continue
            kept += rng.sample(yes, m) + rng.sample(no, m)
            balance[f"{s}:{ind}"] = {"yes": len(yes), "no": len(no), "kept_each": m}
        splits[s] = kept
    for s, rows in splits.items():
        with (out / f"{s}.jsonl").open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    manifest = {"counts": counts, "balance": balance,
                "rows": {s: len(r) for s, r in splits.items()},
                "by_generator": {s: dict(Counter(r["generator"] for r in rows)) for s, rows in splits.items()}}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest["rows"]), json.dumps(manifest["by_generator"]))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--encounters", type=int, default=40000)
    e.add_argument("--patience", type=int, default=30)
    e.add_argument("--max-row-groups", type=int, default=3000)
    e.add_argument("--out", default="data/synthea/osler_v1")
    b = sub.add_parser("build")
    b.add_argument("--out", default="data/synthea/osler_v1")
    b.add_argument("--items", default="data/gen/osler_v1_synthea")
    args = ap.parse_args()
    extract(args) if args.cmd == "extract" else build(args)


if __name__ == "__main__":
    main()
