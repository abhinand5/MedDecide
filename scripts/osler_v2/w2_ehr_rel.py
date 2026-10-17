"""W2 step 2 (osler_v2 ADVISORY W2): EHR-Rel as ``ehr_rel_relatedness_v1`` score items, split by concept pair 80 / 10 / 10.

Downloads the two published TSVs from the EHR-Rel repository (Apache-2.0) into data/gen/osler_v2/ehr_rel/ (gitignored),
converts every line (counting every drop by reason), and writes:
- data/train/osler_v2/ehr_rel_train.jsonl (train rows, for the real-text block C1);
- data/gen/osler_v2/ehr_rel/ehr_rel_dev.jsonl and ehr_rel_test.jsonl (dev for fast dev v2, test for W5);
- outputs/osler_v2/W2/ehr_rel_manifest.json (counts, gold shares, majority baselines, source checksums; no item text).

Usage: source scripts/pod_env.sh && uv run --frozen python scripts/osler_v2/w2_ehr_rel.py
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

from meddecide.mix.ehr_rel import LEVELS, ehr_rel_row
from meddecide.utils.hashing import file_sha256
from meddecide.utils.io import write_json, write_jsonl
from meddecide.utils.provenance import git_commit, utcnow

REPO = Path(__file__).resolve().parents[2]
RAW = "https://raw.githubusercontent.com/babylonhealth/EHR-Rel/master/{name}"
SOURCES = {"RelB": "EHR-RelB.tsv", "RelA": "EHR-RelA.tsv"}  # RelB first: a pair in both keeps RelB (3 raters, the larger set)
GEN_DIR = REPO / "data" / "gen" / "osler_v2" / "ehr_rel"
TRAIN_OUT = REPO / "data" / "train" / "osler_v2" / "ehr_rel_train.jsonl"
OUT_MANIFEST = REPO / "outputs" / "osler_v2" / "W2" / "ehr_rel_manifest.json"


def fetch(name: str) -> bytes:
    GEN_DIR.mkdir(parents=True, exist_ok=True)
    local = GEN_DIR / name
    if not local.exists():
        with urllib.request.urlopen(RAW.format(name=name), timeout=120) as resp:
            local.write_bytes(resp.read())
    return local.read_bytes()


def main() -> None:
    rows_by_split: dict[str, list[dict[str, Any]]] = {"train": [], "dev": [], "test": []}
    report: dict[str, Any] = {"sources": {}}
    seen_pairs: set[str] = set()  # the pair key is order-free: a pair listed twice keeps its first row
    for subset, name in SOURCES.items():
        raw = fetch(name)
        reader = csv.DictReader(io.StringIO(raw.decode("utf-8")), delimiter="\t")
        reasons: Counter[str] = Counter()
        kept = 0
        for rec in reader:
            row, reason = ehr_rel_row(rec, subset)
            if row is not None:
                pair = row["source_record_id"].split("|", 1)[1]
                if pair in seen_pairs:
                    row, reason = None, "duplicate_pair"
                else:
                    seen_pairs.add(pair)
            reasons[reason] += 1
            if row is None:
                continue
            kept += 1
            rows_by_split[row["split"]].append(row)
        report["sources"][subset] = {"file": name, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
                                     "lines_kept": kept, "reasons": dict(reasons)}
        print(subset, dict(reasons), flush=True)

    splits: dict[str, Any] = {}
    for split, rows in rows_by_split.items():
        gold = Counter(r["gold"] for r in rows)
        majority = max(gold.values()) / len(rows) if rows else None
        splits[split] = {"n": len(rows), "gold_counts": {k: gold.get(k, 0) for k, _ in LEVELS},
                         "majority_baseline": None if majority is None else round(majority, 4),
                         "pairs": len({r["source_record_id"].split("|", 1)[1] for r in rows})}
        if split != "train":
            out = GEN_DIR / f"ehr_rel_{split}.jsonl"
            write_jsonl(out, rows)
    write_jsonl(TRAIN_OUT, rows_by_split["train"])
    total = sum(s["n"] for s in splits.values())
    report["splits"] = splits
    report["total_rows"] = total
    report["shares"] = {k: round(v["n"] / total, 4) for k, v in splits.items()}
    report["train_file"] = str(TRAIN_OUT.relative_to(REPO))
    report["train_sha256"] = file_sha256(TRAIN_OUT)
    report["written_utc"] = utcnow()
    report["git_commit"] = git_commit(REPO)
    report["rule"] = ("gold = mean rating rounded half-up on the paper's 0-3 scale; split by unordered concept pair, "
                      "stable hash, 8/1/1 of 10 buckets")
    write_json(OUT_MANIFEST, report)
    print(json.dumps({"total_rows": total, "splits": {k: v["n"] for k, v in splits.items()}}), flush=True)


if __name__ == "__main__":
    main()
