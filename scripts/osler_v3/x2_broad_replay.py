"""osler_v3 X2: build the broad general replay from the pinned sources of scripts/osler_v3/x2_fetch_replay.py (ADVISORY X2; D36).

Train splits only. Each source goes through its converter in meddecide.mix.broad_replay (labels from the answer key or the score
of the correct option; rationales, annotations and formulas never read). Within a source, exact duplicates are dropped and counted.
MathQA's problems are AQuA-RAT's train problems in tokenised form (97.7 % match on normalised text), so AQuA-RAT's copy is kept
and a MathQA row is dropped when its normalised problem is already in AQuA-RAT; the MathQA problems AQuA-RAT lacks are kept. The
whole set is then checked against every protected set (meddecide.mix.protected); a hit stops the build before the training
file is written.

Writes data/train/osler_v3/replay_broad_train.jsonl (training rows; not committed) and outputs/osler_v3/X2/build.json (counts only).

Usage:  source scripts/pod_env.sh && uv run --frozen python scripts/osler_v3/x2_broad_replay.py
"""

from __future__ import annotations

import csv
import json
import sys
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

from meddecide.mix import broad_replay as br
from meddecide.mix.protected import hit_total, leakage_counts, load_protected
from meddecide.utils.io import read_json, write_json, write_jsonl
from meddecide.utils.provenance import git_commit, utcnow

REPO = Path(__file__).resolve().parents[2]
FETCH = REPO / "outputs" / "osler_v3" / "X2" / "fetch.json"
OUT = REPO / "outputs" / "osler_v3" / "X2" / "build.json"
TRAIN = REPO / "data" / "train" / "osler_v3" / "replay_broad_train.jsonl"
BIGBENCH_TASKS = ("snarks", "ruin_names", "sports_understanding", "causal_judgment", "date_understanding")


def fetched(source: str) -> dict[str, Any]:
    for row in read_json(FETCH)["files"]:
        if row["source"] == source:
            return row
    raise FileNotFoundError(f"{source} is not in {FETCH}")


def parquet_records(source: str) -> list[dict[str, Any]]:
    row = fetched(source)
    path = hf_hub_download(row["repo"], row["file"], repo_type="dataset", revision=row["revision"])
    return pq.read_table(path).to_pylist()


def social_iqa_records() -> tuple[list[dict[str, Any]], list[str]]:
    zip_path = REPO / fetched("social_iqa")["file"]
    with zipfile.ZipFile(zip_path) as archive:
        records = [json.loads(line) for line in archive.read("socialiqa-train-dev/train.jsonl").decode("utf-8").splitlines()
                   if line.strip()]
        labels = archive.read("socialiqa-train-dev/train-labels.lst").decode("utf-8").split()
    return records, labels


def cosmos_records() -> list[dict[str, Any]]:
    path = REPO / fetched("cosmos_qa")["file"]
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def math_qa_records() -> list[dict[str, Any]]:
    zip_path = REPO / fetched("math_qa")["file"]
    with zipfile.ZipFile(zip_path) as archive:
        return json.loads(archive.read("train.json").decode("utf-8"))


def main() -> int:
    counts: dict[str, Counter[str]] = {}
    rows_by_source: dict[str, list[dict[str, Any]]] = {}
    duplicates: dict[str, int] = {}

    def add(name: str, rows: list[dict[str, Any]], count: Counter[str]) -> None:
        kept, dropped = br.drop_duplicates(rows)
        count["dropped_duplicate_within_source"] += dropped
        counts[name] = count
        rows_by_source[name] = kept
        duplicates[name] = dropped

    add("strategyqa", *br.strategyqa_rows(parquet_records("strategyqa")))
    add("quartz", *br.quartz_rows(parquet_records("quartz")))
    social, labels = social_iqa_records()
    add("social_iqa", *br.social_iqa_rows(social, labels))
    add("cosmos_qa", *br.cosmos_qa_rows(cosmos_records()))
    aqua_records = parquet_records("aqua_rat")
    add("aqua_rat", *br.aqua_rat_rows(aqua_records))
    aqua_keys = frozenset(br.problem_key(str(r["question"])) for r in aqua_records)
    math_records = math_qa_records()
    math_rows, math_counts = br.math_qa_rows(math_records)
    math_rows, math_dropped = br.drop_rows_with_problem_in(math_rows, aqua_keys)
    math_counts["dropped_problem_also_in_aqua_rat"] = math_dropped
    add("math_qa", math_rows, math_counts)
    for task in BIGBENCH_TASKS:
        add(f"bigbench_{task}", *br.bigbench_rows(task, parquet_records(f"bigbench/{task}")))
    rows = sorted((r for name in rows_by_source for r in rows_by_source[name]), key=lambda r: (r["template_id"], r["item_id"]))
    text, bare, qualified = load_protected(REPO)
    leak = leakage_counts(rows, text, bare, qualified)
    report: dict[str, Any] = {
        "built_utc": utcnow(), "git_commit": git_commit(REPO), "command": "scripts/osler_v3/x2_broad_replay.py",
        "sources": {name: {"template": br.SOURCES[name][0], "licence": br.SOURCES[name][1], "counts": dict(counts[name]),
                           "kept_after_dedupe": len(rows_by_source[name])} for name in counts},
        "rows": len(rows), "distinct_sources": len(counts),
        "templates": dict(sorted(Counter(r["template_id"] for r in rows).items())),
        "math_qa_rows_dropped_as_aqua_rat_problems": math_dropped,
        "mathqa_problems_read": len(math_records),
        "mathqa_problems_matching_aqua_normalised": sum(1 for r in math_records if br.problem_key(str(r["Problem"])) in aqua_keys),
        "aqua_rat_problem_keys": len(aqua_keys),
        "leakage": leak, "fetch_file": str(FETCH.relative_to(REPO)),
    }
    write_json(OUT, report)
    if hit_total(leak):
        print(f"LEAKAGE: {json.dumps(leak, sort_keys=True)}", file=sys.stderr)
        return 1
    write_jsonl(TRAIN, rows)
    print(f"broad replay: {len(rows)} rows from {len(counts)} sources; leakage hits {hit_total(leak)}", flush=True)
    for name in counts:
        print(f"  {name}: kept {len(rows_by_source[name])} of {counts[name]['read']} read", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
