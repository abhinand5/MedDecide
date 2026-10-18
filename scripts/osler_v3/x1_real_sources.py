"""osler_v3 X1: build the RAGTruth and DrugProt items and audit MultiCaRe (ADVISORY X1 steps 1-3; D42).

Reads only the files that scripts/osler_v3/x1_fetch.py pinned (outputs/osler_v3/X1/fetch.json). Writes:
- data/train/osler_v3/x1_sources_train.jsonl: the train items (RAGTruth train contexts; DrugProt train abstracts);
- data/bench/osler_v3/x1_dev.jsonl and x1_test.jsonl: the dev and test items (fast dev v3 draws from dev; test is additional);
- outputs/osler_v3/X1/build.json: every count, balance step and option set (aggregates only);
- outputs/osler_v3/X1/multicare_audit.json: MultiCaRe's structured fields and how often the text states them (aggregates only).

MultiCaRe builds no direct decision item: its only structured fields are age and sex, both stated in the case text in almost every
case (the audit measures how often), so none qualifies under ADVISORY X1 step 3. It remains the real-text substrate of X3.

Usage:  source scripts/pod_env.sh && uv run --frozen python scripts/osler_v3/x1_real_sources.py
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from meddecide.mix import drugprot, ragtruth
from meddecide.utils.io import read_json, write_json, write_jsonl
from meddecide.utils.provenance import git_commit, utcnow

REPO = Path(__file__).resolve().parents[2]
FETCH = REPO / "outputs" / "osler_v3" / "X1" / "fetch.json"
OUT = REPO / "outputs" / "osler_v3" / "X1"
TRAIN = REPO / "data" / "train" / "osler_v3" / "x1_sources_train.jsonl"
DEV = REPO / "data" / "bench" / "osler_v3" / "x1_dev.jsonl"
TEST = REPO / "data" / "bench" / "osler_v3" / "x1_test.jsonl"
AGE_STATED = re.compile(r"\b(\d{1,3})[- ]year[- ]old\b|\baged? (\d{1,3})\b|\b(\d{1,3}) years? old\b", re.IGNORECASE)
GENDERED = re.compile(r"\b(woman|man|female|male|girl|boy|she|he|her|his|lady|gentleman)\b", re.IGNORECASE)


def fetched_path(source: str, filename: str) -> Path:
    for row in read_json(FETCH)["files"]:
        if row["source"] == source and row["file"].endswith(filename):
            if row.get("cache_path_is_outside_repo"):
                from huggingface_hub import (
                    hf_hub_download,  # the cached file of the pinned revision
                )

                return Path(hf_hub_download(row["repo"], row["file"], repo_type="dataset", revision=row["revision"]))
            return REPO / row["file"]
    raise FileNotFoundError(f"{source}/{filename} is not in {FETCH}")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def build_ragtruth() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw = REPO / "data" / "raw" / "ragtruth" / ragtruth.COMMIT
    responses = read_jsonl(raw / "response.jsonl")
    sources = {str(s["source_id"]): s for s in read_jsonl(raw / "source_info.jsonl")}
    out, report = ragtruth.build(responses, sources)
    rows = [row for template in (ragtruth.GROUNDED, ragtruth.SENTENCE, ragtruth.HALLUC_TYPE) for row in out[template]]
    return rows, report


def build_drugprot() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    train = pq.read_table(fetched_path("DrugProt", "train.parquet")).to_pylist()
    validation = pq.read_table(fetched_path("DrugProt", "validation.parquet")).to_pylist()
    options = drugprot.relation_types(train + validation)
    rows_by_split, report = drugprot.build([("train", r) for r in train] + [("validation", r) for r in validation], options)
    rows = rows_by_split["train"] + rows_by_split["dev"] + rows_by_split["test"]
    return rows, report


def audit_multicare() -> dict[str, Any]:
    table = pq.read_table(fetched_path("MultiCaRe", "cases.parquet"), columns=["cases", "article_id"])
    counts: Counter[str] = Counter()
    for article in table.column("cases").to_pylist():
        counts["articles"] += 1
        for case in article:
            counts["cases"] += 1
            text = str(case["case_text"] or "")
            if case["age"] is not None:
                counts["age_known"] += 1
                match = AGE_STATED.search(text)
                if match:
                    counts["age_statement_found"] += 1
                    number = next(g for g in match.groups() if g)
                    counts["age_statement_equals_field"] += int(int(number) == int(case["age"]))
            if case["gender"] in ("Male", "Female"):
                counts["gender_male_or_female"] += 1
                counts["gender_term_in_text"] += int(bool(GENDERED.search(text)))
    counts_dict = dict(counts)
    counts_dict["age_known_share"] = round(counts["age_known"] / counts["cases"], 4)
    counts_dict["age_statement_share_of_age_known"] = round(counts["age_statement_found"] / counts["age_known"], 4)
    counts_dict["gender_term_share_of_gendered"] = round(counts["gender_term_in_text"] / counts["gender_male_or_female"], 4)
    return {"counts": counts_dict, "structured_fields": ["age", "gender"], "direct_items_built": 0,
            "decision": "none qualifies: age and sex are stated in the text in most cases (ADVISORY X1 step 3)"}


def main() -> None:
    rows_rt, report_rt = build_ragtruth()
    rows_dp, report_dp = build_drugprot()
    rows = sorted(rows_rt + rows_dp, key=lambda r: (r["template_id"], r["item_id"]))
    by_split = {split: [r for r in rows if r["split"] == split] for split in ("train", "dev", "test")}
    write_jsonl(TRAIN, by_split["train"])
    write_jsonl(DEV, by_split["dev"])
    write_jsonl(TEST, by_split["test"])
    multicare = audit_multicare()
    write_json(OUT / "multicare_audit.json", multicare)
    summary: dict[str, Any] = {}
    for template in sorted({r["template_id"] for r in rows}):
        summary[template] = {split: dict(Counter(r["gold"] for r in by_split[split] if r["template_id"] == template))
                             for split in by_split}
    write_json(OUT / "build.json", {
        "built_utc": utcnow(), "git_commit": git_commit(REPO), "command": "scripts/osler_v3/x1_real_sources.py",
        "files": {"train": str(TRAIN.relative_to(REPO)), "dev": str(DEV.relative_to(REPO)), "test": str(TEST.relative_to(REPO))},
        "rows": {split: len(v) for split, v in by_split.items()},
        "ragtruth": report_rt, "drugprot": report_dp, "multicare": multicare,
        "gold_by_template_and_split": summary,
        "rows_by_source": dict(Counter(r["source"] for r in rows)),
        "licences": {"ragtruth": ragtruth.LICENCE, "drugprot": drugprot.LICENCE, "multicare": "CC BY 4.0 (not built)"},
    })
    for split, items in by_split.items():
        print(f"{split}: {len(items)} rows; templates {dict(Counter(r['template_id'] for r in items))}", flush=True)


if __name__ == "__main__":
    main()
