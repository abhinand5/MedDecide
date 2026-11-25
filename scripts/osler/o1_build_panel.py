"""O1: build the external clinical panel (eval-only) from pinned public sets, with counted drops.

For each set: download the test files at the pinned revision, rank rows by a stable hash of their
record key (so the subsample is deterministic and needs no seed), convert in that order, and keep the
first ``N_MAX`` valid decisions. Every examined row is either accepted or dropped with a named reason;
rows after the quota are recorded as not examined. Nothing is dropped silently.

Outputs (gitignored, under data/): data/bench/v0.3_ext/panel.jsonl and its section of manifest.json.

Usage:
    source scripts/pod_env.sh && uv run --frozen python scripts/osler/o1_build_panel.py
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
from huggingface_hub import hf_hub_download

from meddecide.panel import converters as C
from meddecide.panel.overlap import item_text_key, row_text_key
from meddecide.panel.quota import take_valid as take_valid_quota
from meddecide.panel.schema import EvalItem
from meddecide.utils.hashing import file_sha256, stable_hash
from meddecide.utils.io import write_json
from meddecide.utils.provenance import git_commit, utcnow

REPO = Path(__file__).resolve().parents[2]
_MEDCONCEPTS: dict[str, Any] = {}
OUT_DIR = REPO / "data/bench/v0.3_ext"
N_MAX = 1000
MEDEXQA_SPECIALTIES = ["biomedical_engineer", "clinical_laboratory_scientist", "clinical_psychologist",
                       "occupational_therapist", "speech_pathologist"]


def fetch(spec: C.SourceSpec, filename: str) -> tuple[Path, str]:
    """Pinned download through the HF cache; returns the local path and its sha256."""
    path = Path(hf_hub_download(spec.repo, filename, revision=spec.revision, repo_type="dataset"))
    return path, file_sha256(path)


def rows_mmlu(filters: dict[str, int]) -> tuple[list[tuple[str, Any]], dict[str, Any]]:
    path, digest = fetch(C.MMLU_PRO, "data/test-00000-of-00001.parquet")
    df = pd.read_parquet(path)
    filters["category_not_health"] = int((df["category"] != "health").sum())
    health = df[df["category"] == "health"]
    rows = [(str(r["question_id"]), r) for r in health.to_dict("records")]
    return rows, {"file": "data/test-00000-of-00001.parquet", "sha256": digest, "rows_in_file": len(df)}


def rows_medxpertqa() -> tuple[list[tuple[str, Any]], dict[str, Any]]:
    path, digest = fetch(C.MEDXPERTQA_TEXT, "Text/test.jsonl")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [(str(r["id"]), r) for r in rows], {"file": "Text/test.jsonl", "sha256": digest, "rows_in_file": len(rows)}


def rows_medexpqa() -> tuple[list[tuple[str, Any]], dict[str, Any]]:
    path, digest = fetch(C.MEDEXPQA_EN, "data/en/test.en.casimedicos.rag.jsonl")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [(str(r["id"]), r) for r in rows], {"file": "data/en/test.en.casimedicos.rag.jsonl",
                                               "sha256": digest, "rows_in_file": len(rows)}


def rows_medconceptsqa() -> tuple[list[tuple[str, Any]], dict[str, Any]]:
    shards = ["all/test-00000-of-00002.parquet", "all/test-00001-of-00002.parquet"]
    frames, digests, total = [], {}, 0
    for name in shards:
        path, digest = fetch(C.MEDCONCEPTSQA, name)
        frame = pd.read_parquet(path)
        frames.append(frame)
        digests[name] = digest
        total += len(frame)
    df = pd.concat(frames, ignore_index=True)
    _MEDCONCEPTS["df"] = df
    rows = [(str(qid), idx) for idx, qid in enumerate(df["question_id"].astype(str).tolist())]
    return rows, {"files": digests, "rows_in_files": total}


def rows_medexqa() -> tuple[list[tuple[str, Any]], dict[str, Any]]:
    rows: list[tuple[str, Any]] = []
    digests: dict[str, str] = {}
    for specialty in MEDEXQA_SPECIALTIES:
        name = f"test/{specialty}_test.tsv"
        path, digest = fetch(C.MEDEXQA, name)
        digests[name] = digest
        with path.open(encoding="utf-8") as fh:
            for line_no, line in enumerate(fh):
                fields = line.rstrip("\r\n").split("\t")
                rows.append((f"{specialty}:{line_no}", (specialty, line_no, fields)))
    return rows, {"files": digests, "rows_in_files": len(rows)}


def rows_symptom() -> tuple[list[tuple[str, Any]], dict[str, Any]]:
    path, digest = fetch(C.SYMPTOM_DIAGNOSIS, "test.jsonl")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    classes = sorted({str(r["output_text"]).strip() for r in rows})
    keyed = [(stable_hash(str(r["input_text"]), length=32), (r, classes)) for r in rows]
    return keyed, {"file": "test.jsonl", "sha256": digest, "rows_in_file": len(rows),
                   "n_classes": len(classes), "classes_sorted": classes}


def rows_pairs() -> tuple[list[tuple[str, Any]], dict[str, Any]]:
    path, digest = fetch(C.QUESTION_PAIRS, "data/test-00000-of-00001.parquet")
    df = pd.read_parquet(path)
    rows = [(str(r["id"]), r) for r in df.to_dict("records")]
    return rows, {"file": "data/test-00000-of-00001.parquet", "sha256": digest, "rows_in_file": len(df)}


def convert_one(name: str, raw: Any) -> EvalItem:
    if name == "mmlu_pro_health":
        return C.mmlu_pro_health(raw)
    if name == "medxpertqa_text":
        return C.medxpertqa_text(raw)
    if name == "medexpqa_en":
        return C.medexpqa_en(raw)
    if name == "medconceptsqa_all":
        return C.medconceptsqa(_MEDCONCEPTS["df"].iloc[raw].to_dict())
    if name == "medexqa_test":
        specialty, line_no, fields = raw
        return C.medexqa_row(fields, specialty, line_no)
    if name == "symptom_to_diagnosis":
        row, classes = raw
        return C.symptom_diagnosis(row, classes)
    if name == "medical_question_pairs":
        return C.question_pair(raw)
    raise KeyError(name)


def v02_text_keys() -> set[str]:
    """Text keys of every v0.2 item (all splits). A panel item with one of these keys is a benchmark item."""
    keys: set[str] = set()
    manifest = json.loads((REPO / "data/bench/v0.2/manifest.json").read_text())
    for rec in manifest["files"].values():
        for line in (REPO / rec["path"]).open(encoding="utf-8"):
            keys.add(row_text_key(json.loads(line)))
    return keys


def take_valid(name: str, rows: list[tuple[str, Any]], n_max: int,
               exclude: set[str]) -> tuple[list[EvalItem], dict[str, Any]]:
    """Rank rows by stable hash of their key; convert in that order; keep the first ``n_max`` valid items
    whose text is not already a v0.2 item (counted as ``duplicate_of_v0_2_item``)."""
    ranked = sorted(rows, key=lambda kv: stable_hash(kv[0], length=32))
    result = take_valid_quota(
        [raw for _, raw in ranked], lambda raw: convert_one(name, raw), n_max,
        exclude_keys=exclude, key_of=item_text_key,
    )
    summary = result.summary(n_max)
    summary["sampling_rule"] = "rank by stable_hash(record key, 32 hex); accept first valid rows up to the quota"
    summary["excluded_if_text_equals_a_v0_2_item"] = True
    if summary["accepted"] + summary["dropped"] + summary["not_examined_after_quota"] != summary["population"]:
        raise RuntimeError(f"{name}: row accounting does not close")
    return result.accepted, summary


def build() -> dict[str, Any]:
    filters: dict[str, int] = {}
    loaders: dict[str, Callable[[], tuple[list[tuple[str, Any]], dict[str, Any]]]] = {
        "mmlu_pro_health": lambda: rows_mmlu(filters),
        "medxpertqa_text": rows_medxpertqa,
        "medexpqa_en": rows_medexpqa,
        "medconceptsqa_all": rows_medconceptsqa,
        "medexqa_test": rows_medexqa,
        "symptom_to_diagnosis": rows_symptom,
        "medical_question_pairs": rows_pairs,
    }
    specs = {
        "mmlu_pro_health": C.MMLU_PRO, "medxpertqa_text": C.MEDXPERTQA_TEXT, "medexpqa_en": C.MEDEXPQA_EN,
        "medconceptsqa_all": C.MEDCONCEPTSQA, "medexqa_test": C.MEDEXQA,
        "symptom_to_diagnosis": C.SYMPTOM_DIAGNOSIS, "medical_question_pairs": C.QUESTION_PAIRS,
    }
    all_items: list[EvalItem] = []
    sets: dict[str, Any] = {}
    benchmark_keys = v02_text_keys()
    for name, load in loaders.items():
        rows, source_info = load()
        items, summary = take_valid(name, rows, N_MAX, benchmark_keys)
        spec = specs[name]
        by_qtype = Counter(item.qtype.value for item in items)
        sets[name] = {
            "repo": spec.repo,
            "revision": spec.revision,
            "licence": spec.licence,
            "licence_note": spec.licence_note,
            "source_url": spec.source_url,
            "split_used": spec.split,
            "template_id": spec.template_id,
            "source": source_info,
            "rows": summary,
            "accepted_by_qtype": dict(sorted(by_qtype.items())),
            "gold_counts": dict(sorted(Counter(item.gold for item in items).items())),
        }
        if name == "mmlu_pro_health":
            sets[name]["filtered_before_ranking"] = {"category_not_health": filters["category_not_health"]}
        print(f"[{utcnow()}] {name}: population={summary['population']} accepted={summary['accepted']} "
              f"dropped={summary['dropped']} {summary['dropped_by_reason']}", flush=True)
        all_items.extend(items)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    panel_path = OUT_DIR / "panel.jsonl"
    with panel_path.open("w", encoding="utf-8") as fh:
        for item in all_items:
            fh.write(item.model_dump_json() + "\n")
    excluded = [
        {"id": "lavita/MedQuAD", "reason": "S5 training source (catalog train-candidate); overlap with training is not allowed"},
        {"id": "LangAGI-Lab/medbullets_op5", "reason": "no licence declared on the Hub"},
        {"id": "ade-benchmark-corpus/ade_corpus_v2", "reason": "no licence declared on the Hub (licence field 'unknown')"},
        {"id": "pietrolesci/pubmed-200k-rct", "reason": "no licence declared on the Hub; GitHub Franck-Dernoncourt/pubmed-rct has no licence"},
        {"id": "tasksource/nli4ct, akkasi/NLI4CT", "reason": "no licence declared on the Hub mirrors; not verified against the task source"},
        {"id": "abachaa/MedQuAD (GitHub)", "reason": "GitHub licence NOASSERTION; also a training source"},
        {"id": "bigbio/ddi_corpus", "reason": "loader-script dataset; data is fetched from outside the Hub, not present in the repo (NOT MEASURED)"},
        {"id": "sixuexing/FAERS-NLP", "reason": "about 5 GB of CSVs with an unverified schema for this loop's timebox (NOT MEASURED)"},
        {"id": "songlab/clinvar", "reason": "numeric feature table (genomic scores), label semantics not documented on the card; not a text decision"},
        {"id": "bigbio/head_qa", "reason": "Spanish; the Osler models are evaluated in English"},
        {"id": "dwadden/healthver_entailment", "reason": "tier-1 source of v0.2 (HealthVer)"},
        {"id": "bigbio/biored, bigbio/mediqa_qa", "reason": "no licence declared on the Hub"},
        {"id": "ncbi/MedCalc-Bench-v1.2, bigbio/gad, bigbio/chemprot", "reason": "catalog train-candidates; not used as panel to keep training and evaluation sets apart"},
    ]
    section = {
        "built_at_utc": utcnow(),
        "git_commit": git_commit(REPO),
        "n_max_per_set": N_MAX,
        "panel_file": str(panel_path.relative_to(REPO)),
        "panel_file_sha256": file_sha256(panel_path),
        "panel_items": len(all_items),
        "by_set": sets,
        "excluded_sets": excluded,
        "panel_note": "evaluation only: none of these splits may enter training; MedExQA and the cc-by-nc sets are for evaluation",
    }
    return section


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.parse_args()
    section = build()
    manifest_path = OUT_DIR / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest["panel"] = section
    write_json(manifest_path, manifest)
    print(f"panel items={section['panel_items']} sha256={section['panel_file_sha256'][:12]}")
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
