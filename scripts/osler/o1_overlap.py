"""O1: leakage check of the external panel against every training file and against v0.2 (ADVISORY O1).

Streams each training file once and reduces every row to the normalised-text key and the
(source, record id) key (meddecide.panel.overlap). Panel items with any exact text or record match in a
training file are listed by id and must be removed before the panel is used; the acceptance is 0.
v0.2 is scanned too, as a benchmark-versus-panel check (reported separately; v0.2 is not training).

Outputs: outputs/osler_v0/O1/overlap.json (counts and per-file hashes) and a summary section in
data/bench/v0.3_ext/manifest.json.

Usage:
    source scripts/pod_env.sh && uv run --frozen python scripts/osler/o1_overlap.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from meddecide.panel.overlap import TrainingKeys, item_text_key
from meddecide.panel.schema import EvalItem
from meddecide.utils.hashing import file_sha256
from meddecide.utils.io import write_json
from meddecide.utils.provenance import git_commit, utcnow

REPO = Path(__file__).resolve().parents[2]
PANEL = REPO / "data/bench/v0.3_ext/panel.jsonl"
MANIFEST = REPO / "data/bench/v0.3_ext/manifest.json"
OUT = REPO / "outputs/osler_v0/O1/overlap.json"

# Every training-side item file of student_v0 and student_v1 (the mixes and their component pools).
# Not scanned: data/train/student_v1/sources/pubmed_prewindow.jsonl.gz (2.8 GB compressed PubMed pool);
# recorded as a limitation in the manifest.
TRAINING_FILES = [
    "data/train/student_v1/train.jsonl",
    "data/train/student_v1/dev.jsonl",
    "data/train/student_v1/components/clinicaltrials_v1.jsonl",
    "data/train/student_v1/components/openfda_v1.jsonl",
    "data/train/student_v1/components/pubmed_v1.jsonl",
    "data/train/student_v0/train.jsonl",
    "data/train/student_v0/dev.jsonl",
    "data/train/student_v0/tier1_train.jsonl",
    "data/train/student_v0/prewindow_clinicaltrials.jsonl",
    "data/train/student_v0/prewindow_consistency.jsonl",
    "data/train/student_v0/prewindow_openfda.jsonl",
    "data/train/student_v0/prewindow_structured.jsonl",
]
NOT_SCANNED = ["data/train/student_v1/sources/pubmed_prewindow.jsonl.gz"]
BENCHMARK_FILES = sorted(str(p.relative_to(REPO)) for p in (REPO / "data/bench/v0.2").rglob("*.jsonl")
                         if "supplementary" not in p.parts)


def scan(path: Path, keys: TrainingKeys) -> int:
    rows = 0
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            keys.add_row(json.loads(line))
            rows += 1
    return rows


def main() -> None:
    panel = [EvalItem.model_validate_json(line) for line in PANEL.read_text(encoding="utf-8").splitlines()
             if line.strip()]
    panel_text = {item.item_id: item_text_key(item) for item in panel}

    train_keys = TrainingKeys()
    per_file: list[dict[str, Any]] = []
    matched: dict[str, str] = {}
    for rel in TRAINING_FILES:
        path = REPO / rel
        before = len(train_keys.text)
        rows = scan(path, train_keys)
        per_file.append({"path": rel, "rows": rows, "sha256": file_sha256(path),
                         "new_distinct_text_keys": len(train_keys.text) - before})
        print(f"[{utcnow()}] scanned {rel}: {rows} rows", flush=True)
    report = train_keys.compare(panel)
    for item in panel:
        if item_text_key(item) in train_keys.text:
            matched[item.item_id] = item.set_name

    # v0.2 as a benchmark-versus-panel check (its rows use the same text fields)
    bench_keys = TrainingKeys()
    bench_rows = 0
    for rel in BENCHMARK_FILES:
        bench_rows += scan(REPO / rel, bench_keys)
    bench_report = bench_keys.compare(panel)

    summary = {
        "built_at_utc": utcnow(),
        "git_commit": git_commit(REPO),
        "panel_items_checked": len(panel),
        "training_files": per_file,
        "training_rows_scanned": train_keys.rows,
        "training_text_hits_total": report.total_text_hits(),
        "training_text_hits_by_set": report.text_hits,
        "training_record_hits_by_set": report.record_hits,
        "matched_item_ids": sorted(matched),
        "not_scanned": NOT_SCANNED,
        "v0_2_files_scanned": BENCHMARK_FILES,
        "v0_2_rows_scanned": bench_rows,
        "v0_2_text_hits_total": bench_report.total_text_hits(),
        "v0_2_text_hits_by_set": bench_report.text_hits,
        "v0_2_record_hits_by_set": bench_report.record_hits,
        "panel_text_keys_distinct": len(set(panel_text.values())),
        "acceptance": "training_text_hits_total == 0 and training_record_hits_by_set empty",
        "verdict": "PASS" if report.total_text_hits() == 0 and not report.record_hits else "FAIL",
    }
    write_json(OUT, summary)
    manifest = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    manifest["overlap"] = {k: summary[k] for k in [
        "built_at_utc", "panel_items_checked", "training_rows_scanned", "training_text_hits_total",
        "training_text_hits_by_set", "training_record_hits_by_set", "v0_2_rows_scanned",
        "v0_2_text_hits_total", "v0_2_text_hits_by_set", "verdict"]}
    manifest["overlap"]["training_files"] = [{"path": f["path"], "rows": f["rows"], "sha256": f["sha256"]}
                                             for f in per_file]
    write_json(MANIFEST, manifest)
    print(f"panel items={len(panel)} training rows={train_keys.rows} training text hits={report.total_text_hits()} "
          f"record hits={report.record_hits} v0.2 text hits={bench_report.total_text_hits()} verdict={summary['verdict']}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
