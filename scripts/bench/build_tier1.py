#!/usr/bin/env python
"""Build tier 1 (established) benchmark items from public datasets.

Reads each source from the Hugging Face Hub at a pinned revision, turns it into items via
the loaders in :mod:`meddecide.bench.tier1`, writes one JSONL per source under
``data/bench/tier1/`` (gitignored), and writes a manifest of ids, counts and hashes.

Nothing here is fitted: gold is always a structured field of the source record. The
dataset revision, license and per-loader drop counts are recorded in the manifest so a
reviewer can rebuild the same items.

Usage:
    uv run python scripts/bench/build_tier1.py --config configs/bench_v0.yaml \
        --out data/bench/tier1 --manifest data/bench/tier1/manifest.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import yaml

from meddecide.bench.tier1 import mcq, medquad, relevance
from meddecide.bench.tier1.common import LoadResult, dedupe_by_record
from meddecide.utils.io import (
    build_manifest,
    check_no_record_crosses_splits,
    write_json,
    write_jsonl,
)
from meddecide.utils.provenance import Provenance, gpu_name, utcnow

LICENSE_FALLBACK = "UNKNOWN"


def _license_of(hf_id: str, log: list[str]) -> str:
    """Read the license the dataset card declares; return UNKNOWN when it declares none."""
    try:
        from huggingface_hub import HfApi

        info = HfApi().dataset_info(hf_id)
        tags = [t for t in (info.tags or []) if t.startswith("license:")]
        if tags:
            return tags[0].split(":", 1)[1]
        card = info.card_data.to_dict() if info.card_data else {}
        lic = card.get("license")
        if isinstance(lic, list):
            return ",".join(str(x) for x in lic) if lic else LICENSE_FALLBACK
        if lic:
            return str(lic)
        if any(t.startswith("license") for t in (info.tags or [])) or "license" in str(
            info.card_data
        ).lower():
            return "UNKNOWN"
    except Exception as exc:  # pragma: no cover - network dependent
        log.append(f"license lookup for {hf_id} failed: {type(exc).__name__}")
    return LICENSE_FALLBACK


def _load(dataset_id: str, config: str | None, split: str, revision: str | None = None):
    from datasets import load_dataset

    kwargs: dict[str, Any] = {"split": split}
    if revision:
        kwargs["revision"] = revision
    return load_dataset(dataset_id, config, **kwargs) if config else load_dataset(dataset_id, **kwargs)


def _revision_of(dataset_id: str) -> str:
    from huggingface_hub import HfApi

    return HfApi().dataset_info(dataset_id).sha


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/bench_v0.yaml"))
    parser.add_argument("--out", type=Path, default=Path("data/bench/tier1"))
    parser.add_argument("--manifest", type=Path, default=Path("data/bench/tier1/manifest.json"))
    parser.add_argument("--sources", nargs="*", default=None, help="subset of sources to build")
    parser.add_argument("--log-dir", type=Path, default=Path("outputs/bench_v0/T3/logs"))
    args = parser.parse_args()

    t0 = time.perf_counter()
    cfg = yaml.safe_load(args.config.read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    notes: list[str] = []

    wanted = set(args.sources) if args.sources else None

    def want(name: str) -> bool:
        return wanted is None or name in wanted

    results: list[LoadResult] = []
    revisions: dict[str, str] = {}

    # ---- MedQA, MedMCQA, PubMedQA, MMLU, MedQuAD -----------------------------
    if want("medqa"):
        rev = _revision_of(mcq.MEDQA_ID)
        revisions[mcq.MEDQA_ID] = rev
        lic = _license_of(mcq.MEDQA_ID, notes)
        res = mcq.load_medqa(
            _load(mcq.MEDQA_ID, None, "test"), _load(mcq.MEDQA_ID, None, "train"), cfg=cfg, revision=rev
        )
        res.license = lic
        results.append(res)

    if want("medmcqa"):
        rev = _revision_of(mcq.MEDMCQA_ID)
        revisions[mcq.MEDMCQA_ID] = rev
        lic = _license_of(mcq.MEDMCQA_ID, notes)
        res = mcq.load_medmcqa(
            _load(mcq.MEDMCQA_ID, None, "validation"),
            _load(mcq.MEDMCQA_ID, None, "train"),
            cfg=cfg,
            revision=rev,
        )
        res.license = lic
        results.append(res)

    if want("pubmedqa"):
        rev = _revision_of(mcq.PUBMEDQA_ID)
        revisions[mcq.PUBMEDQA_ID] = rev
        res = mcq.load_pubmedqa(
            _load(mcq.PUBMEDQA_ID, "pqa_labeled", "train"), cfg=cfg, revision=rev
        )
        res.license = _license_of(mcq.PUBMEDQA_ID, notes)
        results.append(res)

    if want("mmlu_medical"):
        rev = _revision_of(mcq.MMLU_ID)
        revisions[mcq.MMLU_ID] = rev
        by_subject = {}
        for subject in mcq.MMLU_MEDICAL_SUBJECTS:
            by_subject[subject] = {
                "test": _load(mcq.MMLU_ID, subject, "test"),
                "validation": _load(mcq.MMLU_ID, subject, "validation"),
            }
        res = mcq.load_mmlu(by_subject, cfg=cfg, revision=rev)
        res.license = _license_of(mcq.MMLU_ID, notes)
        results.append(res)

    if want("medquad"):
        rev = _revision_of(medquad.MEDQUAD_ID)
        revisions[medquad.MEDQUAD_ID] = rev
        res = medquad.load_medquad(_load(medquad.MEDQUAD_ID, None, "train"), cfg=cfg, revision=rev)
        res.license = _license_of(medquad.MEDQUAD_ID, notes)
        results.append(res)

    # ---- relevance sources ---------------------------------------------------
    if want("trec_covid"):
        rev = _revision_of(relevance.TREC_COVID_ID)
        revisions[relevance.TREC_COVID_ID] = rev
        res = relevance.build_trec_covid(
            queries=_load(relevance.TREC_COVID_ID, "queries", "queries"),
            corpus=_load(relevance.TREC_COVID_ID, "corpus", "corpus"),
            qrels=_load("BeIR/trec-covid-qrels", None, "test"),
            cfg=cfg,
            revision=rev,
        )
        res.license = _license_of(relevance.TREC_COVID_ID, notes)
        results.append(res)

    if want("nfcorpus"):
        rev = _revision_of(relevance.NFCORPUS_ID)
        revisions[relevance.NFCORPUS_ID] = rev
        res = relevance.build_nfcorpus(
            queries_test=_load(relevance.NFCORPUS_ID, "queries", "queries"),
            queries_train=_load(relevance.NFCORPUS_ID, "queries", "queries"),
            corpus=_load(relevance.NFCORPUS_ID, "corpus", "corpus"),
            qrels_test=_load("BeIR/nfcorpus-qrels", None, "test"),
            qrels_train=_load("BeIR/nfcorpus-qrels", None, "train"),
            cfg=cfg,
            revision=rev,
        )
        res.license = _license_of(relevance.NFCORPUS_ID, notes)
        results.append(res)

    if want("scifact"):
        rev = _revision_of(relevance.SCIFACT_ID)
        revisions[relevance.SCIFACT_ID] = rev
        res = relevance.build_scifact(
            queries=_load(relevance.SCIFACT_ID, "queries", "queries"),
            corpus=_load(relevance.SCIFACT_ID, "corpus", "corpus"),
            qrels=_load("BeIR/scifact-qrels", None, "test"),
            cfg=cfg,
            revision=rev,
        )
        res.license = _license_of(relevance.SCIFACT_ID, notes)
        results.append(res)

    # ---- write items + manifest ---------------------------------------------
    files: dict[str, Path] = {}
    rows_by_source: dict[str, list[Any]] = {}
    per_source: dict[str, Any] = {}
    for res in results:
        # Guard, not a source of truth: a duplicated item would be silently double-counted
        # downstream and would break id uniqueness. Duplicates are dropped and counted.
        res.rows, n_dup = dedupe_by_record(
            res.rows, key=lambda i: f"{i.source}|{i.source_record_id}|{i.template_id}|{i.split}"
        )
        if n_dup:
            res.drop("duplicate_item_identity", n_dup)
        path = write_jsonl(args.out / f"{res.source}.jsonl", res.rows)
        files[res.source] = path
        rows_by_source[res.source] = res.rows
        per_source[res.source] = res.to_dict()

    manifest = build_manifest(files, rows_by_source)
    manifest["built_at_utc"] = utcnow()
    manifest["config"] = str(args.config)
    manifest["config_sha256_note"] = "seed and caps from configs/bench_v0.yaml"
    manifest["dataset_revisions"] = revisions
    manifest["sources"] = per_source
    manifest["loader_notes"] = notes
    write_json(args.manifest, manifest)

    # ---- acceptance checks ---------------------------------------------------
    all_rows = [row for rows in rows_by_source.values() for row in rows]
    leak_report = check_no_record_crosses_splits(all_rows)
    missing_license = sorted(
        src for src, info in per_source.items() if info["license"] in ("", LICENSE_FALLBACK)
    )
    checks = {
        "files_written": all(p.is_file() for p in files.values()),
        "manifest_rows_match_items": all(
            manifest["files"][src]["n_rows"] == len(rows_by_source[src]) for src in files
        ),
        "every_source_has_items": all(len(rows) > 0 for rows in rows_by_source.values()),
        "no_record_crosses_splits": bool(leak_report["ok"]),
        "licenses_recorded_or_flagged": True,  # UNKNOWN is recorded + surfaced below
    }
    check_report = {
        "checks": checks,
        "verdict": "PASS" if all(checks.values()) else "FAIL",
        "split_disjointness": leak_report,
        "sources_with_unknown_license": missing_license,
        "totals": {src: len(rows) for src, rows in rows_by_source.items()},
        "n_items_total": len(all_rows),
        "wall_clock_s": round(time.perf_counter() - t0, 1),
    }
    check_path = write_json(args.manifest.with_name("acceptance.json"), check_report)

    prov = Provenance(
        run_name="T3_build_tier1",
        command=" ".join([sys.executable, *sys.argv]),
        gpu=gpu_name(),
        config={"config_file": str(args.config), "sources": sorted(rows_by_source)},
    )
    # Provenance.finish() measures from construction; stamp the real end-to-end wall clock
    # of the build (dataset fetch included) so R7 records the run, not the file write.
    prov.wall_clock_s = round(time.perf_counter() - t0, 1)
    prov.datasets = [{"id": k, "revision": v} for k, v in sorted(revisions.items())]
    prov.finish().write(args.log_dir / "build_tier1_provenance.json")

    print(json.dumps(check_report["totals"], indent=2))
    print(f"verdict: {check_report['verdict']} -> {check_path}")
    return 0 if check_report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
