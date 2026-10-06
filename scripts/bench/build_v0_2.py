#!/usr/bin/env python
"""Build MedDecide-Bench v0.2 (loop `student_v0`, task S1).

v0.2 is v0.1 with three defective templates replaced by `_v2` versions and **everything else
carried through unchanged**, so every earlier prediction on a carried template stays
comparable (the `item_id` is a hash of source, record, template and split, so a carried item
keeps its id exactly).

| superseded (v0.1) | replacement (v0.2) | why |
|---|---|---|
| `nfcorpus_graded_score_v1` | `_v2` | offered "Not relevant", which no passage in the pool had: 0 of 429 items could be answered with it (X029) |
| `pubmed_mesh_major_choice_v1` | `_v2` | distractors were unrelated frequent topics; 0.985-0.996 for every model from 0.8B to 9B |
| `fda_class_choice_v1` | `_v2` | distractors were unrelated class names; 0.84-0.996 for every model |

The new templates are built here rather than by rebuilding the whole benchmark, because the
carried items must not move: a rebuild would re-sample every source and invalidate the v0.1
predictions the loop compares against.

Nothing in this script fits anything: every gold is a structured field of the source record.

Usage:
    uv run python scripts/bench/build_v0_2.py --mesh-index /workspace/tmp/mesh/mesh_index.json \
        --fda-class-index /workspace/tmp/openfda/class_index.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any

import yaml

from meddecide.bench.fresh import openfda, pubmed
from meddecide.bench.schema import Item, QuestionType
from meddecide.bench.tier1 import relevance
from meddecide.bench.tier1.common import balance_classes, finalize_splits
from meddecide.utils.hashing import stable_hash
from meddecide.utils.io import (
    build_manifest,
    check_no_record_crosses_splits,
    read_jsonl,
    write_json,
    write_jsonl,
)
from meddecide.utils.provenance import Provenance, gpu_name, utcnow

SUPERSEDED = {
    "nfcorpus_graded_score_v1": "nfcorpus_graded_score_v2",
    "pubmed_mesh_major_choice_v1": "pubmed_mesh_major_choice_v2",
    "fda_class_choice_v1": "fda_class_choice_v2",
}
TIER1_SOURCES = (
    "medmcqa", "medqa", "medquad", "mmlu_medical", "nfcorpus", "pubmedqa", "scifact", "trec_covid",
)
FRESH_SOURCES = ("clinicaltrials", "openfda", "pubmed")
PUBMED_CACHE = Path("/workspace/tmp/pubmed")


def _load_v0_1(directory: Path, sources: tuple[str, ...]) -> dict[str, list[Item]]:
    out: dict[str, list[Item]] = {}
    for source in sources:
        path = directory / f"{source}.jsonl"
        if not path.is_file():
            continue
        rows, report = read_jsonl(path, Item)
        if report.n_dropped:
            raise SystemExit(f"{path}: {report.n_dropped} unreadable rows — refusing to build on it")
        out[source] = list(rows)
    return out


def _partition(rows: list[Item]) -> tuple[list[Item], dict[str, int]]:
    """Split into carried rows and a per-template count of the superseded rows."""
    carried: list[Item] = []
    dropped: dict[str, int] = defaultdict(int)
    for row in rows:
        if row.template_id in SUPERSEDED:
            dropped[row.template_id] += 1
        else:
            carried.append(row)
    return carried, dict(dropped)


def _load_dataset(dataset_id: str, config: str | None, split: str):
    from datasets import load_dataset

    return load_dataset(dataset_id, config, split=split) if config else load_dataset(dataset_id, split=split)


def build_nfcorpus_score_v2(cfg: dict[str, Any]) -> tuple[list[Item], dict[str, Any]]:
    """The `_v2` graded-score items: the offered levels are exactly the levels present."""
    from huggingface_hub import HfApi

    revision = HfApi().dataset_info(relevance.NFCORPUS_ID).sha
    res = relevance.load_beir_relevance(
        source="nfcorpus",
        dataset_id=relevance.NFCORPUS_ID,
        queries=_load_dataset(relevance.NFCORPUS_ID, "queries", "queries"),
        corpus=_load_dataset(relevance.NFCORPUS_ID, "corpus", "corpus"),
        qrels=_load_dataset("BeIR/nfcorpus-qrels", None, "test"),
        cfg=cfg,
        revision=revision,
        split=relevance.Split.TEST,
        record_date=date(2015, 1, 1),
        graded=True,
        score_template_id="nfcorpus_graded_score_v2",
    )
    score_items = [row for row in res.items if row.qtype is QuestionType.SCORE]
    res.rows = score_items
    res.rows, counts, extra = finalize_splits(
        res.rows, cfg=cfg, source="nfcorpus", salt="nfcorpus_v2", cap_test=5000, cap_dev=2000
    )
    for reason, n in counts.items():
        res.drop(reason, n)
    res.rows, balance = balance_classes(res.rows, cfg=cfg, salt="nfcorpus_v2")
    res.notes.extend(extra)
    res.notes.append(
        "S1 fix: the offered levels are the levels present in the sampled test pool, so every "
        "offered level is the gold of at least one item"
    )
    meta = {"revision": revision, "notes": res.notes, "dropped": res.dropped, "balance": balance}
    return list(res.rows), meta


def build_pubmed_mesh_v2(
    cfg: dict[str, Any], mesh_index_path: Path, window_start: date, window_end: date, n_files: int
) -> tuple[list[Item], dict[str, Any]]:
    from meddecide.bench.mesh import load_index

    index = load_index(mesh_index_path)
    paths = _pubmed_files(n_files, PUBMED_CACHE)
    by_pmid: dict[str, pubmed.PubmedRecord] = {}
    n_parsed = n_copies = 0
    for record in pubmed.iter_records(paths, stop_before=window_start):
        n_parsed += 1
        if not (window_start <= record.entrez_date <= window_end):
            continue
        existing = by_pmid.get(record.pmid)
        if existing is not None:
            n_copies += 1
            if record.entrez_date < existing.entrez_date:
                continue
        by_pmid[record.pmid] = record
    records = list(by_pmid.values())
    rows, drops, notes = pubmed.build_mesh_major_v2_items(
        records,
        cfg=cfg,
        window_start=window_start,
        window_end=window_end,
        mesh_siblings=index.siblings,
        mesh_meta={**index.meta, "index_path": str(mesh_index_path)},
    )
    rows, counts, extra = finalize_splits(
        rows, cfg=cfg, source="pubmed", salt="pubmed_mesh_v2", cap_test=5000, cap_dev=2000
    )
    for reason, n in counts.items():
        drops[reason] = drops.get(reason, 0) + n
    rows, balance = balance_classes(rows, cfg=cfg, salt="pubmed_mesh_v2")
    notes.extend(extra)
    notes.append(
        f"{n_parsed} records parsed, {n_copies} duplicate copies of a PMID merged, "
        f"{len(records)} unique records in window"
    )
    meta = {
        "mesh": index.meta, "files": [p.name for p in paths], "notes": notes, "dropped": drops,
        "balance": balance, "dataset_revision": ",".join(p.name for p in paths),
    }
    return list(rows), meta


def build_openfda_class_v2(
    cfg: dict[str, Any], class_index_path: Path, window_start: date, window_end: date
) -> tuple[list[Item], dict[str, Any]]:
    from meddecide.bench.fresh import fda_classes

    index = fda_classes.load_index(class_index_path)
    labels = openfda.fetch_labels(start=window_start, end=window_end)

    def near_miss(gold_class: str, own: set[str]) -> list[dict[str, Any]]:
        return fda_classes.near_miss_candidates(gold_class, index, exclude=own)

    rows, drops, notes = openfda.build_class_v2_items(
        labels,
        cfg=cfg,
        window_start=window_start,
        window_end=window_end,
        near_miss=near_miss,
        class_index_meta=index.get("meta", {}),
        fallback_pool=openfda.fetch_class_pool(limit=int(cfg["caps"].get("fda_class_pool_size", 800))),
    )
    rows, counts, extra = finalize_splits(
        rows, cfg=cfg, source="openfda", salt="openfda_class_v2", cap_test=5000, cap_dev=2000
    )
    for reason, n in counts.items():
        drops[reason] = drops.get(reason, 0) + n
    rows, balance = balance_classes(rows, cfg=cfg, salt="openfda_class_v2")
    notes.extend(extra)
    notes.append(f"{len(labels)} labels fetched from openFDA for the window")
    meta = {
        "class_index": index.get("meta", {}), "labels_fetched": len(labels), "notes": notes,
        "dropped": drops, "balance": balance,
    }
    return list(rows), meta


def _pubmed_files(n_files: int, cache: Path) -> list[Path]:
    """The same update files the v0.1 build used, re-downloaded only if they are gone."""
    import httpx

    cache.mkdir(parents=True, exist_ok=True)
    base = "https://ftp.ncbi.nlm.nih.gov/pubmed/updatefiles/"
    with httpx.Client(timeout=120.0) as http:
        listing = http.get(base).text
    names = sorted(
        {
            line.split('href="')[-1].split('"')[0]
            for line in listing.splitlines()
            if "pubmed26n" in line and line.split('href="')[-1].split('"')[0].endswith(".xml.gz")
        }
    )
    picked = names[-n_files:]
    paths: list[Path] = []
    for name in picked:
        path = cache / name
        if not path.is_file():
            with (
                httpx.Client(timeout=600.0, follow_redirects=True) as http,
                http.stream("GET", base + name) as response,
                path.open("wb") as fh,
            ):
                response.raise_for_status()
                for chunk in response.iter_bytes(chunk_size=1 << 20):
                    fh.write(chunk)
        paths.append(path)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v0-1", type=Path, default=Path("data/bench/v0.1"))
    parser.add_argument("--out", type=Path, default=Path("data/bench/v0.2"))
    parser.add_argument("--config", type=Path, default=Path("configs/bench_v0_2.yaml"))
    parser.add_argument("--mesh-index", type=Path, default=Path("/workspace/tmp/mesh/mesh_index.json"))
    parser.add_argument("--fda-class-index", type=Path,
                        default=Path("/workspace/tmp/openfda/class_index.json"))
    parser.add_argument("--pubmed-files", type=int, default=12)
    parser.add_argument(
        "--sources", nargs="*", default=None,
        help="which NEW templates to build: nfcorpus_v2 pubmed_mesh_v2 openfda_class_v2 "
             "(carried templates are always written)",
    )
    parser.add_argument("--log-dir", type=Path, default=Path("outputs/student_v0/S1/logs"))
    args = parser.parse_args()

    t0 = time.perf_counter()
    cfg = yaml.safe_load(args.config.read_text())
    wanted = set(args.sources) if args.sources else None

    def want(name: str) -> bool:
        return wanted is None or name in wanted

    v0_1_tier1 = _load_v0_1(args.v0_1 / "tier1", TIER1_SOURCES)
    v0_1_fresh = _load_v0_1(args.v0_1 / "fresh", FRESH_SOURCES)
    fresh_manifest = json.loads((args.v0_1 / "fresh" / "manifest.json").read_text())
    window_start = date.fromisoformat(fresh_manifest["window_start"])
    window_end = date.fromisoformat(fresh_manifest["window_end"])
    strict_start = date.fromisoformat(str(fresh_manifest["strict_slice_start"]))

    # ---- tier 1: carry, then replace the score template ----------------------
    tier1_rows: dict[str, list[Item]] = {}
    superseded_counts: dict[str, int] = {}
    for source, rows in v0_1_tier1.items():
        carried, dropped = _partition(rows)
        for template_id, n in dropped.items():
            superseded_counts[template_id] = superseded_counts.get(template_id, 0) + n
        tier1_rows[source] = carried
    build_meta: dict[str, Any] = {}
    if want("nfcorpus_v2"):
        new_rows, meta = build_nfcorpus_score_v2(cfg)
        tier1_rows["nfcorpus"] = [*tier1_rows.get("nfcorpus", []), *new_rows]
        build_meta["nfcorpus_graded_score_v2"] = meta

    # ---- fresh: carry, then replace the two choice templates -----------------
    fresh_rows: dict[str, list[Item]] = {}
    for source, rows in v0_1_fresh.items():
        carried, dropped = _partition(rows)
        for template_id, n in dropped.items():
            superseded_counts[template_id] = superseded_counts.get(template_id, 0) + n
        fresh_rows[source] = carried
    if want("pubmed_mesh_v2") and args.mesh_index.is_file():
        new_rows, meta = build_pubmed_mesh_v2(
            cfg, args.mesh_index, window_start, window_end, args.pubmed_files
        )
        fresh_rows["pubmed"] = [*fresh_rows.get("pubmed", []), *new_rows]
        build_meta["pubmed_mesh_major_choice_v2"] = meta
    elif want("pubmed_mesh_v2"):
        build_meta["pubmed_mesh_major_choice_v2"] = {
            "status": f"NOT BUILT - mesh index missing at {args.mesh_index}"
        }
    if want("openfda_class_v2") and args.fda_class_index.is_file():
        new_rows, meta = build_openfda_class_v2(
            cfg, args.fda_class_index, window_start, window_end
        )
        fresh_rows["openfda"] = [*fresh_rows.get("openfda", []), *new_rows]
        build_meta["fda_class_choice_v2"] = meta
    elif want("openfda_class_v2"):
        build_meta["fda_class_choice_v2"] = {
            "status": f"NOT BUILT - class index missing at {args.fda_class_index}"
        }

    # ---- write ---------------------------------------------------------------
    files: dict[str, Path] = {}
    rows_by_source: dict[str, list[Item]] = {}
    for tier, rows_by_source_in, sources in (
        ("tier1", tier1_rows, TIER1_SOURCES),
        ("fresh", fresh_rows, FRESH_SOURCES),
    ):
        (args.out / tier).mkdir(parents=True, exist_ok=True)
        for source in sources:
            rows = rows_by_source_in.get(source)
            if rows is None:
                continue
            path = write_jsonl(args.out / tier / f"{source}.jsonl", rows)
            files[f"{tier}/{source}"] = path
            rows_by_source[source] = rows

    manifest = build_manifest(
        {k.split("/")[1]: v for k, v in files.items()}, rows_by_source
    )
    manifest["built_at_utc"] = utcnow()
    manifest["config"] = str(args.config)
    manifest["v0_1_source"] = str(args.v0_1)
    manifest["window_start"] = window_start.isoformat()
    manifest["window_end"] = window_end.isoformat()
    manifest["strict_slice_start"] = strict_start.isoformat()
    manifest["superseded"] = {
        template_id: {
            "superseded_by": SUPERSEDED[template_id],
            "n_items_in_v0_1": n,
            "reason": {
                "nfcorpus_graded_score_v1": "offered a level no passage in its pool had (X029)",
                "pubmed_mesh_major_choice_v1": "unrelated distractors; 0.985-0.996 for every model",
                "fda_class_choice_v1": "unrelated distractors; 0.84-0.996 for every model",
            }[template_id],
        }
        for template_id, n in sorted(superseded_counts.items())
    }
    manifest["builds"] = build_meta
    write_json(args.out / "manifest.json", manifest)
    # copies where the tier-1/fresh tools expect them, and where the baseline runner looks
    # for the strict-slice start
    write_json(args.out / "tier1" / "manifest.json", manifest)
    write_json(args.out / "fresh" / "manifest.json", manifest)

    # ---- acceptance ----------------------------------------------------------
    carried_ids_v0_1 = {
        row.item_id
        for rows in [*v0_1_tier1.values(), *v0_1_fresh.values()]
        for row in rows
        if row.template_id not in SUPERSEDED
    }
    carried_ids_v0_2 = {
        row.item_id
        for rows in rows_by_source.values()
        for row in rows
        if row.template_id not in set(SUPERSEDED.values())
    }
    def _carried_hashes(groups) -> dict[str, str]:
        out: dict[str, str] = {}
        for rows in groups:
            for row in rows:
                if row.template_id in SUPERSEDED:
                    continue
                out[row.item_id] = stable_hash(row.model_dump(mode="json"), length=32)
        return out

    carried_hash_v0_1 = _carried_hashes([*v0_1_tier1.values(), *v0_1_fresh.values()])
    carried_hash_v0_2 = _carried_hashes(rows_by_source.values())
    mismatched = sorted(
        item_id
        for item_id in carried_ids_v0_1 & carried_ids_v0_2
        if carried_hash_v0_1[item_id] != carried_hash_v0_2[item_id]
    )
    new_ids = set(carried_ids_v0_2) - carried_ids_v0_1
    all_rows = [row for rows in rows_by_source.values() for row in rows]
    leak_report = check_no_record_crosses_splits(all_rows)

    score_v2 = [r for r in rows_by_source.get("nfcorpus", []) if r.template_id == "nfcorpus_graded_score_v2"]
    score_levels_offered = {
        str(level) for item in score_v2 for level in item.meta.get("offered_levels", [])
    }
    score_levels_gold = {item.gold for item in score_v2 if str(item.split) == "test"}
    fresh_out_of_window = [
        row.item_id
        for source in FRESH_SOURCES
        for row in rows_by_source.get(source, [])
        if not (window_start <= row.record_date <= window_end)
    ]
    v1_leftovers = sorted(
        {row.template_id for row in all_rows if row.template_id in SUPERSEDED}
    )
    per_template_test = defaultdict(int)
    per_template_split = defaultdict(lambda: defaultdict(int))
    for row in all_rows:
        per_template_split[row.template_id][str(row.split)] += 1
    for row in all_rows:
        if str(row.split) == "test":
            per_template_test[row.template_id] += 1

    checks = {
        "files_written": all(p.is_file() for p in files.values()),
        "manifest_rows_match_items": all(
            manifest["files"][source]["n_rows"] == len(rows_by_source[source])
            for source in rows_by_source
        ),
        "carried_items_identical_to_v0_1": not mismatched and carried_ids_v0_1 == carried_ids_v0_2,
        "no_superseded_template_left": not v1_leftovers,
        "no_record_crosses_splits": bool(leak_report["ok"]),
        "every_offered_score_level_is_a_gold_in_test": score_levels_offered == score_levels_gold,
        "every_fresh_item_inside_window": not fresh_out_of_window,
        "every_v2_template_has_200_test_items": all(
            per_template_test.get(t, 0) >= 200 for t in SUPERSEDED.values()
        ),
    }
    check_report = {
        "checks": checks,
        "verdict": "PASS" if all(checks.values()) else "FAIL",
        "n_items_total": len(all_rows),
        "n_carried": len(carried_ids_v0_2),
        "n_new": len(new_ids),
        "n_carried_hash_mismatches": len(mismatched),
        "superseded": manifest["superseded"],
        "split_disjointness": leak_report,
        "score_v2_levels": {
            "offered": sorted(score_levels_offered, key=int),
            "gold_in_test": sorted(score_levels_gold, key=int),
            "n_items": len(score_v2),
        },
        "per_template_split": {k: dict(v) for k, v in sorted(per_template_split.items())},
        "by_source": {source: len(rows) for source, rows in sorted(rows_by_source.items())},
        "wall_clock_s": round(time.perf_counter() - t0, 1),
    }
    check_path = write_json(args.out / "acceptance.json", check_report)
    write_json(args.out / "fresh" / "acceptance.json", check_report)

    prov = Provenance(
        run_name="S1_build_v0_2",
        command=" ".join([sys.executable, *sys.argv]),
        gpu=gpu_name(),
        seed=int(cfg.get("seed", 0)),
        config={
            "config_file": str(args.config), "window_start": window_start.isoformat(),
            "window_end": window_end.isoformat(), "superseded": sorted(SUPERSEDED),
            "mesh_index": str(args.mesh_index), "fda_class_index": str(args.fda_class_index),
        },
    )
    prov.wall_clock_s = round(time.perf_counter() - t0, 1)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    prov.finish().write(args.log_dir / "build_v0_2_provenance.json")

    print(json.dumps({k: v for k, v in check_report.items() if k != "per_template_split"}, indent=2))
    print(f"verdict: {check_report['verdict']} -> {check_path}")
    return 0 if check_report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
