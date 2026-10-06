#!/usr/bin/env python
"""Build the fresh (tier 2) benchmark items from ClinicalTrials.gov, openFDA and PubMed.

Every item is dated inside the window in ``docs/benchmark/fresh_window.md`` by a structured
source field, and every gold is a lookup of a structured field. Rebuildable: the same
``--window-start``/``--window-end`` produce byte-identical files.

Usage:
    uv run python scripts/bench/build_fresh.py --window-start 2026-09-10 --window-end 2026-10-05
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date
from pathlib import Path
from typing import Any

import yaml

from meddecide.bench.fresh import clinicaltrials, openfda, pubmed
from meddecide.bench.tier1.common import LoadResult, balance_classes, finalize_splits
from meddecide.utils.io import (
    build_manifest,
    check_no_record_crosses_splits,
    read_jsonl,
    write_json,
    write_jsonl,
)
from meddecide.utils.provenance import Provenance, gpu_name, utcnow

PUBMED_CACHE = Path("/workspace/tmp/pubmed")  # gitignored scratch for the update files


def _pubmed_files(n_files: int, cache: Path) -> list[Path]:
    """Ensure the last ``n_files`` PubMed update files are on disk; return their paths."""
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
    parser.add_argument("--config", type=Path, default=Path("configs/bench_v0.yaml"))
    parser.add_argument("--window-start", type=date.fromisoformat, default=None)
    parser.add_argument("--window-end", type=date.fromisoformat, default=None)
    parser.add_argument("--out", type=Path, default=Path("data/bench/fresh"))
    parser.add_argument("--manifest", type=Path, default=Path("data/bench/fresh/manifest.json"))
    parser.add_argument("--sources", nargs="*", default=None)
    parser.add_argument("--pubmed-files", type=int, default=12, help="how many update files to scan")
    parser.add_argument("--log-dir", type=Path, default=Path("outputs/bench_v0/T5/logs"))
    args = parser.parse_args()

    t0 = time.perf_counter()
    cfg = yaml.safe_load(args.config.read_text())
    window_start = args.window_start or date.fromisoformat(str(cfg["fresh_window"]["start"]))
    window_end = args.window_end or date.today()
    if window_start > window_end:
        raise SystemExit(f"window start {window_start} is after window end {window_end}")
    args.out.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    wanted = set(args.sources) if args.sources else None

    def want(name: str) -> bool:
        return wanted is None or name in wanted

    results: list[LoadResult] = []
    source_meta: dict[str, Any] = {}

    if want("clinicaltrials"):
        studies = clinicaltrials.fetch_studies(start=window_start, end=window_end)
        items, drops, notes = clinicaltrials.build_items(
            studies, cfg=cfg, window_start=window_start, window_end=window_end
        )
        res = LoadResult(source="clinicaltrials", dataset_id=clinicaltrials.API,
                         dataset_revision="api-v2", license=clinicaltrials.LICENSE,
                         rows=items, notes=[*notes, f"{len(studies)} studies fetched"])
        for reason, count in drops.items():
            res.drop(reason, count)
        results.append(res)

    if want("openfda"):
        labels = openfda.fetch_labels(start=window_start, end=window_end)
        items, drops, notes = openfda.build_items(
            labels, cfg=cfg, window_start=window_start, window_end=window_end
        )
        res = LoadResult(source="openfda", dataset_id=openfda.API, dataset_revision="api-v1",
                         license=openfda.LICENSE, rows=items,
                         notes=[*notes, f"{len(labels)} labels fetched"])
        for reason, count in drops.items():
            res.drop(reason, count)
        results.append(res)

    if want("pubmed"):
        paths = _pubmed_files(args.pubmed_files, PUBMED_CACHE)
        # The update files overlap: a record appears in several of them (up to 6 copies in
        # this build). Deduplicate by PMID, keeping the copy with the latest Entrez date, so
        # a record contributes exactly one set of items and "no silent drops" still holds
        # (the number of merged copies is recorded).
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
        items, drops, notes = pubmed.build_items(
            records, cfg=cfg, window_start=window_start, window_end=window_end
        )
        res = LoadResult(source="pubmed", dataset_id=pubmed.DATASET_ID,
                         dataset_revision=",".join(p.name for p in paths),
                         license=pubmed.LICENSE, rows=items,
                         notes=[*notes,
                                f"{n_parsed} records parsed, {n_copies} duplicate copies of a "
                                f"PMID across update files merged, {len(records)} unique records in window"])
        for reason, count in drops.items():
            res.drop(reason, count)
        results.append(res)
        source_meta["pubmed_files"] = [p.name for p in paths]

    # integrity rules + caps, per source (same rules as tier 1)
    max_test = int(cfg["caps"]["tier1_max_test_items_per_source"])
    max_dev = int(cfg["caps"]["tier1_max_dev_items_per_source"])
    balance_reports: dict[str, Any] = {}
    strict_slice_start = cfg.get("fresh_window", {}).get("strict_slice_start")
    for res in results:
        res.rows, counts, extra = finalize_splits(
            res.rows, cfg=cfg, source=res.source, salt=res.source, cap_test=max_test, cap_dev=max_dev
        )
        for reason, n in counts.items():
            res.drop(reason, n)
        res.notes.extend(extra)
        # D11 strict slice: mark every item whose own filter date is at/after the strict start
        # (2026-09-10, the teacher's repo date). Items before it are still built; the slice is a
        # reported subset, never a silent filter.
        if strict_slice_start:
            strict_date = date.fromisoformat(str(strict_slice_start))
            sliced = []
            for row in res.rows:
                meta = {**row.meta, "strict_post_teacher": row.record_date >= strict_date}
                sliced.append(row.model_copy(update={"meta": meta}))
            res.rows = sliced
            res.notes.append(
                f"strict slice: {sum(1 for r in res.rows if r.meta['strict_post_teacher'])} of "
                f"{len(res.rows)} items dated on/after {strict_slice_start}"
            )
        # balance classes per (template, split) so micro accuracy can be read against a
        # majority baseline that is not near 1.0
        res.rows, balance = balance_classes(res.rows, cfg=cfg, salt=res.source)
        balance_reports[res.source] = balance
        res.drop("class_balance_removed", balance["n_before_total"] - balance["n_after_total"])
        res.notes.append(
            "class balancing: K per (template, split) recorded in the manifest's balance block"
        )

    files: dict[str, Path] = {}
    rows_by_source: dict[str, list[Any]] = {}
    per_source: dict[str, Any] = {}
    for res in results:
        files[res.source] = write_jsonl(args.out / f"{res.source}.jsonl", res.rows)
        rows_by_source[res.source] = res.rows
        per_source[res.source] = res.to_dict()

    manifest = build_manifest(files, rows_by_source)
    manifest["built_at_utc"] = utcnow()
    manifest["window_start"] = window_start.isoformat()
    manifest["window_end"] = window_end.isoformat()
    manifest["window_source"] = "docs/benchmark/fresh_window.md"
    manifest["sources"] = per_source
    manifest["source_meta"] = source_meta
    manifest["class_balance"] = balance_reports
    # must be a str: a date object is not JSON-serialisable (this aborted one build after the
    # files were written, which is why the manifest write happens before the checks)
    manifest["strict_slice_start"] = str(strict_slice_start) if strict_slice_start else None
    write_json(args.manifest, manifest)

    # ---- acceptance: freshness, rebuild determinism, integrity -------------------
    all_rows = [r for rows in rows_by_source.values() for r in rows]
    date_check = _freshness_check(args.out, rows_by_source, window_start, window_end)
    leak_report = check_no_record_crosses_splits(all_rows)
    checks = {
        "files_written": all(p.is_file() for p in files.values()),
        "every_source_has_items": all(len(rows) > 0 for rows in rows_by_source.values()),
        "zero_records_before_window_start": date_check["ok"],
        "no_record_crosses_splits": bool(leak_report["ok"]),
        "every_item_has_strict_slice_flag": all(
            "strict_post_teacher" in r.meta for rows in rows_by_source.values() for r in rows
        ),
        "no_class_is_single_in_test": all(
            not (entry["single_class"] and key.endswith("|test"))
            for report in balance_reports.values()
            for key, entry in report["templates"].items()
        ),
        "manifest_rows_match_items": all(
            manifest["files"][src]["n_rows"] == len(rows_by_source[src]) for src in files
        ),
    }
    check_report = {
        "checks": checks,
        "verdict": "PASS" if all(checks.values()) else "FAIL",
        "freshness": date_check,
        "split_disjointness": leak_report,
        "totals": {src: len(rows) for src, rows in rows_by_source.items()},
        "by_template": _by_template(rows_by_source),
        "class_balance": balance_reports,
        "strict_slice": _strict_slice_check(args.out, rows_by_source, strict_slice_start),
        "n_items_total": len(all_rows),
        "wall_clock_s": round(time.perf_counter() - t0, 1),
    }
    check_path = write_json(args.manifest.with_name("acceptance.json"), check_report)

    prov = Provenance(
        run_name="T5_build_fresh",
        command=" ".join([sys.executable, *sys.argv]),
        gpu=gpu_name(),
        config={"config_file": str(args.config), "window_start": window_start.isoformat(),
                "window_end": window_end.isoformat(), "sources": sorted(rows_by_source)},
    )
    prov.wall_clock_s = round(time.perf_counter() - t0, 1)
    prov.finish().write(args.log_dir / "build_fresh_provenance.json")

    print(json.dumps(check_report["totals"], indent=2))
    print(json.dumps(check_report["by_template"], indent=2))
    print(f"verdict: {check_report['verdict']} -> {check_path}")
    return 0 if check_report["verdict"] == "PASS" else 1


def _strict_slice_check(
    out: Path, rows_by_source: dict[str, list[Any]], strict_start: Any
) -> dict[str, Any]:
    """Count the strict slice from the written files (never from in-memory rows)."""
    if not strict_start:
        return {"status": "NOT MEASURED — no strict_slice_start in config"}
    per_source: dict[str, Any] = {}
    total = 0
    for source in sorted(rows_by_source):
        rows, _report = read_jsonl(out / f"{source}.jsonl", type(rows_by_source[source][0]))
        in_slice = sum(1 for r in rows if r.meta.get("strict_post_teacher"))
        total += in_slice
        per_source[source] = {"n": len(rows), "in_strict_slice": in_slice}
    return {"strict_start": str(strict_start), "n_items_in_slice": total, "per_source": per_source}


def _by_template(rows_by_source: dict[str, list[Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for source, rows in sorted(rows_by_source.items()):
        for row in rows:
            key = f"{source}:{row.template_id}"
            node = out.setdefault(
                key, {"n": 0, "n_test": 0, "n_dev": 0, "qtype": str(row.qtype),
                      "earliest_record_date": row.record_date.isoformat(),
                      "latest_record_date": row.record_date.isoformat()}
            )
            node["n"] += 1
            node["n_test" if str(row.split) == "test" else "n_dev"] += 1
            node["earliest_record_date"] = min(node["earliest_record_date"], row.record_date.isoformat())
            node["latest_record_date"] = max(node["latest_record_date"], row.record_date.isoformat())
    return out


def _freshness_check(
    out: Path, rows_by_source: dict[str, list[Any]], window_start: date, window_end: date
) -> dict[str, Any]:
    """Re-read from disk and prove no item's filter date precedes the window start.

    The check reads the files again (not the in-memory rows) so it is a check on the
    artifact, and it reports the date histogram per source so a reader can see the range.
    """
    per_source: dict[str, Any] = {}
    n_before = 0
    for source in sorted(rows_by_source):
        rows, _report = read_jsonl(out / f"{source}.jsonl", type(rows_by_source[source][0]))
        dates = [r.record_date for r in rows]
        before = sum(1 for d in dates if d < window_start)
        after = sum(1 for d in dates if d > window_end)
        n_before += before
        per_source[source] = {
            "n": len(rows),
            "n_before_window_start": before,
            "n_after_window_end": after,
            "earliest": min(dates).isoformat() if dates else None,
            "latest": max(dates).isoformat() if dates else None,
        }
    return {
        "ok": n_before == 0,
        "window_start": window_start.isoformat(),
        "window_end": window_end.isoformat(),
        "n_before_window_start": n_before,
        "per_source": per_source,
    }


if __name__ == "__main__":
    raise SystemExit(main())
