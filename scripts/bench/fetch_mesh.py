#!/usr/bin/env python
"""Build the public MeSH descriptor index used for near-miss distractors.

Downloads NLM's 2026 MeSH descriptor file (``desc2026.gz``, 16.8 MB; NLM MeSH is public
domain), streams it into ``{descriptor name: [tree numbers]}``, and saves
``{"meta": ..., "by_name": ...}`` for the benchmark template builder.

    uv run python scripts/bench/fetch_mesh.py \
        --cache-dir /workspace/tmp/mesh --out /workspace/tmp/mesh/mesh_index.json

``--from-file`` parses an already-downloaded local copy (no network); ``--examples`` prints
the sibling lists for named descriptors. Nothing is written outside ``--cache-dir`` except
the ``--out`` index the caller names explicitly.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from meddecide.bench.mesh import (
    DESC_GZ_URL,
    DESC_XML_URL,
    LICENSE,
    MESH_YEAR,
    SOURCE,
    MeshIndex,
    download_descriptors,
    parse_descriptors_with_stats,
    save_index,
)
from meddecide.utils.hashing import file_sha256


def default_cache_dir() -> Path:
    """Cache on the large volume (``TMPDIR``, set by scripts/pod_env.sh)."""
    return Path(os.environ.get("TMPDIR", "/workspace/tmp")) / "mesh"


def build_index(path: Path, *, url: str) -> tuple[MeshIndex, dict[str, Any]]:
    """Parse ``path`` into a :class:`MeshIndex` with a provenance ``meta`` block (R7)."""
    started = time.monotonic()
    by_name, stats = parse_descriptors_with_stats(path)
    parse_seconds = time.monotonic() - started
    n_with_tree_numbers = (
        stats["n_records"] - stats["n_skipped_no_name"] - stats["n_skipped_no_tree_numbers"]
    )
    meta: dict[str, Any] = {
        "year": MESH_YEAR,
        "source": SOURCE,
        "license": LICENSE,
        "url": url,
        "file": path.name,
        "sha256": file_sha256(path),
        "bytes": path.stat().st_size,
        "n_descriptors": len(by_name),
        "n_with_tree_numbers": n_with_tree_numbers,
        "n_records": stats["n_records"],
        "n_skipped_no_name": stats["n_skipped_no_name"],
        "n_skipped_no_tree_numbers": stats["n_skipped_no_tree_numbers"],
        "n_duplicate_names": stats["n_duplicate_names"],
        "n_tree_numbers": stats["n_tree_numbers"],
    }
    timing = {"parse_seconds": round(parse_seconds, 3), **stats}
    return MeshIndex(by_name=by_name, meta=meta), timing


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir", type=Path, default=None, help="download cache (default: $TMPDIR/mesh)"
    )
    parser.add_argument(
        "--out", type=Path, default=None, help="index path (default: cache-dir/mesh_index.json)"
    )
    parser.add_argument(
        "--from-file", type=Path, default=None, help="parse a local copy; no network"
    )
    parser.add_argument(
        "--prefer", choices=("gz", "xml"), default="gz", help="which remote file to fetch"
    )
    parser.add_argument(
        "--examples",
        nargs="+",
        default=None,
        metavar="NAME",
        help="descriptors to print siblings for",
    )
    parser.add_argument("--min-siblings", type=int, default=3)
    parser.add_argument("--max-siblings", type=int, default=None)
    args = parser.parse_args()

    started = time.monotonic()
    cache_dir = Path(args.cache_dir) if args.cache_dir else default_cache_dir()
    out = Path(args.out) if args.out else cache_dir / "mesh_index.json"
    url = DESC_GZ_URL if args.prefer == "gz" else DESC_XML_URL

    if args.from_file is not None:
        source_path = Path(args.from_file)
        if not source_path.is_file():
            parser.error(f"--from-file does not exist: {source_path}")
    else:
        source_path = download_descriptors(cache_dir, prefer_gz=args.prefer == "gz")
    if out.resolve() == source_path.resolve():
        parser.error(f"--out would overwrite the parsed source file: {out}")

    index, timing = build_index(source_path, url=url)
    save_index(index, out)

    report: dict[str, Any] = {
        "meta": index.meta,
        "index_path": str(out),
        "index_bytes": out.stat().st_size,
        "parse_seconds": timing["parse_seconds"],
        "total_seconds": round(time.monotonic() - started, 3),
        "counts": {
            "n_records": timing["n_records"],
            "n_kept": timing["n_kept"],
            "n_skipped_no_name": timing["n_skipped_no_name"],
            "n_skipped_no_tree_numbers": timing["n_skipped_no_tree_numbers"],
            "n_duplicate_names": timing["n_duplicate_names"],
            "n_tree_numbers": timing["n_tree_numbers"],
        },
    }
    if args.examples:
        report["examples"] = {
            name: {
                "known": index.has_descriptor(name),
                "tree_numbers": index.by_name.get(name, []),
                "siblings": index.siblings(
                    name, min_siblings=args.min_siblings, max_siblings=args.max_siblings
                ),
            }
            for name in args.examples
        }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
