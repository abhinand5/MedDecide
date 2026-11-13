#!/usr/bin/env python
"""osler_v7 T6 — the v7 training mix: the v6 distilled mix plus the labelled C1-C4 pool.

ADVISORY §2/§3: the arm trains on "v6 mix + C1-C4" with the teacher labels attached and the KL-to-base term on the
C1-C4 prompts (`v7_c*` templates; the trainer's `kl_include_template_prefixes`). This script writes that mix once:
- verifies the v6 mix by SHA-256 (the 385,536-anchor distilled file osler_v6 trained on);
- verifies the T5 labelled pool by its join-manifest SHA-256;
- appends the C1-C4 rows (each with its pplx-27B `teacher_probs`) after the v6 rows;
- writes the composition manifest (counts by cell, template, source and qtype, labelled share, family shares as
  measured, SHA-256). Rows with a family share above 25 % are listed as measured findings; the mix itself is not
  reweighted here (the loss weighting is the trainer's per-item rule).

Writes `data/train/osler_v7/mix_v7.jsonl` (private, gitignored) and `outputs/osler_v7/T6/mix_manifest.json`.

Run (CPU): ``uv run --frozen python scripts/osler_v7/t6_build_mix.py``
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

V6_MIX = ROOT / "data" / "train" / "osler_v6" / "mix_f2_distill.jsonl"
V6_SHA = "91ff5362d56c7b14aafa1bb62e8d78fd31312c613ff5d310892f7e09c97137ce"
LABELLED = ROOT / "data" / "train" / "osler_v7" / "labelled.jsonl"
JOIN_MANIFEST = ROOT / "outputs" / "osler_v7" / "T5" / "join_manifest.json"
OUT = ROOT / "data" / "train" / "osler_v7" / "mix_v7.jsonl"
MANIFEST = ROOT / "outputs" / "osler_v7" / "T6" / "mix_manifest.json"


def utcnow() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def options_of(row: dict[str, Any]) -> list[Any]:
    options = row["options"]
    return options if isinstance(options, list) else json.loads(options)


def cell_of(row: dict[str, Any]) -> str:
    template = str(row["template_id"])
    for prefix, name in (("v7_c1", "C1"), ("v7_c2", "C2"), ("v7_c3", "C3"), ("v7_c4", "C4")):
        if template.startswith(prefix):
            return name
    return "C0"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    t0 = time.time()
    started = utcnow()
    join_manifest = json.loads(JOIN_MANIFEST.read_text(encoding="utf-8"))
    v6_sha = sha256_file(V6_MIX)
    if v6_sha != V6_SHA:
        raise SystemExit(f"the v6 mix sha256 is {v6_sha}, not {V6_SHA}: stop")
    labelled_sha = sha256_file(LABELLED)
    if labelled_sha != join_manifest["labelled_sha256"]:
        raise SystemExit("the labelled pool sha256 differs from the T5 join manifest: stop")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter()
    cell_rows: Counter[str] = Counter()
    labelled_rows = 0
    seen_ids: set[str] = set()
    v6_duplicate_rows = 0
    v6_duplicate_ids: set[str] = set()
    with args.out.open("w", encoding="utf-8") as out:
        with V6_MIX.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                item_id = str(row["item_id"])
                if item_id in seen_ids:
                    # a pre-existing property of the v6 mix (kept as trained); counted, never silently fixed here
                    v6_duplicate_rows += 1
                    v6_duplicate_ids.add(item_id)
                seen_ids.add(item_id)
                counts["v6_rows"] += 1
                cell_rows["C0"] += 1
                if row.get("teacher_probs") is not None:
                    labelled_rows += 1
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
        with LABELLED.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                item_id = str(row["item_id"])
                if item_id in seen_ids:
                    raise SystemExit(f"C1-C4 item id collides with the v6 mix: {item_id}")
                seen_ids.add(item_id)
                if row.get("teacher_probs") is None:
                    raise SystemExit(f"C1-C4 row {item_id} has no teacher_probs: stop")
                if not 1 <= len(options_of(row)) <= 255:
                    raise SystemExit(f"C1-C4 row {item_id} has an invalid option count")
                counts["c1c4_rows"] += 1
                labelled_rows += 1
                cell_rows[cell_of(row)] += 1
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
    rows = counts["v6_rows"] + counts["c1c4_rows"]
    # Re-read the output for the composition block (one more pass; the file is large but this keeps the manifest
    # derived from the written artifact rather than from counters that could drift).
    templates: Counter[str] = Counter()
    sources: Counter[str] = Counter()
    qtypes: Counter[str] = Counter()
    cell_qtypes: Counter[str] = Counter()
    with args.out.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            templates[str(row["template_id"])] += 1
            sources[str(row["source"])] += 1
            qtypes[str(row["qtype"])] += 1
            cell_qtypes[f"{cell_of(row)}:{row['qtype']}"] += 1
    family_share = {cell: round(n / rows, 8) for cell, n in sorted(cell_rows.items())}
    manifest = {
        "generated_utc": utcnow(),
        "started_utc": started,
        "rows": rows,
        "v6_rows": counts["v6_rows"],
        "c1c4_rows": counts["c1c4_rows"],
        "labelled_rows": labelled_rows,
        "v6_duplicate_rows": v6_duplicate_rows,
        "v6_duplicate_ids": len(v6_duplicate_ids),
        "labelled_share": round(labelled_rows / rows, 8),
        "cells": dict(sorted(cell_rows.items())),
        "family_share": family_share,
        "family_share_above_25pct": [cell for cell, share in family_share.items() if share > 0.25],
        "qtypes": dict(sorted(qtypes.items())),
        "cell_qtypes": dict(sorted(cell_qtypes.items())),
        "sources": dict(sorted(sources.items())),
        "templates": dict(sorted(templates.items())),
        "v6_mix": {"path": str(V6_MIX.relative_to(ROOT)), "sha256": v6_sha, "rows": counts["v6_rows"]},
        "c1c4": {"path": str(LABELLED.relative_to(ROOT)), "sha256": labelled_sha, "rows": counts["c1c4_rows"],
                 "join_manifest": str(JOIN_MANIFEST.relative_to(ROOT))},
        "path": str(args.out.relative_to(ROOT)),
        "sha256": sha256_file(args.out),
        "wall_seconds": round(time.time() - t0, 1),
        "notes": [
            "the KL-to-base pool is the v7_c* rows of this mix (the trainer is configured with that include prefix)",
            "the family share is measured on this mix; the ADVISORY 25% cap is reported, not silently reweighted",
        ],
    }
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"rows": rows, "c1c4": counts["c1c4_rows"], "labelled": labelled_rows,
                      "sha256": manifest["sha256"][:12], "family_share": family_share}, indent=1))


if __name__ == "__main__":
    main()
