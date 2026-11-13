#!/usr/bin/env python
"""osler_v7 T8 — convert the v7 mix and the T2 dev set into the d3 harness's row format (CPU).

The 9B arm trains on the same v7 mix as the 4B arm, through `meddecide.distill.d3_train.D3Student`. That harness takes
row dicts with `row_id`, `source`, `template_id`, `qtype`, `state`, `question`, `options` (list), `gold` (key),
`split` and an optional `teacher_probs` in the row's own option order (the harness itself renders noul as no/yes and
permutes the targets). This script writes:

- `data/train/osler_v7/d3_pool.jsonl`: every v7 mix row, with `row_id = item_id` and the teacher probabilities where the
  mix has them (438,188 rows);
- `data/train/osler_v7/d3_dev.jsonl`: the frozen T2 dev rows (22,934), for evaluation only, split "dev";
- an aggregate manifest with counts and SHA-256.

No item text is committed; the files stay under the gitignored `data/train/osler_v7/`.

Run (CPU): ``uv run --frozen python scripts/osler_v7/t8_prepare_data.py``
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from meddecide.utils.provenance import utcnow  # noqa: E402

MIX = ROOT / "data" / "train" / "osler_v7" / "mix_v7.jsonl"
MIX_MANIFEST = ROOT / "outputs" / "osler_v7" / "T6" / "mix_manifest.json"
DEV = ROOT / "data" / "bench" / "osler_v7" / "dev_format_matched.jsonl"
POOL = ROOT / "data" / "train" / "osler_v7" / "d3_pool.jsonl"
DEV_OUT = ROOT / "data" / "train" / "osler_v7" / "d3_dev.jsonl"
MANIFEST = ROOT / "outputs" / "osler_v7" / "T8" / "data_manifest.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def options_of(row: dict[str, Any]) -> list[dict[str, Any]]:
    options = row["options"]
    return options if isinstance(options, list) else json.loads(options)


def convert(row: dict[str, Any], *, dev: bool) -> dict[str, Any]:
    return {
        "row_id": str(row["item_id"]),
        "source": str(row["source"]),
        "template_id": str(row["template_id"]),
        "qtype": str(row["qtype"]),
        "state": str(row["state"]),
        "question": str(row["question"]),
        "options": [{"key": str(o["key"]), "label": str(o["label"])} for o in options_of(row)],
        "gold": str(row["gold"]),
        "split": "dev" if dev else "train",
        "teacher_probs": [float(p) for p in row["teacher_probs"]] if not dev and row.get("teacher_probs") is not None
        else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mix", type=Path, default=MIX)
    parser.add_argument("--dev", type=Path, default=DEV)
    parser.add_argument("--pool", type=Path, default=POOL)
    parser.add_argument("--dev-out", type=Path, default=DEV_OUT)
    args = parser.parse_args()
    started = time.time()
    mix_manifest = json.loads(MIX_MANIFEST.read_text(encoding="utf-8"))
    if sha256_file(args.mix) != mix_manifest["sha256"]:
        raise SystemExit("the v7 mix sha256 differs from its manifest: stop")
    counts: Counter[str] = Counter()
    qtypes: Counter[str] = Counter()
    labelled = 0
    args.pool.parent.mkdir(parents=True, exist_ok=True)
    with args.mix.open(encoding="utf-8") as handle, args.pool.open("w", encoding="utf-8") as out:
        for line in handle:
            row = json.loads(line)
            converted = convert(row, dev=False)
            out.write(json.dumps(converted, ensure_ascii=False) + "\n")
            counts["pool_rows"] += 1
            qtypes[str(row["qtype"])] += 1
            if converted["teacher_probs"] is not None:
                labelled += 1
    with args.dev.open(encoding="utf-8") as handle, args.dev_out.open("w", encoding="utf-8") as out:
        for line in handle:
            row = json.loads(line)
            out.write(json.dumps(convert(row, dev=True), ensure_ascii=False) + "\n")
            counts["dev_rows"] += 1
    manifest = {
        "generated_utc": utcnow(),
        "mix": {"path": str(args.mix.relative_to(ROOT)), "sha256": mix_manifest["sha256"]},
        "pool": {"path": str(args.pool.relative_to(ROOT)), "sha256": sha256_file(args.pool), "rows": counts["pool_rows"],
                 "labelled_rows": labelled, "qtypes": dict(sorted(qtypes.items()))},
        "dev": {"path": str(args.dev_out.relative_to(ROOT)), "sha256": sha256_file(args.dev_out), "rows": counts["dev_rows"]},
        "wall_seconds": round(time.time() - started, 1),
        "notes": ["teacher_probs are in the row's own option order; the d3 harness permutes noul targets itself",
                  "the dev rows are evaluation-only (split dev)"],
    }
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=1)[:1200])


if __name__ == "__main__":
    main()
