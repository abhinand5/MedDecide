"""V1: build the medical-record audit generators' splits, screen them, and write the manifest (osler_v1 ADVISORY V1).

Same procedure as osler_v0 O3: patients "<generator>-<k>", split by stable hash of the id (80/10/10 by patient, so twins
share a split), quotas per split, the O3 shortcut screen (string presence and bag-of-words Naive Bayes) on dev.

Items about a held-out indicator or ICD family (``HELD_OUT_INDICATORS``, ``HELD_OUT_ICD_FAMILIES``) are moved out of
every split into heldout_test.jsonl (their dev and test patients only; their train patients are dropped and counted).

Outputs (gitignored): data/gen/osler_v1/{train,dev,test,heldout_test}.jsonl and manifest.json;
outputs/osler_v1/V1/screen.json.

Usage:
    source scripts/pod_env.sh && uv run --frozen python scripts/osler_v1/v1_build_mr_generators.py [--train 6000 --dev 300 --test 300]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from meddecide.gen.core import split_of
from meddecide.gen.screen import screen_generator
from meddecide.mr.generators import GENERATORS, is_held_out
from meddecide.utils.hashing import file_sha256
from meddecide.utils.io import write_json
from meddecide.utils.provenance import git_commit, utcnow

REPO = Path(__file__).resolve().parents[2]


def build(gen, quota: dict[str, int]) -> dict[str, list[dict[str, Any]]]:
    got: dict[str, list[dict[str, Any]]] = {"train": [], "dev": [], "test": []}
    remaining = dict(quota)
    k = 0
    while any(v > 0 for v in remaining.values()):
        pid = f"{gen.name}-{k:07d}"
        k += 1
        split = split_of(pid)
        if remaining[split] <= 0:
            continue
        remaining[split] -= 1
        for it in gen.make(pid):
            row = it.as_dict()
            row["held_out"] = gen.held_out or is_held_out(row["meta"])
            row["patient_id"] = pid
            got[split].append(row)
    return got


def separate_held_out(splits: dict[str, list[dict[str, Any]]]) -> tuple[list[dict[str, Any]], int]:
    """Remove held-out rows from every split in place; return (held-out dev+test rows, dropped train rows)."""
    held = [r for s in ("dev", "test") for r in splits[s] if r["held_out"]]
    dropped = sum(r["held_out"] for r in splits["train"])
    for s in splits:
        splits[s] = [r for r in splits[s] if not r["held_out"]]
    return held, dropped


def _rel(path: Path) -> str:
    return str(path.relative_to(REPO)) if path.is_relative_to(REPO) else str(path)


def gold_label_of(row: dict[str, Any]) -> str:
    return next(o["label"] for o in row["options"] if o["key"] == row["gold"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=6000, help="train patients per generator")
    ap.add_argument("--dev", type=int, default=300)
    ap.add_argument("--test", type=int, default=300)
    ap.add_argument("--out", default="data/gen/osler_v1")
    ap.add_argument("--screen", default="outputs/osler_v1/V1/screen.json")
    args = ap.parse_args()
    out = REPO / args.out
    out.mkdir(parents=True, exist_ok=True)
    quota = {"train": args.train, "dev": args.dev, "test": args.test}
    per_split: dict[str, list[dict[str, Any]]] = {"train": [], "dev": [], "test": [], "heldout_test": []}
    counts: dict[str, Any] = {}
    reports = []
    for gen in GENERATORS:
        splits = build(gen, quota)
        held, dropped_train = separate_held_out(splits)
        for s in ("train", "dev", "test"):
            per_split[s].extend(splits[s])
        per_split["heldout_test"].extend(held)
        counts[gen.name] = {
            "family": gen.family, "held_out": gen.held_out,
            "items": {s: len(splits[s]) for s in ("train", "dev", "test")} | {"heldout_test": len(held)},
            "held_out_train_rows_dropped": dropped_train,
            "patients": {s: len({r["patient_id"] for r in splits[s]}) for s in ("train", "dev", "test")},
            "gold_counts_test": dict(sorted(Counter(gold_label_of(r) for r in splits["test"]).items())),
        }
        rep = screen_generator(gen.name, gen.held_out, splits["dev"], splits["train"], "own train split")
        reports.append(rep.as_dict())
        print(f"{gen.name:32s} dev={len(splits['dev'])} sp={rep.string_presence_macro} nb={rep.naive_bayes_macro} "
              f"pass={rep.passes} {rep.failures}", flush=True)
    hashes = {}
    for s, rows in per_split.items():
        path = out / f"{s}.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        hashes[s] = {"file": _rel(path), "rows": len(rows), "sha256": file_sha256(path)}
    screen_path = REPO / args.screen
    screen_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(screen_path, {"built_at_utc": utcnow(), "git_commit": git_commit(REPO), "reports": reports})
    write_json(out / "manifest.json", {
        "built_at_utc": utcnow(), "git_commit": git_commit(REPO), "quota_patients": quota, "generators": counts,
        "files": hashes, "screen_passed": all(r["passes"] for r in reports),
        "screen_file": _rel(screen_path)})
    print(f"wrote {out}: " + ", ".join(f"{s} {h['rows']}" for s, h in hashes.items()))


if __name__ == "__main__":
    main()
