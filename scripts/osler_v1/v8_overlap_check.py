"""V8 overlap check: do the fast dev items share text or records with a block's training file? (CPU; no GPU)

The fast dev set selects checkpoints and decides the keep rule, so any text that also occurs in a block's training rows
would inflate the dev numbers. This script counts, for each block file and each dev partition (MR, general, knowledge),
the dev items whose normalised text key (state, question and option labels, as in meddecide.mix.leakage) equals a training
row's key, and the dev items whose dataset-qualified record id equals a training row's (O4's identity rule). It also counts
the duplicate text keys inside the training file. Counts only; no item text is written.

Writes outputs/osler_v1/V8/overlap_check.json.

Usage:
    source scripts/pod_env.sh && uv run --frozen python scripts/osler_v1/v8_overlap_check.py --blocks B0 B1
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from meddecide.mix.leakage import row_text_key
from meddecide.train.blocks import KNOWLEDGE_SOURCES, MR_SOURCES
from meddecide.utils.io import write_json
from meddecide.utils.provenance import git_commit, utcnow

REPO = Path(__file__).resolve().parents[2]
TRAIN_DIR = REPO / "data" / "train" / "osler_v1"
DEV = REPO / "data" / "bench" / "osler_mr_v0" / "fastdev_mix.jsonl"
OUT = REPO / "outputs" / "osler_v1" / "V8" / "overlap_check.json"


def partition(source: str) -> str:
    if source in MR_SOURCES:
        return "mr"
    return "knowledge" if source in KNOWLEDGE_SOURCES else "general"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--blocks", nargs="+", default=["B0", "B1"])
    args = parser.parse_args()
    dev_rows = [json.loads(line) for line in DEV.open(encoding="utf-8") if line.strip()]
    dev_keys = {r["item_id"]: (row_text_key(r), f"{r['source']}/{r['source_record_id']}", partition(r["source"]))
                for r in dev_rows}
    report: dict[str, Any] = {"written_utc": utcnow(), "git_commit": git_commit(REPO), "dev_items": len(dev_rows),
                              "dev_by_partition": dict(Counter(p for _, _, p in dev_keys.values())), "blocks": {}}
    for name in args.blocks:
        train_text: set[str] = set()
        train_qualified: set[str] = set()
        train_rows = 0
        duplicates = 0
        with (TRAIN_DIR / f"block_{name}.jsonl").open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                key = row_text_key(r)
                train_rows += 1
                if key in train_text:
                    duplicates += 1
                train_text.add(key)
                train_qualified.add(f"{r['source']}/{r['source_record_id']}")
        text_hits: Counter[str] = Counter()
        qualified_hits: Counter[str] = Counter()
        for text_key, qualified, part in dev_keys.values():
            if text_key in train_text:
                text_hits[part] += 1
            if qualified in train_qualified:
                qualified_hits[part] += 1
        report["blocks"][name] = {
            "train_rows": train_rows, "duplicate_text_keys_inside_train": duplicates,
            "dev_items_with_text_in_train": dict(text_hits),
            "dev_items_with_qualified_record_in_train": dict(qualified_hits),
        }
        print(name, report["blocks"][name], flush=True)
    write_json(OUT, report)
    print(f"wrote {OUT.relative_to(REPO)}", flush=True)


if __name__ == "__main__":
    main()
