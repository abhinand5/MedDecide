"""O11: leakage check of the Osler training mix against the O2 common set's v0.2 test items and external panel (CPU).

ADVISORY failure mode 5 (held-out leakage): no training row may be a v0.2 test item or an external-panel item, matched by item id
or by question text (state and question together). Writes outputs/osler_v0/O11/leakage_check.json (gitignored); the CLAIMS row cites
its counts.

Usage:
    uv run --frozen python scripts/osler/o11_leakage_check.py
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from meddecide.utils.io import iter_jsonl, write_json
from meddecide.utils.provenance import git_commit, utcnow

REPO = Path(__file__).resolve().parents[2]
ITEMS = REPO / "data/bench/v0.3_ext/o2_items.jsonl"
TRAIN = REPO / "data/train/osler_v0/train.jsonl"
OUT = REPO / "outputs/osler_v0/O11/leakage_check.json"


def text_key(row: Mapping[str, Any]) -> str:
    """A content hash of the state and the question (the text a model reads with the item)."""
    return hashlib.sha256((str(row["state"]) + "\n" + str(row["question"])).encode("utf-8")).hexdigest()


def overlaps(train: Iterable[Mapping[str, Any]], v02_test: Sequence[Mapping[str, Any]],
             panel: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Counts of training rows that match a v0.2 test item or an external-panel item, by item id and by text."""
    v02_ids = {r["item_id"] for r in v02_test}
    v02_texts = {text_key(r) for r in v02_test}
    panel_ids = {r["item_id"] for r in panel}
    panel_texts = {text_key(r) for r in panel}
    counts = {"train_rows": 0, "v02_test_id": 0, "v02_test_text": 0, "panel_id": 0, "panel_text": 0}
    for row in train:
        counts["train_rows"] += 1
        counts["v02_test_id"] += row["item_id"] in v02_ids
        counts["v02_test_text"] += text_key(row) in v02_texts
        counts["panel_id"] += row["item_id"] in panel_ids
        counts["panel_text"] += text_key(row) in panel_texts
    return counts


def main() -> None:
    items = list(iter_jsonl(ITEMS))
    v02_test = [r for r in items if r["benchmark"] == "v0.2" and r.get("split") == "test"]
    panel = [r for r in items if r["benchmark"] == "ext_panel"]
    counts = overlaps(iter_jsonl(TRAIN), v02_test, panel)
    record = {
        "kind": "osler_v0_O11_leakage_check",
        "built_at_utc": utcnow(),
        "git_commit": git_commit(REPO),
        "command": "scripts/osler/o11_leakage_check.py",
        "items_file": str(ITEMS.relative_to(REPO)),
        "train_file": str(TRAIN.relative_to(REPO)),
        "v02_test_items": len(v02_test),
        "panel_items": len(panel),
        **counts,
    }
    write_json(OUT, record)
    print(json.dumps({k: record[k] for k in ("train_rows", "v02_test_items", "panel_items", "v02_test_id",
                                             "v02_test_text", "panel_id", "panel_text")}))


if __name__ == "__main__":
    main()
