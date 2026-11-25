"""V10 items (osler_v1 ADVISORY V10): the rows Osler-4B is scored on, in the O2 row format (CPU only).

The union of three scopes, each taken from a fixed file, with no model result used to choose a row:
  mr_test       the MR evaluation file's test suite (all rows; the gate's MR suite, D31 (i); MIMIC is absent, so these are the
                generated parts only)
  mr_heldout    the held-out file (dev and test rows; only the test rows are reported, and the dev rows are scored for the record)
  external_panel  the O2 common set's external clinical panel (D31 (ii))
  knowledge     the O2 common set's established MedQA and MedMCQA test items (the knowledge guard, D24 (iii))

Every selected row must convert with ``o2_row_to_item`` (the scorer's preflight) and every item id must be unique. Rows that are not
selected are counted by benchmark, never silently dropped. Writes data/bench/osler_v1/v10_items.jsonl (gitignored, with the
``v10_scope`` field added) and outputs/osler_v1/V10/items_manifest.json (counts and hashes only).

Usage:
    uv run --frozen python scripts/osler_v1/v10_items.py
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from meddecide.eval.osler_scoring import o2_row_to_item
from meddecide.utils.hashing import file_sha256
from meddecide.utils.io import write_json
from meddecide.utils.provenance import git_commit, utcnow

REPO = Path(__file__).resolve().parents[2]
MR_EVAL = REPO / "data/bench/osler_mr_v0/mr_eval.jsonl"
O2_ITEMS = REPO / "data/bench/v0.3_ext/o2_items.jsonl"
OUT_ITEMS = REPO / "data/bench/osler_v1/v10_items.jsonl"
OUT_MANIFEST = REPO / "outputs/osler_v1/V10/items_manifest.json"
KNOWLEDGE_SETS = frozenset({"medqa", "medmcqa"})
MR_SETS = frozenset({"mr_test", "mr_heldout"})


def v10_scope(row: Mapping[str, Any]) -> str | None:
    """The V10 scope of one row, or None when the row is not used in V10."""
    if row["benchmark"] == "mr_v0":
        return row["set_name"] if row["set_name"] in MR_SETS else None
    if row["benchmark"] == "ext_panel":
        return "external_panel"
    if (row["benchmark"] == "v0.2" and (row.get("meta") or {}).get("tier") == "established"
            and row["set_name"] in KNOWLEDGE_SETS and row["split"] == "test"):
        return "knowledge"
    return None


def read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def select(rows_by_file: Mapping[str, list[dict[str, Any]]]) -> tuple[list[dict[str, Any]], Counter[str]]:
    selected: list[dict[str, Any]] = []
    excluded: Counter[str] = Counter()
    for source, rows in rows_by_file.items():
        for row in rows:
            scope = v10_scope(row)
            if scope is None:
                excluded[f"{source}: benchmark {row['benchmark']}, set {row['set_name']}, split {row['split']} (not in V10)"] += 1
                continue
            selected.append({**row, "v10_scope": scope})
    ids = [str(r["item_id"]) for r in selected]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{len(ids) - len(set(ids))} duplicated item ids across the V10 scopes")
    for row in selected:
        try:
            o2_row_to_item(row)
        except (KeyError, ValueError) as exc:
            raise ValueError(f"row {row['item_id']} ({row['set_name']}) does not convert: {exc}") from exc
    return selected, excluded


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.parse_args()
    mr_rows = read_rows(MR_EVAL)
    o2_rows = read_rows(O2_ITEMS)
    selected, excluded = select({"mr_eval": mr_rows, "o2_common_set": o2_rows})
    counts = Counter(f"{r['v10_scope']} / {r['set_name']} / {r['split']}" for r in selected)
    OUT_ITEMS.parent.mkdir(parents=True, exist_ok=True)
    with OUT_ITEMS.open("w", encoding="utf-8") as fh:
        for row in selected:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "built_at_utc": utcnow(), "git_commit": git_commit(REPO), "command": "scripts/osler_v1/v10_items.py",
        "items_file": str(OUT_ITEMS.relative_to(REPO)), "items_sha256": file_sha256(OUT_ITEMS),
        "mr_eval_sha256": file_sha256(MR_EVAL), "mr_eval_rows": len(mr_rows),
        "o2_items_sha256": file_sha256(O2_ITEMS), "o2_items_rows": len(o2_rows),
        "selected_rows": len(selected), "counts_by_scope_set_split": dict(sorted(counts.items())),
        "excluded_rows": sum(excluded.values()), "excluded_by_reason": dict(sorted(excluded.items())),
        "conversion": "every selected row converts with o2_row_to_item; item ids unique",
    }
    write_json(OUT_MANIFEST, manifest)
    print(f"selected {len(selected)} rows of {len(mr_rows) + len(o2_rows)}; excluded {sum(excluded.values())}", flush=True)
    for key, value in sorted(counts.items()):
        print(f"  {key}: {value}", flush=True)
    print(f"wrote {OUT_ITEMS.relative_to(REPO)} and {OUT_MANIFEST.relative_to(REPO)}", flush=True)


if __name__ == "__main__":
    main()
