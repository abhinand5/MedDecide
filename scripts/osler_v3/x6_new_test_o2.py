"""osler_v3 X6 step 4 (ADVISORY): the new test templates (X1, X3, X4) in the O2 item format the baseline runners read.

The five baselines and zero-shot pplx-decider-flash-v1.2 are scored on these items as additional rows only (never in a verdict).
The O2 runners select items by a ``benchmark`` tag (scope "all" takes v0.2, ext_panel and robustness). The new items carry the tag
``v0.2`` for that reason alone; the true source of each item is in ``set_name`` (its source) and ``template_id``, and the manifest
says so. Nothing else of the O2 format is changed.

Inputs: data/bench/osler_v3/x1_test.jsonl (X1), data/bench/osler_v3/x3_test.jsonl (X3, when built), and the X4 test file
(mix rows of scripts/osler_v3/x4_to_mix.py; blocked chart templates are not in it). Output: data/bench/osler_v3/new_test_o2.jsonl (gitignored) and outputs/osler_v3/X6/new_test_manifest.json.

Usage:  source scripts/pod_env.sh && uv run --frozen python scripts/osler_v3/x6_new_test_o2.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from meddecide.utils.io import write_json, write_jsonl
from meddecide.utils.provenance import git_commit, utcnow

REPO = Path(__file__).resolve().parents[2]
INPUTS = {
    "x1": REPO / "data" / "bench" / "osler_v3" / "x1_test.jsonl",
    "x3": REPO / "data" / "bench" / "osler_v3" / "x3_test.jsonl",
    "x4": REPO / "data" / "bench" / "osler_v3" / "x4_test.jsonl",  # mix rows of x4_to_mix.py (blocked templates excluded)
}
OUT = REPO / "data" / "bench" / "osler_v3" / "new_test_o2.jsonl"
MANIFEST = REPO / "outputs" / "osler_v3" / "X6" / "new_test_manifest.json"
SCOPE_TAG = "v0.2"  # the runners' scope "all"; see the module docstring


def to_o2(row: dict[str, Any], part: str) -> dict[str, Any]:
    meta = row.get("meta") or {}
    return {
        "item_id": str(row["item_id"]), "benchmark": SCOPE_TAG, "set_name": str(row.get("source", "")),
        "template_id": str(row["template_id"]), "qtype": str(row["qtype"]), "state": str(row["state"]),
        "question": str(row["question"]), "options": row["options"], "gold": str(row["gold"]),
        "source_record_id": str(row.get("source_record_id", "")), "source_url": str(row.get("source_url", "")),
        "licence": str(row.get("source_license", "")), "revision": f"osler_v3 new test ({part})",
        "split": "test", "meta": json.dumps(meta, ensure_ascii=False, sort_keys=True, default=str),
    }


def main() -> int:
    rows: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for part, path in INPUTS.items():
        if not path.exists():
            counts[f"{part}_missing"] = 1
            continue
        with path.open(encoding="utf-8") as fh:
            part_rows = [json.loads(line) for line in fh if line.strip()]
        test_rows = [r for r in part_rows if r.get("split") == "test"]
        counts[f"{part}_test_rows"] = len(test_rows)
        rows.extend(to_o2(r, part) for r in test_rows)
    ids = [r["item_id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise SystemExit("duplicate item ids across the new test pools")
    write_jsonl(OUT, rows)
    write_json(MANIFEST, {
        "built_utc": utcnow(), "git_commit": git_commit(REPO), "items": len(rows), "scope_tag": SCOPE_TAG,
        "by_template": dict(Counter(r["template_id"] for r in rows)), "counts": dict(counts),
        "file": str(OUT.relative_to(REPO)), "note": "additional rows only; never in a Gate O3 verdict (ADVISORY X6, X10)",
    })
    print(f"new test pools: {len(rows)} items; {dict(counts)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
