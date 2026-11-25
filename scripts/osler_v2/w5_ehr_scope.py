"""osler_v2 W5: the EHR-Rel test items tagged for the O2 runners' existing scopes.

The O2 runners of osler_v1 accept the scopes mr_v0, all and panel_robustness; the mix_v2 scope is read by the zero-shot runner only.
This writes a copy of the EHR-Rel test set (data/bench/osler_v2/ehr_rel_test.jsonl) with benchmark "mr_v0", the scope every runner
accepts. Only the tag changes: the items, options, gold labels and ids are the same. Nothing else reads the tag.

Usage: uv run --frozen python scripts/osler_v2/w5_ehr_scope.py
"""

from __future__ import annotations

import json
from pathlib import Path

from meddecide.utils.hashing import file_sha256
from meddecide.utils.io import write_jsonl

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "data" / "bench" / "osler_v2" / "ehr_rel_test.jsonl"
DST = REPO / "data" / "bench" / "osler_v2" / "ehr_rel_test_mrv0.jsonl"


def main() -> None:
    rows = [json.loads(line) for line in SRC.open(encoding="utf-8")]
    for row in rows:
        row["benchmark"] = "mr_v0"
    write_jsonl(DST, rows)
    print(f"wrote {DST} items={len(rows)} sha256={file_sha256(DST)}")


if __name__ == "__main__":
    main()
