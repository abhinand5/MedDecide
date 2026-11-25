"""Y4 (ADVISORY Y4 §4): the MIMIC partition of fast dev v4, as its own item file for the baselines. Every row of the partition is kept;
the counts are written to outputs/osler_v4/Y4/partition_items_manifest.json (no item text is printed or committed).

Writes data/bench/osler_v4/fastdev_v4_mimic.jsonl (private item text, gitignored).
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from meddecide.utils.hashing import file_sha256
from meddecide.utils.io import iter_jsonl, write_json

FASTDEV = Path("data/bench/osler_v4/fastdev_v4.jsonl")
OUT = Path("data/bench/osler_v4/fastdev_v4_mimic.jsonl")
MANIFEST = Path("outputs/osler_v4/Y4/partition_items_manifest.json")


def main() -> None:
    counts: Counter[str] = Counter()
    kept = 0
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as handle:
        for row in iter_jsonl(FASTDEV):
            counts["rows read"] += 1
            if row["fastdev_partition"] != "mimic":
                counts["other partitions (not written)"] += 1
                continue
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            kept += 1
            counts["mimic partition written"] += 1
    write_json(MANIFEST, {"source": str(FASTDEV), "source_sha256": file_sha256(FASTDEV), "output": str(OUT),
                          "output_sha256": file_sha256(OUT), "written": kept, "counts": dict(sorted(counts.items()))})
    print(dict(sorted(counts.items())))


if __name__ == "__main__":
    main()
