"""Y6 (osler_v4 ADVISORY Y6; D48): the A5 training mix — osler_v3's broad-replay mix (block_A3) with the Y5 teacher distributions
attached to its anchor rows that have one (meddecide.distill.soft_labels.attach_teacher: anchor templates only, matched by
normalised text, option keys the same and in the same order). Matching is by text, not by item id: the item id is not
unique (CLAIMS C059, C062), and the Y5 pool holds one row per text. Every row is written, and every status is counted.

Writes data/train/osler_v4/y6_a5_mix.jsonl (private training data, gitignored) and outputs/osler_v4/Y6/a5_mix_manifest.json.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from meddecide.distill import soft_labels as sl
from meddecide.utils.hashing import file_sha256
from meddecide.utils.io import iter_jsonl, write_json
from meddecide.utils.provenance import utcnow

MIX = Path("data/train/osler_v3/block_A3.jsonl")
SOFT = Path("data/train/osler_v4/y5_anchor_soft.jsonl")
OUT = Path("data/train/osler_v4/y6_a5_mix.jsonl")
MANIFEST = Path("outputs/osler_v4/Y6/a5_mix_manifest.json")


def main() -> None:
    soft: dict[str, tuple[list[str], list[float]]] = {}  # normalised text key -> (option keys, teacher probs)
    for row in iter_jsonl(SOFT):
        key = sl.overlap_keys(row)[0]
        if key in soft:
            raise SystemExit("two soft-label rows share a text key: the Y5 pool should hold one row per text: stop")
        soft[key] = ([str(o["key"]) for o in row["options"]], [float(p) for p in row["teacher_probs"]])
    statuses: Counter[str] = Counter()
    attached_by_qtype: Counter[str] = Counter()
    used: set[str] = set()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as handle:
        for row in iter_jsonl(MIX):
            out, status = sl.attach_teacher(row, soft)
            statuses[status] += 1
            if status == "anchor: attached":
                attached_by_qtype[str(row["qtype"])] += 1
                used.add(sl.overlap_keys(row)[0])
            handle.write(json.dumps(out, ensure_ascii=False) + "\n")
    rows = sum(statuses.values())
    anchors = sum(n for s, n in statuses.items() if s.startswith("anchor"))
    if rows != anchors + sum(n for s, n in statuses.items() if s.startswith("domain")):
        raise SystemExit("statuses do not add up to rows: stop")
    write_json(MANIFEST, {
        "mix": str(MIX),
        "mix_sha256": file_sha256(MIX),
        "soft_label_file": str(SOFT),
        "soft_label_file_sha256": file_sha256(SOFT),
        "soft_label_texts": len(soft),
        "soft_label_texts_used": len(used),
        "output": str(OUT),
        "output_sha256": file_sha256(OUT),
        "rows": rows,
        "anchor_rows": anchors,
        "statuses": dict(sorted(statuses.items())),
        "attached_by_qtype": dict(attached_by_qtype),
        "generated_utc": utcnow(),
    })
    print(f"[{utcnow()}] A5 mix: rows {rows}; anchors {anchors}; statuses {dict(sorted(statuses.items()))}")


if __name__ == "__main__":
    main()
