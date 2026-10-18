"""osler_v3 X7 (ADVISORY X2, X7): the A3 training block = osler_v2's C1 rows plus the broad general replay, with no template above 8 %.

- The C1 rows are copied unchanged from data/train/osler_v2/block_C1.jsonl (osler_v2's kept mix).
- The broad replay rows (data/train/osler_v3/replay_broad_train.jsonl, from X2) are added. Each broad template may keep at most
  8 % of the final block (osler_v1 V7 rule). The cap is solved as a fixed point (meddecide.train.osler_v3.capped_sizes); a capped
  template keeps the rows with the lowest stable hashes of their item ids. Every drop is counted.
- The shares of every template are checked after the build (the cap must hold for all templates, C1's included).

Writes data/train/osler_v3/block_A3.jsonl (training rows; gitignored) and outputs/osler_v3/X7/block_A3.json (counts, shares).

Usage:  source scripts/pod_env.sh && uv run --frozen python scripts/osler_v3/x7_build_a3.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

from meddecide.train.osler_v3 import TEMPLATE_CAP, capped_sizes
from meddecide.utils.hashing import file_sha256, stable_hash
from meddecide.utils.io import write_json
from meddecide.utils.provenance import git_commit, utcnow

REPO = Path(__file__).resolve().parents[2]
C1 = REPO / "data" / "train" / "osler_v2" / "block_C1.jsonl"
BROAD = REPO / "data" / "train" / "osler_v3" / "replay_broad_train.jsonl"
A3 = REPO / "data" / "train" / "osler_v3" / "block_A3.jsonl"
OUT = REPO / "outputs" / "osler_v3" / "X7" / "block_A3.json"
GENERAL_REPLAY = ("snli", "winogrande", "hellaswag", "commonsenseqa", "boolq", "qasc", "arc_easy", "arc_challenge")


def main() -> int:
    c1_templates: Counter[str] = Counter()
    c1_sources: Counter[str] = Counter()
    c1_rows = 0
    with C1.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            c1_templates[str(row["template_id"])] += 1
            c1_sources[str(row["source"])] += 1
            c1_rows += 1
    broad: dict[str, list[dict]] = {}
    with BROAD.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                row = json.loads(line)
                broad.setdefault(str(row["template_id"]), []).append(row)
    # the cap is applied to the broad templates; C1's templates are fixed and checked afterwards
    total, kept_per_template = capped_sizes(fixed_total=c1_rows, sizes={t: len(v) for t, v in broad.items()})
    kept_rows = []
    dropped: dict[str, int] = {}
    for template, rows in sorted(broad.items()):
        keep_n = kept_per_template[template]
        ordered = sorted(rows, key=lambda r: stable_hash({"a3": str(r["item_id"])}, length=16))
        kept_rows.extend(ordered[:keep_n])
        dropped[template] = len(rows) - keep_n
    kept_rows.sort(key=lambda r: (str(r["template_id"]), str(r["item_id"])))
    A3.parent.mkdir(parents=True, exist_ok=True)
    with A3.open("w", encoding="utf-8") as out, C1.open(encoding="utf-8") as fin:
        for line in fin:
            if line.strip():
                out.write(line if line.endswith("\n") else line + "\n")
        for row in kept_rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
    templates = c1_templates + Counter(str(r["template_id"]) for r in kept_rows)
    total_rows = sum(templates.values())
    shares = {t: n / total_rows for t, n in templates.items()}
    largest = max(shares, key=shares.get)
    general = sum(c1_sources[s] for s in GENERAL_REPLAY) + len(kept_rows)  # the broad sources are general replay
    report = {
        "built_utc": utcnow(), "git_commit": git_commit(REPO), "command": "scripts/osler_v3/x7_build_a3.py",
        "c1_file_sha256": file_sha256(C1), "a3_file": str(A3.relative_to(REPO)), "a3_rows": total_rows, "a3_rows_check": total,
        "c1_rows": c1_rows, "broad_rows_read": sum(len(v) for v in broad.values()), "broad_rows_kept": len(kept_rows),
        "broad_dropped_by_cap": dropped, "cap": TEMPLATE_CAP,
        "largest_template": largest, "largest_template_share": round(shares[largest], 4),
        "cap_holds": all(v <= TEMPLATE_CAP + 1e-12 for v in shares.values()),
        "general_replay_rows": general, "general_replay_share": round(general / total_rows, 4),
        "c1_general_replay_share": round(sum(c1_sources[s] for s in GENERAL_REPLAY) / c1_rows, 4),
        "templates": len(templates),
    }
    write_json(OUT, report)
    print(json.dumps({k: report[k] for k in ("a3_rows", "broad_rows_kept", "largest_template", "largest_template_share",
                                             "cap_holds", "general_replay_share")}), flush=True)
    return 0 if report["cap_holds"] and total_rows == total else 1


if __name__ == "__main__":
    sys.exit(main())
