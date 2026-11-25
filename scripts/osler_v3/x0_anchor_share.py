"""osler_v3 X0: the anchor share of the osler_v2 C1 block under the ADVISORY §3.2 domain prefixes (CPU; streams the file).

Counts items per template id in one pass, then the anchor and domain counts with meddecide.train.osler_v3. Writes
outputs/osler_v3/X0/anchor_share_C1.json: totals, per-prefix counts and per-template counts (counts only; no item text).

Usage:  source scripts/pod_env.sh && uv run --frozen python scripts/osler_v3/x0_anchor_share.py
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from meddecide.train.osler_v3 import DOMAIN_PREFIXES, is_anchor
from meddecide.utils.io import write_json
from meddecide.utils.provenance import git_commit, utcnow

REPO = Path(__file__).resolve().parents[2]
TRAIN = REPO / "data" / "train" / "osler_v2" / "block_C1.jsonl"
OUT = REPO / "outputs" / "osler_v3" / "X0" / "anchor_share_C1.json"


def main() -> None:
    counts: Counter[str] = Counter()
    with TRAIN.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                counts[str(json.loads(line)["template_id"])] += 1
    total = sum(counts.values())
    anchors = sum(c for t, c in counts.items() if is_anchor(t))
    by_prefix = {p: sum(c for t, c in counts.items() if t.startswith(p)) for p in DOMAIN_PREFIXES}
    by_prefix = {p: n for p, n in by_prefix.items() if n}
    write_json(OUT, {
        "block": "C1", "train_file": str(TRAIN.relative_to(REPO)), "items": total, "anchors": anchors,
        "domain": total - anchors, "anchor_share": anchors / total, "domain_prefixes": list(DOMAIN_PREFIXES),
        "domain_items_by_prefix": by_prefix, "templates": len(counts),
        "items_by_template": dict(sorted(counts.items())), "git_commit": git_commit(REPO), "written_utc": utcnow(),
        "command": "scripts/osler_v3/x0_anchor_share.py",
    })
    print(f"C1: {total} items, anchors {anchors} ({anchors / total:.4f}), domain {total - anchors}, "
          f"templates {len(counts)}", flush=True)


if __name__ == "__main__":
    main()
