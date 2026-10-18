"""osler_v3 X8 (ADVISORY X8; D34, D36): one Stage B training block = the last kept block plus one new source.

- The base is the last kept block (the chosen Stage A run's mix: osler_v2's C1 or the A3 block; later a kept Stage B block).
  Its lines are copied unchanged (streamed; the base is not held in memory).
- The added source is a file of mix rows (X1 sources for B1; X4 charts for B3). Templates the X4 check blocked are already excluded
  from the X4 file (scripts/osler_v3/x4_to_mix.py), and every count of that exclusion is in its report.
- Replay: general replay (the eight osler_v2 general sources and the broad replay templates `broad_*`) must make up at least 20 % of
  the block (ADVISORY X8). Missing replay rows are drawn from the broad replay pool (X2) in stable-hash order, excluding rows the
  base already holds; R is the smallest count with (replay + R) / (N + R) at least 0.20.
- After the build, no template may exceed 8 % of the block. The build fails otherwise: nothing is adjusted silently.

Writes data/train/osler_v3/block_<name>.jsonl (training rows; gitignored) and outputs/osler_v3/X8/block_<name>.json (counts, shares,
file hashes). Usage:
    source scripts/pod_env.sh && uv run --frozen python scripts/osler_v3/x8_build_block.py --name B1 \
        --base data/train/osler_v2/block_C1.jsonl --add data/train/osler_v3/x1_sources_train.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from meddecide.train.osler_v3 import TEMPLATE_CAP
from meddecide.utils.hashing import file_sha256, stable_hash
from meddecide.utils.io import write_json
from meddecide.utils.provenance import git_commit, utcnow

REPO = Path(__file__).resolve().parents[2]
BROAD = REPO / "data" / "train" / "osler_v3" / "replay_broad_train.jsonl"
OUT = REPO / "outputs" / "osler_v3" / "X8"
GENERAL_REPLAY = ("snli", "winogrande", "hellaswag", "commonsenseqa", "boolq", "qasc", "arc_easy", "arc_challenge")
REPLAY_MIN = 0.20  # D36 and ADVISORY X8: replay at least 20 % of every block


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPO / path


def is_replay(row: dict[str, Any]) -> bool:
    return str(row["source"]) in GENERAL_REPLAY or str(row["template_id"]).startswith("broad_")


def top_up_count(rows: int, replay: int, minimum: float = REPLAY_MIN) -> int:
    """The smallest R with (replay + R) / (rows + R) >= minimum (R is 0 when the share already meets it)."""
    if replay >= minimum * rows:
        return 0
    return math.ceil((minimum * rows - replay) / (1.0 - minimum))


def scan(path: Path) -> tuple[int, int, Counter[str], set[str]]:
    """Rows, replay rows, template counts and item ids of a JSONL file, read in one streamed pass."""
    rows = replay = 0
    templates: Counter[str] = Counter()
    ids: set[str] = set()
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            rows += 1
            replay += is_replay(row)
            templates[str(row["template_id"])] += 1
            ids.add(str(row["item_id"]))
    return rows, replay, templates, ids


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--name", required=True, help="the block name, e.g. B1")
    ap.add_argument("--base", required=True, type=Path, help="the last kept block (mix rows, JSONL)")
    ap.add_argument("--add", required=True, type=Path, help="the new source (mix rows, JSONL)")
    args = ap.parse_args()
    base_path, add_path = resolve(args.base), resolve(args.add)
    base_rows, base_replay, base_templates, base_ids = scan(base_path)
    add = [json.loads(line) for line in add_path.open(encoding="utf-8") if line.strip()]
    add_replay = sum(is_replay(r) for r in add)
    add_templates = Counter(str(r["template_id"]) for r in add)
    n = base_rows + len(add)
    replay = base_replay + add_replay
    pool = [json.loads(line) for line in BROAD.open(encoding="utf-8") if line.strip()]
    pool = [r for r in pool if str(r["item_id"]) not in base_ids]
    need = top_up_count(n, replay)
    if need > len(pool):
        raise SystemExit(f"{args.name}: replay top-up needs {need} rows but the pool holds {len(pool)}")
    topup = sorted(pool, key=lambda r: stable_hash({"x8": args.name, "item": str(r["item_id"])}, length=16))[:need]
    total = n + len(topup)
    templates = base_templates + add_templates + Counter(str(r["template_id"]) for r in topup)
    largest = max(templates, key=templates.get)
    share = templates[largest] / total
    replay_total = replay + sum(is_replay(r) for r in topup)
    cap_holds = all(c / total <= TEMPLATE_CAP + 1e-12 for c in templates.values())
    report: dict[str, Any] = {
        "built_utc": utcnow(), "git_commit": git_commit(REPO), "command": " ".join(sys.argv), "block": args.name,
        "base": str(args.base), "base_sha256": file_sha256(base_path), "base_rows": base_rows, "base_replay_rows": base_replay,
        "add": str(args.add), "add_rows": len(add), "add_replay_rows": add_replay,
        "topup_rows": len(topup), "topup_pool_rows": len(pool),
        "rows": total, "replay_rows": replay_total, "replay_share": round(replay_total / total, 4), "replay_min": REPLAY_MIN,
        "replay_holds": replay_total / total >= REPLAY_MIN - 1e-12,
        "largest_template": largest, "largest_template_share": round(share, 4), "cap": TEMPLATE_CAP, "cap_holds": cap_holds,
        "templates": len(templates),
    }
    if not (report["replay_holds"] and cap_holds):
        raise SystemExit(f"{args.name}: the block breaks the replay or the template cap ({report['replay_share']}, {share:.4f})")
    out = REPO / "data" / "train" / "osler_v3" / f"block_{args.name}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh, base_path.open(encoding="utf-8") as src:
        for line in src:
            if line.strip():
                fh.write(line if line.endswith("\n") else line + "\n")
        for row in add + topup:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    report["file"] = str(out.relative_to(REPO))
    report["file_sha256"] = file_sha256(out)
    write_json(OUT / f"block_{args.name}.json", report)
    print(json.dumps({k: report[k] for k in ("block", "rows", "replay_share", "largest_template", "largest_template_share",
                                             "cap_holds", "replay_holds")}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
