"""osler_v4 Stage B block (ADVISORY Y7; D47): the last kept block plus one new source, with the caps of the ADVISORY.

- The base is the last kept block (for B1: the chosen Stage A recipe's mix, osler_v2's C1, which is A0's mix). Its lines are copied
  unchanged (streamed).
- The added source is one or more files of mix rows (B1: the Y2 MIMIC structured-gold training rows of eleven templates).
- Replay: the general replay (the eight osler_v2 general sources and the broad replay templates) must make up at least 20 % of the
  block. Missing rows are drawn from osler_v3's broad replay pool in stable-hash order, excluding rows the base already holds
  (the same rule as osler_v3's X8 builder).
- Caps (ADVISORY Y7): no template above 8 % of the block; the MIMIC templates as a whole at most 35 % of the block. The build fails
  if any cap or the replay share does not hold; nothing is adjusted silently.

Writes data/train/osler_v4/block_<name>.jsonl (training rows, gitignored) and outputs/osler_v4/Y7/block_<name>.json (counts, shares,
file hashes). Usage (CPU):
    uv run --frozen python scripts/osler_v4/y7_build_block.py --name B1 --base data/train/osler_v2/block_C1.jsonl \
        --add data/mimic_build/items/mimic_apr_severity_v1__train.jsonl ...
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
OUT = REPO / "outputs" / "osler_v4" / "Y7"
GENERAL_REPLAY = ("snli", "winogrande", "hellaswag", "commonsenseqa", "boolq", "qasc", "arc_easy", "arc_challenge")
REPLAY_MIN = 0.20  # D36 and ADVISORY X8 / Y7: replay at least 20 % of every block
MIMIC_CAP = 0.35  # ADVISORY Y7: the MIMIC templates as a whole at most 35 % of a block


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPO / path


def is_replay(row: dict[str, Any]) -> bool:
    return str(row["source"]) in GENERAL_REPLAY or str(row["template_id"]).startswith("broad_")


def is_mimic(row: dict[str, Any]) -> bool:
    return str(row["template_id"]).startswith("mimic_")


def top_up_count(rows: int, replay: int, minimum: float = REPLAY_MIN) -> int:
    """The smallest R with (replay + R) / (rows + R) at least the minimum (0 when the share already meets it)."""
    if replay >= minimum * rows:
        return 0
    return math.ceil((minimum * rows - replay) / (1.0 - minimum))


def scan(path: Path) -> tuple[int, int, int, Counter[str], set[str]]:
    """Rows, replay rows, MIMIC rows, template counts and item ids of a JSONL file, in one streamed pass."""
    rows = replay = mimic = 0
    templates: Counter[str] = Counter()
    ids: set[str] = set()
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            rows += 1
            replay += is_replay(row)
            mimic += is_mimic(row)
            templates[str(row["template_id"])] += 1
            ids.add(str(row["item_id"]))
    return rows, replay, mimic, templates, ids


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--name", required=True, help="the block name, e.g. B1")
    ap.add_argument("--base", required=True, type=Path, help="the last kept block (mix rows, JSONL)")
    ap.add_argument("--add", required=True, type=Path, nargs="+", help="the new source files (mix rows, JSONL)")
    args = ap.parse_args()
    base_path = resolve(args.base)
    add_paths = [resolve(p) for p in args.add]
    base_rows, base_replay, base_mimic, base_templates, base_ids = scan(base_path)
    add: list[dict[str, Any]] = []
    for path in add_paths:
        add.extend(json.loads(line) for line in path.open(encoding="utf-8") if line.strip())
    add_replay = sum(is_replay(r) for r in add)
    add_mimic = sum(is_mimic(r) for r in add)
    add_templates = Counter(str(r["template_id"]) for r in add)
    n = base_rows + len(add)
    replay = base_replay + add_replay
    pool = [json.loads(line) for line in BROAD.open(encoding="utf-8") if line.strip()]
    pool = [r for r in pool if str(r["item_id"]) not in base_ids]
    need = top_up_count(n, replay)
    if need > len(pool):
        raise SystemExit(f"{args.name}: replay top-up needs {need} rows but the pool holds {len(pool)}")
    topup = sorted(pool, key=lambda r: stable_hash({"y7": args.name, "item": str(r["item_id"])}, length=16))[:need]
    total = n + len(topup)
    templates = base_templates + add_templates + Counter(str(r["template_id"]) for r in topup)
    largest = max(templates, key=templates.get)
    replay_total = replay + sum(is_replay(r) for r in topup)
    mimic_total = base_mimic + add_mimic
    cap_holds = all(c / total <= TEMPLATE_CAP + 1e-12 for c in templates.values())
    mimic_holds = mimic_total / total <= MIMIC_CAP + 1e-12
    report: dict[str, Any] = {
        "built_utc": utcnow(), "git_commit": git_commit(REPO), "command": " ".join(sys.argv), "block": args.name,
        "base": str(args.base), "base_sha256": file_sha256(base_path), "base_rows": base_rows, "base_replay_rows": base_replay,
        "add": [str(p) for p in args.add], "add_rows": len(add), "add_replay_rows": add_replay, "add_mimic_rows": add_mimic,
        "topup_rows": len(topup), "topup_pool_rows": len(pool),
        "rows": total, "replay_rows": replay_total, "replay_share": round(replay_total / total, 4), "replay_min": REPLAY_MIN,
        "replay_holds": replay_total / total >= REPLAY_MIN - 1e-12,
        "mimic_rows": mimic_total, "mimic_share": round(mimic_total / total, 4), "mimic_cap": MIMIC_CAP, "mimic_holds": mimic_holds,
        "largest_template": largest, "largest_template_share": round(templates[largest] / total, 4), "cap": TEMPLATE_CAP,
        "cap_holds": cap_holds, "templates": len(templates),
        "template_rows": dict(sorted(templates.items())),
    }
    if not (report["replay_holds"] and cap_holds and mimic_holds):
        raise SystemExit(f"{args.name}: the block breaks a cap or the replay share (replay {report['replay_share']}, "
                         f"MIMIC {report['mimic_share']}, largest template {report['largest_template_share']})")
    out = REPO / "data" / "train" / "osler_v4" / f"block_{args.name}.jsonl"
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
    print(json.dumps({k: report[k] for k in ("block", "rows", "replay_share", "mimic_share", "largest_template_share",
                                             "cap_holds", "mimic_holds", "replay_holds")}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
