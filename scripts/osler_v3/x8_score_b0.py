"""osler_v3 X8 (ADVISORY X8; operator ruling D-X8-2): the chosen Stage A checkpoint (B0) scored on the EXTENDED fast dev v3.

Stage B keep decisions compare each block with B0 on the same dev items. fast dev v3 was extended with the restyle dev pages
after Stage A ran (D-X8-2), so B0's dev predictions must be recomputed on the extended set. The checkpoint is the chosen arm's own
best checkpoint (outputs/osler_v3/stage_a/<chosen>/checkpoints/best); the scoring is the trainer's (as run_arm's final scoring).

Validation (reported, never used to choose): on the items the old fast dev v3 held, the rescored correctness must agree with the arm's
own dev predictions item by item; the agreement is recorded as a count and must be complete for the run to be used.

Writes outputs/osler_v3/X8/b0_dev/dev_predictions.jsonl (the keep rule's current-block predictions) and validation.json (counts only).
Usage (one GPU job; after the Gemma server is stopped):  source scripts/pod_env.sh && uv run --frozen python scripts/osler_v3/x8_score_b0.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

from meddecide.model.pplx import load_any
from meddecide.train.config import load_config
from meddecide.train.mix_items import mix_row_to_item
from meddecide.train.osler_arm import dev_prediction_rows
from meddecide.train.trainer import Trainer
from meddecide.utils.io import read_json, write_json, write_jsonl
from meddecide.utils.provenance import git_commit, utcnow

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "configs" / "osler_v3" / "block_arm.yaml"
DEV = REPO / "data" / "bench" / "osler_v3" / "fastdev_v3.jsonl"
CHOICE = REPO / "outputs" / "osler_v3" / "X7" / "choice.json"
STAGE_A = REPO / "outputs" / "osler_v3" / "stage_a"
OUT = REPO / "outputs" / "osler_v3" / "X8" / "b0_dev"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def main() -> int:
    started = time.time()
    chosen = read_json(CHOICE)["chosen"]
    best = STAGE_A / chosen / "checkpoints" / "best"
    config = load_config(CONFIG)
    model = load_any(best, device=config.device, dtype=config.dtype)
    items = [mix_row_to_item(row) for row in read_jsonl(DEV)]
    scored = Trainer(model, config).score(items)
    rows = dev_prediction_rows(scored)
    OUT.mkdir(parents=True, exist_ok=True)
    write_jsonl(OUT / "dev_predictions.jsonl", rows)
    own = {str(r["item_id"]): bool(r["correct"]) for r in read_jsonl(STAGE_A / chosen / "dev_predictions.jsonl")}
    new = {str(r["item_id"]): bool(r["correct"]) for r in rows}
    shared = [i for i in own if i in new]
    write_json(OUT / "validation.json", {
        "built_utc": utcnow(), "git_commit": git_commit(REPO), "chosen": chosen, "checkpoint": str(best.relative_to(REPO)),
        "dev_file": str(DEV.relative_to(REPO)), "items_scored": len(rows), "items_in_own_predictions": len(own),
        "shared_items": len(shared), "agree_on_shared": sum(own[i] == new[i] for i in shared),
        "accuracy_all_scored": round(sum(new.values()) / len(new), 4),
        "wall_clock_s": round(time.time() - started, 1),
    })
    print(f"B0 ({chosen}) scored {len(rows)} items; shared {len(shared)}; agreement "
          f"{sum(own[i] == new[i] for i in shared)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
