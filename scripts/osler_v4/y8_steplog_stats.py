"""Y8 SELF_AUDIT inputs from the step log (osler_v4; read-only; CPU).

Counts and medians only: the health of the logged steps on the 50-step grid, the largest per-step loss and its batch, the large
per-step losses by batch size (a logged loss is the mean over the batch's items, so a one-item batch shows one item's loss), the
gradient clipping of the run against the earlier Stage A and B arms (same recipe, grad_clip from configs/osler_v3/block_arm.yaml),
and the allocator warnings in the run's log. Writes outputs/osler_v4/Y8/steplog_stats.json.

Usage (CPU): CUDA_VISIBLE_DEVICES= uv run --frozen python scripts/osler_v4/y8_steplog_stats.py
"""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
from typing import Any

from meddecide.train.config import load_config
from meddecide.utils.io import write_json
from meddecide.utils.provenance import utcnow

REPO = Path(__file__).resolve().parents[2]
RECIPE = REPO / "configs" / "osler_v3" / "block_arm.yaml"
OUT = REPO / "outputs" / "osler_v4" / "Y8"
EARLIER = {"Y6/A4": "outputs/osler_v4/Y6/A4", "Y6/A5": "outputs/osler_v4/Y6/A5",
           "Y7/B1": "outputs/osler_v4/Y7/B1", "Y7/B3": "outputs/osler_v4/Y7/B3"}
GRID = 50
LARGE_CE = 5.0


def step_rows(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [r for r in rows if r.get("event") == "step"]


def grad_summary(rows: list[dict[str, Any]], clip: float) -> dict[str, Any]:
    grads = [r["grad_norm"] for r in rows]
    return {"rows": len(rows), "median": round(statistics.median(grads), 2),
            "share_above_clip": round(sum(g > clip for g in grads) / len(grads), 4)}


def main() -> None:
    clip = load_config(RECIPE).grad_clip
    rows = step_rows(OUT / "logs" / "train.jsonl")
    steps = [r["step"] for r in rows]
    expected = set(range(min(steps), max(steps) + 1, GRID))
    by_step = {r["step"]: r for r in rows}
    top = max(rows, key=lambda r: r["ce"])
    large = [r for r in rows if r["ce"] > LARGE_CE]
    single = [r for r in rows if r["n_items"] == 1]
    multi = [r for r in rows if r["n_items"] > 1]
    large_single = [r for r in large if r["n_items"] == 1]
    text = (OUT / "logs" / "final_train.log").read_text(encoding="utf-8", errors="replace")
    stats: dict[str, Any] = {
        "generated_utc": utcnow(),
        "grad_clip": clip,
        "logged": {"rows": len(rows), "first": min(steps), "last": max(steps),
                   "missing": len(expected - set(steps)), "duplicates": len(steps) - len(set(steps)),
                   "non_finite": sum(1 for r in rows for v in r.values() if isinstance(v, float) and not math.isfinite(v))},
        "largest_ce": {"step": top["step"], "ce": round(top["ce"], 4), "n_items": top["n_items"],
                       "template_ids": top["template_ids"], "accuracy": top["accuracy"], "brier": round(top["brier"], 4),
                       "grad_norm": round(top["grad_norm"], 2),
                       "neighbour_ce": [round(by_step[s]["ce"], 4) if s in by_step else None
                                        for s in (top["step"] - GRID, top["step"] + GRID)]},
        "large_ce": {"threshold": LARGE_CE, "rows": len(large), "one_item_rows": len(large_single),
                     "more_than_one_item_rows": len(large) - len(large_single)},
        "batch_size": {"one_item_rows": len(single), "share_of_rows": round(len(single) / len(rows), 4),
                       "large_ce_rate_one_item": round(len(large_single) / len(single), 4),
                       "large_ce_rate_more_than_one_item": round((len(large) - len(large_single)) / len(multi), 4),
                       "median_max_item_tokens_one_item": statistics.median(r["max_item_tokens"] for r in single),
                       "median_max_item_tokens_more_than_one_item": statistics.median(r["max_item_tokens"] for r in multi)},
        "grad_norm": {"Y8": grad_summary(rows, clip)},
        "allocator": {"CUDACachingAllocator_lines": text.count("CUDACachingAllocator"),
                      "Traceback_lines": text.count("Traceback"), "OutOfMemoryError_lines": text.count("OutOfMemoryError")},
    }
    for arm, folder in EARLIER.items():
        stats["grad_norm"][arm] = grad_summary(step_rows(REPO / folder / "logs" / "train.jsonl"), clip)
    write_json(OUT / "steplog_stats.json", stats)
    print(json.dumps({k: stats[k] for k in ("logged", "large_ce", "batch_size", "allocator")}, sort_keys=True))


if __name__ == "__main__":
    main()
