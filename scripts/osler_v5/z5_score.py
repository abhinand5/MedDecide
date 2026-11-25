"""osler_v5 Z5 scoring (GPU; test sets): one osler_v5 checkpoint scored on the frozen osler_v4 sets, calibrated with its saved temperatures.

The sets are osler_v4's Y9 files (data/bench/osler_v4/y9/*.jsonl). Each file's SHA-256 must equal the value in osler_v4's manifest
(outputs/osler_v4/Y9/sets_manifest.json) before the set is read; a mismatch stops the run (ADVISORY §3.3). Items are keyed by the
recomputed item id (meddecide.train.mix_items.mix_row_to_item), as the Y9 gate reads them. Predictions are written as
outputs/osler_v5/Z5/<prefix>_<set>/predictions.jsonl (meddecide.train.osler_arm.dev_prediction_rows), and a manifest with the counts.

Usage (GPU; one job; detached):
    source scripts/pod_env.sh && setsid nohup uv run --frozen python scripts/osler_v5/z5_score.py \
        --checkpoint outputs/osler_v5/Z4/final_blended --prefix v5blend > outputs/osler_v5/Z5/logs/score_v5blend.log 2>&1 < /dev/null &
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path
from typing import Any

from meddecide.model.pplx import PplxDecisionModel
from meddecide.train.config import load_config
from meddecide.train.mix_items import mix_row_to_item
from meddecide.train.osler_arm import dev_prediction_rows
from meddecide.utils.hashing import file_sha256
from meddecide.utils.io import iter_jsonl, write_json, write_jsonl
from meddecide.utils.provenance import git_commit, utcnow

REPO = Path(__file__).resolve().parents[2]
SETS_DIR = REPO / "data" / "bench" / "osler_v4" / "y9"
MANIFEST = REPO / "outputs" / "osler_v4" / "Y9" / "sets_manifest.json"
OUT = REPO / "outputs" / "osler_v5" / "Z5"
RECIPE = REPO / "configs" / "osler_v5" / "arm_R0.yaml"  # the evaluation settings (batch sizes, prompt cap) are the same in every arm
SETS = ("mr_verdict", "mr_reported", "mr_heldout_test", "mimic_suite", "mimic_heldout", "panel", "knowledge", "mednli_test", "pairs_test")


def verified_path(name: str, manifest: dict[str, Any]) -> Path:
    """The set's file, after its SHA-256 is checked against osler_v4's manifest; raises on a mismatch or a missing entry."""
    entry = manifest["sets"][name]
    path = REPO / entry["file"]
    digest = file_sha256(path)
    if digest != entry["sha256"]:
        raise SystemExit(f"{name}: SHA-256 {digest[:12]} differs from osler_v4's manifest {entry['sha256'][:12]}; not scored")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--sets", nargs="*", default=list(SETS))
    parser.add_argument("--limit", type=int, default=0, help="score only the first N items of each set (a code-path check)")
    args = parser.parse_args()
    started = utcnow()
    config = dataclasses.replace(load_config(RECIPE), device="cuda:0", dtype="bfloat16", backbone="pplx")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    model = PplxDecisionModel.load(REPO / args.checkpoint, device="cuda:0", dtype="bfloat16")
    model.eval_mode()
    if not model.calibration:
        raise SystemExit("the checkpoint carries no temperatures; an uncalibrated model is not scored for Gate O5 (ADVISORY Z5)")
    counts: dict[str, Any] = {}
    for name in args.sets:
        t0 = time.time()
        path = verified_path(name, manifest)
        rows = list(iter_jsonl(path))
        if args.limit:
            rows = rows[: args.limit]
        items = [mix_row_to_item(row) for row in rows]
        if len({item.item_id for item in items}) != len(items):
            raise SystemExit(f"{name}: the recomputed item ids are not unique")
        scored = model.score_items(items, batch_size=config.eval_batch_size or config.batch_size,
                                   max_batch_tokens=config.eval_max_batch_tokens or config.max_batch_tokens,
                                   max_prompt_tokens=config.max_prompt_tokens, round_to_chunk=config.eval_round_to_chunk,
                                   length_buckets=config.eval_length_buckets)
        calibrated = model.apply_temperature(scored)
        out_dir = OUT / f"{args.prefix}_{name}"
        out_dir.mkdir(parents=True, exist_ok=True)
        write_jsonl(out_dir / "predictions.jsonl", dev_prediction_rows(calibrated))
        counts[name] = {"file": str(path.relative_to(REPO)), "rows": len(rows), "scored": len(items),
                        "seconds": round(time.time() - t0, 1)}
        print(f"{name}: {len(items)} items scored in {counts[name]['seconds']} s", flush=True)
    write_json(OUT / f"{args.prefix}_manifest.json", {
        "started_utc": started, "finished_utc": utcnow(), "git_commit": git_commit(REPO), "checkpoint": args.checkpoint,
        "temperatures": dict(model.calibration), "limit": args.limit, "sets": counts,
    })
    print("done", flush=True)


if __name__ == "__main__":
    main()
