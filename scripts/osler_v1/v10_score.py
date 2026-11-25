"""V10: score one Osler checkpoint on the V10 items, in both option orders and in the original order (GPU).

Writes O2-format prediction rows, the same shape as the baselines' files, so every row pairs with a baseline row by item id:
    outputs/osler_v1/V10/<name>/both_orders.jsonl    both-order averages for choice items (the gate's input)
    outputs/osler_v1/V10/<name>/single_order.jsonl   the original order only (reported beside the gate, never in it)
    outputs/osler_v1/V10/<name>/run.json             provenance, calibration and counts

The calibration is the arm's own dev fits (``temperature_fits`` in its arm_result.json), applied to every pass (ADVISORY V10).
The item file is written by scripts/osler_v1/v10_items.py. Resumable: rows already present in both files are not rescored.
One GPU job at a time: launch detached, with logs under outputs/osler_v1/V10/logs/.

Usage:
    source scripts/pod_env.sh && setsid nohup uv run --frozen python scripts/osler_v1/v10_score.py \
        --checkpoint outputs/osler_v1/V9/osler_4b/checkpoints/best --name osler_4b \
        --temperatures outputs/osler_v1/V9/osler_4b/arm_result.json \
        > outputs/osler_v1/V10/logs/osler_4b.log 2>&1 < /dev/null &
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from meddecide.eval.o2_io import completed_ids, open_predictions, write_row
from meddecide.eval.osler_scoring import dev_temperatures, o2_row_to_item, score_rows
from meddecide.model.meddecide_model import MedDecideModel
from meddecide.utils.hashing import file_sha256
from meddecide.utils.io import iter_jsonl, write_json
from meddecide.utils.provenance import git_commit, utcnow

REPO = Path(__file__).resolve().parents[2]
ITEMS = REPO / "data/bench/osler_v1/v10_items.jsonl"
OUT = REPO / "outputs/osler_v1/V10"
MAX_BATCH_TOKENS = 8192
MAX_PROMPT_TOKENS = 16384


def preflight(rows: list[dict]) -> None:
    """Every row must convert to an Item, and every item id must be unique (resuming skips rows by id), before the model loads."""
    for row in rows:
        try:
            o2_row_to_item(row)
        except (KeyError, ValueError) as exc:
            raise ValueError(f"row {row['item_id']} ({row['set_name']}, {row['qtype']}) does not convert: {exc}") from exc
    ids = [row["item_id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{len(ids) - len(set(ids))} duplicated item ids; resuming by id would drop rows")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--checkpoint", type=Path, required=True, help="a saved MedDecide checkpoint directory")
    parser.add_argument("--name", required=True, help="the output directory name under outputs/osler_v1/V10")
    parser.add_argument("--temperatures", type=Path, required=True,
                        help="the arm_result.json whose dev temperature fits calibrate every pass")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16", help="the arm's training dtype")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--rows-per-call", type=int, default=512)
    parser.add_argument("--limit", type=int, default=0, help="score only the first N remaining rows (0 = all; smoke runs)")
    parser.add_argument("--items", type=Path, default=ITEMS, help="the V10 item file (smoke runs only)")
    parser.add_argument("--out-root", type=Path, default=OUT, help="output root (smoke runs only)")
    args = parser.parse_args()
    temperatures = dev_temperatures(json.loads(args.temperatures.read_text(encoding="utf-8")))
    # ADVISORY V10: the checkpoint carries its temperatures; they must equal the arm's dev fits used for scoring
    carried = json.loads((args.checkpoint / "model.json").read_text(encoding="utf-8")).get("calibration") or {}
    carried_match = bool(carried) and set(carried) == set(temperatures) and all(
        abs(float(carried[k]) - temperatures[k]) < 1e-9 for k in temperatures)
    if not carried_match:
        raise SystemExit(f"the checkpoint's calibration {carried} does not match the dev fits {temperatures}")
    started = time.time()

    rows = list(iter_jsonl(args.items))
    preflight(rows)
    out_dir = args.out_root / args.name
    both_path = out_dir / "both_orders.jsonl"
    single_path = out_dir / "single_order.jsonl"
    done = completed_ids(both_path) & completed_ids(single_path)
    remaining = [row for row in rows if row["item_id"] not in done]
    if args.limit:
        remaining = remaining[: args.limit]
    print(f"{args.name}: pre-flight passed: {len(rows)} rows convert, item ids unique; {len(done)} already written, "
          f"{len(remaining)} to score; calibration {temperatures}", flush=True)

    model = MedDecideModel.load(args.checkpoint, device=args.device, dtype=args.dtype)
    model.eval_mode()
    scored = 0
    with open_predictions(both_path) as both_fh, open_predictions(single_path) as single_fh:
        for start in range(0, len(remaining), args.rows_per_call):
            chunk = remaining[start : start + args.rows_per_call]
            both, single = score_rows(model, chunk, rows_per_call=len(chunk), batch_size=args.batch_size,
                                      max_batch_tokens=MAX_BATCH_TOKENS, max_prompt_tokens=MAX_PROMPT_TOKENS,
                                      round_to_chunk=True, temperatures=temperatures)
            for row in both:
                write_row(both_fh, row)
            for row in single:
                write_row(single_fh, row)
            scored += len(chunk)
            print(f"{args.name}: {scored} of {len(remaining)} scored, {time.time() - started:.0f} s", flush=True)

    record = {
        "kind": "osler_v1_V10_scoring",
        "built_at_utc": utcnow(),
        "git_commit": git_commit(REPO),
        "command": f"scripts/osler_v1/v10_score.py --checkpoint {args.checkpoint} --name {args.name} --temperatures {args.temperatures}",
        "checkpoint": str(args.checkpoint),
        "calibration": f"dev per-question-type temperatures from {args.temperatures}",
        "temperatures": temperatures,
        "checkpoint_calibration": carried,
        "checkpoint_calibration_matches_dev_fits": carried_match,
        "items_file":str(args.items.relative_to(REPO)) if args.items.is_relative_to(REPO) else str(args.items),
        "items_sha256": file_sha256(args.items),
        "rows_in_items_file": len(rows),
        "rows_written_before": len(done),
        "rows_scored_now": scored,
        "device": args.device,
        "dtype": args.dtype,
        "batch_size": args.batch_size,
        "max_batch_tokens": MAX_BATCH_TOKENS,
        "max_prompt_tokens": MAX_PROMPT_TOKENS,
        "wall_clock_s": round(time.time() - started, 1),
    }
    write_json(out_dir / "run.json", record)
    print(f"{args.name}: done; run.json written", flush=True)


if __name__ == "__main__":
    main()
