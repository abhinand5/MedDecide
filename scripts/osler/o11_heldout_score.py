"""O11: score one Osler checkpoint on the held-out generators' test items (ADVISORY O11; GPU).

The held-out items are O2-format rows written by o11_heldout_items.py (data/gen/osler_v0/o11_heldout_test.jsonl). They are scored
as o11_score.py scores the O2 common set (both option orders averaged for choice items, and the single order), and written to
their own directory, because heldout.md rule 3 keeps held-out results apart from the seen results:
    outputs/osler_v0/O11/heldout/<name>/both_orders.jsonl    both-order averages for choice items
    outputs/osler_v0/O11/heldout/<name>/single_order.jsonl   the original order only
    outputs/osler_v0/O11/heldout/<name>/run.json             provenance and counts
Resumable, and one GPU job at a time: launch detached, with logs under outputs/osler_v0/O11/logs/.

Usage:
    source scripts/pod_env.sh && setsid nohup uv run --frozen python scripts/osler/o11_heldout_score.py \
        --checkpoint outputs/osler_v0/O6/arm_L/checkpoints/best --name osler_4b_L \
        > outputs/osler_v0/O11/logs/heldout_osler_4b_L.log 2>&1 < /dev/null &
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from meddecide.eval.o2_io import Scope, completed_ids, iter_items, open_predictions, write_row
from meddecide.eval.osler_scoring import o2_row_to_item, score_rows
from meddecide.model.meddecide_model import MedDecideModel
from meddecide.utils.io import write_json
from meddecide.utils.provenance import git_commit, utcnow

REPO = Path(__file__).resolve().parents[2]
ITEMS = REPO / "data/gen/osler_v0/o11_heldout_test.jsonl"
OUT = REPO / "outputs/osler_v0/O11/heldout"
SCOPE = Scope("heldout", ("osler_v0_heldout",))  # the benchmark name that o11_heldout_items.py writes
MAX_BATCH_TOKENS = 8192
MAX_PROMPT_TOKENS = 16384


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True, help="a saved MedDecide checkpoint directory")
    parser.add_argument("--name", required=True, help="the output directory name under outputs/osler_v0/O11/heldout")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16", help="the arm's training dtype (ADVISORY O6-O8: bf16)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--rows-per-call", type=int, default=512)
    args = parser.parse_args()
    started = time.time()

    out_dir = OUT / args.name
    both_path = out_dir / "both_orders.jsonl"
    single_path = out_dir / "single_order.jsonl"
    done = completed_ids(both_path) & completed_ids(single_path)
    in_scope = list(iter_items(ITEMS, SCOPE))
    # pre-flight (CPU, before the model loads): every held-out row converts, and every item id is unique
    for row in in_scope:
        o2_row_to_item(row)
    if len({row["item_id"] for row in in_scope}) != len(in_scope):
        raise ValueError("held-out item ids are not unique")
    remaining = [row for row in in_scope if row["item_id"] not in done]
    print(f"{args.name}: pre-flight passed: {len(in_scope)} held-out rows convert, item ids unique; "
          f"{len(done)} already written, {len(remaining)} to score", flush=True)

    model = MedDecideModel.load(args.checkpoint, device=args.device, dtype=args.dtype)
    model.eval_mode()
    scored = 0
    with open_predictions(both_path) as both_fh, open_predictions(single_path) as single_fh:
        for start in range(0, len(remaining), args.rows_per_call):
            chunk = remaining[start : start + args.rows_per_call]
            both, single = score_rows(model, chunk, rows_per_call=len(chunk), batch_size=args.batch_size,
                                      max_batch_tokens=MAX_BATCH_TOKENS, max_prompt_tokens=MAX_PROMPT_TOKENS,
                                      round_to_chunk=True)
            for row in both:
                write_row(both_fh, row)
            for row in single:
                write_row(single_fh, row)
            scored += len(chunk)
            print(f"{args.name}: {scored} of {len(remaining)} scored, {time.time() - started:.0f} s", flush=True)

    record = {
        "kind": "osler_v0_O11_heldout_scoring",
        "built_at_utc": utcnow(),
        "git_commit": git_commit(REPO),
        "command": f"scripts/osler/o11_heldout_score.py --checkpoint {args.checkpoint} --name {args.name}",
        "checkpoint": str(args.checkpoint),
        "items_file": str(ITEMS.relative_to(REPO)),
        "scope": list(SCOPE.benchmarks),
        "items_in_scope": len(in_scope),
        "rows_converted_preflight": len(in_scope),
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
