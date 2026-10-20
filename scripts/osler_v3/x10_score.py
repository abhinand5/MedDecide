"""osler_v3 X10 (operator ruling, 2026-10-20): score one checkpoint on the V10 items with its calibration, as an additional row.

Stage A runs (A0 to A3) and Stage B block B1 are scored here on the same item file that osler_v2's Gate O3 used (30,621 rows:
MR suite v1 verdict and reported-only templates, held-out MR, the external panel, the knowledge items). No Gate O3 verdict is
written from these rows (the operator ruled no new final model, so no verdict; D35/D39/D44 are not applied here).

The scoring functions are osler_v1's (scripts/osler_v1/v10_score.py) and the file layout is osler_v2's W8 scorer
(scripts/osler_v2/w8_score.py), copied here with two changes: the output root is outputs/osler_v3/X10, and the record names this
loop. The temperatures are the run's own dev fits (arm_result.json ``temperature_fits``), and the checkpoint's saved calibration must
match them (the osler_v2 check). Resumable by item id.

Writes outputs/osler_v3/X10/<name>/{both_orders.jsonl, single_order.jsonl, record.json}.

Usage (one GPU job at a time; detached):
    source scripts/pod_env.sh && uv run --frozen python scripts/osler_v3/x10_score.py --checkpoint <dir> --name <name> \
        --temperatures <arm_result.json>
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

from meddecide.model.pplx import load_any
from meddecide.utils.hashing import file_sha256
from meddecide.utils.io import iter_jsonl, write_json
from meddecide.utils.provenance import git_commit, utcnow

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "outputs" / "osler_v3" / "X10"
ITEMS = REPO / "data" / "bench" / "osler_v1" / "v10_items.jsonl"


def _v10():
    spec = importlib.util.spec_from_file_location("v10_score", REPO / "scripts" / "osler_v1" / "v10_score.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["v10_score"] = module
    spec.loader.exec_module(module)
    return module


v10 = _v10()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--temperatures", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--rows-per-call", type=int, default=512)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    temperatures = v10.dev_temperatures(json.loads(args.temperatures.read_text(encoding="utf-8")))
    carried = json.loads((args.checkpoint / "model.json").read_text(encoding="utf-8")).get("calibration") or {}
    carried_match = bool(carried) and set(carried) == set(temperatures) and all(
        abs(float(carried[k]) - temperatures[k]) < 1e-9 for k in temperatures)
    if not carried_match:
        raise SystemExit(f"the checkpoint's calibration {carried} does not match the dev fits {temperatures}")
    started = time.time()
    rows = list(iter_jsonl(ITEMS))
    v10.preflight(rows)
    out_dir = OUT / args.name
    both_path = out_dir / "both_orders.jsonl"
    single_path = out_dir / "single_order.jsonl"
    done = v10.completed_ids(both_path) & v10.completed_ids(single_path)
    remaining = [row for row in rows if row["item_id"] not in done]
    if args.limit:
        remaining = remaining[: args.limit]
    print(f"{args.name}: pre-flight passed: {len(rows)} rows; {len(done)} already written, {len(remaining)} to score; "
          f"calibration {temperatures}", flush=True)
    model = load_any(args.checkpoint, device=args.device, dtype=args.dtype)
    model.eval_mode()
    scored = 0
    with v10.open_predictions(both_path) as both_fh, v10.open_predictions(single_path) as single_fh:
        for start in range(0, len(remaining), args.rows_per_call):
            chunk = remaining[start: start + args.rows_per_call]
            both, single = v10.score_rows(model, chunk, rows_per_call=len(chunk), batch_size=args.batch_size,
                                          max_batch_tokens=v10.MAX_BATCH_TOKENS, max_prompt_tokens=v10.MAX_PROMPT_TOKENS,
                                          round_to_chunk=True, temperatures=temperatures)
            for row in both:
                v10.write_row(both_fh, row)
            for row in single:
                v10.write_row(single_fh, row)
            scored += len(chunk)
            print(f"{args.name}: {scored} of {len(remaining)} scored, {time.time() - started:.0f} s", flush=True)
    write_json(out_dir / "record.json", {
        "kind": "osler_v3_X10_scoring", "built_at_utc": utcnow(), "git_commit": git_commit(REPO),
        "checkpoint": str(args.checkpoint), "temperatures": temperatures,
        "checkpoint_calibration_matches_dev_fits": carried_match, "items_file": str(ITEMS.relative_to(REPO)),
        "items_sha256": file_sha256(ITEMS), "rows_in_items_file": len(rows), "rows_scored_now": scored,
        "device": args.device, "dtype": args.dtype, "loader": "meddecide.model.pplx.load_any",
        "wall_clock_s": round(time.time() - started, 1),
    })
    print(f"{args.name}: done, {scored} rows scored now", flush=True)


if __name__ == "__main__":
    main()
