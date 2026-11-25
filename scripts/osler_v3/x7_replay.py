"""osler_v3 replay of a training file: the batches the trainer will draw, with the pplx encoder and no weights (CPU; ADVISORY X7).

The trainer's own batcher (meddecide.train.data.iter_batches over shuffled_order, seed 0, epoch 0) runs on the file with the pplx
prompt encoder, exactly as osler_v2's replay (scripts/osler_v2/w6_replay.py, whose functions this reuses). The recipe is the
osler_v3 config (identical to osler_v2's in every field the batcher reads). The file written records the per-step sizes and token
counts that the block run checks against its logged steps.

Usage:  source scripts/pod_env.sh && uv run --frozen python scripts/osler_v3/x7_replay.py \
            --train data/train/osler_v3/block_A3.jsonl --name A3 --steps 13992
"""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import sys
import time
from collections import Counter
from pathlib import Path

from meddecide.train.config import load_config
from meddecide.train.data import iter_batches, shuffled_order
from meddecide.utils.hashing import file_sha256
from meddecide.utils.io import write_json
from meddecide.utils.provenance import git_commit, utcnow

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "configs" / "osler_v3" / "block_arm.yaml"
OUT = REPO / "outputs" / "osler_v3" / "replays"


def _w6():
    spec = importlib.util.spec_from_file_location("w6_replay", REPO / "scripts" / "osler_v2" / "w6_replay.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--train", required=True, type=Path)
    ap.add_argument("--name", required=True)
    ap.add_argument("--steps", type=int, default=13_992)
    args = ap.parse_args()
    rp = _w6()
    config = load_config(CONFIG)
    started = time.time()
    items = rp.read_block(args.train)
    encoder = rp.encoder_for("pplx", config)
    order = shuffled_order(len(items), seed=config.seed, epoch=0)
    batches = iter_batches(encoder, items, batch_size=config.batch_size, max_batch_tokens=config.max_batch_tokens,
                           max_prompt_tokens=config.max_prompt_tokens, order=order, augment_options=config.shuffle_options,
                           epoch=0, seed=config.seed)
    records = [rp.batch_record(batch) for _, batch in itertools.islice(batches, args.steps)]
    batches.close()
    if len(records) != args.steps:
        raise SystemExit(f"replayed {len(records)} batches for {args.steps} steps: the file is too small for the step budget")
    sizes = [r[0] for r in records]
    write_json(OUT / f"{args.name}_pplx.json", {
        "block": args.name, "backbone": "pplx", "steps": args.steps, "train_file": str(args.train),
        "train_file_sha256": file_sha256(args.train), "train_items": len(items), "sizes": sizes,
        "real_tokens": [r[1] for r in records], "padded_tokens": [r[2] for r in records], "realised_items": sum(sizes),
        "items_per_step_counts": {str(k): v for k, v in sorted(Counter(sizes).items())},
        "wall_clock_s": round(time.time() - started, 1), "written_utc": utcnow(), "git_commit": git_commit(REPO),
    })
    print(f"{args.name}: {args.steps} steps, realised items {sum(sizes)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
