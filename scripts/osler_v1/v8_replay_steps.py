"""V8: the step count of the block runs, and the realised items and tokens of every block at that count (ADVISORY V8).

The step count is the smallest multiple of four whose batches hold at least 80,000 realised items on B0. It is found by
replaying the trainer's own batcher (meddecide.train.data.iter_batches over shuffled_order, seed 0, epoch 0, with the same
arguments as the trainer's train loop; the O6 replay pattern), without loading model weights or using the GPU. Every block
then runs that same number of steps, and its realised items and tokens at that count are recorded.

Writes outputs/osler_v1/V8/replay_steps.json (aggregates only) and outputs/osler_v1/V8/replay/<block>.json (per-step sizes
and token counts; gitignored). The block runs check their logged step sizes against these files.

Usage (CPU; no GPU):
    source scripts/pod_env.sh && uv run --frozen python scripts/osler_v1/v8_replay_steps.py --blocks B0 B1 B3
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from meddecide.bench.schema import Item
from meddecide.model.meddecide_model import LINEAR_ATTENTION_CHUNK, MedDecideModel
from meddecide.train.blocks import EVAL_QUARTERS, REALISED_ITEM_TARGET, steps_for_items
from meddecide.train.config import load_config
from meddecide.train.data import iter_batches, shuffled_order
from meddecide.train.mix_items import mix_row_to_item
from meddecide.utils.hashing import file_sha256
from meddecide.utils.io import write_json
from meddecide.utils.provenance import git_commit, utcnow

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "configs" / "osler_v1" / "block_arm.yaml"
TRAIN_DIR = REPO / "data" / "train" / "osler_v1"
OUT = REPO / "outputs" / "osler_v1" / "V8"
BLOCKS = ("B0", "B1", "B2", "B3")


class _Encoder:
    """The attributes ``MedDecideModel.encode_item`` reads, and nothing else (no weights are loaded; as in O6)."""

    def __init__(self, tokenizer: Any, variant: str, max_prompt_tokens: int) -> None:
        self.tokenizer = tokenizer
        self.variant = variant
        self.max_prompt_tokens = max_prompt_tokens

    def encode_item(self, item: Item, *, permutation: Any = None, max_prompt_tokens: int | None = None) -> Any:
        return MedDecideModel.encode_item(self, item, permutation=permutation, max_prompt_tokens=max_prompt_tokens)


def read_block(name: str) -> list[Item]:
    """The block's rows as trainer items, in file order (the batcher's positions refer to this order)."""
    path = TRAIN_DIR / f"block_{name}.jsonl"
    with path.open(encoding="utf-8") as fh:
        return [mix_row_to_item(json.loads(line)) for line in fh if line.strip()]


def batch_record(batch: list[Any]) -> tuple[int, int, int]:
    """(items, real tokens, padded tokens) of one batch, as the trainer's step record counts them: the real tokens are the
    attention mask's sum, and the padded tokens are batch x the longest item rounded up to the linear-attention chunk
    (ModelBatch.padded_tokens after MedDecideModel.collate with its default rounding)."""
    lengths = [int(e.n_tokens) for e in batch]
    longest = -(-max(lengths) // LINEAR_ATTENTION_CHUNK) * LINEAR_ATTENTION_CHUNK
    return len(batch), sum(lengths), len(batch) * longest


def replay_block(config: Any, items: list[Item], encoder: _Encoder, *, steps: int | None = None,
                 target_items: int | None = None, multiple: int = EVAL_QUARTERS) -> tuple[int, list[tuple[int, int, int]]]:
    """Replay the batches of one block: exactly ``steps`` batches, or the batches up to the smallest valid step count
    holding ``target_items`` items. Returns the step count and one record per consumed batch."""
    order = shuffled_order(len(items), seed=config.seed, epoch=0)
    batches = iter_batches(
        encoder,
        items,
        batch_size=config.batch_size,
        max_batch_tokens=config.max_batch_tokens,
        max_prompt_tokens=config.max_prompt_tokens,
        order=order,
        augment_options=config.shuffle_options,
        epoch=0,
        seed=config.seed,
    )
    records: list[tuple[int, int, int]] = []

    def stream():
        for _, batch in batches:
            record = batch_record(batch)
            records.append(record)
            yield record[0]

    if steps is None:
        if target_items is None:
            raise ValueError("replay_block needs a step count or a target")
        steps = steps_for_items(stream(), target_items, multiple=multiple)
    else:
        for _ in itertools.islice(stream(), steps):
            pass
    batches.close()
    if len(records) != steps:
        raise RuntimeError(f"replayed {len(records)} batches for {steps} steps")
    return steps, records


def summarise(name: str, steps: int, records: list[tuple[int, int, int]]) -> dict[str, Any]:
    sizes = [r[0] for r in records]
    return {
        "block": name,
        "steps": steps,
        "realised_items": sum(sizes),
        "real_tokens": sum(r[1] for r in records),
        "padded_tokens": sum(r[2] for r in records),
        "mean_items_per_step": round(sum(sizes) / steps, 4),
        "items_per_step_counts": {str(k): v for k, v in sorted(Counter(sizes).items())},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--blocks", nargs="+", default=["B0", "B1", "B3"], choices=list(BLOCKS),
                        help="the first block sets the step count for the rest (B0 for the block runs of V8)")
    parser.add_argument("--target-items", type=int, default=REALISED_ITEM_TARGET,
                        help="the realised-item target of the first block (80,000 for V8; 250,000 for the final run V9)")
    parser.add_argument("--multiple", type=int, default=EVAL_QUARTERS,
                        help="the step count is a multiple of this (4 for V8's quarter evaluations; 8 for V9)")
    parser.add_argument("--out", default=str(OUT), help="where the replay files go (default: the V8 directory)")
    args = parser.parse_args()
    out = Path(args.out)
    started = time.time()
    config = load_config(CONFIG)
    tokenizer = AutoTokenizer.from_pretrained(config.base_model, revision=config.revision, local_files_only=True)
    encoder = _Encoder(tokenizer, config.variant, config.max_prompt_tokens)
    (out / "replay").mkdir(parents=True, exist_ok=True)

    steps: int | None = None
    summaries: dict[str, Any] = {}
    for name in args.blocks:
        t0 = time.time()
        items = read_block(name)
        if steps is None:
            steps, records = replay_block(config, items, encoder, target_items=args.target_items, multiple=args.multiple)
        else:
            steps, records = replay_block(config, items, encoder, steps=steps)
        summary = summarise(name, steps, records)
        summary["train_items_in_block"] = len(items)
        summary["train_file_sha256"] = file_sha256(TRAIN_DIR / f"block_{name}.jsonl")
        summary["replay_wall_clock_s"] = round(time.time() - t0, 1)
        summaries[name] = summary
        write_json(out / "replay" / f"{name}.json", {**summary, "sizes": [r[0] for r in records],
                                                     "real_tokens": [r[1] for r in records],
                                                     "padded_tokens": [r[2] for r in records]})
        print(f"{name}: {steps} steps, {summary['realised_items']} realised items, "
              f"{summary['real_tokens']} real tokens, mean {summary['mean_items_per_step']} items/step", flush=True)
    write_json(out / "replay_steps.json", {
        "written_utc": utcnow(), "git_commit": git_commit(REPO), "config": str(CONFIG.relative_to(REPO)),
        "first_block": args.blocks[0], "target_realised_items": args.target_items, "multiple": args.multiple,
        "steps": steps, "blocks": summaries, "wall_clock_s": round(time.time() - started, 1),
    })


if __name__ == "__main__":
    main()
