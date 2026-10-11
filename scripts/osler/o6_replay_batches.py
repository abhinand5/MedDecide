"""Replay an O6-O8 arm's batcher offline (CPU only) to count the examples it really sees (deviation 51; Question 21).

The trainer forms each step's batch with ``meddecide.train.data.iter_batches`` over ``shuffled_order(seed, epoch 0)``, and
one step is one batch. This script calls the same function with the same arguments and the same item encoding
(``MedDecideModel.encode_item``, which reads only the tokenizer, the letter variant and the prompt cap), without loading the
model weights. It counts the items of every batch up to the step budget, then checks the batch sizes against the arm's
logged step records (``logs/train.jsonl``, one per ``log_every`` steps): every logged size must agree exactly.

Writes ``outputs/osler_v0/O6/replay/<arm>/replay.json`` (gitignored); the committed record is the CLAIMS row.

Usage (CPU; does not use the GPU):
    source scripts/pod_env.sh && uv run --frozen python scripts/osler/o6_replay_batches.py --config configs/osler_v0/arm_L.yaml
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from meddecide.model.meddecide_model import MedDecideModel
from meddecide.train.config import load_config
from meddecide.train.data import iter_batches, shuffled_order
from meddecide.train.mix_items import mix_row_to_item
from meddecide.train.osler_arm import example_budget, step_count
from meddecide.utils.io import write_json
from meddecide.utils.provenance import git_commit, utcnow

REPO = Path(__file__).resolve().parents[2]


class _Encoder:
    """The attributes ``MedDecideModel.encode_item`` reads, and nothing else (no weights are loaded)."""

    def __init__(self, tokenizer: Any, variant: str, max_prompt_tokens: int) -> None:
        self.tokenizer = tokenizer
        self.variant = variant
        self.max_prompt_tokens = max_prompt_tokens

    def encode_item(self, item: Any, *, permutation: Any = None, max_prompt_tokens: int | None = None) -> Any:
        return MedDecideModel.encode_item(self, item, permutation=permutation, max_prompt_tokens=max_prompt_tokens)


def logged_step_sizes(log: Path) -> dict[int, int]:
    """The batch size of every logged step. A last line that the running trainer has not finished writing is skipped."""
    sizes: dict[int, int] = {}
    lines = [line for line in log.read_text(encoding="utf-8").splitlines() if line.strip()]
    for i, line in enumerate(lines):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            if i == len(lines) - 1:
                break
            raise
        if row.get("event") == "step":
            sizes[int(row["step"])] = int(row["n_items"])
    return sizes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=None, help="the step budget (default: the arm's budget)")
    args = parser.parse_args()
    started = time.time()
    config_path = args.config if args.config.is_absolute() else REPO / args.config
    config = load_config(config_path)
    train_path = REPO / config.train_path
    items = []
    with train_path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                items.append(mix_row_to_item(json.loads(line)))
    steps = args.steps if args.steps is not None else step_count(example_budget(len(items)), config.batch_size)
    tokenizer = AutoTokenizer.from_pretrained(
        config.base_model, revision=config.revision, trust_remote_code=False, local_files_only=True
    )
    encoder = _Encoder(tokenizer, config.variant, config.max_prompt_tokens)
    order = shuffled_order(len(items), seed=config.seed, epoch=0)
    sizes: list[int] = []
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
    for _, batch in batches:
        if len(sizes) >= steps:
            break
        sizes.append(len(batch))
    batches.close()

    out_dir = REPO / "outputs/osler_v0/O6/replay" / config_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    logged = {}
    log = REPO / config.output_dir / "logs" / "train.jsonl"
    if log.exists():
        logged = logged_step_sizes(log)
    checked = [s for s in sorted(logged) if s <= len(sizes)]
    mismatches = [s for s in checked if sizes[s - 1] != logged[s]]
    record = {
        "kind": "osler_v0_O6_replay_batches",
        "built_at_utc": utcnow(),
        "git_commit": git_commit(REPO),
        "config": str(config_path.relative_to(REPO)),
        "train_items": len(items),
        "steps": len(sizes),
        "batch_size_nominal": config.batch_size,
        "total_examples": sum(sizes),
        "mean_items_per_step": sum(sizes) / len(sizes),
        "items_per_step_counts": {str(k): v for k, v in sorted(Counter(sizes).items())},
        "validation": {"logged_steps_in_log": len(logged), "logged_steps_checked": len(checked),
                       "matches": len(checked) - len(mismatches), "mismatches": len(mismatches),
                       "first_mismatches": mismatches[:5]},
        "wall_clock_s": round(time.time() - started, 1),
    }
    write_json(out_dir / "replay.json", record)
    print(f"{config_path.name}: steps {len(sizes)}, total examples {sum(sizes)}, mean items/step "
          f"{record['mean_items_per_step']:.3f}; logged steps checked {len(checked)}, mismatches {len(mismatches)}; "
          f"wrote {out_dir / 'replay.json'}", flush=True)


if __name__ == "__main__":
    main()
