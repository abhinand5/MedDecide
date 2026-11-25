"""osler_v2 W6: replay the batches of each block at the fixed step count (13,992, as osler_v1's blocks), per backbone.

The trainer's own batcher (meddecide.train.data.iter_batches over shuffled_order, seed 0, epoch 0) runs without weights, with
each backbone's own encoder: Qwen3.5-4B's prompt (the MedDecideModel encoder) or pplx-decider-flash's prompt (the
PplxDecisionModel.encode_item method, applied to an object that holds only the tokenizer, the readout marker, the option codes
and the prompt length, so no weights are loaded). The realised items and tokens at 13,992 steps are recorded per block and
backbone; the block runs check their logged step sizes against these files.

Writes outputs/osler_v2/W6/replay/<block>_<backbone>.json (per-step sizes and token counts; aggregates; no item text).

Usage (CPU; no GPU): source scripts/pod_env.sh && uv run --frozen python scripts/osler_v2/w6_replay.py --blocks C0 C1 C2 --backbones qwen pplx
"""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from meddecide.model.meddecide_model import MedDecideModel
from meddecide.model.pplx import MARKER, PplxDecisionModel, checkpoint_dir
from meddecide.model.pplx import REPO as PPLX_REPO
from meddecide.train.config import load_config
from meddecide.train.data import iter_batches, shuffled_order
from meddecide.train.mix_items import mix_row_to_item
from meddecide.utils.hashing import file_sha256
from meddecide.utils.io import write_json
from meddecide.utils.provenance import git_commit, utcnow

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "configs" / "osler_v2" / "block_arm.yaml"
TRAIN_DIR = REPO / "data" / "train" / "osler_v2"
OUT = REPO / "outputs" / "osler_v2" / "W6" / "replay"
REPLAY_STEPS = 13_992  # osler_v1's block step count (outputs/osler_v1/V8/replay_steps.json), fixed for osler_v2 (ADVISORY W6)


def _v8_helpers():
    """batch_record and the encoder shape of osler_v1's replay (the trainer's own record semantics)."""
    spec = importlib.util.spec_from_file_location("v8_replay_steps", REPO / "scripts" / "osler_v1" / "v8_replay_steps.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.batch_record


batch_record = _v8_helpers()


class _QwenEncoder:
    """The attributes MedDecideModel.encode_item reads (as osler_v1's V8 replay)."""

    def __init__(self, tokenizer: Any, variant: str, max_prompt_tokens: int) -> None:
        self.tokenizer = tokenizer
        self.variant = variant
        self.max_prompt_tokens = max_prompt_tokens

    def encode_item(self, item: Any, *, permutation: Any = None, max_prompt_tokens: int | None = None) -> Any:
        return MedDecideModel.encode_item(self, item, permutation=permutation, max_prompt_tokens=max_prompt_tokens)


def pplx_encoder(max_prompt_tokens: int) -> Any:
    """PplxDecisionModel.encode_item on an object with the four attributes it reads; no weights are loaded."""
    ckpt = checkpoint_dir(PPLX_REPO)  # the pplx checkpoint's repository id, not this project's root
    tokenizer = AutoTokenizer.from_pretrained(str(ckpt))
    obj = PplxDecisionModel.__new__(PplxDecisionModel)
    obj.tokenizer = tokenizer
    obj.marker_id = int(tokenizer.encode(MARKER, add_special_tokens=False)[0])
    obj.optcode_names = list(json.loads((ckpt / "decision_config.json").read_text(encoding="utf-8"))["codes"])
    obj.max_prompt_tokens = int(max_prompt_tokens)
    return obj


def encoder_for(backbone: str, config: Any) -> Any:
    if backbone == "qwen":
        tokenizer = AutoTokenizer.from_pretrained(config.base_model, revision=config.revision, local_files_only=True)
        return _QwenEncoder(tokenizer, config.variant, config.max_prompt_tokens)
    return pplx_encoder(config.max_prompt_tokens)


def read_block(path: Path) -> list[Any]:
    with path.open(encoding="utf-8") as fh:
        return [mix_row_to_item(json.loads(line)) for line in fh if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--blocks", nargs="+", default=["C0"], choices=["C0", "C1", "C2"])
    ap.add_argument("--backbones", nargs="+", default=["qwen", "pplx"], choices=["qwen", "pplx"])
    args = ap.parse_args()
    config = load_config(CONFIG)
    for block in args.blocks:
        path = TRAIN_DIR / f"block_{block}.jsonl"
        items = read_block(path)
        sha = file_sha256(path)
        for backbone in args.backbones:
            started = time.time()
            encoder = encoder_for(backbone, config)
            order = shuffled_order(len(items), seed=config.seed, epoch=0)
            batches = iter_batches(encoder, items, batch_size=config.batch_size, max_batch_tokens=config.max_batch_tokens,
                                   max_prompt_tokens=config.max_prompt_tokens, order=order,
                                   augment_options=config.shuffle_options, epoch=0, seed=config.seed)
            records = [batch_record(batch) for _, batch in itertools.islice(batches, REPLAY_STEPS)]
            batches.close()
            if len(records) != REPLAY_STEPS:
                raise RuntimeError(f"{block}/{backbone}: replayed {len(records)} batches for {REPLAY_STEPS} steps")
            sizes = [r[0] for r in records]
            write_json(OUT / f"{block}_{backbone}.json", {
                "block": block, "backbone": backbone, "steps": REPLAY_STEPS, "train_file": str(path.relative_to(REPO)),
                "train_file_sha256": sha, "train_items": len(items), "sizes": sizes,
                "real_tokens": [r[1] for r in records], "padded_tokens": [r[2] for r in records],
                "realised_items": sum(sizes), "mean_items_per_step": round(sum(sizes) / REPLAY_STEPS, 4),
                "items_per_step_counts": {str(k): v for k, v in sorted(Counter(sizes).items())},
                "wall_clock_s": round(time.time() - started, 1), "written_utc": utcnow(), "git_commit": git_commit(REPO),
            })
            print(f"{block}/{backbone}: {REPLAY_STEPS} steps, realised items {sum(sizes)}, "
                  f"real tokens {sum(r[1] for r in records)}", flush=True)


if __name__ == "__main__":
    main()
