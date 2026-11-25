"""W0 speed check (osler_v2 ADVISORY W0 step 5): phase timing of real training steps on osler_v1 B1 rows.

The advisor's profile of 2026-10-17 (D40), as a script. Each step is timed by phase with CUDA synchronisation, so the
numbers are slightly slower than an unsynchronised training run.

Usage:
    uv run --frozen python scripts/osler_v2/w0_speed.py --steps 300 --dyn 8192           # Qwen3.5-4B
    uv run --frozen python scripts/osler_v2/w0_speed.py --steps 300 --dyn 8192 --pplx    # pplx-decider-flash backbone
    --all: all-layer LoRA (the osler_v1 recipe); --dyn 0: checkpoint every batch; --nogc: never checkpoint
"""

from __future__ import annotations

import argparse
import itertools
import json
import time

import torch

from meddecide.model.meddecide_model import LoraSettings, MedDecideModel
from meddecide.model.pplx import ATTENTION_TARGETS, PplxDecisionModel
from meddecide.train.config import load_config
from meddecide.train.data import iter_batches, shuffled_order
from meddecide.train.mix_items import mix_row_to_item
from meddecide.train.trainer import Trainer, ce_plus_brier

PHASES = ("batching", "collate", "forward", "loss", "backward", "clip", "opt")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--dyn", type=int, default=8192, help="checkpoint batches above this many padded tokens; 0 = all")
    ap.add_argument("--nogc", action="store_true")
    ap.add_argument("--all", action="store_true", help="all-layer LoRA instead of attention-only")
    ap.add_argument("--pplx", action="store_true")
    ap.add_argument("--rows", default="data/train/osler_v1/block_B1.jsonl")
    ap.add_argument("--config", default="configs/osler_v1/block_arm.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    with open(args.rows, encoding="utf-8") as fh:
        items = [mix_row_to_item(json.loads(line)) for line in itertools.islice(fh, 40000)]
    lora = LoraSettings(r=32, alpha=32) if args.all else LoraSettings(r=32, alpha=32, target_modules=ATTENTION_TARGETS)
    if args.pplx:
        model = PplxDecisionModel(lora=lora, dtype="bfloat16", device="cuda:0", max_prompt_tokens=cfg.max_prompt_tokens)
    else:
        model = MedDecideModel(cfg.base_model, revision=cfg.revision, lora=lora, readout="option_code",
                               dtype="bfloat16", device="cuda:0", max_prompt_tokens=cfg.max_prompt_tokens)
    if not args.nogc:
        model.enable_gradient_checkpointing()
    trainer = Trainer(model, cfg)
    model.train_mode()
    batches = iter_batches(model, items, batch_size=cfg.batch_size, max_batch_tokens=cfg.max_batch_tokens,
                           max_prompt_tokens=cfg.max_prompt_tokens, order=shuffled_order(len(items), seed=0, epoch=0),
                           augment_options=True)
    params = [p for group in trainer.optimizer.param_groups for p in group["params"]]
    spent = dict.fromkeys(PHASES, 0.0)
    real = padded = max_padded = 0

    def timed(fn):
        torch.cuda.synchronize()
        start = time.time()
        out = fn()
        torch.cuda.synchronize()
        return out, time.time() - start

    for step in range(args.steps + 5):
        start = time.time()
        _, batch = next(batches)
        t_batch = time.time() - start
        mb, t_collate = timed(lambda batch=batch: model.collate(batch).to(model.device))
        if not args.nogc and args.dyn > 0:
            model.checkpoint_this_batch(mb.padded_tokens > args.dyn)
        (_, probs), t_fwd = timed(lambda mb=mb: model.batch_logits(mb))
        loss, t_loss = timed(lambda mb=mb, probs=probs: ce_plus_brier(probs, mb.gold_flat, mb.marker_item,
                                                                      mb.batch_size, lam=1.0))
        trainer.optimizer.zero_grad(set_to_none=True)
        _, t_bwd = timed(lambda loss=loss: loss.total.backward())
        _, t_clip = timed(lambda: torch.nn.utils.clip_grad_norm_(params, 1.0))
        _, t_opt = timed(trainer.optimizer.step)
        if step >= 5:  # the first steps compile kernels
            for key, value in zip(PHASES, (t_batch, t_collate, t_fwd, t_loss, t_bwd, t_clip, t_opt), strict=True):
                spent[key] += value
            real += mb.real_tokens
            padded += mb.padded_tokens
            max_padded = max(max_padded, mb.padded_tokens)
    total = sum(spent.values())
    print(json.dumps({
        "backbone": "pplx" if args.pplx else "qwen", "lora": "all" if args.all else "attention",
        "checkpointing": "never" if args.nogc else ("every batch" if args.dyn <= 0 else f"above {args.dyn} tokens"),
        "steps": args.steps, "s_per_step": round(total / args.steps, 3), "real_tok_per_s": round(real / total),
        "padded_tok_per_s": round(padded / total), "max_padded": max_padded,
        "share": {k: round(v / total, 3) for k, v in spent.items()},
        "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 1)}))


if __name__ == "__main__":
    main()
