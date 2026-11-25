"""osler_v3 X0: KL-to-base step timing on the osler_v2 C1 rows, pplx-decider-flash-v1.2 (ADVISORY X0 step 3; D41).

Runs the trainer for --steps steps with --kl-to-base (0 for the control) and the ADVISORY §3.2 domain prefixes, using the
osler_v2 recipe (configs/osler_v3/block_arm.yaml). No dev evaluation and no temperature fit: this run measures time, the
anchor share and memory only. The batches are the trainer's own (the same order as osler_v2's C1 replay).

Writes outputs/osler_v3/X0/<name>/timing.json (aggregates, from meddecide.train.osler_v3.timing_summary) and
step_records.jsonl (per-step numbers only; no item text).

Usage (detached; check the GPU is free first):
    source scripts/pod_env.sh && setsid nohup uv run --frozen python scripts/osler_v3/x0_kl_timing.py \
        --kl-to-base 0.5 --steps 152 --name kl_0p5 > outputs/osler_v3/X0/logs/kl_0p5.log 2>&1 < /dev/null &
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

import torch

from meddecide.model.pplx import ATTENTION_TARGETS, PplxDecisionModel
from meddecide.train.config import load_config
from meddecide.train.mix_items import mix_row_to_item
from meddecide.train.osler_v3 import DOMAIN_PREFIXES, timing_summary
from meddecide.train.trainer import Trainer
from meddecide.utils.io import write_json, write_jsonl
from meddecide.utils.provenance import git_commit, utcnow

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "configs" / "osler_v3" / "block_arm.yaml"
TRAIN = REPO / "data" / "train" / "osler_v2" / "block_C1.jsonl"
OUT = REPO / "outputs" / "osler_v3" / "X0"
STEP_FIELDS = ("step", "elapsed_s", "n_items", "n_anchor", "kl", "loss", "ce", "brier", "accuracy", "grad_norm",
               "real_tokens", "padded_tokens", "gpu_peak_gb")


def read_items(path: Path) -> list:
    with path.open(encoding="utf-8") as fh:
        return [mix_row_to_item(json.loads(line)) for line in fh if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--kl-to-base", type=float, required=True, help="0 for the control arm")
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--name", required=True)
    args = ap.parse_args()
    started = time.time()
    base = load_config(CONFIG)
    out_dir = OUT / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    config = dataclasses.replace(
        base,
        train_path=str(TRAIN.relative_to(REPO)),
        output_dir=str(out_dir.relative_to(REPO)),
        backbone="pplx",
        kl_to_base=args.kl_to_base,
        kl_exclude_template_prefixes=DOMAIN_PREFIXES if args.kl_to_base > 0 else (),
        lora=dataclasses.replace(base.lora, target_modules=ATTENTION_TARGETS),
    )
    torch.manual_seed(config.seed)  # run_arm's convention: the initialisation is seeded before the model is built
    items = read_items(TRAIN)
    model = PplxDecisionModel(lora=config.lora, dtype=config.dtype, device=config.device,
                              max_prompt_tokens=config.max_prompt_tokens)
    if config.gradient_checkpointing:
        model.enable_gradient_checkpointing()
    trainer = Trainer(model, config, log_path=out_dir / "logs" / "train.jsonl")
    result = trainer.train(items, steps=args.steps, schedule_steps=args.steps)
    records = [r.to_dict() for r in result.history]
    summary = timing_summary(records)
    write_jsonl(out_dir / "step_records.jsonl", [{k: r[k] for k in STEP_FIELDS} for r in records])
    write_json(out_dir / "timing.json", {
        "name": args.name, "kl_to_base": args.kl_to_base, "kl_exclude_template_prefixes": list(config.kl_exclude_template_prefixes),
        "steps_requested": args.steps, "steps_taken": result.steps, "stopped_early": result.stopped_early,
        "train_file": str(TRAIN.relative_to(REPO)), "train_items": len(items), "summary": summary,
        "train_wall_clock_s": round(result.wall_clock_s, 1), "total_wall_clock_s": round(time.time() - started, 1),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none", "torch": torch.__version__,
        "git_commit": git_commit(REPO), "written_utc": utcnow(), "config": config.to_dict(),
        "command": f"scripts/osler_v3/x0_kl_timing.py --kl-to-base {args.kl_to_base} --steps {args.steps} --name {args.name}",
    })
    print(f"{args.name}: {result.steps} steps, s/step (excl. first) {summary['s_per_step_mean_excl_first']:.3f}, "
          f"anchor share {summary['anchor_share']:.4f}, peak {summary['gpu_peak_gb_max']:.1f} GB, "
          f"KL mean {summary['kl_mean_over_anchor_steps']:.4f}, non-finite {summary['nonfinite_losses']}", flush=True)


if __name__ == "__main__":
    main()
