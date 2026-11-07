"""Z2 (ADVISORY §2.4, D57): the λ calibration probe — 200 steps of G1's recipe at λ = 1, pre-clip.

The probe trains the F2 recipe on the distilled mix (``data/train/osler_v6/mix_f2_distill.jsonl``) for 200 steps with the
D57 dosing sampler (3 labelled anchors in every batch) and the distillation term at λ = 1. For every step it records the
pre-clip gradient norms of the CE+Brier term and the KL term and their share

    s = ‖∇KL‖ / (‖∇CE‖ + ‖∇KL‖)

Then s0 = the mean of the 200 per-step shares, and the λ that puts the distillation term at 40 % of the applied gradient
is the pre-registered closed form

    λ = (0.4 / 0.6) · (1 - s0) / s0.

No weights are kept. The probe's own steps are written to ``steps.jsonl`` and the summary to ``probe.json``; λ is frozen
there and used by Z3. s0 at 0 or 1 makes the closed form undefined: that is BLOCKED, not clamped (STATE §2b).

Usage (GPU; detached, one job):
    source scripts/pod_env.sh && setsid nohup uv run --frozen python scripts/osler_v6/z2_probe.py \
        --out outputs/osler_v6/Z2/probe > outputs/osler_v6/Z2/probe/logs/probe.log 2>&1 < /dev/null &
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import time
from collections import Counter
from pathlib import Path

import torch

from meddecide.model.pplx import PplxDecisionModel
from meddecide.train.config import StudentConfig, load_config
from meddecide.train.mix_items import mix_row_to_item
from meddecide.train.trainer import Trainer
from meddecide.utils.hashing import file_sha256
from meddecide.utils.io import iter_jsonl, write_json, write_jsonl
from meddecide.utils.provenance import git_commit, gpu_name, utcnow

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG = REPO_ROOT / "configs" / "osler_v5" / "arm_F2.yaml"
MIX = REPO_ROOT / "data" / "train" / "osler_v6" / "mix_f2_distill.jsonl"
MIX_MANIFEST = REPO_ROOT / "outputs" / "osler_v6" / "Z2" / "mix_manifest.json"
STEPS = 200
SCHEDULE_STEPS = 13_992  # the arm's length: the probe sees the same LR schedule the arm's first 200 steps do
TARGET_SHARE = 0.40  # ADVISORY §2.4: the distillation term at 40 % of the applied gradient


def probe_config(base: StudentConfig, out: Path) -> StudentConfig:
    """G1's recipe at λ = 1 with the dosing quota and the probe recording on; every path under ``out``."""
    return dataclasses.replace(
        base,
        output_dir=str(out.relative_to(REPO_ROOT)),
        distill_weight=1.0,
        distill_min_labelled_per_batch=3,
        distill_probe=True,
        distill_probe_steps=0,
        eval_every=0,
        eval_steps=(),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="outputs/osler_v6/Z2/probe")
    parser.add_argument("--smoke", action="store_true",
                        help="code-path check only: every 500th mix row, steps 3; never reported")
    args = parser.parse_args()
    out = (REPO_ROOT / args.out) if not Path(args.out).is_absolute() else Path(args.out)
    for directory in (out, out / "logs"):
        directory.mkdir(parents=True, exist_ok=True)
    probe_path = out / ("smoke_probe.json" if args.smoke else "probe.json")
    provenance_path = out / ("smoke_provenance.json" if args.smoke else "run_provenance.json")
    if probe_path.exists():
        raise SystemExit(f"{probe_path} exists: the probe is not run twice (refuse-twice guard)")
    if not MIX.exists() or not MIX_MANIFEST.exists():
        raise SystemExit(f"{MIX} or {MIX_MANIFEST} is missing: run `scripts/osler_v6/build_mix.py build` first")
    manifest = json.loads(MIX_MANIFEST.read_text())
    if file_sha256(MIX) != manifest["output_sha256"]:
        raise SystemExit("the mix file does not match its manifest sha256: stop")
    if manifest["labelled_rows"] < 3:
        raise SystemExit("fewer than 3 labelled mix rows: the dosing quota cannot be met")

    base = load_config(CONFIG)
    config = probe_config(base, out)
    if config.distill_min_labelled_per_batch > config.batch_size:
        raise SystemExit("the dosing quota exceeds the batch size: stop")
    started_utc = utcnow()
    started = time.time()
    rows = list(iter_jsonl(MIX))
    if args.smoke:
        rows = rows[::500]
    items = [mix_row_to_item(row) for row in rows]
    del rows
    labelled = sum(1 for item in items if getattr(item, "teacher_probs", None) is not None)
    if labelled == 0:
        raise SystemExit("no mix item carries a teacher distribution: stop")
    if labelled < config.distill_min_labelled_per_batch * 2:
        raise SystemExit(f"only {labelled} labelled items for a quota of {config.distill_min_labelled_per_batch}: stop")

    write_json(provenance_path, {
        "started_utc": started_utc,
        "git_commit": git_commit(REPO_ROOT),
        "command": " ".join(["python", "scripts/osler_v6/z2_probe.py", *([] if not args.smoke else ["--smoke"])]),
        "config": config.to_dict(),
        "seed": config.seed,
        "mix_file": str(MIX.relative_to(REPO_ROOT)),
        "mix_sha256": manifest["output_sha256"],
        "soft_label_file": manifest["soft_label_file"],
        "soft_label_file_sha256": manifest["soft_label_file_sha256"],
        "teacher": manifest["teacher"],
        "gpu": gpu_name(),
    })
    torch.manual_seed(config.seed)
    model = PplxDecisionModel(lora=config.lora, dtype=config.dtype, device=config.device,
                              max_prompt_tokens=config.max_prompt_tokens)
    if config.gradient_checkpointing:
        model.enable_gradient_checkpointing()
    trainer = Trainer(model, config, log_path=out / "logs" / "train.jsonl")
    steps = 3 if args.smoke else STEPS
    result = trainer.train(items, steps=steps, schedule_steps=3 if args.smoke else SCHEDULE_STEPS)
    wall = time.time() - started
    shares = [record.grad_share for record in result.history]
    teachers = [record.n_teacher for record in result.history]
    losses = [record.loss for record in result.history]
    summary: dict[str, object] = {
        "task": "Z2",
        "smoke": bool(args.smoke),
        "teacher": manifest["teacher"],
        "started_utc": started_utc,
        "finished_utc": utcnow(),
        "git_commit": git_commit(REPO_ROOT),
        "command": " ".join(["python", "scripts/osler_v6/z2_probe.py", *([] if not args.smoke else ["--smoke"])]),
        "config": config.to_dict(),
        "mix": {"file": str(MIX.relative_to(REPO_ROOT)), "sha256": manifest["output_sha256"],
                "rows": manifest["rows"], "labelled_rows": manifest["labelled_rows"],
                "labelled_share_of_rows": manifest["labelled_share_of_rows"]},
        "items_read": len(items),
        "labelled_items": labelled,
        "steps_requested": steps,
        "steps_completed": result.steps,
        "model": model.describe(),
        "checkpoint_path": str(model.checkpoint_path),
        "hardware": {"gpu": gpu_name()},
        "wall_clock_s": round(wall, 1),
        "loss": {"first": losses[0] if losses else None, "last": losses[-1] if losses else None,
                 "all_finite": all(math.isfinite(value) for value in losses)},
        "labelled_per_step": {"min": min(teachers) if teachers else None,
                              "max": max(teachers) if teachers else None,
                              "histogram": {str(k): v for k, v in sorted(Counter(teachers).items())},
                              "steps_meeting_quota": sum(1 for n in teachers if n >= config.distill_min_labelled_per_batch)},
        "dosing": result.dosing,
        "gradient_share": {
            "definition": "per step, pre-clip: ||grad KL|| / (||grad CE+Brier|| + ||grad KL||)",
            "steps": len(shares),
            "s0_mean": sum(shares) / len(shares) if shares else None,
            "min": min(shares) if shares else None,
            "max": max(shares) if shares else None,
            "all_finite": all(math.isfinite(value) for value in shares),
        },
    }
    s0 = summary["gradient_share"]["s0_mean"]  # type: ignore[index]
    if args.smoke:
        summary["status"] = "SMOKE — not reported"
        write_json(probe_path, summary)
        print(f"[{utcnow()}] smoke probe: {summary['gradient_share']}; {summary['labelled_per_step']}")
        return
    if result.steps != STEPS:
        raise SystemExit(f"the probe stopped after {result.steps} steps, not {STEPS}: {result.stopped_early}")
    problems = []
    if not summary["loss"]["all_finite"]:
        problems.append("non-finite loss")
    if not summary["gradient_share"]["all_finite"]:
        problems.append("non-finite gradient share")
    if summary["labelled_per_step"]["min"] < config.distill_min_labelled_per_batch:
        problems.append(f"a step carried {summary['labelled_per_step']['min']} labelled items")
    if problems:
        summary["status"] = "BLOCKED — " + "; ".join(problems)
        write_json(probe_path, summary)
        raise SystemExit(f"BLOCKED: {'; '.join(problems)}")
    if not 0.0 < s0 < 1.0:
        summary["status"] = f"BLOCKED — s0 {s0} makes the closed form undefined"
        write_json(probe_path, summary)
        raise SystemExit(f"BLOCKED: s0 = {s0}, the λ closed form is undefined")
    lam = (TARGET_SHARE / (1.0 - TARGET_SHARE)) * (1.0 - s0) / s0
    summary["lambda"] = {
        "formula": "(0.4 / 0.6) * (1 - s0) / s0",
        "target_share": TARGET_SHARE,
        "s0": s0,
        "value": lam,
        "frozen": True,
        "note": "λ is frozen here; it is not re-tuned inside any arm (ADVISORY §2.4)",
    }
    summary["status"] = "DONE"
    write_jsonl(out / "steps.jsonl", [record.to_dict() for record in result.history])
    write_json(probe_path, summary)
    print(f"[{utcnow()}] probe done: s0 {s0:.4f} -> λ {lam:.4f}; steps {result.steps}; "
          f"labelled/step {summary['labelled_per_step']['min']}-{summary['labelled_per_step']['max']}; "
          f"wall {wall / 60:.1f} min")


if __name__ == "__main__":
    main()
