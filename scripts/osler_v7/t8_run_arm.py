#!/usr/bin/env python
"""osler_v7 T8 — 9B arm on d3-flash: the v7 mix, CE + teacher distillation + KL-to-base on the C1-C4 pool.

ADVISORY §2/§3 with the 9B base (`vllm-sr/d3-flash`, pinned revision): attention-only LoRA + the 255-row readout at the
last prompt position through the student's own runtime (`meddecide.distill.d3_train`). One arm = 13,992 steps on the
same v7 mix as T6:
- CE(gold) on every row;
- lambda_teacher * KL(teacher || student) on rows with a teacher distribution (the mix's row-order targets are permuted
  by the harness's own `render_targets`);
- lambda_KL * KL(base || student) on the `v7_c*` rows only (the Lever-1 pool). The base is the same checkpoint with the
  LoRA adapters disabled (scaling 0) and the checkpoint's base readout; it is computed per batch without gradients.

Dev evaluations run on the frozen T2 dev set (22,934 rows) at the requested steps: osler_v5's S composite (MR and MIMIC
template macros on fast dev v4 rows, CG v2 micro), knowledge-dev accuracy and the D52b positional check on the original
order of the DDXPlus items. No test split, panel, MIMIC audit suite or held-out item is read.

Run (GPU; one job; detached)::

    source scripts/pod_env.sh && setsid nohup uv run --frozen --with pillow==12.3.0 --with "torchvision==0.29.0" \
        python scripts/osler_v7/t8_run_arm.py --out outputs/osler_v7/T8/arm --steps 13992 \
        > outputs/osler_v7/T8/logs/arm.log 2>&1 < /dev/null &
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
from torch import nn
from transformers import AutoModel

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from meddecide.distill.d3_train import (  # noqa: E402
    BatchPlan,
    D3Student,
    TrainRow,
    batch_tensors,
    gradient_norm,
    loss_from_logits,
    lr_at,
    prepare_rows,
)
from meddecide.eval.composite import composite, positional_health  # noqa: E402
from meddecide.utils.provenance import git_commit, utcnow  # noqa: E402

POOL = ROOT / "data" / "train" / "osler_v7" / "d3_pool.jsonl"
DEV = ROOT / "data" / "train" / "osler_v7" / "d3_dev.jsonl"
REVISION = "67e03e1bc157961c582ee7c6a35d9b15fada3dec"
SNAP = Path(os.environ.get("HF_HOME", "/workspace/.hf_home")) / "hub" / "models--vllm-sr--d3-flash" / "snapshots" / REVISION
KL_PREFIX = "v7_c"
DDX_PREFIX = "cg_v2_ddxplus"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def kl_to_reference(ref: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
    """Mean KL(ref || probs) over rows; ref rows are already normalised over the offered codes."""
    logp = torch.log(probs.clamp_min(1e-12))
    return (ref * (torch.log(ref.clamp_min(1e-12)) - logp)).sum(dim=-1).mean()


def base_probabilities(student: D3Student, sequences: list[list[int]], counts: list[int]) -> torch.Tensor:
    """The zero-shot base's probabilities: adapters disabled (scaling 0) and the checkpoint's base readout."""
    runtime = student.runtime
    saved_scaling = {name: adapter.scaling for name, adapter in student.adapters.items()}
    saved_readout = runtime.readout
    for adapter in student.adapters.values():
        adapter.scaling = 0.0
    runtime.readout = student.base_readout
    try:
        with torch.inference_mode():
            logits = runtime.logits(sequences, counts) / student.temperature
            probs = torch.softmax(logits.float(), dim=-1)
    finally:
        for name, adapter in student.adapters.items():
            adapter.scaling = saved_scaling[name]
        runtime.readout = saved_readout
    return probs


def step_losses(
    student: D3Student,
    rows: list[TrainRow],
    templates: list[str],
    *,
    lam_teacher: float,
    lam_kl: float,
    params: list[nn.Parameter],
    probe: bool,
) -> dict[str, Any]:
    """One forward: CE, teacher KL and KL-to-base (v7_c rows); sets gradients on ``params`` and returns the values."""
    ids, mask, counts, golds = batch_tensors(rows, student.pad_id, student.device)
    logits = student.forward_logits(ids, mask)
    ce_sum, kl_teacher_sum = loss_from_logits(logits, counts, golds, [row.teacher for row in rows],
                                               temperature=student.temperature)
    batch = len(rows)
    ce = ce_sum / batch
    kl_teacher = kl_teacher_sum / batch
    kl_base = logits.new_zeros(())
    idx = [i for i, template in enumerate(templates) if template.startswith(KL_PREFIX)]
    if idx and lam_kl > 0:
        seqs = [list(rows[i].sequence) for i in idx]
        cnt = [rows[i].count for i in idx]
        ref = base_probabilities(student, seqs, cnt)
        masked = torch.full_like(logits, float("-inf"))
        for row_index, count in zip(idx, cnt, strict=True):
            masked[row_index, :count] = logits[row_index, :count]
        probs = torch.softmax(masked.float() / student.temperature, dim=-1)
        kl_base = kl_to_reference(ref, probs[torch.tensor(idx, device=student.device)])
    total = ce + lam_teacher * kl_teacher + lam_kl * kl_base
    share = None
    if probe:
        g_ce = torch.autograd.grad(ce, params, retain_graph=True, allow_unused=True)
        g_kl = torch.autograd.grad(lam_teacher * kl_teacher + lam_kl * kl_base, params, retain_graph=True,
                                   allow_unused=True)
        n_ce = gradient_norm(params, g_ce)
        n_kl = gradient_norm(params, g_kl)
        share = n_kl / (n_ce + n_kl) if n_ce + n_kl > 0 else float("nan")
        grads = [None if a is None and b is None else (a if b is None else b if a is None else a + b)
                 for a, b in zip(g_ce, g_kl, strict=True)]
    else:
        grads = torch.autograd.grad(total, params, allow_unused=True)
    for parameter, grad in zip(params, grads, strict=True):
        parameter.grad = None if grad is None else grad.detach()
    pre_clip = float(torch.nn.utils.clip_grad_norm_(params, 1.0)) if grads is not None else 0.0
    return {"ce": float(ce.detach()), "kl_teacher": float(kl_teacher.detach()),
            "kl_base": float(kl_base.detach()), "share": share, "pre_clip": pre_clip}


def evaluate(
    student: D3Student,
    dev_prepared: list[TrainRow],
    dev_templates: list[str],
    dev_origins: list[str],
    *,
    batch_size: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Prediction rows and the dev metrics (S composite, knowledge accuracy, positional original order)."""
    probs = student.probabilities([list(row.sequence) for row in dev_prepared],
                                  [row.count for row in dev_prepared], batch_size=batch_size)
    preds: list[dict[str, Any]] = []
    for row, template, values in zip(dev_prepared, dev_templates, probs, strict=True):
        predicted = max(range(len(values)), key=values.__getitem__)
        preds.append({"item_id": row.row_id, "template_id": template, "source": row.source, "n_options": row.count,
                      "gold_index": row.gold, "predicted_index": predicted, "correct": predicted == row.gold})
    fast = [row for row, origin in zip(preds, dev_origins, strict=True) if origin == "fastdev_v4"]
    cg = [row for row, origin in zip(preds, dev_origins, strict=True) if origin == "cg_v2"]
    know = [row for row, origin in zip(preds, dev_origins, strict=True) if origin == "knowledge_dev_v2"]
    parts = composite(fast, cg) if fast and cg else None
    ddx_rows = [preds[i] for i, template in enumerate(dev_templates) if template.startswith(DDX_PREFIX)]
    positional = positional_health(ddx_rows).to_dict() if ddx_rows else None
    metrics = {
        "composite": parts.to_dict() if parts is not None else None,
        "knowledge_dev_accuracy": sum(bool(row["correct"]) for row in know) / len(know) if know else None,
        "positional_original": positional,
        "items": len(preds),
    }
    return metrics, preds


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--snapshot", type=Path, default=SNAP)
    ap.add_argument("--pool", type=Path, default=POOL)
    ap.add_argument("--dev", type=Path, default=DEV)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=13_992)
    ap.add_argument("--eval-every", type=int, default=6_996)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--window", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lam-teacher", type=float, default=3.243598)
    ap.add_argument("--lam-kl", type=float, default=0.1)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-lr", type=float, default=1e-4)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--eval-batch", type=int, default=32)
    ap.add_argument("--max-train-rows", type=int, default=0)
    ap.add_argument("--max-dev-rows", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="tiny run for a code-path check")
    args = ap.parse_args()
    out = args.out if args.out.is_absolute() else ROOT / args.out
    if (out / "arm_result.json").exists() and not args.resume:
        raise SystemExit(f"{out}: arm_result.json exists; the arm is not run twice")
    for directory in (out, out / "logs", out / "checkpoints", out / "evals"):
        directory.mkdir(parents=True, exist_ok=True)
    started = time.time()
    pool_rows = read_jsonl(args.pool)
    dev_rows = read_jsonl(args.dev)
    if args.max_train_rows:
        pool_rows = pool_rows[: args.max_train_rows]
    if args.max_dev_rows:
        dev_rows = dev_rows[: args.max_dev_rows]
    if args.smoke:
        args.steps = min(args.steps, 5)
        args.eval_every = args.steps
    model = AutoModel.from_pretrained(str(args.snapshot), trust_remote_code=True, dtype=torch.bfloat16,
                                      device="cuda:0")
    runtime = model.runtime
    student = D3Student(runtime, lora_r=args.lora_r, lora_alpha=args.lora_alpha).to(runtime.device)
    student.base_readout = runtime.readout.detach().clone()
    lora_params = list(student.lora_parameters().values())
    params = [*lora_params, student.readout]
    optimizer = torch.optim.AdamW(
        [{"params": lora_params, "lr": args.lora_lr}, {"params": [student.readout], "lr": args.head_lr}],
        betas=(0.9, 0.95), weight_decay=0.0,
    )
    print(f"[{utcnow()}] preparing {len(pool_rows)} pool rows and {len(dev_rows)} dev rows", flush=True)
    t0 = time.time()
    train_rows = prepare_rows(runtime, pool_rows)
    train_templates = [str(row["template_id"]) for row in pool_rows]
    dev_prepared = prepare_rows(runtime, dev_rows)
    dev_templates = [str(row["template_id"]) for row in dev_rows]
    dev_origins = [str(row.get("dev_origin", "?")) for row in dev_rows]
    print(f"[{utcnow()}] prepared in {time.time() - t0:.1f}s", flush=True)
    plan = BatchPlan([row.length for row in train_rows], batch=args.batch, window=args.window, seed=args.seed)
    start_step = 0
    if args.resume:
        existing = sorted((out / "checkpoints").glob("step_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
        if existing:
            payload = torch.load(existing[-1], map_location=student.device, weights_only=False)
            student.load_checkpoint_state(payload["model"])
            optimizer.load_state_dict(payload["optimizer"])
            start_step = int(payload["step"])
            print(f"[{utcnow()}] resumed from {existing[-1]} at step {start_step}", flush=True)
    evaluations: dict[str, Any] = {}
    log = (out / "logs" / "train.jsonl").open("a", encoding="utf-8")
    for step in range(start_step + 1, args.steps + 1):
        for group, base_lr in zip(optimizer.param_groups, (args.lora_lr, args.head_lr), strict=True):
            group["lr"] = lr_at(step, args.steps, base_lr)
        indices = plan.batch(step - 1)
        rows = [train_rows[i] for i in indices]
        templates = [train_templates[i] for i in indices]
        t_step = time.time()
        stats = step_losses(student, rows, templates, lam_teacher=args.lam_teacher, lam_kl=args.lam_kl,
                            params=params, probe=step <= 200)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        record = {"event": "step", "step": step, **stats, "seconds": round(time.time() - t_step, 3),
                  "templates": templates}
        log.write(json.dumps(record) + "\n")
        log.flush()
        if step % 50 == 0:
            print(f"[{utcnow()}] step {step}/{args.steps} ce {stats['ce']:.4f} kl_t {stats['kl_teacher']:.4f} "
                  f"kl_b {stats['kl_base']:.4f}", flush=True)
        if step % 2_500 == 0:
            torch.save({"step": step, "model": student.state_dict_for_checkpoint(),
                        "optimizer": optimizer.state_dict()}, out / "checkpoints" / f"step_{step:06d}.pt")
        if step % args.eval_every == 0 or step == args.steps:
            metrics, preds = evaluate(student, dev_prepared, dev_templates, dev_origins,
                                      batch_size=args.eval_batch)
            evaluations[str(step)] = metrics
            folder = out / "evals" / f"step_{step:06d}"
            folder.mkdir(parents=True, exist_ok=True)
            with (folder / "dev.jsonl").open("w", encoding="utf-8") as handle:
                for row in preds:
                    handle.write(json.dumps(row) + "\n")
            torch.save({"step": step, "model": student.state_dict_for_checkpoint(),
                        "optimizer": optimizer.state_dict()}, out / "checkpoints" / f"step_{step:06d}.pt")
            (out / "evaluations.json").write_text(json.dumps(evaluations, indent=2, sort_keys=True) + "\n",
                                                  encoding="utf-8")
            composite_text = (f"S {metrics['composite']['S']:.4f} (MR {metrics['composite']['mr']:.4f}, "
                              f"MIMIC {metrics['composite']['mimic']:.4f}, CG {metrics['composite']['cg_micro']:.4f})"
                              if metrics["composite"] is not None else "S NOT MEASURED (incomplete dev partition)")
            print(f"[{utcnow()}] eval step {step}: {composite_text}; knowledge "
                  f"{metrics['knowledge_dev_accuracy']}; positional "
                  f"{metrics['positional_original']['passes'] if metrics['positional_original'] else None}",
                  flush=True)
    result = {
        "generated_utc": utcnow(), "git_commit": git_commit(ROOT),
        "command": "uv run --frozen python scripts/osler_v7/t8_run_arm.py",
        "steps": args.steps, "start_step": start_step,
        "evaluations": evaluations,
        "lam_teacher": args.lam_teacher, "lam_kl": args.lam_kl,
        "train_rows": len(train_rows), "dev_rows": len(dev_prepared),
        "wall_seconds": round(time.time() - started, 1),
    }
    (out / "arm_result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"done: {args.steps} steps", flush=True)


if __name__ == "__main__":
    main()
