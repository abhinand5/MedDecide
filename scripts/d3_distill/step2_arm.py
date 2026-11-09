"""d3_distill step 2 driver: train one student arm (or the final) on the teacher-labelled pool.

One arm = one run of the pre-registered recipe (ADVISORY §2-4): attention-only LoRA + trained readout, CE(gold) +
λ·KL(teacher ‖ student) on every row, batch 8, deterministic windowed sampler, dev evaluations, checkpoints.

Modes:
  --probe   200 steps at λ = 1, measure the pre-clip gradient share per step, write the probe JSON (no weights kept).
            λ = (0.4/0.6)·(1-s0)/s0 is computed by the caller from the JSON.
  (default) train with --lam for --steps steps; the first 200 steps measure the share (the D57 band).

Resume from the newest checkpoint with --resume. Dev = the train-side dev carve (never trained on). No test-side row is
read here.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ.setdefault("TRITON_CACHE_AUTOTUNING", "1")

import torch
from torch import nn
from transformers import AutoModel

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from meddecide.distill.d3_train import (  # noqa: E402
    BatchPlan,
    D3Student,
    TrainRow,
    batch_tensors,
    gradient_norm,
    lr_at,
    prepare_rows,
)
from meddecide.utils.hashing import file_sha256  # noqa: E402

POOL = REPO / "data" / "train" / "d3_distill" / "pool_200k.jsonl"
LABELS = REPO / "outputs" / "d3_distill" / "step1" / "labels.jsonl"
PROBE_STEPS = 200
SHARE_BAND = (0.30, 0.50)


def utcnow() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_pool(pool: Path, labels: Path, *, max_train: int = 0, max_dev: int = 0) -> tuple[list[dict], list[dict], dict]:
    """Join the pool with the teacher labels; returns the train and dev row dicts and a summary."""
    labelled: dict[str, list[float] | None] = {}
    invalid = 0
    for row in read_jsonl(labels):
        if row["valid"]:
            labelled[row["row_id"]] = row["teacher_probs"]
        else:
            invalid += 1
    train, dev, missing = [], [], 0
    for row in read_jsonl(pool):
        if row["row_id"] not in labelled:
            missing += 1
            continue
        row = dict(row, teacher_probs=labelled[row["row_id"]])
        (train if row["split"] == "train" else dev).append(row)
    if invalid:
        raise SystemExit(f"{invalid} invalid label rows in {labels}: the pool is not usable")
    if missing:
        raise SystemExit(f"{missing} pool rows have no label row: {labels} is incomplete")
    if max_train:
        train = train[:max_train]
    if max_dev:
        dev = dev[:max_dev]
    summary = {
        "labels_file": str(labels),
        "labels_sha256": file_sha256(labels),
        "pool_file": str(pool),
        "pool_sha256": file_sha256(pool),
        "train_rows": len(train),
        "dev_rows": len(dev),
        "train_source_counts": dict(sorted(Counter(r["source"] for r in train).items())),
        "dev_source_counts": dict(sorted(Counter(r["source"] for r in dev).items())),
    }
    return train, dev, summary


def accuracy(probs: list[list[float]], rows: list[TrainRow]) -> tuple[int, dict[str, int], dict[str, int]]:
    correct, per_source, total = 0, Counter(), Counter()
    for value, row in zip(probs, rows, strict=True):
        best = max(range(len(value)), key=value.__getitem__)
        hit = int(best == row.gold)
        correct += hit
        per_source[row.source] += hit
        total[row.source] += 1
    return correct, dict(sorted(per_source.items())), dict(sorted(total.items()))


def evaluate(student: D3Student, dev_rows: list[TrainRow], *, batch_size: int) -> dict:
    probs = student.probabilities([r.sequence for r in dev_rows], [r.count for r in dev_rows], batch_size=batch_size)
    correct, per_source, totals = accuracy(probs, dev_rows)
    kls = []
    for value, row in zip(probs, dev_rows, strict=True):
        if row.teacher is None:
            continue
        target = torch.tensor(row.teacher, dtype=torch.float32)
        student_p = torch.tensor(value, dtype=torch.float32).clamp_min(1e-12)
        kls.append(float((target * (torch.log(target.clamp_min(1e-12)) - torch.log(student_p))).sum()))
    return {
        "items": len(dev_rows),
        "correct": correct,
        "accuracy": correct / len(dev_rows),
        "per_source_accuracy": {s: per_source.get(s, 0) / totals[s] for s in totals},
        "teacher_kl_mean": sum(kls) / len(kls) if kls else None,
        "per_source_items": totals,
    }


def backward_losses(student: D3Student, rows: list[TrainRow], lam: float, *, probe: bool, params: list[nn.Parameter]):
    """One forward; returns (ce, kl, share, grad_norm_preclip) and leaves grads on the parameters."""
    ids, mask, counts, golds = batch_tensors(rows, student.pad_id, student.device)
    teachers = [row.teacher for row in rows]
    loss_ce, loss_kl = student.loss_terms(ids, mask, counts, golds, teachers)
    batch = len(rows)
    ce = loss_ce / batch
    kl = loss_kl / batch
    if probe:
        g_ce = torch.autograd.grad(ce, params, retain_graph=True, allow_unused=True)
        g_kl = torch.autograd.grad(lam * kl, params, retain_graph=True, allow_unused=True)
        norm_ce = gradient_norm(params, g_ce)
        norm_kl = gradient_norm(params, g_kl)
        share = norm_kl / (norm_ce + norm_kl) if norm_ce + norm_kl > 0 else float("nan")
        grads = [
            None if a is None and b is None else (a if b is None else b if a is None else a + b)
            for a, b in zip(g_ce, g_kl, strict=True)
        ]
    else:
        grads = torch.autograd.grad(ce + lam * kl, params, allow_unused=True)
        share, norm_ce, norm_kl = None, None, None
    for parameter, grad in zip(params, grads, strict=True):
        parameter.grad = None if grad is None else grad.detach()
    pre_clip = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
    return float(ce.detach()), float(kl.detach()), share, pre_clip, norm_ce, norm_kl


def save_checkpoint(path: Path, student: D3Student, optimizer: torch.optim.Optimizer, step: int, meta: dict) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save({"step": step, "model": student.state_dict_for_checkpoint(),
                "optimizer": optimizer.state_dict(), "meta": meta}, tmp)
    tmp.replace(path)


def load_checkpoint(path: Path, student: D3Student, optimizer: torch.optim.Optimizer | None) -> dict:
    payload = torch.load(path, map_location=student.device, weights_only=False)
    student.load_checkpoint_state(payload["model"])
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer"])
    return payload


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--snapshot", type=Path, required=True)
    ap.add_argument("--pool", type=Path, default=POOL)
    ap.add_argument("--labels", type=Path, default=LABELS)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--schedule-steps", type=int, default=0,
                    help="length of the LR schedule (default: --steps); a probe passes the arm's 13,992")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lam", type=float, default=0.0, help="frozen λ; ignored with --probe")
    ap.add_argument("--probe", action="store_true", help="200 steps at λ=1; no weights are kept")
    ap.add_argument("--probe-steps", type=int, default=PROBE_STEPS)
    ap.add_argument("--eval-every", type=int, default=3500)
    ap.add_argument("--eval-batch", type=int, default=32)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--window", type=int, default=64)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-lr", type=float, default=1e-4)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--max-train-rows", type=int, default=0)
    ap.add_argument("--max-dev-rows", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="tiny data and steps for a code-path check")
    args = ap.parse_args()

    if args.smoke:
        args.max_train_rows = args.max_train_rows or 512
        args.max_dev_rows = args.max_dev_rows or 256
        args.eval_every = min(args.eval_every, max(1, args.steps))
    schedule_steps = args.schedule_steps or args.steps
    out = args.out if args.out.is_absolute() else REPO / args.out
    if (out / "arm_result.json").exists() and not args.resume:
        raise SystemExit(f"{out}: arm_result.json exists; the arm is not run twice (refuse-twice guard)")
    for directory in (out, out / "logs", out / "checkpoints", out / "evals"):
        directory.mkdir(parents=True, exist_ok=True)
    started_utc = utcnow()
    t_start = time.time()

    train_rows_dict, dev_rows_dict, data_summary = load_pool(
        args.pool, args.labels, max_train=args.max_train_rows, max_dev=args.max_dev_rows
    )
    print(f"[{utcnow()}] data: {len(train_rows_dict)} train, {len(dev_rows_dict)} dev", flush=True)

    model = AutoModel.from_pretrained(
        str(args.snapshot.resolve()), trust_remote_code=True, dtype=torch.bfloat16, device="cuda:0"
    )
    runtime = model.runtime
    student = D3Student(runtime, lora_r=args.lora_r, lora_alpha=args.lora_alpha).to(runtime.device)
    lora_params = list(student.lora_parameters().values())
    params = [*lora_params, student.readout]
    optimizer = torch.optim.AdamW(
        [{"params": lora_params, "lr": args.lora_lr}, {"params": [student.readout], "lr": args.head_lr}],
        betas=(0.9, 0.95), weight_decay=0.0,
    )
    print(f"[{utcnow()}] student loaded: {len(lora_params)} LoRA tensors, {len(student.adapters)} adapters, "
          f"readout {tuple(student.readout.shape)}", flush=True)

    t_prepare = time.time()
    train_rows = prepare_rows(runtime, train_rows_dict)
    dev_rows = prepare_rows(runtime, dev_rows_dict)
    prepare_s = time.time() - t_prepare
    print(f"[{utcnow()}] prepared {len(train_rows)} train + {len(dev_rows)} dev rows in {prepare_s:.1f}s", flush=True)
    plan = BatchPlan([r.length for r in train_rows], batch=args.batch, window=args.window, seed=args.seed)

    # Zero-shot dev reference (the frozen student, before any step).
    with torch.inference_mode():
        zero_shot = evaluate(student, dev_rows, batch_size=args.eval_batch) if dev_rows else None
    if zero_shot:
        print(f"[{utcnow()}] zero-shot dev accuracy {zero_shot['accuracy']:.4f} "
              f"({zero_shot['correct']}/{zero_shot['items']})", flush=True)

    probe_steps = min(args.probe_steps, args.steps) if (args.probe or args.lam > 0) else 0
    lam = 1.0 if args.probe else args.lam
    if not args.probe and args.lam <= 0:
        raise SystemExit("a training arm needs --lam > 0 (run --probe first); λ=0 is not a distill arm")

    start_step = 0
    best_accuracy, best_step = None, None
    evals_path = out / "evals.jsonl"
    if args.resume and evals_path.exists():
        prior = [json.loads(line) for line in evals_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if prior:
            best = max(prior, key=lambda e: e["accuracy"])
            best_accuracy, best_step = best["accuracy"], best["step"]
    steps_path = out / "steps.jsonl"
    if args.resume:
        checkpoints = sorted((out / "checkpoints").glob("step_*.pt"))
        if not checkpoints:
            raise SystemExit(f"{out}: --resume but no checkpoint")
        payload = load_checkpoint(checkpoints[-1], student, optimizer)
        start_step = int(payload["step"]) + 1
        print(f"[{utcnow()}] resumed at step {start_step} from {checkpoints[-1].name}", flush=True)

    cfg = {
        "student": str(args.snapshot), "revision": args.snapshot.name, "steps": args.steps, "seed": args.seed,
        "lam": lam, "probe": args.probe, "batch": args.batch, "window": args.window,
        "schedule_steps": schedule_steps,
        "lora": {"r": args.lora_r, "alpha": args.lora_alpha, "targets": ["q_proj", "k_proj", "v_proj", "o_proj"],
                 "lr": args.lora_lr},
        "head_lr": args.head_lr, "grad_clip": args.grad_clip, "temperature": student.temperature,
        "started_utc": started_utc, "prepare_s": round(prepare_s, 1), **data_summary,
    }
    (out / "config.json").write_text(json.dumps(cfg, indent=1), encoding="utf-8")

    shares, band_shares = [], []
    log_every = 1 if args.probe else 10
    for step in range(start_step, args.steps):
        t_step = time.time()
        for group, base in ((optimizer.param_groups[0], args.lora_lr), (optimizer.param_groups[1], args.head_lr)):
            group["lr"] = lr_at(step, schedule_steps, base)
        rows = [train_rows[i] for i in plan.batch(step)]
        labelled = sum(1 for r in rows if r.teacher is not None)
        if labelled < 3:
            raise SystemExit(f"step {step}: {labelled} labelled rows, below the D57 minimum of 3")
        probe_this = step < probe_steps
        ce, kl, share, pre_clip, norm_ce, norm_kl = backward_losses(
            student, rows, lam, probe=probe_this, params=params
        )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if probe_this:
            shares.append(share)
            band_shares.append(share)
        if step % log_every == 0 or step == args.steps - 1 or probe_this:
            entry = {
                "step": step, "ce": round(ce, 6), "kl": round(kl, 6), "lam": lam,
                "share": None if share is None else round(share, 6),
                "grad_norm_preclip": round(pre_clip, 4),
                "norm_ce": None if norm_ce is None else round(norm_ce, 6),
                "norm_kl": None if norm_kl is None else round(norm_kl, 6),
                "labelled": labelled, "batch": len(rows),
                "lr_lora": optimizer.param_groups[0]["lr"], "lr_head": optimizer.param_groups[1]["lr"],
                "sec": round(time.time() - t_step, 4),
            }
            with steps_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry) + "\n")
        if (step + 1) % 250 == 0 or step == args.steps - 1:
            print(f"[{utcnow()}] step {step + 1}/{args.steps} ce {ce:.4f} kl {kl:.4f} "
                  f"share {'-' if share is None else f'{share:.3f}'} preclip {pre_clip:.2f} "
                  f"{time.time() - t_start:.0f}s", flush=True)

        last = step == args.steps - 1
        eval_now = bool(dev_rows) and not args.probe and (last or ((step + 1) % args.eval_every == 0))
        checkpoint_now = not args.probe and (last or ((step + 1) % args.eval_every == 0))
        if checkpoint_now:
            save_checkpoint(out / "checkpoints" / f"step_{step + 1:06d}.pt", student, optimizer, step, cfg)
            keep = sorted((out / "checkpoints").glob("step_*.pt"))[-3:]
            for old in sorted((out / "checkpoints").glob("step_*.pt"))[:-len(keep)]:
                old.unlink()
        if eval_now:
            result = evaluate(student, dev_rows, batch_size=args.eval_batch)
            result.update({"step": step + 1, "eval_utc": utcnow()})
            (out / "evals" / f"eval_{step + 1:06d}.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
            with (out / "evals.jsonl").open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(result) + "\n")
            print(f"[{utcnow()}] eval at {step + 1}: dev accuracy {result['accuracy']:.4f} "
                  f"({result['correct']}/{result['items']})", flush=True)
            if best_accuracy is None or result["accuracy"] > best_accuracy:
                best_accuracy, best_step = result["accuracy"], step + 1
                save_checkpoint(out / "best.pt", student, optimizer, step, {**cfg, "selected_for": "best dev accuracy"})

    wall = time.time() - t_start
    final_eval = None
    evals_path = out / "evals.jsonl"
    if evals_path.exists():
        lines = [json.loads(line) for line in evals_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if lines:
            final_eval = lines[-1]
    band = None
    if band_shares:
        mean_band = sum(band_shares) / len(band_shares)
        band = {"steps": len(band_shares), "mean_share": mean_band,
                "passed": SHARE_BAND[0] <= mean_band <= SHARE_BAND[1], "band": list(SHARE_BAND)}
    probe_share = sum(shares) / len(shares) if (args.probe and shares) else None
    result = {
        "mode": "probe" if args.probe else "arm",
        "out": str(out),
        "steps_run": args.steps - start_step,
        "start_step": start_step,
        "wall_s": round(wall, 1),
        "s_per_step": round(wall / max(1, args.steps - start_step), 3),
        "band": band,
        "probe_share": probe_share,
        "best_dev_accuracy": best_accuracy,
        "best_dev_step": best_step,
        "final_eval": final_eval,
        "zero_shot_dev": zero_shot,
        "finished_utc": utcnow(),
        "cfg": cfg,
        "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
        "git_commit": subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True,
                                     text=True).stdout.strip(),
    }
    (out / ("probe_result.json" if args.probe else "arm_result.json")).write_text(
        json.dumps(result, indent=1), encoding="utf-8"
    )
    print(json.dumps({k: result[k] for k in ("mode", "steps_run", "wall_s", "s_per_step", "band",
                                             "probe_share", "peak_gpu_gb")}, indent=1), flush=True)


if __name__ == "__main__":
    main()
