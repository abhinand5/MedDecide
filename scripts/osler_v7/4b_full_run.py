#!/usr/bin/env python
"""osler_v7 4B full run (operator request, ``loops/osler_v7/4B_FULL_RUN.md``): the 9B final's steps and batching.

The 4B arm (T6) was trained for 13,992 steps with the chunked, item-capped batching of ``meddecide.train.data``; the 9B
final (T11) ran 63,304 steps with the token-capped batch plan of ``scripts/osler_v7/t8_run_arm.py``. This driver runs the 4B
with the same steps and the same batching, and changes nothing else: the v7 mix, ``arm_4b.yaml`` (LoRA, learning rates,
cosine schedule, per-item weighting, KL 0.1 on ``v7_c*``), ``lambda_teacher`` and the seed are read as they are.

Two batch sources:

``--batching old``
    T6's path exactly: ``iter_batches`` / ``dosed_batches`` over the mix, chunked length bucketing, the D57 quota.
``--batching token``
    The 9B plan: every row encoded once into a prepared cache (``meddecide.train.prepared``), batches packed over the
    length-sorted rows under ``batch_size`` items and ``max_batch_tokens`` **padded** tokens, batch order shuffled with
    the seed, steps indexing the plan. This mode carries **no D57 quota** — the plan is the 9B final's batch composition
    (see the parity report: the quota, not the batching, is what sets the KL-to-base active share).

Step 1 of the instruction is the 500-step parity of the two modes (``--no-eval``); Step 2 is the 63,304-step run with
evaluations at 13,992 / 27,984 / 41,976 / 55,968 / 63,304. Dev evaluations use the frozen T2 dev set only (fast dev v4,
CG v2, knowledge dev v2, and the DDXPlus items for the D52b positional check); no test split, panel, MIMIC audit suite or
held-out file is read.

Usage (one GPU job, detached; Step 1 parity)::

    source scripts/pod_env.sh && setsid nohup uv run --frozen python scripts/osler_v7/4b_full_run.py \
        --batching token --steps 500 --no-eval --out outputs/osler_v7/4B_FULL/parity/token \
        > outputs/osler_v7/4B_FULL/parity/token/logs/arm.log 2>&1 < /dev/null &
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from meddecide.eval.composite import composite, permutation_for, positional_health
from meddecide.model.meddecide_model import reorder_options
from meddecide.model.pplx import REPO as PPLX_REPO
from meddecide.model.pplx import PplxDecisionModel, PplxPromptEncoder, checkpoint_dir
from meddecide.train.config import StudentConfig, load_config
from meddecide.train.data import option_permutation
from meddecide.train.mix_items import mix_row_to_item
from meddecide.train.osler_arm import dev_prediction_rows
from meddecide.train.prepared import PreparedRows, build_prepared, verify_rows
from meddecide.train.resume import latest_full_checkpoint, truncate_log_after
from meddecide.train.token_plan import (
    TokenBatchPlan,
    build_token_plan,
    plan_statistics,
    save_plan,
)
from meddecide.train.trainer import Trainer, evaluation_grid
from meddecide.utils.hashing import file_sha256
from meddecide.utils.io import write_json, write_jsonl
from meddecide.utils.provenance import git_commit, utcnow

REPO = Path(__file__).resolve().parents[2]
DEV = REPO / "data" / "bench" / "osler_v7" / "dev_format_matched.jsonl"
FULL_CHECKPOINT_EVERY = 2_500  # ADVISORY Z3: full checkpoints every 2,500 steps
FULL_CHECKPOINTS_KEPT = 2
DDX_PREFIX = "cg_v2_ddxplus"
POSITIONAL_SEEDS = (1, 2)
TWO_LETTER_FROM = 26
KL_PREFIX = "v7_c"
DEFAULT_CACHE = REPO / "outputs" / "osler_v7" / "4B_FULL" / "prepared.npz"
DEFAULT_PLAN = REPO / "outputs" / "osler_v7" / "4B_FULL" / "plan.npz"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def resolve(path: str | Path) -> Path:
    return Path(path) if Path(path).is_absolute() else REPO / Path(path)


def relative(path: Path) -> str:
    return str(path.relative_to(REPO)) if path.is_relative_to(REPO) else str(path)


def compose_rows(config: StudentConfig, out: Path, mix_path: Path, manifest_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Read the v7 mix (``t6_build_mix.py``) and check it against its manifest before any step (T6's check, verbatim)."""
    if not mix_path.exists() or not manifest_path.exists():
        raise SystemExit(f"{mix_path} or {manifest_path} is missing: run scripts/osler_v7/t6_build_mix.py first")
    manifest = json.loads(manifest_path.read_text())
    sha = file_sha256(mix_path)
    if sha != manifest["sha256"]:
        raise SystemExit(f"the mix file sha256 {sha} differs from its manifest {manifest['sha256']}: stop")
    rows = read_jsonl(mix_path)
    labelled = sum(1 for row in rows if row.get("teacher_probs") is not None)
    if len(rows) != int(manifest["rows"]) or labelled != int(manifest["labelled_rows"]):
        raise SystemExit(f"the mix file holds {len(rows)} rows ({labelled} labelled), the manifest "
                         f"{manifest['rows']} ({manifest['labelled_rows']}): stop")
    if labelled < config.distill_min_labelled_per_batch:
        raise SystemExit(f"only {labelled} labelled rows for a quota of {config.distill_min_labelled_per_batch}: stop")
    if manifest["c1c4_rows"] <= 0:
        raise SystemExit("the mix has no C1-C4 rows: the Lever-1 pool would be empty")
    info = {"file": relative(mix_path), "sha256": sha, "rows": len(rows), "labelled_rows": labelled,
            "labelled_share_of_rows": labelled / len(rows), "c1c4_rows": manifest["c1c4_rows"],
            "cells": manifest["cells"], "family_share": manifest.get("family_share")}
    write_json(out / "mix_check.json", info)
    return rows, info


def plan_dose(plan: TokenBatchPlan, prepared: PreparedRows) -> dict[str, Any]:
    """The plan's step-level dose: the share of batches that carry a ``v7_c`` row (KL) or a teacher row (distillation)."""
    pool_by_template = np.array([template.startswith(KL_PREFIX) for template in prepared.templates], dtype=bool)
    pool = pool_by_template[prepared.template_index]
    labelled = prepared.labelled()
    pool_per_batch = np.array([int(pool[list(batch)].sum()) for batch in plan.batches])
    labelled_per_batch = np.array([int(labelled[list(batch)].sum()) for batch in plan.batches])
    return {
        "rows": int(plan.n_items),
        "rows_v7_c": int(pool.sum()),
        "rows_v7_c_share": float(pool.mean()),
        "rows_labelled": int(labelled.sum()),
        "rows_labelled_share": float(labelled.mean()),
        "steps_with_v7_c": int((pool_per_batch > 0).sum()),
        "steps_with_v7_c_share": float((pool_per_batch > 0).mean()),
        "steps_with_teacher": int((labelled_per_batch > 0).sum()),
        "steps_with_teacher_share": float((labelled_per_batch > 0).mean()),
        "mean_v7_c_rows_per_step": float(pool_per_batch.mean()),
        "mean_labelled_rows_per_step": float(labelled_per_batch.mean()),
        "max_labelled_rows_per_step": int(labelled_per_batch.max()),
    }


def prepare_cache(args: argparse.Namespace, config: StudentConfig, out: Path) -> tuple[PreparedRows, Path]:
    """Build (or load) the prepared-row cache: one CPU pass over the mix with pplx's own prompt encoder."""
    cache = resolve(args.cache)
    if cache.exists():
        if args.max_train_rows:
            raise SystemExit(f"{cache} exists: --max-train-rows is a build-time slice and needs a fresh --cache path")
        started = time.time()
        prepared = PreparedRows.load(cache)
        print(f"[{utcnow()}] loaded {len(prepared)} prepared rows from {cache} in {time.time() - started:.1f}s",
              flush=True)
        return prepared, cache
    mix_path = resolve(args.mix)
    rows = read_jsonl(mix_path)
    if args.max_train_rows:
        rows = rows[: args.max_train_rows]
    encoder = PplxPromptEncoder(args.snapshot or PPLX_REPO, max_prompt_tokens=config.max_prompt_tokens)
    print(f"[{utcnow()}] preparing {len(rows)} rows with {encoder.describe()}", flush=True)

    def encode(position: int) -> Any:
        item = mix_row_to_item(rows[position])
        permutation = (option_permutation(item.n_options, seed=config.seed, epoch=0, index=position)
                       if config.shuffle_options else None)
        return encoder.encode_item(item, permutation=permutation, max_prompt_tokens=config.max_prompt_tokens)

    started = time.time()
    prepared = build_prepared(len(rows), encode, option_codes=encoder.optcode_names)
    prepared.save(cache)
    print(f"[{utcnow()}] cache written to {cache} in {time.time() - started:.0f}s", flush=True)
    write_json(out / "cache_build.json", {"rows": len(prepared), "tokens": int(prepared.ids.shape[0]),
                                          "file": relative(cache), "seconds": round(time.time() - started, 1),
                                          "snapshot": str(encoder.checkpoint_path),
                                          "shuffle_options": bool(config.shuffle_options)})
    return prepared, cache


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", default="configs/osler_v7/arm_4b.yaml")
    parser.add_argument("--mix", default="data/train/osler_v7/mix_v7.jsonl")
    parser.add_argument("--mix-manifest", default="outputs/osler_v7/T6/mix_manifest.json")
    parser.add_argument("--batching", choices=("old", "token"), default="token")
    parser.add_argument("--out", required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--distill-weight", type=float, default=3.243598, help="the frozen lambda (D57)")
    parser.add_argument("--min-labelled", type=int, default=None,
                        help="D57 quota; old defaults to the config (3), token must be 0 (the plan carries no quota)")
    parser.add_argument("--cache", default=str(DEFAULT_CACHE))
    parser.add_argument("--plan", default=str(DEFAULT_PLAN))
    parser.add_argument("--snapshot", default=None, help="pplx snapshot directory (default: the pinned HF cache)")
    parser.add_argument("--no-eval", action="store_true", help="training only (the 500-step parity)")
    parser.add_argument("--eval-steps", default="", help="comma-separated evaluation steps (Step 2's five points)")
    parser.add_argument("--evaluations", type=int, default=2, help="evaluations spread evenly when --eval-steps is empty")
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--probe-steps", type=int, default=200,
                        help="D57 gradient-share probe on the first N steps (0 = off; T6 used 200)")
    parser.add_argument("--max-train-rows", type=int, default=0, help="smoke only: slice the training rows")
    parser.add_argument("--verify-cache", type=int, default=32, help="re-encode N cached rows with the model and compare")
    parser.add_argument("--smoke", action="store_true", help="cut every dev set to a few hundred items (positional keeps both groups)")
    parser.add_argument("--prepare-only", action="store_true", help="build the cache and the plan, then stop before the model")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    out = resolve(args.out)
    if (out / "arm_result.json").exists() and not args.resume and not args.prepare_only:
        raise SystemExit(f"{out}: arm_result.json exists; the arm is not run twice (refuse-twice guard)")
    for directory in (out, out / "logs", out / "evals", out / "checkpoints"):
        directory.mkdir(parents=True, exist_ok=True)
    started = utcnow()
    config = load_config(resolve(args.config))
    if args.min_labelled is None:
        min_labelled = 0 if args.batching == "token" else int(config.distill_min_labelled_per_batch)
    else:
        min_labelled = int(args.min_labelled)
    if args.batching == "token" and min_labelled > 0:
        raise SystemExit("the token plan carries its own batch composition: --min-labelled must be 0 in --batching token "
                         "(see loops/osler_v7/STATE.md: the quota, not the batching, sets the KL-to-base share)")
    if args.no_eval:
        grid: tuple[int, ...] = ()
    elif args.eval_steps:
        grid = tuple(int(value) for value in args.eval_steps.split(",") if value.strip())
        if any(step < 1 or step > args.steps for step in grid):
            raise SystemExit(f"evaluation steps {grid} outside 1..{args.steps}")
    else:
        grid = evaluation_grid(args.steps, max(1, args.evaluations))
    probe = args.probe_steps > 0 and args.distill_weight > 0
    config = dataclasses.replace(
        config, output_dir=relative(out), eval_steps=grid, eval_every=0, backbone="pplx",
        kl_to_base=config.kl_to_base, distill_weight=float(args.distill_weight),
        distill_min_labelled_per_batch=min_labelled, distill_probe=probe,
        distill_probe_steps=int(args.probe_steps) if probe else 0, log_every=int(args.log_every),
    )
    if config.distill_min_labelled_per_batch > 0 and config.distill_weight <= 0:
        raise SystemExit("the D57 quota needs a positive --distill-weight")

    rows, mix_info = compose_rows(config, out, resolve(args.mix), resolve(args.mix_manifest))
    prepared: PreparedRows | None = None
    plan: TokenBatchPlan | None = None
    train_items: list[Any] = []
    plan_info: dict[str, Any] = {}
    cache_path: Path | None = None
    if args.batching == "token":
        del rows  # the token path reads the mix only through its cache (one row at a time)
        prepared, cache_path = prepare_cache(args, config, out)
        plan = build_token_plan(prepared.lengths, batch_size=config.batch_size,
                                max_tokens=config.max_batch_tokens, seed=config.seed)
        save_plan(plan, resolve(args.plan))
        plan_info = {"mode": "token", "quota": 0, **plan.to_dict(), **plan_statistics(plan, prepared.lengths),
                     "dose": plan_dose(plan, prepared)}
        write_json(resolve(args.plan).with_suffix(".json"), plan_info)
        print(f"[{utcnow()}] plan: {plan_info['batches']} batches, mean {plan_info['mean_items']:.2f} items and "
              f"{plan_info['mean_padded_tokens']:.0f} padded tokens a step; KL-to-base active on "
              f"{plan_info['dose']['steps_with_v7_c_share']:.2%} of steps, distillation on "
              f"{plan_info['dose']['steps_with_teacher_share']:.2%}", flush=True)
        if args.prepare_only:
            write_json(out / "prepare_only.json", {"utc": utcnow(), "cache": relative(cache_path), "plan": plan_info,
                                                   "mix": mix_info})
            return
    else:
        if args.max_train_rows:
            rows = rows[: args.max_train_rows]
        train_items = [mix_row_to_item(row) for row in rows]
        del rows
        plan_info = {"mode": "old", "quota": min_labelled, "rows": len(train_items)}
        if args.prepare_only:
            write_json(out / "prepare_only.json", {"utc": utcnow(), "batching": "old", "rows": len(train_items),
                                                   "mix": mix_info})
            return

    snapshot = Path(args.snapshot) if args.snapshot else checkpoint_dir(PPLX_REPO)
    torch.manual_seed(config.seed)  # the initialisation, as arm_run.run_arm seeds it
    model = PplxDecisionModel(snapshot, lora=config.lora, dtype=config.dtype, device=config.device,
                              max_prompt_tokens=config.max_prompt_tokens)
    if config.gradient_checkpointing:
        model.enable_gradient_checkpointing()
    write_json(out / "run_provenance.json", {
        "started_utc": started, "git_commit": git_commit(REPO), "command": " ".join(sys.argv), "config": config.to_dict(),
        "steps": args.steps, "evaluation_steps": list(grid), "mix": mix_info, "batching": args.batching,
        "plan": plan_info, "model": {"checkpoint": str(model.checkpoint_path), "base_model_id": model.base_model_id,
                                     "saved_temperature": model.saved_temperature},
        "cache": ({"file": relative(cache_path), "rows": len(prepared), "tokens": int(prepared.ids.shape[0])}
                  if cache_path is not None and prepared is not None else None),
    })
    if args.batching == "token" and args.verify_cache and prepared is not None:
        rng = np.random.default_rng([config.seed, 4242])
        positions = [int(p) for p in rng.choice(len(prepared), size=min(args.verify_cache, len(prepared)), replace=False)]
        rows_for_verify = read_jsonl(resolve(args.mix))
        live = []
        for position in positions:
            item = mix_row_to_item(rows_for_verify[position])
            permutation = (option_permutation(item.n_options, seed=config.seed, epoch=0, index=position)
                           if config.shuffle_options else None)
            live.append(model.encode_item(item, permutation=permutation, max_prompt_tokens=config.max_prompt_tokens))
        check = verify_rows(prepared, live, positions)
        write_json(out / "cache_verify.json", check)
        print(f"[{utcnow()}] cache verification: {check['checked']} rows, {len(check['mismatched'])} mismatched",
              flush=True)
        if not check["identical"]:
            raise SystemExit("the prepared cache does not reproduce the model's encoding: stop")

    fast_items: list[Any] = []
    cg_items: list[Any] = []
    know_items: list[Any] = []
    ddx_items: list[Any] = []
    ddx_index: list[int] = []
    if grid:
        dev_rows = read_jsonl(DEV)
        fast_rows = [row for row in dev_rows if str(row.get("dev_origin")) == "fastdev_v4"]
        cg_rows = [row for row in dev_rows if str(row.get("dev_origin")) == "cg_v2"]
        know_rows = [row for row in dev_rows if str(row.get("dev_origin")) == "knowledge_dev_v2"]
        if not fast_rows or not cg_rows or not know_rows:
            raise SystemExit("the dev file is missing a partition (fastdev_v4 / cg_v2 / knowledge_dev_v2)")
        if args.smoke:
            fast_rows, cg_rows, know_rows = fast_rows[::50], cg_rows[::20], know_rows[::20]
        fast_items = [mix_row_to_item(r) for r in fast_rows]
        cg_items = [mix_row_to_item(r) for r in cg_rows]
        know_items = [mix_row_to_item(r) for r in know_rows]
        ddx_index = [i for i, item in enumerate(cg_items) if str(item.template_id).startswith(DDX_PREFIX)]
        if not ddx_index or any(cg_items[i].n_options <= TWO_LETTER_FROM for i in ddx_index):
            raise SystemExit("the positional check needs the DDXPlus items of CG v2 with more than 26 options")
        ddx_items = [cg_items[i] for i in ddx_index]
        write_json(out / "dev_sets.json", {"fast": len(fast_items), "cg": len(cg_items), "knowledge": len(know_items),
                                           "positional": len(ddx_items)})

    resume_from = None
    if args.resume:
        resume_from = latest_full_checkpoint(out / "full")
        if resume_from is None:
            raise SystemExit("--resume, but no complete full checkpoint exists")
        step = int(resume_from.name.split("_")[1])
        truncate_log_after(out / "logs" / "train.jsonl", step)
        with (out / "logs" / "resumes.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"resumed_from": relative(resume_from), "step": step, "utc": utcnow()}) + "\n")

    trainer = Trainer(model, config, log_path=out / "logs" / "train.jsonl")
    evaluations: dict[int, dict[str, Any]] = {}
    if args.resume and (out / "evaluations.json").exists():
        evaluations = {int(key): value for key, value in json.loads((out / "evaluations.json").read_text()).items()}

    def evaluate_dev(step: int, _metrics: dict[str, Any]) -> None:
        """Every dev set at one evaluation: rows saved, composite and positional health recorded; the checkpoint kept."""
        started_eval = time.time()
        fast_scored = cached.pop("fast") if "fast" in cached else trainer.score(fast_items)
        cg_scored = trainer.score(cg_items)
        know_scored = trainer.score(know_items)
        fast_pred = dev_prediction_rows(fast_scored)
        cg_pred = dev_prediction_rows(cg_scored)
        know_pred = dev_prediction_rows(know_scored)
        perm_health: dict[str, Any] = {}
        ddx_pred_by_seed: dict[str, list[dict[str, Any]]] = {}
        for seed in POSITIONAL_SEEDS:
            permuted = [reorder_options(item, permutation_for(item.n_options, seed=seed, index=i))
                        for item, i in zip(ddx_items, ddx_index, strict=True)]
            pred = dev_prediction_rows(trainer.score(permuted))
            ddx_pred_by_seed[str(seed)] = pred
            perm_health[str(seed)] = positional_health(pred).to_dict()
        parts = composite(fast_pred, cg_pred)
        original = positional_health([r for r in cg_pred if r["template_id"].startswith(DDX_PREFIX)]).to_dict()
        eval_dir = out / "evals" / f"step_{step:06d}"
        eval_dir.mkdir(parents=True, exist_ok=True)
        write_jsonl(eval_dir / "fast_dev.jsonl", fast_pred)
        write_jsonl(eval_dir / "cg_v2.jsonl", cg_pred)
        write_jsonl(eval_dir / "knowledge_dev.jsonl", know_pred)
        for seed, pred in ddx_pred_by_seed.items():
            write_jsonl(eval_dir / f"positional_seed{seed}.jsonl", pred)
        model.save(out / "checkpoints" / f"step_{step:06d}")
        knowledge = sum(bool(r["correct"]) for r in know_pred) / len(know_pred)
        record = {"step": step, "composite": parts.to_dict(), "knowledge_dev_accuracy": knowledge,
                  "positional_original": original, "positional_permuted": perm_health,
                  "positional_passes_all_orders": bool(original["passes"] and all(h["passes"] for h in perm_health.values())),
                  "seconds": round(time.time() - started_eval, 1)}
        evaluations[step] = record
        write_json(out / "evaluations.json", {str(k): v for k, v in sorted(evaluations.items())})
        print(f"eval step {step}: S {parts.s:.4f} (MR {parts.mr:.4f}, MIMIC {parts.mimic:.4f}, CG micro {parts.cg_micro:.4f}); "
              f"knowledge {knowledge:.4f}; positional original {original['passes']} seeds "
              f"{[h['passes'] for h in perm_health.values()]}", flush=True)

    cached: dict[str, Any] = {}
    if grid:
        trainer.eval_hook = evaluate_dev
        score_direct = trainer.score

        def score_and_keep_fast_dev(items: list[Any], *, batch_size: int | None = None) -> Any:
            scored = score_direct(items, batch_size=batch_size)
            if items is fast_items:
                cached["fast"] = scored
            return scored

        trainer.score = score_and_keep_fast_dev  # type: ignore[method-assign]

    result = trainer.train(
        train_items, steps=args.steps, eval_items=(fast_items if grid else None), schedule_steps=args.steps,
        full_checkpoint_dir=out / "full", full_checkpoint_every=FULL_CHECKPOINT_EVERY,
        resume_from=resume_from, full_checkpoint_keep_last=FULL_CHECKPOINTS_KEPT,
        batch_plan=(plan.batches if plan is not None else None), encoded=prepared,
    )
    finals: dict[str, Any] = {}
    if grid:
        model.eval_mode()
        fits = Trainer(model, config).fit_calibration(fast_items)
        finals = {
            "temperature_fits": {k: v.to_dict() if hasattr(v, "to_dict") else str(v) for k, v in fits.items()},
            "final_fast_dev": dev_prediction_rows(Trainer(model, config).score(fast_items)),
            "final_cg_v2": dev_prediction_rows(Trainer(model, config).score(cg_items)),
        }
        write_jsonl(out / "final_fast_dev.jsonl", finals.pop("final_fast_dev"))
        write_jsonl(out / "final_cg_v2.jsonl", finals.pop("final_cg_v2"))
        model.save(out / "checkpoints" / f"final_step_{result.steps:06d}")
    write_json(out / "arm_result.json", {
        "steps": result.steps, "stopped_early": result.stopped_early, "evaluation_steps": list(grid),
        "evaluations": {str(k): v for k, v in sorted(evaluations.items())},
        "batching": args.batching, "quota": min_labelled, "plan": plan_info,
        "final_step": result.steps, "wall_clock_s": round(result.wall_clock_s, 1),
        "finished_utc": utcnow(), "config": config.to_dict(), "git_commit": git_commit(REPO), **finals,
    })
    print(f"done: {result.steps} steps; evaluations at {list(grid)}", flush=True)


if __name__ == "__main__":
    main()
