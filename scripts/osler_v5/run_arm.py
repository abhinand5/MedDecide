"""osler_v5 arm driver (ADVISORY Z1-Z3): train one recipe arm or the final on osler_v4's exact mix, and at every evaluation score the
dev sets only: fast dev v4 (MR, MIMIC and the rest), CG v2 (micro), knowledge dev v2, and the DDXPlus items of CG v2 under the original
order and two seeded option permutations (D52b). The per-evaluation checkpoints are kept, for the Z3 selection by S among the
evaluations that pass positional health. No test split, panel, MIMIC audit suite or held-out file is read here.

The mix is composed in memory with osler_v4's Y8 composer (composition 9d25e30f…, checked before any step). A config with
option_augment_probability > 0 rebuilds the closed-list rows of the TRAINING rows before they become items (D52a); the report of
the rebuild is written next to the arm. The run refuses to start when the arm has an arm_result.json (refuse-twice guard) and
resumes from the newest full checkpoint with --resume (the watchdog rule of ADVISORY §3.7).

Usage (GPU; one job; detached):
    source scripts/pod_env.sh && setsid nohup uv run --frozen python scripts/osler_v5/run_arm.py \
        --config configs/osler_v5/arm_R0.yaml --out outputs/osler_v5/Z2/R0 --steps 13992 --evaluations 2 \
        > outputs/osler_v5/Z2/R0/logs/run.log 2>&1 < /dev/null &
    (after an interruption: the same command with --resume, appended to the same log)
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

from meddecide.eval.composite import composite, permutation_for, positional_health
from meddecide.model.meddecide_model import reorder_options
from meddecide.model.pplx import PplxDecisionModel
from meddecide.train.config import StudentConfig, load_config
from meddecide.train.final_mix import compose_final_mix
from meddecide.train.mix_items import mix_row_to_item
from meddecide.train.option_augment import augment_rows, build_pools
from meddecide.train.osler_arm import dev_prediction_rows
from meddecide.train.resume import latest_full_checkpoint, truncate_log_after
from meddecide.train.trainer import Trainer, evaluation_grid
from meddecide.utils.io import write_json, write_jsonl
from meddecide.utils.provenance import git_commit, utcnow

REPO = Path(__file__).resolve().parents[2]
DEV = REPO / "data" / "bench" / "osler_v4" / "fastdev_v4.jsonl"
CG = REPO / "data" / "bench" / "osler_v4" / "cg_v2.jsonl"
KNOW = REPO / "data" / "bench" / "osler_v2" / "knowledge_dev_v2.jsonl"
EXPECTED_COMPOSITION = "9d25e30f53d7823d051b9c71334c248660318c86ce904c26f855b9a9dbc618b8"
FULL_CHECKPOINT_EVERY = 2_500  # ADVISORY Z3: full checkpoints every 2,500 steps
FULL_CHECKPOINTS_KEPT = 2
DDX_PREFIX = "cg_v2_ddxplus"  # the CG v2 items that offer more than 26 options (49 options)
POSITIONAL_SEEDS = (1, 2)
TWO_LETTER_FROM = 26


def _module(name: str, relative: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, REPO / relative)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def compose_rows(config: StudentConfig, out: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """osler_v4's final mix in memory (the Y8 composer), then the closed-list augmentation of its training rows if configured."""
    y8_mix = _module("y8_final_mix", "scripts/osler_v4/y8_final_mix.py")
    decisions = {name: y8_mix.entering(name) for name in ("B1", "B2", "B3", "B4")}
    family = {name: y8_mix.add_files(name) for name in y8_mix.FAMILY if decisions[name][0]}
    other = y8_mix.add_files("B4") if decisions["B4"][0] else []
    rows, composition = compose_final_mix(y8_mix.BASE, family, other, y8_mix.BROAD)
    if composition["composition_sha256"] != EXPECTED_COMPOSITION:
        raise SystemExit(f"the composed mix hash is {composition['composition_sha256']}, not osler_v4's {EXPECTED_COMPOSITION}")
    info: dict[str, Any] = {"composition_sha256": composition["composition_sha256"], "rows": len(rows)}
    if config.option_augment_probability > 0:
        pools = build_pools(rows, config.option_augment_templates)
        rows, report = augment_rows(rows, pools, seed=config.seed, probability=config.option_augment_probability)
        info["augmentation"] = {"probability": config.option_augment_probability, "templates": list(config.option_augment_templates),
                                "pool_sizes": {t: len(p.entries) for t, p in pools.items()},
                                "label_set_sizes": {t: p.label_set_size for t, p in pools.items()}, **report.to_dict()}
        write_json(out / "augmentation.json", info["augmentation"])
    return rows, info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--evaluations", type=int, default=2, help="evaluations spread evenly over the steps, the last at the budget")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    out = (REPO / args.out) if not Path(args.out).is_absolute() else Path(args.out)
    if (out / "arm_result.json").exists():
        raise SystemExit(f"{out}: arm_result.json exists; the arm is not run twice (refuse-twice guard)")
    for directory in (out, out / "logs", out / "evals", out / "checkpoints"):
        directory.mkdir(parents=True, exist_ok=True)
    started = utcnow()
    config = load_config(REPO / args.config)
    grid = evaluation_grid(args.steps, args.evaluations)
    config = dataclasses.replace(config, output_dir=str(out.relative_to(REPO)), eval_steps=grid, eval_every=args.steps // args.evaluations,
                                 backbone="pplx", kl_to_base=config.kl_to_base, distill_weight=0.0)

    rows, mix_info = compose_rows(config, out)
    train_items = [mix_row_to_item(row) for row in rows]
    del rows
    fast_rows, cg_rows, know_rows = read_jsonl(DEV), read_jsonl(CG), read_jsonl(KNOW)
    fast_items = [mix_row_to_item(r) for r in fast_rows]
    cg_items = [mix_row_to_item(r) for r in cg_rows]
    know_items = [mix_row_to_item(r) for r in know_rows]
    ddx_index = [i for i, item in enumerate(cg_items) if str(item.template_id).startswith(DDX_PREFIX)]
    if not ddx_index or any(cg_items[i].n_options <= TWO_LETTER_FROM for i in ddx_index):
        raise SystemExit("the positional check needs the DDXPlus items of CG v2 with more than 26 options")
    ddx_items = [cg_items[i] for i in ddx_index]
    write_json(out / "run_provenance.json", {
        "started_utc": started, "git_commit": git_commit(REPO), "command": " ".join(sys.argv), "config": config.to_dict(),
        "steps": args.steps, "evaluation_steps": list(grid), "mix": mix_info, "dev_items": len(fast_items), "cg_items": len(cg_items),
        "knowledge_items": len(know_items), "positional_items": len(ddx_items),
    })

    resume_from = None
    if args.resume:
        resume_from = latest_full_checkpoint(out / "full")
        if resume_from is None:
            raise SystemExit("--resume, but no complete full checkpoint exists")
        step = int(resume_from.name.split("_")[1])
        truncate_log_after(out / "logs" / "train.jsonl", step)
        with (out / "logs" / "resumes.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"resumed_from": str(resume_from.relative_to(REPO)), "step": step, "utc": utcnow()}) + "\n")

    torch.manual_seed(config.seed)  # the initialisation, as arm_run.run_arm seeds it (matched arms start from the same weights)
    model = PplxDecisionModel(lora=config.lora, dtype=config.dtype, device=config.device,
                              max_prompt_tokens=config.max_prompt_tokens)
    if config.gradient_checkpointing:
        model.enable_gradient_checkpointing()
    trainer = Trainer(model, config, log_path=out / "logs" / "train.jsonl")
    evaluations: dict[int, dict[str, Any]] = {}

    def evaluate_dev(step: int, _metrics: dict[str, Any]) -> None:
        """Every dev set at one evaluation: rows saved, composite and positional health recorded; the model's checkpoint kept."""
        started_eval = time.time()
        fast_scored = trainer.score(fast_items)
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

    trainer.eval_hook = evaluate_dev
    # the trainer evaluates the full fast dev v4 at each evaluation step, as in osler_v4 Y8: the dev-macro tripwire reads that
    # measurement; the hook then scores the dev sets for the composite, the knowledge set and the positional check
    result = trainer.train(train_items, steps=args.steps, eval_items=fast_items, schedule_steps=args.steps,
                           full_checkpoint_dir=out / "full", full_checkpoint_every=FULL_CHECKPOINT_EVERY,
                           resume_from=resume_from, full_checkpoint_keep_last=FULL_CHECKPOINTS_KEPT)
    model.eval_mode()
    fits = Trainer(model, config).fit_calibration(fast_items)
    final_step = result.steps
    eval_steps = [int(s) for s in evaluations]
    final_scores = {
        "fast_dev": dev_prediction_rows(Trainer(model, config).score(fast_items)),
        "cg_v2": dev_prediction_rows(Trainer(model, config).score(cg_items)),
    }
    write_jsonl(out / "final_fast_dev.jsonl", final_scores["fast_dev"])
    write_jsonl(out / "final_cg_v2.jsonl", final_scores["cg_v2"])
    model.save(out / "checkpoints" / f"final_step_{final_step:06d}")
    write_json(out / "arm_result.json", {
        "steps": result.steps, "stopped_early": result.stopped_early, "evaluation_steps": eval_steps,
        "evaluations": {str(k): v for k, v in sorted(evaluations.items())},
        "temperature_fits": {k: v.to_dict() if hasattr(v, "to_dict") else str(v) for k, v in fits.items()},
        "final_step": final_step, "finished_utc": utcnow(), "config": config.to_dict(), "git_commit": git_commit(REPO),
    })
    print(f"done: {result.steps} steps; evaluations at {eval_steps}", flush=True)


if __name__ == "__main__":
    main()
