"""osler_v6 arm driver (ADVISORY §2.5-2.7): train one recipe arm or the final on the prepared distilled mix, and at every
evaluation score the dev sets only: fast dev v4 (MR, MIMIC and the rest), CG v2 (micro), knowledge dev v2, and the DDXPlus items of
CG v2 under the original order and two seeded option permutations (D52b). The per-evaluation checkpoints are kept, for the
selection by S among the evaluations that pass positional health. No test split, panel, MIMIC audit suite or held-out file is read
here.

The mix is the file `scripts/osler_v6/build_mix.py` built: osler_v5's corrected F2 mix (composition 78d7e547…, checked by the
builder) with the Z1 teacher distributions attached and the D52a option augmentation already applied at the arm's seed. This driver
reads that file and checks it against its manifest sha256 before the first step. The distillation enters through
``distill_weight`` (the frozen λ) and the D57 quota sampler (``distill_min_labelled_per_batch``); with a weight of 0 the same
driver runs the G0 reference. The run refuses to start when the arm has an arm_result.json (refuse-twice guard) and resumes from
the newest full checkpoint with --resume (the watchdog rule of ADVISORY §3.7).

Usage (GPU; one job; detached):
    source scripts/pod_env.sh && setsid nohup uv run --frozen python scripts/osler_v6/run_arm.py \
        --config configs/osler_v5/arm_F2.yaml --mix data/train/osler_v6/mix_f2_distill.jsonl \
        --distill-weight <λ> --min-labelled 3 --out outputs/osler_v6/Z3/G1s0 --steps 13992 --evaluations 2 \
        > outputs/osler_v6/Z3/G1s0/logs/run.log 2>&1 < /dev/null &
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

from meddecide.bench.schema import Item
from meddecide.eval.composite import composite, permutation_for, positional_health
from meddecide.model.meddecide_model import reorder_options
from meddecide.model.pplx import PplxDecisionModel
from meddecide.train.config import StudentConfig, load_config
from meddecide.train.mix_items import mix_row_to_item
from meddecide.train.osler_arm import dev_prediction_rows
from meddecide.train.resume import latest_full_checkpoint, truncate_log_after
from meddecide.train.trainer import Trainer, evaluation_grid
from meddecide.utils.hashing import file_sha256
from meddecide.utils.io import write_json, write_jsonl
from meddecide.utils.provenance import git_commit, utcnow

REPO = Path(__file__).resolve().parents[2]
DEV = REPO / "data" / "bench" / "osler_v4" / "fastdev_v4.jsonl"
CG = REPO / "data" / "bench" / "osler_v4" / "cg_v2.jsonl"
KNOW = REPO / "data" / "bench" / "osler_v2" / "knowledge_dev_v2.jsonl"
EXPECTED_COMPOSITION = "78d7e5478060fc828784cb3365569e177c6961270d38a76a9cff144684e9f3be"  # the D56a-corrected mix
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


def compose_rows(config: StudentConfig, out: Path, mix_path: Path, manifest_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Read the prepared distilled mix (``build_mix.py``) and check it against its manifest before any step."""
    if not mix_path.exists() or not manifest_path.exists():
        raise SystemExit(f"{mix_path} or {manifest_path} is missing: run scripts/osler_v6/build_mix.py build first")
    manifest = json.loads(manifest_path.read_text())
    sha = file_sha256(mix_path)
    if sha != manifest["output_sha256"]:
        raise SystemExit(f"the mix file sha256 {sha} differs from its manifest {manifest['output_sha256']}: stop")
    if manifest["mix"]["composition_sha256"] != EXPECTED_COMPOSITION:
        raise SystemExit(f"the mix manifest's composition hash is {manifest['mix']['composition_sha256']}, "
                         f"not {EXPECTED_COMPOSITION}: stop")
    if int(manifest["seed"]) != int(config.seed):
        raise SystemExit(f"the mix was built at augmentation seed {manifest['seed']}, the arm runs at {config.seed}: stop")
    rows = read_jsonl(mix_path)
    labelled = sum(1 for row in rows if row.get("teacher_probs") is not None)
    if len(rows) != int(manifest["rows"]) or labelled != int(manifest["labelled_rows"]):
        raise SystemExit(f"the mix file holds {len(rows)} rows ({labelled} labelled), the manifest "
                         f"{manifest['rows']} ({manifest['labelled_rows']}): stop")
    if labelled < config.distill_min_labelled_per_batch:
        raise SystemExit(f"only {labelled} labelled rows for a quota of {config.distill_min_labelled_per_batch}: stop")
    info = {"file": str(mix_path.relative_to(REPO)), "sha256": sha, "rows": len(rows), "labelled_rows": labelled,
            "labelled_share_of_rows": labelled / len(rows), "seed": int(manifest["seed"]),
            "composition_sha256": manifest["mix"]["composition_sha256"],
            "soft_label_file_sha256": manifest["soft_label_file_sha256"],
            "augmentation": manifest["mix"].get("augmentation")}
    write_json(out / "mix_check.json", info)
    return rows, info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--mix", required=True)
    parser.add_argument("--mix-manifest", required=True)
    parser.add_argument("--distill-weight", type=float, required=True, help="the frozen λ (D57); 0 runs the G0 reference")
    parser.add_argument("--min-labelled", type=int, default=3, help="D57 quota: labelled anchors per batch")
    parser.add_argument("--out", required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--evaluations", type=int, default=2, help="evaluations spread evenly over the steps, the last at the budget")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true",
                        help="a code-path check only: every dev set is cut to a few hundred items (the positional check keeps both groups)")
    parser.add_argument("--dry-run", action="store_true",
                        help="compose the mix, apply the augmentation and convert every item on the CPU, then stop before the model")
    args = parser.parse_args()
    out = (REPO / args.out) if not Path(args.out).is_absolute() else Path(args.out)
    if (out / "arm_result.json").exists() and not args.dry_run:
        raise SystemExit(f"{out}: arm_result.json exists; the arm is not run twice (refuse-twice guard)")
    for directory in (out, out / "logs", out / "evals", out / "checkpoints"):
        directory.mkdir(parents=True, exist_ok=True)
    started = utcnow()
    config = load_config(REPO / args.config)
    grid = evaluation_grid(args.steps, args.evaluations)
    config = dataclasses.replace(config, output_dir=str(out.relative_to(REPO)), eval_steps=grid, eval_every=args.steps // args.evaluations,
                                 backbone="pplx", kl_to_base=config.kl_to_base, distill_weight=float(args.distill_weight),
                                 distill_min_labelled_per_batch=int(args.min_labelled), distill_probe=False)
    if config.distill_min_labelled_per_batch > 0 and config.distill_weight <= 0:
        raise SystemExit("the D57 quota needs a positive --distill-weight")

    rows, mix_info = compose_rows(config, out, (REPO / args.mix) if not Path(args.mix).is_absolute() else Path(args.mix),
                                  (REPO / args.mix_manifest) if not Path(args.mix_manifest).is_absolute() else Path(args.mix_manifest))
    train_items = [mix_row_to_item(row) for row in rows]
    del rows
    fast_rows, cg_rows, know_rows = read_jsonl(DEV), read_jsonl(CG), read_jsonl(KNOW)
    if args.smoke:
        fast_rows, cg_rows, know_rows = fast_rows[::50], cg_rows[::20], know_rows[::20]
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

    if args.dry_run:
        counts = {"train_items": len(train_items), "dev_items": len(fast_items), "cg_items": len(cg_items),
                  "knowledge_items": len(know_items), "positional_items": len(ddx_items), "grid": list(grid)}
        write_json(out / "dry_run.json", {"utc": utcnow(), "config": config.to_dict(), "mix": mix_info, "counts": counts})
        print(json.dumps({"dry_run": True, **counts, "labelled_rows": mix_info["labelled_rows"]}, sort_keys=True), flush=True)
        return

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
        # the trainer has just scored the full fast dev set for its tripwire (the same model, items and settings): reuse that result
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

    trainer.eval_hook = evaluate_dev
    cached: dict[str, Any] = {}
    score_direct = trainer.score

    def score_and_keep_fast_dev(items: list[Item], *, batch_size: int | None = None) -> Any:
        scored = score_direct(items, batch_size=batch_size)
        if items is fast_items:
            cached["fast"] = scored
        return scored

    trainer.score = score_and_keep_fast_dev  # type: ignore[method-assign]
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
