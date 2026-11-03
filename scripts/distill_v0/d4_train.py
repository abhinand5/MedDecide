"""D4/D5: train one distillation student (distill_v0) to the pre-registered 63,304-step budget.

Composes the D2 pool (the same composition hashes are checked), attaches the frozen d1-ultra
distributions by text key, applies the F2 option augmentation, and trains with
``CE + 1.0 * Brier + lambda * KL(teacher || model)``. The distillation weight is calibrated over the
first 200 steps (ADVISORY §3) and frozen; the per-step teacher/MIMIC dose is logged and the
pre-registered 1,000-step tripwire stops the run if either share leaves its band. Ten evaluations at
the osler_v4 Y8 grid score fast dev v4, CG v2, knowledge dev v2 and positional health, and full
checkpoints are written every 2,500 steps (newest two kept).

Usage (GPU; one job; detached):
    source scripts/pod_env.sh && setsid nohup uv run --frozen python scripts/distill_v0/d4_train.py \
        --config configs/distill_v0/student_4b.yaml --out outputs/distill_v0/D4 --steps 63304 \
        > outputs/distill_v0/D4/logs/run.log 2>&1 < /dev/null &
    (after an interruption: the same command with --resume)
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

from meddecide.bench.schema import Item
from meddecide.distill.d1_teacher import MODEL as TEACHER_MODEL
from meddecide.distill.d1_teacher import file_sha256
from meddecide.eval.composite import composite, permutation_for, positional_health
from meddecide.model.meddecide_model import MedDecideModel, reorder_options
from meddecide.model.pplx import PplxDecisionModel
from meddecide.train.config import load_config
from meddecide.train.distill_mix import POOL_ROWS_EXPECTED, select_teacher_rows
from meddecide.train.distill_pool import attach_teacher_probs, compose_pool, load_labels
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
LABELS = REPO / "outputs" / "distill_v0" / "D2" / "labels.jsonl"
LABEL_MANIFEST = REPO / "outputs" / "distill_v0" / "D2" / "label_manifest.json"
FULL_CHECKPOINT_EVERY = 2_500
FULL_CHECKPOINTS_KEPT = 2
DDX_PREFIX = "cg_v2_ddxplus"
POSITIONAL_SEEDS = (1, 2)
TWO_LETTER_FROM = 26
DOSE_TRIPWIRE_STEP = 1_000
TEACHER_BAND = (0.38, 0.45)
MIMIC_BAND = (0.37, 0.42)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--steps", type=int, default=63_304)
    parser.add_argument("--evaluations", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="a code-path check (dev sets cut, few steps)")
    args = parser.parse_args()
    out = (REPO / args.out) if not Path(args.out).is_absolute() else Path(args.out)
    for directory in (out, out / "logs", out / "evals", out / "checkpoints", out / "full"):
        directory.mkdir(parents=True, exist_ok=True)
    started = utcnow()
    config = load_config(REPO / args.config)
    steps = args.steps
    grid = evaluation_grid(steps, args.evaluations)
    config = dataclasses.replace(config, output_dir=str(out.relative_to(REPO)), eval_steps=grid,
                                 eval_every=max(1, steps // args.evaluations), kl_to_base=0.0)
    if (out / "arm_result.json").exists() and not args.dry_run:
        raise SystemExit(f"{out}: arm_result.json exists; the run is not repeated (refuse-twice guard)")
    if config.distill_weight <= 0:
        raise SystemExit("D4 is a distillation run: distill_weight must be > 0")

    print("composing the D2 pool ...", flush=True)
    pool, keys, pool_info = compose_pool()
    if not pool_info["pool_matches_expected"]:
        raise SystemExit(f"pool rows {pool_info['pool_rows']} != {POOL_ROWS_EXPECTED}")
    if not LABEL_MANIFEST.exists():
        raise SystemExit(f"missing {LABEL_MANIFEST}; D2 must finish first")
    label_manifest = json.loads(LABEL_MANIFEST.read_text(encoding="utf-8"))
    if file_sha256(LABELS) != label_manifest["labels_sha256"]:
        raise SystemExit("the frozen label file does not match the D2 manifest's SHA-256")
    labels = load_labels(LABELS)
    teacher_indices, teacher_report = select_teacher_rows(pool)
    basket_keys = {keys[index] for index in teacher_indices}
    print(f"pool {pool_info['pool_rows']} rows; frozen labels {len(labels)}; basket keys {len(basket_keys)}; attaching ...",
          flush=True)
    attach = attach_teacher_probs(pool, keys, labels, model_id=TEACHER_MODEL, basket_keys=basket_keys)
    print(f"attach: {attach}", flush=True)
    if (attach["attached_keys"] != len(labels) or attach["missing_label"] or attach["template_mismatch"]
            or attach["option_order_mismatch"]):
        raise SystemExit(f"the teacher attachment is not complete: {attach}")
    del keys, labels

    pools = build_pools(pool, config.option_augment_templates)
    rows, augmentation = augment_rows(pool, pools, seed=config.seed, probability=config.option_augment_probability)
    write_json(out / "augmentation.json", {"probability": config.option_augment_probability,
                                           "pool_sizes": {t: len(p.entries) for t, p in pools.items()},
                                           "label_set_sizes": {t: p.label_set_size for t, p in pools.items()},
                                           **augmentation.to_dict()})
    del pool
    print("converting rows to items ...", flush=True)
    train_items = [mix_row_to_item(row) for row in rows]
    del rows
    n_teacher_items = sum(1 for item in train_items if getattr(item, "teacher_probs", None) is not None)
    if n_teacher_items != attach["attached"]:
        raise SystemExit(f"teacher items in the training set ({n_teacher_items}) != attached rows ({attach['attached']})")
    if n_teacher_items != teacher_report["selected_total"]:
        raise SystemExit(f"teacher items ({n_teacher_items}) != the basket's {teacher_report['selected_total']} rows")

    fast_rows, cg_rows, know_rows = read_jsonl(DEV), read_jsonl(CG), read_jsonl(KNOW)
    if args.smoke:
        fast_rows, cg_rows, know_rows = fast_rows[::50], cg_rows[::20], know_rows[::20]
    fast_items = [mix_row_to_item(row) for row in fast_rows]
    cg_items = [mix_row_to_item(row) for row in cg_rows]
    know_items = [mix_row_to_item(row) for row in know_rows]
    ddx_index = [i for i, item in enumerate(cg_items) if str(item.template_id).startswith(DDX_PREFIX)]
    if not ddx_index or any(cg_items[i].n_options <= TWO_LETTER_FROM for i in ddx_index):
        raise SystemExit("the positional check needs the DDXPlus items of CG v2 with more than 26 options")
    ddx_items = [cg_items[i] for i in ddx_index]
    provenance = {
        "task": "D4 distillation training", "started_utc": started, "git_commit": git_commit(REPO),
        "command": " ".join(sys.argv), "config": config.to_dict(), "steps": steps,
        "evaluation_steps": list(grid), "pool": pool_info, "labels": {"path": str(LABELS.relative_to(REPO)),
                                                                     "sha256": label_manifest["labels_sha256"]},
        "attach": attach, "train_items": len(train_items), "teacher_items": n_teacher_items,
        "dev_items": len(fast_items), "cg_items": len(cg_items), "knowledge_items": len(know_items),
        "positional_items": len(ddx_items),
    }
    write_json(out / "run_provenance.json", provenance)
    print(json.dumps({"train_items": len(train_items), "teacher_items": n_teacher_items,
                      "teacher_share": n_teacher_items / len(train_items)}, indent=1), flush=True)
    if args.dry_run:
        write_json(out / "dry_run.json", {"utc": utcnow(), "counts": {k: provenance[k] for k in
                                                                     ("train_items", "teacher_items", "dev_items",
                                                                      "cg_items", "knowledge_items", "positional_items")},
                                          "config": config.to_dict()})
        print("dry run: no model loaded", flush=True)
        return

    torch.manual_seed(config.seed)
    if config.backbone == "pplx":
        model = PplxDecisionModel(lora=config.lora, dtype=config.dtype, device=config.device,
                                  max_prompt_tokens=config.max_prompt_tokens, revision=config.revision)
    elif config.backbone == "clef":
        from meddecide.model.clef import load_clef_text_backbone

        backbone, clef_info = load_clef_text_backbone(dtype=config.dtype, device=config.device)
        provenance["clef"] = clef_info
        model = MedDecideModel(config.base_model, revision=config.revision, lora=config.lora,
                               readout=config.readout, dtype=config.dtype, device=config.device,
                               variant=config.variant, max_prompt_tokens=config.max_prompt_tokens,
                               base_override=backbone, backbone="clef")
    else:
        base = Path(config.base_model)
        base = base if base.is_absolute() else REPO / base
        if not (base / "config.json").exists():
            raise SystemExit(f"the base model directory does not exist: {base}")
        model = MedDecideModel(str(base), lora=config.lora, readout=config.readout, dtype=config.dtype,
                               device=config.device, variant=config.variant,
                               max_prompt_tokens=config.max_prompt_tokens)
    if config.gradient_checkpointing:
        model.enable_gradient_checkpointing()
    trainer = Trainer(model, config, log_path=out / "logs" / "train.jsonl")
    evaluations: dict[int, dict[str, Any]] = {}

    def evaluate_dev(step: int, _metrics: dict[str, Any]) -> None:
        eval_started = time.time()
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
        original = positional_health([row for row in cg_pred if row["template_id"].startswith(DDX_PREFIX)]).to_dict()
        eval_dir = out / "evals" / f"step_{step:06d}"
        eval_dir.mkdir(parents=True, exist_ok=True)
        write_jsonl(eval_dir / "fast_dev.jsonl", fast_pred)
        write_jsonl(eval_dir / "cg_v2.jsonl", cg_pred)
        write_jsonl(eval_dir / "knowledge_dev.jsonl", know_pred)
        for seed, pred in ddx_pred_by_seed.items():
            write_jsonl(eval_dir / f"positional_seed{seed}.jsonl", pred)
        model.save(out / "checkpoints" / f"step_{step:06d}")
        knowledge = sum(bool(row["correct"]) for row in know_pred) / len(know_pred)
        record = {"step": step, "composite": parts.to_dict(), "knowledge_dev_accuracy": knowledge,
                  "positional_original": original, "positional_permuted": perm_health,
                  "positional_passes_all_orders": bool(original["passes"]
                                                       and all(h["passes"] for h in perm_health.values())),
                  "seconds": round(time.time() - eval_started, 1)}
        evaluations[step] = record
        write_json(out / "evaluations.json", {str(key): value for key, value in sorted(evaluations.items())})
        print(f"eval step {step}: S {parts.s:.4f} (MR {parts.mr:.4f}, MIMIC {parts.mimic:.4f}, CG {parts.cg_micro:.4f}); "
              f"knowledge {knowledge:.4f}; positional {original['passes']} seeds "
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

    dose_path = out / "logs" / "dose.jsonl"
    dose_handle = dose_path.open("w", encoding="utf-8")
    first_thousand: list[dict[str, Any]] = []

    def on_step(_step: int) -> None:
        record = trainer.history[-1]
        entry = {"step": int(record.step), "n_teacher": int(record.n_teacher), "n_items": int(record.n_items),
                 "templates": list(record.template_ids)}
        dose_handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
        if entry["step"] <= DOSE_TRIPWIRE_STEP:
            first_thousand.append(entry)
        if entry["step"] == DOSE_TRIPWIRE_STEP:
            items = sum(row["n_items"] for row in first_thousand)
            teacher = sum(row["n_teacher"] for row in first_thousand) / items
            mimic = sum(1 for row in first_thousand for template in row["templates"]
                        if template.startswith("mimic_")) / items
            tripwire = {"step": DOSE_TRIPWIRE_STEP, "items": items, "teacher_share": teacher, "mimic_share": mimic,
                        "teacher_band": list(TEACHER_BAND), "mimic_band": list(MIMIC_BAND),
                        "mean_teacher_per_batch": sum(row["n_teacher"] for row in first_thousand) / len(first_thousand)}
            write_json(out / "dose_tripwire_1000.json", tripwire)
            print(f"dose tripwire at 1,000 steps: teacher {teacher:.4f}, MIMIC {mimic:.4f}", flush=True)
            if not (TEACHER_BAND[0] <= teacher <= TEACHER_BAND[1] and MIMIC_BAND[0] <= mimic <= MIMIC_BAND[1]):
                dose_handle.close()
                raise SystemExit(f"dose tripwire: teacher {teacher:.4f} MIMIC {mimic:.4f} outside the pre-registered bands")

    resume_from = None
    if args.resume:
        resume_from = latest_full_checkpoint(out / "full")
        if resume_from is None:
            raise SystemExit("--resume, but no complete full checkpoint exists")
        step = int(resume_from.name.split("_")[1])
        truncate_log_after(out / "logs" / "train.jsonl", step)
        with (out / "logs" / "resumes.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"resumed_from": str(resume_from.relative_to(REPO)), "step": step,
                                     "utc": utcnow()}) + "\n")

    result = trainer.train(train_items, steps=steps, eval_items=fast_items, schedule_steps=steps,
                           full_checkpoint_dir=out / "full", full_checkpoint_every=FULL_CHECKPOINT_EVERY,
                           resume_from=resume_from, full_checkpoint_keep_last=FULL_CHECKPOINTS_KEPT, on_step=on_step)
    dose_handle.close()
    model.eval_mode()
    fits = Trainer(model, config).fit_calibration(fast_items)
    final_step = result.steps
    final_scores = {"fast_dev": dev_prediction_rows(Trainer(model, config).score(fast_items)),
                    "cg_v2": dev_prediction_rows(Trainer(model, config).score(cg_items))}
    write_jsonl(out / "final_fast_dev.jsonl", final_scores["fast_dev"])
    write_jsonl(out / "final_cg_v2.jsonl", final_scores["cg_v2"])
    model.save(out / "checkpoints" / f"final_step_{final_step:06d}")

    dose = [json.loads(line) for line in dose_path.open(encoding="utf-8") if line.strip()]
    total_items = sum(row["n_items"] for row in dose)
    total_teacher = sum(row["n_teacher"] for row in dose)
    total_mimic = sum(1 for row in dose for template in row["templates"] if template.startswith("mimic_"))
    per_batch = [row["n_teacher"] for row in dose]
    dose_summary = {
        "steps": len(dose), "items": total_items, "teacher_items": total_teacher,
        "teacher_share": total_teacher / total_items, "mimic_share": total_mimic / total_items,
        "mean_teacher_per_batch": sum(per_batch) / len(per_batch),
        "share_batches_below_3": sum(1 for value in per_batch if value < 3) / len(per_batch),
        "per_batch_histogram": {str(k): per_batch.count(k) for k in sorted(set(per_batch))},
        "teacher_band": list(TEACHER_BAND), "mimic_band": list(MIMIC_BAND),
    }
    write_json(out / "dose_summary.json", dose_summary)
    lambda_path = out / "logs" / "lambda_calibration.json"
    lambda_value = json.loads(lambda_path.read_text(encoding="utf-8"))["lambda"] if lambda_path.exists() else None
    write_json(out / "arm_result.json", {
        "task": "D4", "steps": result.steps, "stopped_early": result.stopped_early,
        "evaluation_steps": [int(step) for step in evaluations],
        "evaluations": {str(key): value for key, value in sorted(evaluations.items())},
        "temperature_fits": {key: value.to_dict() if hasattr(value, "to_dict") else str(value)
                             for key, value in fits.items()},
        "final_step": final_step, "finished_utc": utcnow(), "config": config.to_dict(),
        "git_commit": git_commit(REPO), "pool_text_sha256": pool_info["pool_text_sha256"],
        "labels_sha256": label_manifest["labels_sha256"], "attach": attach,
        "lambda": lambda_value, "dose": dose_summary,
    })
    print(f"done: {result.steps} steps; lambda {lambda_value}; dose {dose_summary['mean_teacher_per_batch']:.2f}/batch; "
          f"evaluations at {list(evaluations)}", flush=True)


if __name__ == "__main__":
    main()
