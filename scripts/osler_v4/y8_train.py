"""osler_v4 Y8 (ADVISORY Y8; operator ruling 2026-10-23, items 1 to 3; operator steer 2026-10-23): the final Osler-4B (GPU; one job).

The recipe is A0's (configs/osler_v3/block_arm.yaml with KL to base 0 and no distillation). The final mix is composed in memory from its
recorded sources (osler_v2's C1 as the base; the added files of each MIMIC-family block that enters by the operator's item 1; the
replay top-up from osler_v3's broad pool; meddecide.train.final_mix). The disk quota cannot hold the ~9.5 GB mix file, so the composition
is identified by the SHA-256 of its item ids and by the hashes of its sources, both recorded in outputs/osler_v4/Y8/final_mix.json.
The replay (the batches the trainer will draw) is computed in process from the same items and recorded in outputs/osler_v4/Y8/replays/.
The step budget is 63,304 with ten evaluations on fast dev v4 at evaluation_grid(63304, 10), the last at the budget. The selected checkpoint
is the trainer's (the pooled template macro accuracy, Brier tie-break). At every evaluation the CG v2 set and the MIMIC partition (the fast
dev v4 items whose template id starts with mimic_) are scored and recorded; they are not in the selection. After training, the selected
checkpoint is scored on CG v2 and the predictions are written. The temperatures are fitted on fast dev v4 and saved with the checkpoint.
The run refuses to start when outputs/osler_v4/Y8/arm_result.json exists (the osler_v3 X9 guard).

Resumable (operator steer, 2026-10-23): a full checkpoint is written every 5,000 steps under outputs/osler_v4/Y8/full/; only the newest two are
kept (the disk quota). `--resume` continues from the latest complete one with the same data order; the logs are cut back to that step first.

Usage (GPU; one job; detached; nothing else on the GPU):
    source scripts/pod_env.sh && setsid nohup uv run --frozen python scripts/osler_v4/y8_train.py \
        > outputs/osler_v4/Y8/logs/final_train.log 2>&1 < /dev/null &
    (after an interruption: the same command with --resume, appended to the same log)
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import itertools
import json
import sys
from pathlib import Path
from typing import Any

from meddecide.model.pplx import ATTENTION_TARGETS, load_any
from meddecide.train.arm_run import run_arm
from meddecide.train.config import load_config
from meddecide.train.data import iter_batches, shuffled_order
from meddecide.train.final_mix import compose_final_mix
from meddecide.train.mix_items import mix_row_to_item
from meddecide.train.osler_arm import dev_prediction_rows
from meddecide.train.resume import latest_full_checkpoint, truncate_log_after
from meddecide.train.trainer import Trainer, evaluation_grid
from meddecide.utils.hashing import file_sha256
from meddecide.utils.io import write_json, write_jsonl
from meddecide.utils.provenance import git_commit, utcnow

REPO = Path(__file__).resolve().parents[2]
RECIPE = REPO / "configs" / "osler_v3" / "block_arm.yaml"
BASE = REPO / "data" / "train" / "osler_v2" / "block_C1.jsonl"
BROAD = REPO / "data" / "train" / "osler_v3" / "replay_broad_train.jsonl"
DEV = REPO / "data" / "bench" / "osler_v4" / "fastdev_v4.jsonl"
CG = REPO / "data" / "bench" / "osler_v4" / "cg_v2.jsonl"
OUT = REPO / "outputs" / "osler_v4" / "Y8"
STEPS = 63_304  # ADVISORY Y8: the smallest multiple of 8 at or above 1.5 x 42,200
EVALUATIONS = 10  # ADVISORY Y8
MIMIC_PREFIX = "mimic_"
CHECKPOINT_EVERY = 5_000  # the operator steer (2026-10-23): a full checkpoint every 5,000 steps
CHECKPOINTS_KEPT = 2  # the disk quota: the newest two full checkpoints are kept


def _module(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, REPO / relative)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def numbers(metrics: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in metrics.items() if isinstance(v, (int, float))}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def replay_records(items: list, config: Any, steps: int) -> list[tuple[int, int, int]]:
    """The batches the trainer will draw, as (items, real tokens, padded tokens) per step: the osler_v3 replay on the same items."""
    rp = _module("w6_replay", "scripts/osler_v2/w6_replay.py")
    encoder = rp.encoder_for("pplx", config)
    order = shuffled_order(len(items), seed=config.seed, epoch=0)
    batches = iter_batches(encoder, items, batch_size=config.batch_size, max_batch_tokens=config.max_batch_tokens,
                           max_prompt_tokens=config.max_prompt_tokens, order=order, augment_options=config.shuffle_options,
                           epoch=0, seed=config.seed)
    records = [rp.batch_record(batch) for _, batch in itertools.islice(batches, steps)]
    batches.close()
    if len(records) != steps:
        raise SystemExit(f"Y8: the replay drew {len(records)} batches for {steps} steps")
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--resume", action="store_true",
                        help="continue from the latest complete full checkpoint under outputs/osler_v4/Y8/full")
    args = parser.parse_args()
    if (OUT / "arm_result.json").exists():
        raise SystemExit("Y8: outputs/osler_v4/Y8/arm_result.json exists; the final run is not repeated (osler_v3 X9 guard)")
    started = utcnow()
    for directory in (OUT, OUT / "logs", OUT / "replays"):
        directory.mkdir(parents=True, exist_ok=True)
    y8_mix = _module("y8_final_mix", "scripts/osler_v4/y8_final_mix.py")
    decisions = {name: y8_mix.entering(name) for name in ("B1", "B2", "B3", "B4")}
    family = {name: y8_mix.add_files(name) for name in y8_mix.FAMILY if decisions[name][0]}
    other = y8_mix.add_files("B4") if decisions["B4"][0] else []
    rows, composition = compose_final_mix(BASE, family, other, BROAD)
    composition_id = composition["composition_sha256"]
    sources = {"base": str(BASE.relative_to(REPO)), "base_sha256": file_sha256(BASE),
               "broad_pool": str(BROAD.relative_to(REPO)), "broad_pool_sha256": file_sha256(BROAD),
               "family": {name: {str(p.relative_to(REPO)): file_sha256(p) for p in paths} for name, paths in family.items()},
               "other": {str(p.relative_to(REPO)): file_sha256(p) for p in other}}
    write_json(OUT / "final_mix.json", {
        "built_utc": started, "git_commit": git_commit(REPO), "command": " ".join(sys.argv), "in_memory": True,
        "composition_sha256": composition_id, "sources": sources,
        "entering": {name: {"enters": decisions[name][0], "reason": decisions[name][1], **decisions[name][2]}
                     for name in decisions},
        **{k: v for k, v in composition.items() if k != "family_kept_per_file"}, "family_kept_per_file": composition["family_kept_per_file"],
    })
    config = load_config(RECIPE)
    grid = evaluation_grid(STEPS, EVALUATIONS)
    config = dataclasses.replace(
        config,
        train_path=str((OUT / "final_mix.json").relative_to(REPO)),
        dev_path=str(DEV.relative_to(REPO)),
        output_dir=str(OUT.relative_to(REPO)),
        eval_every=STEPS // EVALUATIONS,
        eval_steps=grid,
        backbone="pplx",
        lora=dataclasses.replace(config.lora, target_modules=ATTENTION_TARGETS),
        kl_to_base=0.0,
        kl_exclude_template_prefixes=(),
        distill_weight=0.0,
    )
    train_items = [mix_row_to_item(row) for row in rows]
    del rows
    dev_rows = read_jsonl(DEV)
    cg_rows = read_jsonl(CG)
    dev_items = [mix_row_to_item(row) for row in dev_rows]
    cg_items = [mix_row_to_item(row) for row in cg_rows]
    mimic_items = [mix_row_to_item(row) for row in dev_rows if str(row["template_id"]).startswith(MIMIC_PREFIX)]
    records = replay_records(train_items, config, STEPS)
    replay = {"block": "final_mix", "backbone": "pplx", "steps": STEPS, "composition_sha256": composition_id,
              "train_items": len(train_items), "sizes": [r[0] for r in records],
              "real_tokens": [r[1] for r in records], "padded_tokens": [r[2] for r in records],
              "realised_items": sum(r[0] for r in records), "written_utc": utcnow(), "git_commit": git_commit(REPO)}
    write_json(OUT / "replays" / "final_mix_replay.json", replay)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    resume_from = None
    if args.resume:
        resume_from = latest_full_checkpoint(OUT / "full")
        if resume_from is None:
            raise SystemExit("Y8: --resume, but no complete full checkpoint exists under outputs/osler_v4/Y8/full")
        step = int(resume_from.name.split("_")[1])
        dropped = (truncate_log_after(OUT / "logs" / "train.jsonl", step)
                   + truncate_log_after(OUT / "logs" / "partitions.jsonl", step))
        with (OUT / "logs" / "resumes.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"resumed_from": str(resume_from.relative_to(REPO)), "step": step,
                                 "dropped_records": dropped, "utc": utcnow()}) + "\n")
    write_json(OUT / "provenance.json", {
        "built_at_utc": started, "git_commit": git_commit(REPO), "command": " ".join(sys.argv),
        "config_file": str(RECIPE.relative_to(REPO)), "composition_sha256": composition_id,
        "train_items": len(train_items), "dev_file": str(DEV.relative_to(REPO)), "dev_items": len(dev_items),
        "cg_items": len(cg_items), "mimic_partition_items": len(mimic_items), "steps": STEPS,
        "evaluation_steps": list(grid), "kl_to_base": 0.0, "distill_weight": 0.0,
        "base_model": "pplx-decider-flash-v1.2 (D39)", "recipe": "A0 (the chosen Stage A recipe)",
    })

    def partitions(step: int, trainer: Trainer) -> dict[str, Any]:
        record = {"step": step, "cg_v2": numbers(trainer.score(cg_items).metrics()),
                  "mimic_partition": numbers(trainer.score(mimic_items).metrics())}
        with (OUT / "logs" / "partitions.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
        return record

    summary = run_arm(config, train_items=train_items, dev_items=dev_items, out_dir=OUT, steps=STEPS,
                      eval_subset_size=len(dev_items), partition_hook=partitions,
                      full_checkpoint_dir=OUT / "full", full_checkpoint_every=CHECKPOINT_EVERY, resume_from=resume_from,
                      full_checkpoint_keep_last=CHECKPOINTS_KEPT)
    config_scoring = dataclasses.replace(config, kl_to_base=0.0, distill_weight=0.0)
    model = load_any(OUT / "checkpoints" / "best", device=config_scoring.device, dtype=config_scoring.dtype)
    scored = Trainer(model, config_scoring).score(cg_items)
    write_jsonl(OUT / "cg_v2_predictions.jsonl", dev_prediction_rows(scored))
    # the complete record of the run: after a resume, the summary holds only the evaluations run since the resume
    evaluations = [{"step": r["step"], **numbers(r["metrics"])}
                   for r in read_jsonl(OUT / "logs" / "train.jsonl") if r.get("event") == "eval"]
    partitions_record = read_jsonl(OUT / "logs" / "partitions.jsonl")
    eval_steps = [e["step"] for e in evaluations]
    validation = {
        "steps_taken": summary["steps"], "steps_as_planned": summary["steps"] == STEPS,
        "stopped_early": summary["stopped_early"], "evaluation_steps": eval_steps,
        "evaluation_steps_as_planned": tuple(eval_steps) == grid,
        "logged_steps": _module("v8_block_run", "scripts/osler_v1/v8_block_run.py").validate_logged_steps(
            OUT / "logs" / "train.jsonl", replay, STEPS),
    }
    write_json(OUT / "final_result.json", {
        "steps": STEPS, "selected": summary["selected"], "best_step": summary["best_step"],
        "evaluations": evaluations, "partitions": partitions_record,
        "cg_v2_at_selected": numbers(scored.metrics()), "cg_v2_items": len(cg_items),
        "full_dev_metrics": summary["full_dev_metrics"], "temperature_fits": summary["temperature_fits"],
        "validation": validation, "realised_items": replay["realised_items"], "composition_sha256": composition_id,
        "started_utc": started, "finished_utc": utcnow(), "git_commit": git_commit(REPO),
    })
    print(f"Y8: steps {summary['steps']}, selected {summary['selected']}, validation "
          f"{validation['logged_steps']['mismatches']} log mismatches, "
          f"{'evaluations on the planned steps' if validation['evaluation_steps_as_planned'] else 'EVALUATION STEPS OFF PLAN'}",
          flush=True)


if __name__ == "__main__":
    main()
