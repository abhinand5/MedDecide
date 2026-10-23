"""osler_v4 Y8 (ADVISORY Y8; operator ruling 2026-10-23, items 2 and 3): the final Osler-4B on the final mix (GPU; one job).

The recipe is A0's (configs/osler_v3/block_arm.yaml with KL to base 0 and no distillation). The mix is data/train/osler_v4/final_mix.jsonl
(scripts/osler_v4/y8_final_mix.py) and its replay (scripts/osler_v4/y8_replay.py). The step budget is 63,304 with ten evaluations on fast
dev v4 at evaluation_grid(63304, 10), the last at the budget. The selected checkpoint is the trainer's (the pooled template macro accuracy,
Brier tie-break). At every evaluation the CG v2 set and the MIMIC partition (the fast dev v4 items whose template id starts with mimic_)
are scored and recorded; they are not in the selection. After training, the selected checkpoint is scored on CG v2 and the predictions
are written. The temperatures are fitted on fast dev v4 and saved with the checkpoint (run_arm). The run refuses to start when
outputs/osler_v4/Y8/arm_result.json exists (the osler_v3 X9 guard), and when the replay was made from another file.

Usage (GPU; one job; detached; nothing else on the GPU):
    source scripts/pod_env.sh && setsid nohup uv run --frozen python scripts/osler_v4/y8_train.py \
        > outputs/osler_v4/Y8/logs/final_train.log 2>&1 < /dev/null &
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

from meddecide.model.pplx import ATTENTION_TARGETS, load_any
from meddecide.train.arm_run import run_arm
from meddecide.train.config import load_config
from meddecide.train.mix_items import mix_row_to_item
from meddecide.train.osler_arm import dev_prediction_rows
from meddecide.train.trainer import Trainer, evaluation_grid
from meddecide.utils.hashing import file_sha256
from meddecide.utils.io import read_json, write_json, write_jsonl
from meddecide.utils.provenance import git_commit, utcnow

REPO = Path(__file__).resolve().parents[2]
RECIPE = REPO / "configs" / "osler_v3" / "block_arm.yaml"
MIX = REPO / "data" / "train" / "osler_v4" / "final_mix.jsonl"
REPLAY = REPO / "outputs" / "osler_v4" / "Y8" / "replays" / "final_mix_pplx.json"
DEV = REPO / "data" / "bench" / "osler_v4" / "fastdev_v4.jsonl"
CG = REPO / "data" / "bench" / "osler_v4" / "cg_v2.jsonl"
OUT = REPO / "outputs" / "osler_v4" / "Y8"
STEPS = 63_304  # ADVISORY Y8: the smallest multiple of 8 at or above 1.5 x 42,200
EVALUATIONS = 10  # ADVISORY Y8
MIMIC_PREFIX = "mimic_"


def _v8():
    spec = importlib.util.spec_from_file_location("v8_block_run", REPO / "scripts" / "osler_v1" / "v8_block_run.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def numbers(metrics: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in metrics.items() if isinstance(v, (int, float))}


def main() -> None:
    if (OUT / "arm_result.json").exists():
        raise SystemExit("Y8: outputs/osler_v4/Y8/arm_result.json exists; the final run is not repeated (osler_v3 X9 guard)")
    replay = read_json(REPLAY)
    if int(replay["steps"]) != STEPS:
        raise SystemExit(f"Y8: the replay has {replay['steps']} steps, not {STEPS}")
    if replay.get("train_file_sha256") != file_sha256(MIX):
        raise SystemExit("Y8: the replay was made from another file (replay the final mix again)")
    started = utcnow()
    base = load_config(RECIPE)
    grid = evaluation_grid(STEPS, EVALUATIONS)
    config = dataclasses.replace(
        base,
        train_path=str(MIX.relative_to(REPO)),
        dev_path=str(DEV.relative_to(REPO)),
        output_dir=str(OUT.relative_to(REPO)),
        eval_every=STEPS // EVALUATIONS,
        eval_steps=grid,
        backbone="pplx",
        lora=dataclasses.replace(base.lora, target_modules=ATTENTION_TARGETS),
        kl_to_base=0.0,
        kl_exclude_template_prefixes=(),
        distill_weight=0.0,
    )
    train_rows = [json.loads(line) for line in MIX.open(encoding="utf-8") if line.strip()]
    dev_rows = [json.loads(line) for line in DEV.open(encoding="utf-8") if line.strip()]
    cg_rows = [json.loads(line) for line in CG.open(encoding="utf-8") if line.strip()]
    train_items = [mix_row_to_item(row) for row in train_rows]
    dev_items = [mix_row_to_item(row) for row in dev_rows]
    cg_items = [mix_row_to_item(row) for row in cg_rows]
    mimic_items = [mix_row_to_item(row) for row in dev_rows if str(row["template_id"]).startswith(MIMIC_PREFIX)]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    write_json(OUT / "provenance.json", {
        "built_at_utc": started, "git_commit": git_commit(REPO), "command": " ".join(sys.argv),
        "config_file": str(RECIPE.relative_to(REPO)), "train_file": str(MIX.relative_to(REPO)), "train_sha256": file_sha256(MIX),
        "train_items": len(train_items), "replay_file": str(REPLAY.relative_to(REPO)), "dev_file": str(DEV.relative_to(REPO)),
        "dev_items": len(dev_items), "cg_items": len(cg_items), "mimic_partition_items": len(mimic_items), "steps": STEPS,
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
                      eval_subset_size=len(dev_items), partition_hook=partitions)
    config_scoring = dataclasses.replace(config, kl_to_base=0.0, distill_weight=0.0)
    model = load_any(OUT / "checkpoints" / "best", device=config_scoring.device, dtype=config_scoring.dtype)
    scored = Trainer(model, config_scoring).score(cg_items)
    write_jsonl(OUT / "cg_v2_predictions.jsonl", dev_prediction_rows(scored))
    eval_steps = [e["step"] for e in summary["evaluations"]]
    validation = {
        "steps_taken": summary["steps"], "steps_as_planned": summary["steps"] == STEPS,
        "stopped_early": summary["stopped_early"], "evaluation_steps": eval_steps,
        "evaluation_steps_as_planned": tuple(eval_steps) == grid,
        "logged_steps": _v8().validate_logged_steps(OUT / "logs" / "train.jsonl", replay, STEPS),
    }
    write_json(OUT / "final_result.json", {
        "steps": STEPS, "selected": summary["selected"], "best_step": summary["best_step"],
        "evaluations": summary["evaluations"], "partitions": summary["partitions"],
        "cg_v2_at_selected": numbers(scored.metrics()), "cg_v2_items": len(cg_items),
        "full_dev_metrics": summary["full_dev_metrics"], "temperature_fits": summary["temperature_fits"],
        "validation": validation, "realised_items": sum(replay["sizes"][:STEPS]),
        "started_utc": started, "finished_utc": utcnow(), "git_commit": git_commit(REPO),
    })
    print(f"Y8: steps {summary['steps']}, selected {summary['selected']}, validation "
          f"{validation['logged_steps']['mismatches']} log mismatches, "
          f"{'evaluations on the planned steps' if validation['evaluation_steps_as_planned'] else 'EVALUATION STEPS OFF PLAN'}",
          flush=True)


if __name__ == "__main__":
    main()
