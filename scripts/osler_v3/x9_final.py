"""osler_v3 X9 (ADVISORY X9): the final Osler-4B on the last kept Stage B block. One GPU job.

- The mix is the last kept block (outputs/osler_v3/X8/last_kept.txt): B0 is the chosen Stage A run's mix (osler_v2's C1 for A0-A2,
  the A3 block for A3), B1 or B3 when kept. The recipe is the chosen Stage A arm's (its KL setting, D41; attention-only LoRA, D33;
  the D40 speed recipe; base pplx-decider-flash-v1.2, D39).
- One pass capped at 250,000 realised items. The step count is the smallest multiple of eight whose batches hold at least that many
  realised items (the trainer's own batcher with the pplx encoder, written first as a replay file, as osler_v2's W7).
- Eight evaluations on the whole fast dev v3. The selected checkpoint is the trainer's (best pooled template macro, ties by Brier);
  the temperatures are fitted on dev and saved with that checkpoint (osler_v0 deviation 59).
- The clinical-general (CG) partition is scored by the same model at every evaluation (labelled: outside the selection and the
  verdict; ADVISORY X9 and section 7.1). It is scored inside the trainer's hook, before the trainer's selection, so it cannot move it.

Writes outputs/osler_v3/X9/: replay_250k.json, provenance.json, arm_result.json, cg_trajectory.json, dev_predictions.jsonl,
cg_predictions.jsonl (raw; gitignored), logs/, checkpoints/best.

Usage:
    source scripts/pod_env.sh && uv run --frozen python scripts/osler_v3/x9_final.py --replay-only
    setsid nohup uv run --frozen python scripts/osler_v3/x9_final.py > outputs/osler_v3/X9/logs/final.log 2>&1 < /dev/null &
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch

from meddecide.model.pplx import ATTENTION_TARGETS, PplxDecisionModel, load_any
from meddecide.train.blocks import steps_for_items
from meddecide.train.config import load_config
from meddecide.train.data import iter_batches, shuffled_order
from meddecide.train.mix_items import mix_row_to_item
from meddecide.train.osler_arm import dev_prediction_rows
from meddecide.train.osler_v3 import DOMAIN_PREFIXES
from meddecide.train.trainer import Trainer
from meddecide.utils.hashing import file_sha256
from meddecide.utils.io import read_json, write_json, write_jsonl
from meddecide.utils.provenance import git_commit, utcnow

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "configs" / "osler_v3" / "block_arm.yaml"
FASTDEV_V3 = REPO / "data" / "bench" / "osler_v3" / "fastdev_v3.jsonl"
CG = REPO / "data" / "bench" / "osler_v3" / "cg_v3.jsonl"
OUT = REPO / "outputs" / "osler_v3" / "X9"
CHOICE = REPO / "outputs" / "osler_v3" / "X7" / "choice.json"
LAST_KEPT = REPO / "outputs" / "osler_v3" / "X8" / "last_kept.txt"
REPLAY = OUT / "replay_250k.json"
EVALUATIONS = 8  # ADVISORY X9
TARGET_ITEMS = 250_000  # ADVISORY X9: one pass capped at 250,000 realised items
MULTIPLE = 8
KL_BY_ARM = {"A0": 0.0, "A1": 0.1, "A2": 0.5, "A3": 0.0}  # the Stage A settings (x7_stage_a.sh)


def _module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def mix_of_last_kept() -> tuple[str, Path, float]:
    """The last kept block's name, its training file and the KL setting of the chosen Stage A arm."""
    arm = read_json(CHOICE)["chosen"]
    last = LAST_KEPT.read_text(encoding="utf-8").strip()
    if last == "B0":
        path = REPO / ("data/train/osler_v3/block_A3.jsonl" if arm == "A3" else "data/train/osler_v2/block_C1.jsonl")
    else:
        path = REPO / "data" / "train" / "osler_v3" / f"block_{last}.jsonl"
    return last, path, KL_BY_ARM[arm]


def read_mix(path: Path) -> list[Any]:
    with path.open(encoding="utf-8") as fh:
        return [mix_row_to_item(json.loads(line)) for line in fh if line.strip()]


def replay_to_target(path: Path, config: Any) -> dict[str, Any]:
    """The step count (a multiple of eight) at which the batches reach TARGET_ITEMS realised items, from the trainer's batcher."""
    rp = _module("w6_replay", REPO / "scripts" / "osler_v2" / "w6_replay.py")
    items = rp.read_block(path)
    encoder = rp.encoder_for("pplx", config)
    order = shuffled_order(len(items), seed=config.seed, epoch=0)
    batches = iter_batches(encoder, items, batch_size=config.batch_size, max_batch_tokens=config.max_batch_tokens,
                           max_prompt_tokens=config.max_prompt_tokens, order=order, augment_options=config.shuffle_options,
                           epoch=0, seed=config.seed)
    records: list[tuple[int, int, int]] = []

    def stream():
        for _, batch in batches:
            record = rp.batch_record(batch)
            records.append(record)
            yield record[0]

    steps = steps_for_items(stream(), TARGET_ITEMS, multiple=MULTIPLE)
    batches.close()
    if len(records) != steps:
        raise RuntimeError(f"replayed {len(records)} batches for {steps} steps")
    sizes = [r[0] for r in records]
    replay = {"train_file": str(path.relative_to(REPO)), "train_file_sha256": file_sha256(path), "train_items": len(items),
              "backbone": "pplx", "steps": steps, "target_items": TARGET_ITEMS, "realised_items": sum(sizes),
              "sizes": sizes, "real_tokens": [r[1] for r in records], "padded_tokens": [r[2] for r in records],
              "items_per_step_counts": {str(k): v for k, v in sorted(Counter(sizes).items())},
              "written_utc": utcnow(), "git_commit": git_commit(REPO)}
    write_json(REPLAY, replay)
    return replay


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--replay-only", action="store_true", help="write the replay file and stop (CPU)")
    args = ap.parse_args()
    last, train_path, kl = mix_of_last_kept()
    base = load_config(CONFIG)
    if REPLAY.exists() and read_json(REPLAY)["train_file_sha256"] == file_sha256(train_path):
        replay = read_json(REPLAY)
    else:
        replay = replay_to_target(train_path, base)
    print(f"final mix {last} ({train_path.name}): steps {replay['steps']}, realised items {replay['realised_items']}, kl {kl}",
          flush=True)
    if args.replay_only:
        return 0
    if (OUT / "arm_result.json").exists():
        raise SystemExit("the final run already has an arm_result.json; it is not run twice")
    started = time.time()
    steps = int(replay["steps"])
    if steps % EVALUATIONS:
        raise SystemExit("the replay does not give eight evaluations on whole steps")
    config = dataclasses.replace(
        base, train_path=str(train_path.relative_to(REPO)), dev_path=str(FASTDEV_V3.relative_to(REPO)),
        output_dir=str(OUT.relative_to(REPO)), eval_every=steps // EVALUATIONS, backbone="pplx",
        lora=dataclasses.replace(base.lora, target_modules=ATTENTION_TARGETS), kl_to_base=kl,
        kl_exclude_template_prefixes=DOMAIN_PREFIXES if kl > 0 else ())
    out_dir = OUT
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)
    train_items = read_mix(train_path)
    dev_items = read_mix(FASTDEV_V3)
    cg_items = read_mix(CG)
    write_json(out_dir / "provenance.json", {
        "built_at_utc": utcnow(), "git_commit": git_commit(REPO), "command": " ".join(sys.argv), "mix": last,
        "train_file": str(train_path.relative_to(REPO)), "train_sha256": replay["train_file_sha256"],
        "dev_file": str(FASTDEV_V3.relative_to(REPO)), "dev_sha256": file_sha256(FASTDEV_V3), "dev_items": len(dev_items),
        "cg_file": str(CG.relative_to(REPO)), "cg_sha256": file_sha256(CG), "cg_items": len(cg_items), "steps": steps,
        "evaluations": EVALUATIONS, "target_realised_items": TARGET_ITEMS, "realised_items": replay["realised_items"],
        "kl_to_base": kl, "kl_exclude_template_prefixes": list(config.kl_exclude_template_prefixes),
        "lora_target_modules": list(config.lora.target_modules), "base_model": "pplx-decider-flash-v1.2 (D39)",
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none", "torch": torch.__version__,
    })
    torch.manual_seed(config.seed)
    model = PplxDecisionModel(lora=config.lora, dtype=config.dtype, device=config.device,
                              max_prompt_tokens=config.max_prompt_tokens)
    if config.gradient_checkpointing:
        model.enable_gradient_checkpointing()
    trainer = Trainer(model, config, log_path=out_dir / "logs" / "train.jsonl")
    dev_trajectory: list[dict[str, Any]] = []
    cg_trajectory: list[dict[str, Any]] = []

    def observe(step: int, metrics: dict[str, Any]) -> None:
        dev_trajectory.append({"step": step, "macro_accuracy": float(metrics["macro_accuracy"]),
                               "template_macro_accuracy": float(metrics["template_macro_accuracy"]),
                               "accuracy": float(metrics["accuracy"]), "brier": float(metrics["brier"])})
        cg = trainer.score(cg_items, batch_size=config.eval_batch_size or config.batch_size).metrics()
        cg_trajectory.append({"step": step, "cg_accuracy": float(cg["accuracy"]), "cg_brier": float(cg["brier"]),
                              "cg_items": len(cg_items), "labelled": "CG partition: outside selection and verdict"})
        write_json(out_dir / "cg_trajectory.json", cg_trajectory)

    trainer.eval_hook = observe
    best_dir = out_dir / "checkpoints" / "best"
    result = trainer.train(train_items, steps=steps, eval_items=dev_items, checkpoint_dir=best_dir, schedule_steps=steps)
    selected = "final model (no checkpoint was saved)"
    final_model = model
    if result.best_step is not None and best_dir.exists():
        final_model = load_any(best_dir, device=config.device, dtype=config.dtype)
        selected = f"best checkpoint at step {result.best_step}"
    calibration: dict[str, Any] = {}
    if config.fit_temperature:
        fits = Trainer(final_model, config).fit_calibration(dev_items)
        calibration = {k: v.to_dict() if hasattr(v, "to_dict") else str(v) for k, v in fits.items()}
        if final_model is not model:
            final_model.save(best_dir)
    scored_dev = Trainer(final_model, config).score(dev_items)
    full_dev = scored_dev.metrics()
    full_dev["per_qtype"] = scored_dev.per_qtype_metrics()
    write_jsonl(out_dir / "dev_predictions.jsonl", dev_prediction_rows(scored_dev))
    scored_cg = Trainer(final_model, config).score(cg_items)
    write_jsonl(out_dir / "cg_predictions.jsonl", dev_prediction_rows(scored_cg))
    expected = [steps * q // EVALUATIONS for q in range(1, EVALUATIONS + 1)]
    eval_steps = [e["step"] for e in dev_trajectory]
    logged = _module("v8_block_run", REPO / "scripts" / "osler_v1" / "v8_block_run.py").validate_logged_steps(
        out_dir / "logs" / "train.jsonl", replay, steps)
    summary = {
        "mix": last, "kl_to_base": kl, "steps": steps, "evaluations": EVALUATIONS, "realised_items": replay["realised_items"],
        "selected": selected, "best_step": result.best_step, "stopped_early": result.stopped_early,
        "dev_evaluations": dev_trajectory, "cg_trajectory": cg_trajectory, "full_dev_metrics": full_dev,
        "cg_metrics_of_selected": {k: v for k, v in scored_cg.metrics().items() if isinstance(v, (int, float))},
        "temperature_fits_keys": sorted(calibration), "validation": {
            "steps_taken": result.steps, "steps_as_planned": result.steps == steps, "evaluation_steps": eval_steps,
            "evaluation_steps_as_planned": eval_steps == expected, "logged_steps": logged},
        "train_wall_clock_s": round(result.wall_clock_s, 1), "total_wall_clock_s": round(time.time() - started, 1),
    }
    write_json(out_dir / "arm_result.json", summary)
    print(f"X9: steps {result.steps}, selected {selected}, evaluations at {eval_steps}, "
          f"logged mismatches {logged.get('mismatches')}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
