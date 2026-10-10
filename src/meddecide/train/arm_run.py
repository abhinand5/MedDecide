"""Run one matched Osler-4B arm (ADVISORY O6-O8): train on mix v2, evaluate on the dev subset at the cadence, log the tier-1
diagnostic, keep the best checkpoint, fit temperatures on dev, and evaluate the selected checkpoint on the full dev split.

Only the mix's dev split is used. The test splits, the external panel and the held-out generators and templates never reach
this function. The tier-1 diagnostic is reported at every evaluation and is never a selection signal (the selection is the
trainer's: dev macro, Brier tie-break).
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from meddecide.bench.schema import Item
from meddecide.model.meddecide_model import MedDecideModel
from meddecide.train.config import StudentConfig
from meddecide.train.osler_arm import (
    DEV_SUBSET_SIZE,
    accuracy_by_source,
    dev_prediction_rows,
    example_budget,
    step_count,
    stratified_subset,
    tier1_items,
)
from meddecide.train.trainer import Trainer
from meddecide.utils.io import write_json, write_jsonl


def run_arm(
    config: StudentConfig,
    *,
    train_items: Sequence[Item],
    dev_items: Sequence[Item],
    out_dir: str | Path,
    steps: int | None = None,
    eval_subset_size: int = DEV_SUBSET_SIZE,
    base_model: str | None = None,
    revision: str | None = None,
    device: str | None = None,
    dtype: str | None = None,
) -> dict[str, Any]:
    """Train one arm and return (and write to ``out_dir/arm_result.json``) its summary."""
    if not train_items:
        raise ValueError("no training items")
    if not dev_items:
        raise ValueError("no dev items")
    out = Path(out_dir)
    (out / "logs").mkdir(parents=True, exist_ok=True)
    started = time.time()
    model = MedDecideModel(
        base_model or config.base_model,
        revision=revision if revision is not None else config.revision,
        lora=config.lora,
        head_settings=config.head,
        dtype=dtype or config.dtype,
        device=device or config.device,
        variant=config.variant,
        readout=config.readout,
        bidirectional_full_attention=config.bidirectional_full_attention,
        max_prompt_tokens=config.max_prompt_tokens,
    )
    if config.gradient_checkpointing:
        model.enable_gradient_checkpointing()
    trainer = Trainer(model, config, log_path=out / "logs" / "train.jsonl")
    tier1 = tier1_items(dev_items)
    tier1_trajectory: list[dict[str, Any]] = []
    eval_trajectory: list[dict[str, Any]] = []

    def observe(step: int, metrics: dict[str, Any]) -> None:
        eval_trajectory.append({"step": step, "macro_accuracy": float(metrics["macro_accuracy"]),
                                "template_macro_accuracy": float(metrics["template_macro_accuracy"]),
                                "accuracy": float(metrics["accuracy"]), "brier": float(metrics["brier"])})
        if tier1:
            decisions = model.predict(tier1, batch_size=config.eval_batch_size or config.batch_size)
            per_source = accuracy_by_source(decisions, tier1)
            tier1_trajectory.append({"step": step, "accuracy_by_source": per_source})
            # written at every eval, so the diagnostic can be watched while the arm runs (ADVISORY §8.3); the diagnostic
            # is still never a selection signal
            write_jsonl(out / "logs" / "tier1.jsonl", tier1_trajectory)

    trainer.eval_hook = observe
    eval_subset = stratified_subset(dev_items, key="template_id", size=min(eval_subset_size, len(dev_items)),
                                    seed=config.seed)
    total_steps = steps if steps is not None else step_count(example_budget(len(train_items)), config.batch_size)
    best_dir = out / "checkpoints" / "best"
    result = trainer.train(train_items, steps=total_steps, eval_items=eval_subset, checkpoint_dir=best_dir,
                           schedule_steps=total_steps)
    calibration: dict[str, Any] = {}
    if config.fit_temperature:
        fits = trainer.fit_calibration(dev_items)
        calibration = {k: v.to_dict() if hasattr(v, "to_dict") else str(v) for k, v in fits.items()}
    selected = "final model (no checkpoint was saved)"
    final_model = model
    if result.best_step is not None and best_dir.exists():
        final_model = MedDecideModel.load(best_dir, device=device or config.device, dtype=dtype or config.dtype)
        selected = f"best checkpoint at step {result.best_step}"
    scored_dev = Trainer(final_model, config).score(dev_items)
    full_dev = scored_dev.metrics()
    full_dev["per_qtype"] = scored_dev.per_qtype_metrics()
    write_jsonl(out / "dev_predictions.jsonl", dev_prediction_rows(scored_dev))
    summary = {
        "readout": config.readout,
        "bidirectional_full_attention": config.bidirectional_full_attention,
        "steps": result.steps,
        "examples": result.steps * config.batch_size,
        "example_budget": example_budget(len(train_items)),
        "stopped_early": result.stopped_early,
        "best_step": result.best_step,
        "selected": selected,
        "schedule": result.schedule,
        "eval_subset_items": len(eval_subset),
        "evaluations": eval_trajectory,
        "tier1_trajectory": tier1_trajectory,
        "full_dev_items": len(dev_items),
        "full_dev_metrics": full_dev,
        "temperature_fits": calibration,
        "train_wall_clock_s": round(result.wall_clock_s, 1),
        "total_wall_clock_s": round(time.time() - started, 1),
        "config": config.to_dict(),
    }
    write_json(out / "arm_result.json", summary)
    return summary
