#!/usr/bin/env python
"""V8 arm C: Unsloth's Clef-style joint decision head on the V4 mix (run in envs/unsloth; one GPU job).

The training items are the V4 train rows converted by ``src/meddecide/train/unsloth_rows.py`` and built
by Unsloth's ``build_dataset`` at ``max_seq_length`` 8,192 (the ADVISORY value). Unsloth's documented
defaults are used except where noted in the ADVISORY or in the audit: LoRA r=16 and alpha 16, learning
rate 2e-4 with a head learning rate of 1e-4, cosine schedule, dropout 0. Warmup is 3 % of the step budget,
the same fraction as arms A and B. The step budget is 15,000 steps of batch 8 (120,000 items, about a
quarter of the mix); the run saves an adapter and head at every 500 steps. Selection and calibration are
separate scripts (``v8_unsloth_predict.py`` and ``v8_select.py``), so training never switches the model
between training and inference modes.

Run: ``UNSLOTH_COMPILE_LOCATION=/workspace/tmp/unsloth_compiled_cache \
      envs/unsloth/.venv/bin/python -I scripts/student/v8_unsloth_train.py``
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = "Qwen/Qwen3.5-0.8B"
TRAIN = ROOT / "data" / "train" / "student_v1" / "train.jsonl"
OUT = ROOT / "outputs" / "student_v1" / "V8" / "arm_c"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-steps", type=int, default=15000)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--max-seq-length", type=int, default=8192)
    parser.add_argument("--limit-rows", type=int, default=None, help="smoke test: first N training rows")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    # Unsloth must be imported before transformers, peft and torch (its import warning says so)
    from unsloth import FastDecisionModel  # noqa: I001
    from unsloth.models.decision import DecisionTrainer

    import torch
    from transformers import TrainerCallback, TrainingArguments

    converter = load_module("unsloth_rows", ROOT / "src" / "meddecide" / "train" / "unsloth_rows.py")
    report: dict = {"task": "V8", "kind": "arm_c_training", "base_model": BASE,
                    "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "max_seq_length": args.max_seq_length, "max_steps": args.max_steps}

    rows, refused = [], {}
    with TRAIN.open(encoding="utf-8") as handle:
        sample = []
        for index, line in enumerate(handle):
            if args.limit_rows is not None and index >= args.limit_rows:
                break
            sample.append(json.loads(line))
    converted, refused = converter.to_unsloth_rows(sample)
    rows = converted
    report["train_rows_read"] = len(sample)
    report["refused_by_conversion"] = refused

    model, tokenizer = FastDecisionModel.from_pretrained(
        BASE, max_seq_length=args.max_seq_length, dtype=torch.bfloat16, load_in_4bit=False
    )
    t0 = time.time()
    items, build_report = FastDecisionModel.build_dataset(rows, tokenizer, model)
    report["build_dataset"] = {"items": len(items), "report": {k: (v if not isinstance(v, dict) else dict(v))
                                                               for k, v in build_report.items()},
                               "seconds": round(time.time() - t0, 1)}
    print("built:", report["build_dataset"], flush=True)

    model = FastDecisionModel.get_peft_model(
        model, r=16, lora_alpha=16, lora_dropout=0,
        use_gradient_checkpointing="unsloth", random_state=3407,
    )

    class SaveEvery(TrainerCallback):
        def on_step_end(self, arguments, state, control, **kwargs):
            if state.global_step > 0 and state.global_step % args.save_every == 0:
                target = args.out / "checkpoints" / f"step_{state.global_step}"
                model.save_pretrained(target)
                print(f"saved {target.name}", flush=True)
            return control

    training_args = TrainingArguments(
        per_device_train_batch_size=8,
        gradient_accumulation_steps=1,
        max_steps=args.max_steps,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_steps=round(0.03 * args.max_steps),
        weight_decay=0.0,
        bf16=True,
        logging_steps=1,  # every step: the grad-norm tripwire (ADVISORY V6-V8) needs each pre-clip norm
        output_dir=str(args.out / "trainer"),
        report_to="none",
        seed=3407,
        save_strategy="no",
        # random batches padded to their longest item ran at about 2 s per step (1.5 h for 2,700 steps);
        # grouping items of similar length cuts the padding (documented deviation, recorded in the audit)
        group_by_length=True,
    )
    trainer = DecisionTrainer(
        model=model,
        args=training_args,
        processing_class=tokenizer,
        head_learning_rate=1e-4,
        train_dataset=items,
        callbacks=[SaveEvery()],
    )
    report["head_learning_rate"] = 1e-4
    report["warmup_steps"] = training_args.warmup_steps
    t_train = time.time()
    result = trainer.train()
    report["train_seconds"] = round(time.time() - t_train, 1)
    report["train_loss"] = float(result.training_loss)
    report["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    logs_dir = args.out / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    with (logs_dir / "train_steps.jsonl").open("w", encoding="utf-8") as steps_file:
        for record in trainer.state.log_history:
            steps_file.write(json.dumps(record, default=str) + "\n")
    (args.out / "training.json").write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    print("done:", {k: report[k] for k in ("train_seconds", "train_loss")}, flush=True)


if __name__ == "__main__":
    main()
