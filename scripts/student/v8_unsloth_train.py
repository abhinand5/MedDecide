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

Batch order (D15): arms A and B bucket by length inside seeded chunks (``src/meddecide/train/data.py``).
Arm C uses the same chunking and seeded streams (``src/meddecide/train/batch_order.py``) with fixed batches
of 8 items, in place of Unsloth's random sampler: the random sampler padded every batch to its longest
item and ran at about 2 s per step. The order is passed to the trainer as a fixed sampler, and its hash
over the used indices is recorded in ``training.json``.

Option order (D16): the option-order augmentation that arms A and B apply per epoch is applied to the
choice and score items before conversion (``batch_order.permute_options``, the same permutation
function). Noul items keep their order: the Unsloth noul question has no option list to reorder.

Run: ``UNSLOTH_COMPILE_LOCATION=/workspace/tmp/unsloth_compiled_cache \
      envs/unsloth/.venv/bin/python -I scripts/student/v8_unsloth_train.py``
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import pickle
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = "Qwen/Qwen3.5-0.8B"
TRAIN = ROOT / "data" / "train" / "student_v1" / "train.jsonl"
OUT = ROOT / "outputs" / "student_v1" / "V8" / "arm_c"
# the arms A/B batch order: batch size 8, chunk factor 100 (data.DEFAULT_BATCH_CHUNK_FACTOR), seed 0
BATCH_SIZE = 8
CHUNK_FACTOR = 100
ORDER_SEED = 0


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
    batch_order = load_module("batch_order", ROOT / "src" / "meddecide" / "train" / "batch_order.py")
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
    # option-order augmentation as arms A and B apply it (D16): choice and score items, one permutation per
    # (seed, epoch, file index); noul is not permuted, because the Unsloth noul question has no option list
    sample, reordered = batch_order.permute_options(sample, seed=ORDER_SEED, epoch=0)
    converted, refused = converter.to_unsloth_rows(sample)
    rows = converted
    report["train_rows_read"] = len(sample)
    report["refused_by_conversion"] = refused
    report["option_order"] = {"rule": "arms A/B option-order permutation on choice and score items (D16)",
                              "seed": ORDER_SEED, "epoch": 0, "items_changed": reordered}

    model, tokenizer = FastDecisionModel.from_pretrained(
        BASE, max_seq_length=args.max_seq_length, dtype=torch.bfloat16, load_in_4bit=False
    )

    # the built items are cached (the build takes about 17 min); a cache of another row set is rebuilt
    cache = args.out / "cache" / f"built_items_{args.max_seq_length}.pkl"
    t0 = time.time()
    loaded = None
    if cache.exists():
        with cache.open("rb") as handle:
            loaded = pickle.load(handle)
    if loaded is not None and loaded[0] == len(rows):
        _, items, build_report = loaded
        report["build_dataset_cache"] = str(cache)
    else:
        items, build_report = FastDecisionModel.build_dataset(rows, tokenizer, model)
        try:  # best effort: a cache that cannot be written must not cost the build
            cache.parent.mkdir(parents=True, exist_ok=True)
            with cache.open("wb") as handle:
                pickle.dump((len(rows), items, build_report), handle, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as exc:
            report["build_dataset_cache_error"] = f"{type(exc).__name__}: {exc}"
            cache.unlink(missing_ok=True)
    report["build_dataset"] = {"items": len(items), "report": {k: (v if not isinstance(v, dict) else dict(v))
                                                               for k, v in build_report.items()},
                               "seconds": round(time.time() - t0, 1)}
    print("built:", report["build_dataset"], flush=True)

    # the arms A/B batch order over the built items (built items keep the row order: skipped is 0)
    lengths = [len(item["input_ids"]) for item in items]
    order = batch_order.length_bucketed_order(
        lengths, batch_size=BATCH_SIZE, chunk_factor=CHUNK_FACTOR, seed=ORDER_SEED, epoch=0
    )
    used = order[: args.max_steps * BATCH_SIZE]
    report["batch_order"] = {
        "rule": "arms A/B length bucketing (meddecide.train.batch_order); fixed batches of 8 (D15)",
        "batch_size": BATCH_SIZE, "chunk_factor": CHUNK_FACTOR, "seed": ORDER_SEED, "epoch": 0,
        "items_used": len(used), "distinct_items_used": len(set(used)),
        "sha256_used_indices": hashlib.sha256(json.dumps(used).encode("utf-8")).hexdigest(),
    }
    print("batch order:", {k: v for k, v in report["batch_order"].items() if k != "rule"}, flush=True)

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

    class FixedOrder(torch.utils.data.Sampler):
        """A fixed sequence of item indices: each run of ``BATCH_SIZE`` consecutive indices is one batch."""

        def __init__(self, indices):
            self.indices = indices

        def __iter__(self):
            return iter(self.indices)

        def __len__(self):
            return len(self.indices)

    class BucketedDecisionTrainer(DecisionTrainer):
        """DecisionTrainer with the arms A/B batch order in place of Unsloth's sampler (D15)."""

        def _get_train_sampler(self, train_dataset=None):
            return FixedOrder(order)

    training_args = TrainingArguments(
        per_device_train_batch_size=BATCH_SIZE,
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
    )
    trainer = BucketedDecisionTrainer(
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
