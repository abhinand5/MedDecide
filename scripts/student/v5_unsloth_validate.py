#!/usr/bin/env python
"""V5: validate Unsloth's decision pipeline (run in envs/unsloth; one GPU job).

Three parts, each written to outputs/student_v1/V5/:

A. Round trip of MedDecide rows: ``to_unsloth_row`` (src/meddecide/train/unsloth_rows.py) then
   ``FastDecisionModel.build_dataset``. Expected skipped count 0; any skip is counted with its reason.
B. Chance check: the untrained head's answers on 200 seen-template v0.2 dev items. The metrics are
   computed in the main environment (``scripts/student/v5_chance_check.py``) with the repository's code.
C. Unsloth's documented recipe on LocalLLaMA/typed-decisions (all subset, Qwen3.5-0.8B, bf16, 300 steps):
   held-out accuracy before and after on its test split. This validates the integration only. The model
   is discarded; it is not a MedDecide result.

Run: ``envs/unsloth/.venv/bin/python -I scripts/student/v5_unsloth_validate.py``
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "student_v1" / "V5"
BASE = "Qwen/Qwen3.5-0.8B"
os.environ.setdefault("UNSLOTH_COMPILE_LOCATION", "/workspace/tmp/unsloth_compiled_cache")


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    return str(value)


def seen_dev_sample(n: int) -> list[dict]:
    held = {"ct_phase_choice_v1", "fda_boxed_warning_noul_v1", "pubmed_humans_noul_v1", "ct_arm_role_noul_v1"}
    rows = []
    for folder in ("tier1", "fresh"):
        for path in sorted((ROOT / "data" / "bench" / "v0.2" / folder).glob("*.jsonl")):
            for line in path.open(encoding="utf-8"):
                row = json.loads(line)
                if row["split"] == "dev" and row["template_id"] not in held:
                    rows.append(row)
    rows.sort(key=lambda r: r["item_id"])
    step = max(1, len(rows) // n)
    return rows[::step][:n]


def train_rows_sample(n: int) -> list[dict]:
    path = ROOT / "data" / "train" / "student_v1" / "train.jsonl"
    every = 240  # about 2,000 of 482,889 rows, spread over the file
    rows = []
    with path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if index % every == 0:
                rows.append(json.loads(line))
                if len(rows) >= n:
                    break
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--chance-items", type=int, default=200)
    parser.add_argument("--round-trip-items", type=int, default=2000)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    # Unsloth must be imported before transformers, peft and torch (its import warning says so);
    # the isort rule would reorder these, so it is suppressed for this block only
    from unsloth import FastDecisionModel  # noqa: I001
    from unsloth.models.decision import DecisionTrainer

    import torch
    from datasets import load_dataset
    from transformers import TrainingArguments

    converter = load_module("unsloth_rows", ROOT / "src" / "meddecide" / "train" / "unsloth_rows.py")
    report: dict = {"task": "V5", "kind": "unsloth_validation", "base_model": BASE,
                    "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    t0 = time.time()
    model, tokenizer = FastDecisionModel.from_pretrained(
        BASE, max_seq_length=2048, dtype=torch.bfloat16, load_in_4bit=False
    )
    report["load_seconds"] = round(time.time() - t0, 1)
    report["dtype"] = str(next(model.parameters()).dtype)

    # ---- A. round trip of MedDecide rows ------------------------------------
    sample = train_rows_sample(args.round_trip_items)
    rows, refused = converter.to_unsloth_rows(sample)
    items, build_report = FastDecisionModel.build_dataset(rows, tokenizer, model)
    report["round_trip"] = {
        "sampled_items": len(sample),
        "converted_rows": len(rows),
        "refused_by_conversion": refused,
        "build_dataset_report": jsonable(build_report),
        "items_built": len(items),
    }
    print("round trip:", report["round_trip"]["build_dataset_report"], flush=True)

    # ---- B. chance check: untrained head on seen-template v0.2 dev items -----
    dev = seen_dev_sample(args.chance_items)
    predictions = []
    with torch.no_grad():
        for row in dev:
            rows1, _ = converter.to_unsloth_rows([row])
            if not rows1:
                predictions.append({"item_id": row["item_id"], "refused": True})
                continue
            unsloth_row = rows1[0]
            answers = FastDecisionModel.predict(model, tokenizer, unsloth_row["state"],
                                                unsloth_row["questions"])
            answer = answers["q"]["answer"]
            predictions.append({
                "item_id": row["item_id"],
                "template_id": row["template_id"],
                "qtype": row["qtype"],
                "option_keys": [o["key"] for o in row["options"]],
                "gold": row["gold"],
                "answer_raw": jsonable(answer),
                "probabilities": jsonable(answers["q"].get("probabilities")),
            })
    (OUT / "chance_predictions.json").write_text(json.dumps(predictions, indent=1) + "\n", encoding="utf-8")
    report["chance_check"] = {"items": len(dev), "predictions_file": "outputs/student_v1/V5/chance_predictions.json"}
    print("chance predictions written:", len(predictions), flush=True)

    # ---- C. typed-decisions validation ----------------------------------------
    ds_all = load_dataset("LocalLLaMA/typed-decisions", "all", split="train")
    ds_test = load_dataset("LocalLLaMA/typed-decisions", "all", split="test")
    train_items, rep_train = FastDecisionModel.build_dataset(ds_all, tokenizer, model)
    test_items, rep_test = FastDecisionModel.build_dataset(ds_test, tokenizer, model)
    report["typed_decisions"] = {"train_rows": len(ds_all), "test_rows": len(ds_test),
                                 "build_report_train": jsonable(rep_train),
                                 "build_report_test": jsonable(rep_test)}
    fit_items, eval_items = FastDecisionModel.split_holdout(train_items, seed=3407)
    before = FastDecisionModel.evaluate(model, tokenizer, test_items)
    report["typed_decisions"]["before"] = jsonable(before)
    print("before:", {k: v for k, v in jsonable(before).items() if not isinstance(v, (dict, list))}, flush=True)

    model = FastDecisionModel.get_peft_model(
        model, r=16, lora_alpha=16, lora_dropout=0,
        use_gradient_checkpointing="unsloth", random_state=3407,
    )
    training_args = TrainingArguments(
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        max_steps=args.max_steps,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_steps=10,
        weight_decay=0.01,
        bf16=True,
        logging_steps=10,
        output_dir=str(ROOT / "outputs" / "student_v1" / "V5" / "trainer"),
        report_to="none",
        seed=3407,
    )
    trainer = DecisionTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=fit_items,
        eval_dataset=eval_items,
        args=training_args,
    )
    t_train = time.time()
    train_result = trainer.train()
    report["typed_decisions"]["train_seconds"] = round(time.time() - t_train, 1)
    report["typed_decisions"]["train_loss"] = float(train_result.training_loss)
    report["typed_decisions"]["head_learning_rate"] = "Unsloth default (1e-4)"
    after = FastDecisionModel.evaluate(model, tokenizer, test_items)
    report["typed_decisions"]["after"] = jsonable(after)
    print("after:", {k: v for k, v in jsonable(after).items() if not isinstance(v, (dict, list))}, flush=True)
    try:
        cal = FastDecisionModel.calibrate(model, tokenizer, eval_items)
        report["typed_decisions"]["calibrate_on_holdout"] = jsonable(cal)
    except Exception as exc:  # recorded, not hidden
        report["typed_decisions"]["calibrate_on_holdout"] = f"NOT MEASURED — {type(exc).__name__}: {exc}"

    report["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
    report["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    (OUT / "validation.json").write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    print("wrote", (OUT / "validation.json").relative_to(ROOT), flush=True)


if __name__ == "__main__":
    main()
