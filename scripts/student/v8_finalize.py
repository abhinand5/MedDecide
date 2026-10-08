#!/usr/bin/env python
"""V8 finalize (arm C): calibrate the selected checkpoint on seen dev, then predict the v0.2 test items.

Both steps run in one process, so the calibration is in effect for the test predictions (the calibration is
applied in memory by ``predict``; it is not relied on to persist through ``save_pretrained``). Calibration uses
Unsloth's own ``FastDecisionModel.calibrate`` on the V4 dev set (every template there is a seen template;
no held-out template is in it). Writes the calibration and the raw test answers.

Run: ``envs/unsloth/.venv/bin/python -I scripts/student/v8_finalize.py --checkpoint <dir> --out-dir <dir>``
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dev", type=Path, default=ROOT / "data" / "train" / "student_v1" / "dev.jsonl")
    parser.add_argument("--test-items", type=Path, default=ROOT / "outputs" / "student_v1" / "V9" / "test_items.jsonl")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-seq-length", type=int, default=8192)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Unsloth must be imported before transformers, peft and torch
    from unsloth import FastDecisionModel  # noqa: I001

    import torch

    converter = load_module("unsloth_rows", ROOT / "src" / "meddecide" / "train" / "unsloth_rows.py")
    report: dict = {"task": "V8", "kind": "arm_c_finalize", "checkpoint": str(args.checkpoint),
                    "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    model, tokenizer = FastDecisionModel.from_pretrained(
        str(args.checkpoint), max_seq_length=args.max_seq_length, dtype=torch.bfloat16, load_in_4bit=False
    )

    dev_sample = [json.loads(line) for line in args.dev.open(encoding="utf-8")]
    dev_rows, dev_refused = converter.to_unsloth_rows(dev_sample)
    dev_items, dev_build = FastDecisionModel.build_dataset(dev_rows, tokenizer, model)
    report["calibration_dev"] = {"rows": len(dev_rows), "refused": dev_refused,
                                 "build_report": jsonable(dev_build)}
    calibration = FastDecisionModel.calibrate(model, tokenizer, dev_items)
    report["calibration"] = jsonable(calibration)
    print("calibrated:", {k: v for k, v in jsonable(calibration).items() if not isinstance(v, (dict, list))},
          flush=True)

    FastDecisionModel.for_inference(model)
    raw_path = args.out_dir / "test_raw.jsonl"
    count, refused = 0, 0
    started = time.time()
    with args.test_items.open(encoding="utf-8") as src, raw_path.open("w", encoding="utf-8") as dst:
        for line in src:
            item = json.loads(line)
            rows, _ = converter.to_unsloth_rows([item])
            if not rows:
                refused += 1
                continue
            t0 = time.time()
            answers = FastDecisionModel.predict(model, tokenizer, rows[0]["state"], rows[0]["questions"])
            answer = answers["q"]
            dst.write(json.dumps({
                "item_id": item["item_id"],
                "qtype": item["qtype"],
                "answer": jsonable(answer.get("answer")),
                "probabilities": {str(k): float(v) for k, v in (answer.get("probabilities") or {}).items()},
                "latency_s": time.time() - t0,
            }) + "\n")
            count += 1
    report.update({"test_items_predicted": count, "test_items_refused": refused,
                   "predict_seconds": round(time.time() - started, 1),
                   "raw_test": str(raw_path.relative_to(ROOT)),
                   "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    (args.out_dir / "finalize.json").write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"test_items_predicted": count, "refused": refused}), flush=True)


if __name__ == "__main__":
    main()
