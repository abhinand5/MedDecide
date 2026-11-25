#!/usr/bin/env python
"""Predict every item of a MedDecide JSONL with one saved Unsloth decision checkpoint (envs/unsloth).

One item per call (no padding, the evaluation path used for every arm). Writes one raw JSON line per item:
the item id, the question type, and Unsloth's answer and probabilities. Converting these answers to
run_student-format rows and scoring them happens in the main environment (``v8_select.py``).

Run: ``envs/unsloth/.venv/bin/python -I scripts/student/v8_unsloth_predict.py --checkpoint <dir> \
      --items <jsonl> --out <raw.jsonl>``
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--items", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-seq-length", type=int, default=8192)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    # Unsloth must be imported before transformers, peft and torch
    from unsloth import FastDecisionModel  # noqa: I001

    import torch

    converter = load_module("unsloth_rows", ROOT / "src" / "meddecide" / "train" / "unsloth_rows.py")
    model, tokenizer = FastDecisionModel.from_pretrained(
        str(args.checkpoint), max_seq_length=args.max_seq_length, dtype=torch.bfloat16, load_in_4bit=False
    )
    FastDecisionModel.for_inference(model)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    count = 0
    refused = 0
    with args.items.open(encoding="utf-8") as src, args.out.open("w", encoding="utf-8") as dst:
        for line in src:
            if args.limit is not None and count >= args.limit:
                break
            item = json.loads(line)
            rows, _ = converter.to_unsloth_rows([item])
            if not rows:
                refused += 1
                continue
            row = rows[0]
            t0 = time.time()
            answers = FastDecisionModel.predict(model, tokenizer, row["state"], row["questions"])
            seconds = time.time() - t0
            answer = answers["q"]
            dst.write(json.dumps({
                "item_id": item["item_id"],
                "qtype": item["qtype"],
                "answer": answer.get("answer"),
                "probabilities": {str(k): float(v) for k, v in (answer.get("probabilities") or {}).items()},
                "latency_s": seconds,
            }) + "\n")
            count += 1
    print(json.dumps({"checkpoint": str(args.checkpoint), "items": count, "refused": refused,
                      "seconds": round(time.time() - started, 1)}), flush=True)


if __name__ == "__main__":
    main()
