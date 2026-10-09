#!/usr/bin/env python
"""V0 smoke: load Qwen3.5-0.8B through Unsloth's FastDecisionModel in bf16 (load_in_4bit=False).

Run with the isolated Unsloth environment:  envs/unsloth/.venv/bin/python -I scripts/student/v1_unsloth_load.py
Writes outputs/student_v1/V0/unsloth_load.json. PASS requires every parameter in bf16 and one
forward pass that returns finite logits of the vocabulary's width.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import torch

BASE = "Qwen/Qwen3.5-0.8B"
OUT = Path("outputs/student_v1/V0/unsloth_load.json")


def main() -> int:
    from unsloth import FastDecisionModel

    t0 = time.time()
    model, tokenizer = FastDecisionModel.from_pretrained(
        BASE, max_seq_length=8192, dtype=torch.bfloat16, load_in_4bit=False
    )
    load_seconds = time.time() - t0

    dtypes = sorted({str(p.dtype) for p in model.parameters()})
    n_params = sum(p.numel() for p in model.parameters())
    fp32_names = [name for name, p in model.named_parameters() if p.dtype != torch.bfloat16]
    bf16_names = [name for name, p in model.named_parameters() if p.dtype == torch.bfloat16]
    non_bf16_outside_norm_or_head = [
        n for n in fp32_names if "norm" not in n and "head" not in n
    ]
    state = "Record: the trial enrolled adults aged 18 to 65 with type 2 diabetes."
    questions = {
        "supported": {"type": "noul", "instructions": "Does the record support the claim that enrolment was adults?"},
        "choice": {
            "type": "choice",
            "instructions": "Which population is studied?",
            "criteria": ["adults", "children", "animals"],
        },
    }
    with torch.inference_mode():
        answers = FastDecisionModel.predict(model, tokenizer, state, questions)
    probs = {name: list(ans["probabilities"].values()) for name, ans in answers.items()}
    finite = all(math.isfinite(v) for vals in probs.values() for v in vals)
    sums = {name: round(sum(vals), 6) for name, vals in probs.items()}

    passed = (
        not non_bf16_outside_norm_or_head
        and bool(bf16_names)
        and finite
        and all(abs(v - 1.0) < 1e-3 for v in sums.values())
    )
    result = {
        "status": "PASS" if passed else "FAIL",
        "base": BASE,
        "load_in_4bit": False,
        "load_seconds": round(load_seconds, 2),
        "parameter_dtypes": dtypes,
        "n_bf16_parameter_tensors": len(bf16_names),
        "n_non_bf16_parameter_tensors": len(fp32_names),
        "non_bf16_outside_norm_or_head": non_bf16_outside_norm_or_head,
        "non_bf16_all_norm_or_head": all("norm" in n or "head" in n for n in fp32_names),
        "n_parameters": n_params,
        "predict_probabilities_finite": finite,
        "predict_probability_sums": sums,
        "predict_answers": {name: ans.get("answer") for name, ans in answers.items()},
        "model_class": type(model).__name__,
        "gpu": torch.cuda.get_device_name(0),
        "peak_cuda_memory_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3),
        "torch": torch.__version__,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
