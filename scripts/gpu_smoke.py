#!/usr/bin/env python
"""GPU smoke test (T1): load a ladder model in bf16 and run one forward pass.

Proves the CUDA 13 / Blackwell wheel stack works before any task depends on it. This
script is a thin CLI wrapper; the model loading lives in
:mod:`meddecide.eval.readout` so the harness (T6) reuses the same code path.

Usage:
    uv run python scripts/gpu_smoke.py --model Qwen/Qwen3.5-0.8B --out outputs/bench_v0/T1/gpu_smoke.txt
"""

from __future__ import annotations

import argparse
import platform
import sys
from pathlib import Path

from meddecide.utils.provenance import Provenance, gpu_name, utcnow


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--out", type=Path, default=Path("outputs/bench_v0/T1/gpu_smoke.txt"))
    parser.add_argument("--prompt", default="The capital of France is")
    args = parser.parse_args()

    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    lines: list[str] = []

    def log(message: str) -> None:
        print(message, flush=True)
        lines.append(message)

    prov = Provenance(
        run_name="T1_gpu_smoke",
        command=" ".join([sys.executable, *sys.argv]),
        gpu=gpu_name(),
    )
    try:  # R7: record the resolved model revision, not just the model id
        from huggingface_hub import HfApi

        prov.models.append({"id": args.model, "revision": HfApi().model_info(args.model).sha})
        log(f"model_revision: {prov.models[0]['revision']}")
    except Exception as exc:  # pragma: no cover - network dependent
        log(f"model_revision: UNKNOWN ({type(exc).__name__})")
    log(f"time_utc: {utcnow()}")
    log(f"python: {platform.python_version()}")
    log(f"torch: {torch.__version__}")
    log(f"transformers: {transformers.__version__}")
    log(f"cuda_available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        log("FAIL: CUDA is not available — the GPU stack is broken")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text("\n".join(lines) + "\n")
        return 1
    log(f"cuda_version: {torch.version.cuda}")
    log(f"device: {torch.cuda.get_device_name(0)}")
    log(f"capability: {torch.cuda.get_device_capability(0)}")
    log(f"device_memory_total_gb: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}")
    log(f"arch_list: {torch.cuda.get_arch_list()}")

    log(f"loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    log(f"loading model (bf16): {args.model}")
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda:0")
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    log(f"model_params: {n_params:,}")
    log(f"model_dtype: {next(model.parameters()).dtype}")

    inputs = tokenizer(args.prompt, return_tensors="pt").to("cuda:0")
    log(f"prompt_tokens: {int(inputs['input_ids'].shape[1])}")

    with torch.inference_mode():
        out = model(**inputs)
        logits = out.logits
        log(f"logits_shape: {tuple(logits.shape)}")
        next_token_logits = logits[0, -1, :]
        top = torch.topk(next_token_logits.float(), k=5)
        top_tokens = [tokenizer.decode([int(i)]) for i in top.indices]
        log(f"top5_next_tokens: {top_tokens}")
        log(f"top5_logits: {[round(float(v), 4) for v in top.values]}")
        # one generated token, to prove the decode path also works
        generated = model.generate(**inputs, max_new_tokens=1, do_sample=False)
        log(f"greedy_next_token: {tokenizer.decode(generated[0, inputs['input_ids'].shape[1]:])!r}")

    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    log(f"peak_gpu_memory_gb: {peak_gb:.2f}")
    log("PASS: bf16 forward pass completed on GPU")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    prov.notes = "gpu smoke test: bf16 load + one forward pass + one greedy token"
    prov.finish().write(args.out.with_name("gpu_smoke_provenance.json"))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
