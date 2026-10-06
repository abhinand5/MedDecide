#!/usr/bin/env python
"""S0 — before/after fused-kernel measurement on a ladder model (prefill only).

The missing kernels (`causal_conv1d`, `flash-linear-attention`) are a known ~3x
slowdown and the cause of the 30 GB allocations on long prompts (bench_v0_fix0 P4).
This script measures one prefill pass at a given sequence length and reports
tokens/s, peak allocated/reserved memory, and which kernel implementation each
decorated function resolved to.

`--kernels off` reproduces the **pre-install** state exactly: a meta-path finder makes
`causal_conv1d` and `fla` unimportable *before* `transformers` is imported, so the
modeling module binds its PyTorch fallbacks. That is the same code path the previous
loop ran on; it does not modify the environment.

Usage:
    uv run python scripts/student/s0_kernels.py --kernels off --out outputs/student_v0/S0/kernels_before.json
    uv run python scripts/student/s0_kernels.py --kernels auto --out outputs/student_v0/S0/kernels_after.json
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from pathlib import Path

BLOCKED_PACKAGES = ("causal_conv1d", "fla", "fla_core")


def block_kernel_imports() -> None:
    """Make the kernel packages unimportable in this process (pre-install state)."""
    import importlib.abc

    class _Blocker(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):  # noqa: ANN001
            if fullname.split(".")[0] in BLOCKED_PACKAGES:
                raise ModuleNotFoundError(f"{fullname} blocked by --kernels off")
            return None

    sys.meta_path.insert(0, _Blocker())


def kernel_state() -> dict:
    """Importability + version of each kernel package, in this process."""
    state: dict = {}
    for name in ("causal_conv1d", "fla", "triton"):
        try:
            module = importlib.import_module(name)
            state[name] = {
                "importable": True,
                "version": getattr(module, "__version__", None),
                "file": getattr(module, "__file__", None),
            }
        except Exception as exc:  # noqa: BLE001 - report the exact failure
            state[name] = {"importable": False, "error": f"{type(exc).__name__}: {exc}"}
    return state


def _bound_implementation(func):
    """Find the implementation a kernel wrapper actually calls.

    ``use_kernel_func_from_hub_with_fallback`` wraps the chosen implementation with
    ``functools.wraps(torch_function)``, which copies ``__module__``/``__name__`` from the
    *torch* function — so reading ``__module__`` reports the fallback even when the fast
    kernel is bound. The implementation lives in the wrapper's closure instead.
    """
    seen = set()
    stack = [func]
    while stack:
        current = stack.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        closure = getattr(current, "__closure__", None) or ()
        for cell in closure:
            try:
                value = cell.cell_contents
            except ValueError:
                continue
            if callable(value) and value is not getattr(current, "__wrapped__", None):
                module = getattr(value, "__module__", "") or ""
                if not module.startswith("transformers."):
                    return {"module": module, "name": getattr(value, "__name__", None)}
                stack.append(value)
    return None


def resolved_implementations() -> dict:
    """Which implementation each decorated kernel function is bound to.

    Reads the wrapper closure (see ``_bound_implementation``) and records the library's
    own runtime fallback warning separately.
    """
    out: dict = {}
    for model_module in ("transformers.models.qwen3_5.modeling_qwen3_5",):
        try:
            module = importlib.import_module(model_module)
        except Exception as exc:  # noqa: BLE001
            out[model_module] = {"error": f"{type(exc).__name__}: {exc}"}
            continue
        funcs = {}
        for func_name in (
            "causal_conv1d_fn",
            "causal_conv1d_update",
            "torch_chunk_gated_delta_rule",
            "fused_recurrent_gated_delta_rule",
        ):
            func = getattr(module, func_name, None)
            if func is None:
                continue
            bound = _bound_implementation(func)
            funcs[func_name] = {
                "wrapped_name": getattr(func, "__name__", None),
                "bound_implementation": bound,
                "uses_reference_torch_path": bound is None,
            }
        out[model_module] = funcs
    return out


class _WarningCapture:
    """Collect transformers' own 'falling back to reference PyTorch' warnings."""

    def __init__(self) -> None:
        self.messages: list[str] = []

    def __enter__(self):
        import logging

        capture = self

        class _Handler(logging.Handler):
            def emit(self, record):  # noqa: ANN001
                capture.messages.append(record.getMessage())

        self._handler = _Handler()
        logging.getLogger("transformers").addHandler(self._handler)
        return self

    def __exit__(self, *exc):  # noqa: ANN002
        import logging

        logging.getLogger("transformers").removeHandler(self._handler)
        return False

    @property
    def fallback_warnings(self) -> list[str]:
        return [m for m in self.messages if "falling back" in m]


def build_input_ids(tokenizer, length: int):
    import torch

    seed = (
        "A randomised, double-blind, placebo-controlled trial enrolled adults with "
        "moderate disease. Participants received the study drug or matching placebo "
        "once daily for twelve weeks. The primary endpoint was change from baseline "
        "in the composite score at week twelve. "
    )
    seed_ids = tokenizer(seed, add_special_tokens=False)["input_ids"]
    if not seed_ids:
        raise RuntimeError("tokenizer produced no ids for the seed text")
    reps = (length // len(seed_ids)) + 1
    ids = (seed_ids * reps)[:length]
    return torch.tensor([ids], dtype=torch.long, device="cuda")


def measure(model, tokenizer, length: int, warmup: int = 1024, reps: int = 1, warmup_at_length: bool = False) -> dict:
    import torch

    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # Warmup on a short prompt so lazy init is not charged to the measured length.
    short = build_input_ids(tokenizer, warmup)
    with torch.no_grad():
        model(input_ids=short, use_cache=False, logits_to_keep=1)
    torch.cuda.synchronize()
    del short
    torch.cuda.empty_cache()

    ids = build_input_ids(tokenizer, length)
    if warmup_at_length:
        with torch.no_grad():
            model(input_ids=ids, use_cache=False, logits_to_keep=1)
        torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    times: list[float] = []
    logits_shape: list[int] | None = None
    for _ in range(reps):
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            out = model(input_ids=ids, use_cache=False, logits_to_keep=1)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
        logits_shape = list(out.logits.shape)
        del out
    peak_allocated = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()

    best = min(times)
    result = {
        "prompt_tokens": int(ids.shape[1]),
        "reps": reps,
        "warmup_at_length": warmup_at_length,
        "seconds_first": round(times[0], 4),
        "seconds_min": round(best, 4),
        "seconds_median": round(sorted(times)[len(times) // 2], 4),
        "seconds_all": [round(t, 4) for t in times],
        "tokens_per_s_min": round(ids.shape[1] / best, 2),
        "tokens_per_s_first": round(ids.shape[1] / times[0], 2),
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
        "logits_shape": logits_shape,
    }
    del ids
    torch.cuda.empty_cache()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--lengths", type=int, nargs="+", default=[8192, 16384])
    parser.add_argument("--kernels", choices=["auto", "off"], default="auto")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=1024)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--warmup-at-length", action="store_true")
    args = parser.parse_args()

    if args.kernels == "off":
        block_kernel_imports()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    kernels = kernel_state()
    implementations = resolved_implementations()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda")
    model.eval()

    capture = _WarningCapture()
    capture.__enter__()

    record: dict = {
        "model": args.model,
        "model_class": type(model).__name__,
        "model_revision": getattr(model.config, "_name_or_path", None),
        "dtype": "bfloat16",
        "device": torch.cuda.get_device_name(0),
        "kernels_mode": args.kernels,
        "kernels": kernels,
        "warmup_tokens": args.warmup,
        "reps": args.reps,
        "warmup_at_length": bool(args.warmup_at_length),
        "resolved_implementations": implementations,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "measurements": [],
        "errors": [],
    }

    for length in args.lengths:
        try:
            record["measurements"].append(
                measure(
                    model,
                    tokenizer,
                    length,
                    args.warmup,
                    reps=args.reps,
                    warmup_at_length=args.warmup_at_length,
                )
            )
        except Exception as exc:  # noqa: BLE001 - an OOM here is a result
            torch.cuda.empty_cache()
            record["errors"].append({"prompt_tokens": length, "error": f"{type(exc).__name__}: {exc}"})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    record["fallback_warnings"] = capture.fallback_warnings
    capture.__exit__()
    args.out.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({k: record[k] for k in ("kernels_mode", "measurements", "errors", "fallback_warnings")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
