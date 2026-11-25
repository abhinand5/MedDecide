"""Numerical-precision settings for fp32 runs of the Qwen3.5 linear-attention kernels (ADVISORY O5 diagnostics).

On the Ampere-or-newer GPUs used here, fla's chunked gated-delta solve asks for TF32 explicitly and Triton's default
precision for fp32 dot products is TF32, so an fp32 forward that goes through the kernels is not full fp32. The settings
below make that choice explicit:

    as_run     the default: fla and causal_conv1d kernels with their own precision (the O5 verdict run)
    reference  kernels blocked: the pure PyTorch reference path (fp32 matmuls, no TF32)
    ieee       kernels on, with Triton's fp32 dot default set to IEEE and fla's triangular solve set to IEEE

``prepare_precision`` must run before transformers or fla is imported, because both read the environment and the module
table at import time. ``apply_after_import`` must run once fla is importable and before the first forward pass. This module
imports only torch and the standard library, so it can be imported first.
"""

from __future__ import annotations

import importlib
import inspect
import os
import sys

import torch

PRECISIONS = ("as_run", "reference", "ieee")
KERNEL_MODULES = ("causal_conv1d", "fla")
QWEN_MODULE = "transformers.models.qwen3_5.modeling_qwen3_5"
KERNEL_FUNCTIONS = ("torch_chunk_gated_delta_rule", "causal_conv1d_fn")


def prepare_precision(name: str) -> None:
    """Set the environment and module table for one precision setting (call before importing transformers or fla)."""
    if name not in PRECISIONS:
        raise ValueError(f"unknown precision {name!r}; expected one of {PRECISIONS}")
    if name == "reference":
        os.environ["USE_HUB_KERNELS"] = "NO"
        for module in KERNEL_MODULES:
            sys.modules[module] = None
    elif name == "ieee":
        os.environ["TRITON_F32_DEFAULT"] = "ieee"


def apply_after_import(name: str) -> None:
    """Finish the ieee setting: force fla's triangular-solve dots to IEEE (a diagnostic override of a module constant)."""
    if name != "ieee":
        return
    import fla.ops.gated_delta_rule.chunk_fwd as chunk_fwd
    import triton.language as tl

    chunk_fwd.SOLVE_TRIL_DOT_PRECISION = tl.constexpr("ieee")


def kernel_implementations() -> dict[str, str]:
    """Which implementation each Qwen3.5 kernel entry point resolved to, read from the decorator's closure."""
    module = importlib.import_module(QWEN_MODULE)
    resolved: dict[str, str] = {}
    for function in KERNEL_FUNCTIONS:
        implementation = inspect.getclosurevars(getattr(module, function)).nonlocals.get("implementation")
        if implementation is None:
            resolved[function] = "torch reference"
        else:
            resolved[function] = f"{implementation.__module__}.{implementation.__qualname__}"
    return resolved


def max_abs_gap(a: torch.Tensor, b: torch.Tensor) -> float:
    """Largest absolute element-wise difference, computed in float32."""
    return float((a.float() - b.float()).abs().max().item())


def snapshot_layer_weights(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Copies of the two-dimensional decoder-layer weights, for a perturbation that is later undone."""
    return {name: p.detach().clone() for name, p in module.named_parameters() if "layers." in name and p.dim() == 2}


def perturb_layer_weights(module: torch.nn.Module, snapshot: dict[str, torch.Tensor], relative_scale: float,
                          seed: int) -> None:
    """Set each snapshotted weight to its snapshot times (1 + relative_scale * N(0, 1)), with a fixed seed."""
    if not snapshot:
        raise ValueError("empty snapshot: no decoder-layer weights to perturb")
    device = next(iter(snapshot.values())).device
    generator = torch.Generator(device=device).manual_seed(seed)
    with torch.no_grad():
        for name, p in module.named_parameters():
            if name in snapshot:
                p.copy_(snapshot[name])
                p.mul_(1 + relative_scale * torch.randn(p.shape, generator=generator, device=p.device))


def restore_layer_weights(module: torch.nn.Module, snapshot: dict[str, torch.Tensor]) -> None:
    """Undo ``perturb_layer_weights``: copy the snapshot back into the decoder-layer weights."""
    with torch.no_grad():
        for name, p in module.named_parameters():
            if name in snapshot:
                p.copy_(snapshot[name])

