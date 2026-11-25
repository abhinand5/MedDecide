"""Unit tests for the O5 precision settings and the weight-perturbation helpers (CPU only)."""

from __future__ import annotations

import os
import sys

import pytest
import torch

from meddecide.precision import (
    PRECISIONS,
    max_abs_gap,
    perturb_layer_weights,
    prepare_precision,
    restore_layer_weights,
    snapshot_layer_weights,
)


class _Tiny(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList([torch.nn.Linear(4, 4, bias=False)])
        self.norm = torch.nn.Parameter(torch.ones(4))


def test_precision_names_are_the_three_documented_settings() -> None:
    assert PRECISIONS == ("as_run", "reference", "ieee")


def test_unknown_precision_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown precision"):
        prepare_precision("bf16")


def test_as_run_leaves_the_environment_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TRITON_F32_DEFAULT", raising=False)
    monkeypatch.delenv("USE_HUB_KERNELS", raising=False)
    prepare_precision("as_run")
    assert "TRITON_F32_DEFAULT" not in os.environ
    assert "USE_HUB_KERNELS" not in os.environ


def test_ieee_sets_the_triton_fp32_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TRITON_F32_DEFAULT", raising=False)
    prepare_precision("ieee")
    assert os.environ["TRITON_F32_DEFAULT"] == "ieee"


def test_reference_blocks_the_kernel_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("USE_HUB_KERNELS", raising=False)
    for module in ("fla", "causal_conv1d"):
        monkeypatch.setitem(sys.modules, module, sys.modules.get(module))
    prepare_precision("reference")
    assert os.environ["USE_HUB_KERNELS"] == "NO"
    assert sys.modules["fla"] is None
    assert sys.modules["causal_conv1d"] is None


def test_max_abs_gap_is_the_largest_elementwise_difference() -> None:
    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([1.0, 2.5, 2.0])
    assert max_abs_gap(a, b) == pytest.approx(1.0)
    assert max_abs_gap(a, a) == 0.0


def test_perturb_is_seeded_and_restore_undoes_it() -> None:
    torch.manual_seed(0)
    model = _Tiny()
    original = model.layers[0].weight.detach().clone()
    snapshot = snapshot_layer_weights(model)
    assert set(snapshot) == {"layers.0.weight"}

    perturb_layer_weights(model, snapshot, 1e-3, seed=1)
    first = model.layers[0].weight.detach().clone()
    assert not torch.equal(first, original)
    assert torch.equal(model.norm.detach(), torch.ones(4))

    restore_layer_weights(model, snapshot)
    assert torch.equal(model.layers[0].weight.detach(), original)

    perturb_layer_weights(model, snapshot, 1e-3, seed=1)
    assert torch.equal(model.layers[0].weight.detach(), first)


def test_perturb_refuses_an_empty_snapshot() -> None:
    with pytest.raises(ValueError, match="empty snapshot"):
        perturb_layer_weights(_Tiny(), {}, 1e-3, seed=0)
