"""osler_v5 D53: the weight-space blend. alpha = 0 reproduces the base model's probabilities through the reference path
(MedDecideModel.reference_option_probs), alpha = 1 reproduces the trained model, and the readout and the LoRA scalings interpolate.
CPU, a tiny random Qwen3.5 model built from the pinned snapshot's configuration (as in test_train_step_o5.py)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("USE_HUB_KERNELS", "NO")
sys.modules["causal_conv1d"] = None
sys.modules["fla"] = None

import pytest  # noqa: E402
import torch  # noqa: E402

from meddecide.bench.schema import Item  # noqa: E402
from meddecide.model.blend import Blend  # noqa: E402
from meddecide.model.meddecide_model import LoraSettings, MedDecideModel  # noqa: E402
from meddecide.model.readout_checks import synthetic_choice_item  # noqa: E402

REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
HUB = Path(os.environ.get("HF_HOME", "/workspace/.hf_home")) / "hub"
SNAPSHOT = HUB / "models--Qwen--Qwen3.5-4B" / "snapshots" / REVISION


@pytest.fixture(scope="module")
def tiny_dir(tmp_path_factory) -> Path:
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    if not SNAPSHOT.exists():
        pytest.skip(f"the pinned Qwen3.5-4B snapshot is not in the cache: {SNAPSHOT}")
    cfg = AutoConfig.from_pretrained(str(SNAPSHOT))
    text = cfg.text_config
    text.hidden_size = 64
    text.intermediate_size = 128
    text.num_hidden_layers = 4
    text.layer_types = ["linear_attention", "linear_attention", "linear_attention", "full_attention"]
    text.num_attention_heads = 4
    text.num_key_value_heads = 2
    text.head_dim = 16
    text.linear_num_key_heads = 2
    text.linear_num_value_heads = 4
    text.linear_key_head_dim = 16
    text.linear_value_head_dim = 16
    cfg.vision_config.depth = 1
    torch.manual_seed(0)
    out = tmp_path_factory.mktemp("tiny_blend")
    AutoModelForCausalLM.from_config(cfg, dtype=torch.float32).save_pretrained(out)
    AutoTokenizer.from_pretrained(str(SNAPSHOT)).save_pretrained(out)
    return out


def _trained_model(tiny_dir: Path) -> MedDecideModel:
    """A model whose adapter and readout have moved away from the base (so every blend weight changes the output)."""
    model = MedDecideModel(str(tiny_dir), lora=LoraSettings(r=4, alpha=8), device="cpu", dtype="float32", readout="option_code")
    gen = torch.Generator().manual_seed(7)
    with torch.no_grad():
        for name, parameter in model.peft_model.named_parameters():
            if "lora_B" in name:
                parameter.copy_(torch.randn(parameter.shape, generator=gen) * 0.05)
        model.optcode.weight.add_(torch.randn(model.optcode.weight.shape, generator=gen) * 0.02)
    return model


def _probs(model: MedDecideModel, items: list[Item]) -> torch.Tensor:
    batch = model.collate([model.encode_item(item) for item in items])
    _, probs = model.batch_logits(batch)
    return probs.detach().clone()


def _reference(model: MedDecideModel, items: list[Item]) -> torch.Tensor:
    batch = model.collate([model.encode_item(item) for item in items])
    return model.reference_option_probs(batch).detach().clone()


def test_alpha_zero_reproduces_the_reference_path_and_alpha_one_the_trained_model(tiny_dir) -> None:
    model = _trained_model(tiny_dir)
    items = [synthetic_choice_item(n, seed) for seed, n in enumerate([2, 4, 5, 3])]
    blend = Blend(model)
    trained = _probs(model, items)
    blend.apply(0.0)
    assert (_probs(model, items) - _reference(model, items)).abs().max().item() < 1e-5
    assert (_probs(model, items) - trained).abs().max().item() > 1e-4  # the trained model is not the base
    blend.apply(1.0)
    assert (_probs(model, items) - trained).abs().max().item() < 1e-6


def test_the_readout_interpolates_between_the_base_and_the_trained_rows(tiny_dir) -> None:
    model = _trained_model(tiny_dir)
    blend = Blend(model)
    base, trained = blend.base_readout.clone(), blend.trained_readout.clone()
    blend.apply(0.25)
    assert torch.allclose(model.optcode.weight.detach(), 0.75 * base + 0.25 * trained, atol=1e-7)
    blend.apply(1.0)
    assert torch.equal(model.optcode.weight.detach().float(), trained)


def test_the_lora_scalings_scale_with_alpha_and_restore_exactly(tiny_dir) -> None:
    model = _trained_model(tiny_dir)
    blend = Blend(model)
    originals = [(module, dict(scales)) for module, scales in blend.base_scaling]
    assert originals, "the blend must see the LoRA layers"
    blend.apply(0.5)
    for module, scales in originals:
        for name, scale in scales.items():
            assert module.scaling[name] == pytest.approx(0.5 * scale)
    blend.apply(1.0)
    for module, scales in originals:
        for name, scale in scales.items():
            assert module.scaling[name] == scale


def test_alpha_outside_the_unit_interval_is_refused(tiny_dir) -> None:
    blend = Blend(_trained_model(tiny_dir))
    with pytest.raises(ValueError, match="alpha must be in"):
        blend.apply(1.2)
    with pytest.raises(ValueError, match="alpha must be in"):
        blend.apply(-0.1)
