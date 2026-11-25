"""O5 readiness for training: a gradient step on the option-code model moves the loss, and gradients reach both the
option-code head and the adapter (CPU, tiny random Qwen3.5 model, kernels blocked as in test_readouts_o5.py)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("USE_HUB_KERNELS", "NO")
sys.modules["causal_conv1d"] = None
sys.modules["fla"] = None

import pytest  # noqa: E402
import torch  # noqa: E402

from meddecide.model.meddecide_model import LoraSettings, MedDecideModel  # noqa: E402
from meddecide.model.readout_checks import synthetic_choice_item  # noqa: E402
from meddecide.train.losses import ce_plus_brier  # noqa: E402

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
    out = tmp_path_factory.mktemp("tiny_train")
    AutoModelForCausalLM.from_config(cfg, dtype=torch.float32).save_pretrained(out)
    AutoTokenizer.from_pretrained(str(SNAPSHOT)).save_pretrained(out)
    return out


def test_gradient_steps_lower_the_loss_and_reach_the_head_and_the_adapter(tiny_dir) -> None:
    model = MedDecideModel(str(tiny_dir), lora=LoraSettings(r=4, alpha=8), device="cpu", dtype="float32",
                           readout="option_code")
    names = {id(p): n for n, p in model.peft_model.named_parameters()}
    groups = model.trainable_param_groups(lr=1e-2, lora_lr=1e-2)
    assert {g["name"] for g in groups} == {"optcode", "lora"}
    assert model.trainable_parameter_count == sum(p.numel() for g in groups for p in g["params"])
    optimizer = torch.optim.AdamW(groups)
    items = [synthetic_choice_item(n, seed) for seed, n in enumerate([2, 4, 5, 3])]
    losses = []
    for step in range(3):
        batch = model.collate([model.encode_item(item) for item in items])
        _, probs = model.batch_logits(batch)
        breakdown = ce_plus_brier(probs, batch.gold_flat, batch.marker_item, batch.batch_size)
        optimizer.zero_grad(set_to_none=True)
        breakdown.total.backward()
        if step == 0:
            assert model.optcode.weight.grad is not None
            assert float(model.optcode.weight.grad.abs().sum()) > 0.0
            lora_grads = [p.grad for p in model.peft_model.parameters() if p.requires_grad and p.grad is not None]
            assert any(float(g.abs().sum()) > 0.0 for g in lora_grads)
            assert any("lora_B" in name for name in names.values())  # the adapter is the trained LoRA
        optimizer.step()
        losses.append(breakdown.total.item())
    assert losses[-1] < losses[0]
