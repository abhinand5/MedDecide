"""Gradient checkpointing for the O6-O8 arms (CPU, tiny Qwen3.5): the gradients are the same with and without it, and the
decoder layers stay in training mode across ``train_mode`` calls (the trainer calls it after every evaluation)."""

from __future__ import annotations

import os
import sys
from datetime import date
from pathlib import Path

os.environ.setdefault("USE_HUB_KERNELS", "NO")
sys.modules["causal_conv1d"] = None
sys.modules["fla"] = None

import pytest  # noqa: E402
import torch  # noqa: E402

from meddecide.bench.schema import Option, QuestionType, Tier, make_item  # noqa: E402
from meddecide.model.meddecide_model import LoraSettings, MedDecideModel  # noqa: E402

REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
HUB = Path(os.environ.get("HF_HOME", "/workspace/.hf_home")) / "hub"
SNAPSHOT = HUB / "models--Qwen--Qwen3.5-4B" / "snapshots" / REVISION


def _item(seed: int):
    options = [Option(key=chr(ord("A") + i), label=f"option {i}") for i in range(3)]
    return make_item(
        tier=Tier.FRESH, source="pubmed", source_record_id=f"ck-{seed}", source_url="https://example.org/ck",
        source_license="test", record_date=date(2026, 3, 1), split="train", template_id="ck_t", skill="ck",
        qtype=QuestionType.CHOICE, state=f"A synthetic stem number {seed} about a patient with a headache.",
        question=f"Which option applies to case {seed}?", options=options, gold=options[seed % 3].key,
        option_order_seed=seed)


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
    out = tmp_path_factory.mktemp("tiny_ckpt")
    AutoModelForCausalLM.from_config(cfg, dtype=torch.float32).save_pretrained(out)
    AutoTokenizer.from_pretrained(str(SNAPSHOT)).save_pretrained(out)
    return out


def _model(tiny: Path) -> MedDecideModel:
    return MedDecideModel(str(tiny), revision=None, lora=LoraSettings(r=4, alpha=8), device="cpu", dtype="float32",
                          readout="option_code", variant="bare")


def _gradients(model: MedDecideModel, items) -> tuple[float, dict[str, torch.Tensor]]:
    model.train_mode()
    batch = model.collate([model.encode_item(item) for item in items])
    flat_logits, _ = model.batch_logits(batch)
    loss = flat_logits.float().logsumexp(-1).sum()
    loss.backward()
    grads = {name: p.grad.detach().clone() for name, p in model.peft_model.named_parameters()
             if p.requires_grad and p.grad is not None}
    model.peft_model.zero_grad(set_to_none=True)
    return float(loss.item()), grads


def test_checkpointing_leaves_the_gradients_unchanged(tiny_dir: Path) -> None:
    plain = _model(tiny_dir)
    checkpointed = _model(tiny_dir)
    checkpointed.peft_model.load_state_dict(plain.peft_model.state_dict())
    checkpointed.optcode.load_state_dict(plain.optcode.state_dict())
    checkpointed.enable_gradient_checkpointing()
    items = [_item(i) for i in range(3)]
    loss_plain, grads_plain = _gradients(plain, items)
    loss_ckpt, grads_ckpt = _gradients(checkpointed, items)
    assert loss_ckpt == pytest.approx(loss_plain, rel=1e-6)
    assert grads_plain and set(grads_plain) == set(grads_ckpt)
    for name in grads_plain:
        assert torch.allclose(grads_plain[name], grads_ckpt[name], rtol=1e-4, atol=1e-6), name


def test_the_checkpoint_flag_survives_train_mode(tiny_dir: Path) -> None:
    plain = _model(tiny_dir)
    plain.train_mode()
    assert not any(layer.training for layer in plain.base.model.layers)  # no checkpointing: the base stays in eval
    checkpointed = _model(tiny_dir)
    checkpointed.enable_gradient_checkpointing()
    assert all(layer.training for layer in checkpointed.base.model.layers)
    checkpointed.eval_mode()
    checkpointed.train_mode()  # what the trainer does after an evaluation
    assert all(layer.training for layer in checkpointed.base.model.layers)


def test_checkpointing_needs_the_adapter(tiny_dir: Path) -> None:
    model = _model(tiny_dir)
    model.peft_model = None
    with pytest.raises(ValueError, match="needs the LoRA adapter"):
        model.enable_gradient_checkpointing()
