"""osler_v4 Y8 resume (operator steer, 2026-10-23): a run interrupted at a full checkpoint and resumed matches the uninterrupted run.

The tiny model is the one of tests/test_trainer_o6.py (a four-layer Qwen3.5-shaped model on CPU, built from the cached snapshot). The run
is interrupted at step 3, inside the first epoch, so the resume must skip the batches already taken in that epoch, and it continues into
the second epoch. Every trainable tensor and every optimizer moment must then equal the uninterrupted run's exactly.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from meddecide.model.meddecide_model import LoraSettings, MedDecideModel
from meddecide.model.readout_checks import synthetic_choice_item
from meddecide.train.config import StudentConfig
from meddecide.train.resume import latest_full_checkpoint, optimizer_params
from meddecide.train.trainer import Trainer

REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
HUB = Path(os.environ.get("HF_HOME", "/workspace/.hf_home")) / "hub"
SNAPSHOT = HUB / "models--Qwen--Qwen3.5-4B" / "snapshots" / REVISION


def build_tiny_dir(tmp_path_factory) -> Path:
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    if not SNAPSHOT.exists():
        pytest.skip(f"the pinned Qwen3.5-4B snapshot is not in the cache: {SNAPSHOT}")
    cfg = AutoConfig.from_pretrained(str(SNAPSHOT))
    text = cfg.text_config
    text.hidden_size = 64
    text.intermediate_size = 128
    text.num_hidden_layers = 4
    # full attention only: the linear-attention layers need the CUDA causal-conv kernel and cannot run on CPU
    text.layer_types = ["full_attention"] * 4
    text.num_attention_heads = 4
    text.num_key_value_heads = 2
    text.head_dim = 16
    text.linear_num_key_heads = 2
    text.linear_num_value_heads = 4
    text.linear_key_head_dim = 16
    text.linear_value_head_dim = 16
    cfg.vision_config.depth = 1
    torch.manual_seed(0)
    out = tmp_path_factory.mktemp("tiny_resume")
    AutoModelForCausalLM.from_config(cfg, dtype=torch.float32).save_pretrained(out)
    AutoTokenizer.from_pretrained(str(SNAPSHOT)).save_pretrained(out)
    return out


@pytest.fixture(scope="module")
def tiny_dir(tmp_path_factory) -> Path:
    return build_tiny_dir(tmp_path_factory)


def _trainer(tiny: Path) -> tuple[Trainer, MedDecideModel]:
    torch.manual_seed(0)  # as run_arm does: the LoRA A matrices start from the same weights in every run
    model = MedDecideModel(str(tiny), lora=LoraSettings(r=4, alpha=8), device="cpu", dtype="float32", readout="option_code")
    config = StudentConfig(device="cpu", dtype="float32", batch_size=2, eval_every=1000, log_every=0, lr=1e-2, lora_lr=1e-2)
    return Trainer(model, config), model


def _items() -> list:
    return [synthetic_choice_item(n, seed) for seed, n in enumerate([2, 4, 3, 5, 2, 4, 3, 5])]


def test_resumed_run_matches_the_uninterrupted_run(tiny_dir: Path, tmp_path: Path) -> None:
    items = _items()
    steps, schedule, first_steps = 6, 6, 3  # the run is interrupted after step 3, inside the first epoch

    straight, _ = _trainer(tiny_dir)
    straight.train(items, steps=steps, schedule_steps=schedule)

    first, _ = _trainer(tiny_dir)
    first.train(items, steps=first_steps, schedule_steps=schedule, full_checkpoint_dir=tmp_path / "full",
                full_checkpoint_every=first_steps)
    assert latest_full_checkpoint(tmp_path / "full") == tmp_path / "full" / f"step_{first_steps:06d}"

    resumed, _ = _trainer(tiny_dir)  # a new model and trainer: nothing carries over but the checkpoint
    result = resumed.train(items, steps=steps, schedule_steps=schedule,
                           resume_from=tmp_path / "full" / f"step_{first_steps:06d}")
    assert result.steps == steps

    assert [r.loss for r in straight.history[first_steps:]] == [r.loss for r in resumed.history]
    expected, actual = optimizer_params(straight.optimizer), optimizer_params(resumed.optimizer)
    assert set(expected) == set(actual)
    for name in expected:
        assert torch.equal(expected[name], actual[name]), name
    straight_state = straight.optimizer.state_dict()["state"]
    resumed_state = resumed.optimizer.state_dict()["state"]
    assert set(straight_state) == set(resumed_state)
    for key, entry in straight_state.items():
        for field, value in entry.items():
            assert torch.equal(value, resumed_state[key][field]), (key, field)


