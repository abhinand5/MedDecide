"""O6 trainer additions: the cosine floor, the two divergence tripwires and the evaluation hook (CPU, tiny model).

The student path keeps its defaults (floor 0, no tripwires, no hook); these tests cover the new options only.
"""

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
from meddecide.train.config import StudentConfig  # noqa: E402
from meddecide.train.trainer import Trainer  # noqa: E402

REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
HUB = Path(os.environ.get("HF_HOME", "/workspace/.hf_home")) / "hub"
SNAPSHOT = HUB / "models--Qwen--Qwen3.5-4B" / "snapshots" / REVISION


def test_cosine_floor_keeps_the_rate_at_the_floor_at_the_end_and_is_one_at_the_peak() -> None:
    peak = Trainer.lr_factor(3, total_steps=100, warmup_steps=3, floor=0.1)
    end = Trainer.lr_factor(100, total_steps=100, warmup_steps=3, floor=0.1)
    assert peak == pytest.approx(1.0)
    assert end == pytest.approx(0.1)
    assert Trainer.lr_factor(100, total_steps=100, warmup_steps=3) == pytest.approx(0.0)  # the student default
    middle = Trainer.lr_factor(51, total_steps=100, warmup_steps=3, floor=0.1)
    assert 0.1 < middle < 1.0


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
    out = tmp_path_factory.mktemp("tiny_trainer")
    AutoModelForCausalLM.from_config(cfg, dtype=torch.float32).save_pretrained(out)
    AutoTokenizer.from_pretrained(str(SNAPSHOT)).save_pretrained(out)
    return out


def _trainer(tiny_dir: Path, **config_overrides) -> tuple[Trainer, MedDecideModel, StudentConfig]:
    model = MedDecideModel(str(tiny_dir), lora=LoraSettings(r=4, alpha=8), device="cpu", dtype="float32",
                           readout="option_code")
    config = StudentConfig(device="cpu", dtype="float32", batch_size=2, eval_every=1, log_every=0, lr=1e-2,
                           lora_lr=1e-2, **config_overrides)
    return Trainer(model, config), model, config


def test_gradient_norm_tripwire_stops_before_the_optimizer_step(tiny_dir) -> None:
    trainer, model, _ = _trainer(tiny_dir, tripwire_grad_norm=0.0)
    before = model.optcode.weight.detach().clone()
    items = [synthetic_choice_item(n, seed) for seed, n in enumerate([2, 4, 3, 5])]
    result = trainer.train(items, steps=5)
    assert result.steps == 0
    assert result.stopped_early is not None and result.stopped_early.startswith("gradient-norm tripwire")
    assert torch.equal(model.optcode.weight.detach(), before)  # no update was applied on the tripped step


def test_dev_macro_tripwire_stops_after_consecutive_low_evaluations(tiny_dir, monkeypatch) -> None:
    trainer, _, _ = _trainer(tiny_dir, tripwire_macro_floor=0.5, tripwire_consecutive_evals=2)
    monkeypatch.setattr(trainer, "evaluate", lambda items, batch_size=None: {"macro_accuracy": 0.3, "brier": 0.5})
    items = [synthetic_choice_item(n, seed) for seed, n in enumerate([2, 4, 3, 5])]
    result = trainer.train(items, steps=10, eval_items=items[:2])
    assert result.steps == 2
    assert result.stopped_early is not None and result.stopped_early.startswith("dev-macro tripwire")


def test_dev_macro_tripwire_resets_on_a_good_evaluation(tiny_dir, monkeypatch) -> None:
    trainer, _, _ = _trainer(tiny_dir, tripwire_macro_floor=0.5, tripwire_consecutive_evals=2)
    values = iter([0.3, 0.9, 0.3, 0.9, 0.3, 0.9])
    monkeypatch.setattr(trainer, "evaluate", lambda items, batch_size=None: {"macro_accuracy": next(values),
                                                                             "brier": 0.5})
    items = [synthetic_choice_item(n, seed) for seed, n in enumerate([2, 4, 3, 5])]
    result = trainer.train(items, steps=6, eval_items=items[:2])
    assert result.steps == 6 and result.stopped_early is None


def test_eval_hook_observes_every_evaluation(tiny_dir, monkeypatch) -> None:
    trainer, _, _ = _trainer(tiny_dir)
    seen: list[tuple[int, float]] = []
    trainer.eval_hook = lambda step, metrics: seen.append((step, float(metrics["macro_accuracy"])))
    monkeypatch.setattr(trainer, "evaluate", lambda items, batch_size=None: {"macro_accuracy": 0.6, "brier": 0.4})
    items = [synthetic_choice_item(n, seed) for seed, n in enumerate([2, 4, 3, 5])]
    trainer.train(items, steps=2, eval_items=items[:2])
    assert seen == [(1, 0.6), (2, 0.6)]
