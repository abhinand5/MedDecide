"""End-to-end smoke test of one O6-O8 arm on CPU with the tiny Qwen3.5 model: training, evaluation at the cadence, the
tier-1 diagnostic, the best checkpoint, the temperature fit and the final full-dev evaluation of the selected checkpoint."""

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
from meddecide.model.meddecide_model import LoraSettings  # noqa: E402
from meddecide.train.arm_run import run_arm  # noqa: E402
from meddecide.train.config import StudentConfig  # noqa: E402

REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
HUB = Path(os.environ.get("HF_HOME", "/workspace/.hf_home")) / "hub"
SNAPSHOT = HUB / "models--Qwen--Qwen3.5-4B" / "snapshots" / REVISION


def _item(seed: int, source: str, n_options: int, template: str):
    options = [Option(key=chr(ord("A") + i), label=f"option {i}") for i in range(n_options)]
    return make_item(
        tier=Tier.FRESH, source=source, source_record_id=f"smoke-{seed}", source_url="https://example.org/smoke",
        source_license="test", record_date=date(2026, 1, 1), split="dev", template_id=template, skill="smoke",
        qtype=QuestionType.CHOICE, state=f"A synthetic stem number {seed} about a patient with a headache.",
        question=f"Which option applies to case {seed}?", options=options, gold=options[seed % n_options].key,
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
    out = tmp_path_factory.mktemp("tiny_arm")
    AutoModelForCausalLM.from_config(cfg, dtype=torch.float32).save_pretrained(out)
    AutoTokenizer.from_pretrained(str(SNAPSHOT)).save_pretrained(out)
    return out


def test_one_arm_trains_evaluates_selects_and_reports_on_cpu(tiny_dir, tmp_path) -> None:
    config = StudentConfig(device="cpu", dtype="float32", batch_size=2, eval_every=2, log_every=0, lr=1e-3,
                           lora_lr=1e-4, readout="option_code", lora=LoraSettings(r=4, alpha=8), eval_batch_size=2,
                           fit_temperature=True, temperature_per_qtype=True)
    train = [_item(i, "pubmed", 2 + (i % 3), "t_a" if i % 2 else "t_b") for i in range(8)]
    dev = [_item(100 + i, "medqa" if i < 2 else "pubmed", 3, "d_a" if i % 2 else "d_b") for i in range(6)]
    summary = run_arm(config, train_items=train, dev_items=dev, out_dir=tmp_path / "arm", steps=4,
                      eval_subset_size=4, base_model=str(tiny_dir), revision=None, device="cpu", dtype="float32")
    assert summary["steps"] == 4 and summary["examples"] == 8
    assert [e["step"] for e in summary["evaluations"]] == [2, 4]
    assert [t["step"] for t in summary["tier1_trajectory"]] == [2, 4]
    assert set(summary["tier1_trajectory"][0]["accuracy_by_source"]) == {"medqa"}
    assert summary["eval_subset_items"] == 4 and summary["full_dev_items"] == 6
    assert "macro_accuracy" in summary["full_dev_metrics"]
    assert summary["selected"].startswith("best checkpoint")
    assert (tmp_path / "arm" / "checkpoints" / "best" / "optcode.pt").exists()
    assert summary["temperature_fits"]  # the per-qtype temperature was fitted on dev
    assert (tmp_path / "arm" / "arm_result.json").exists()
    assert summary["stopped_early"] is None
