"""O5 readout checks on a tiny random Qwen3.5 model, on CPU.

The tiny model keeps the 4B's layer pattern (three linear-attention layers, then one full-attention layer), its tokenizer
and its tied LM head, with random weights. The CUDA kernel packages are blocked before the modelling file is imported, so
the pure-torch reference path runs. The checks themselves live in ``meddecide.model.readout_checks``; the same functions
run on the real Qwen3.5-4B on the GPU in ``scripts/osler/o5_checks.py``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# CPU only: the CUDA kernel packages would be called on CPU tensors. Blocking them makes transformers use the torch path.
os.environ.setdefault("USE_HUB_KERNELS", "NO")
sys.modules["causal_conv1d"] = None
sys.modules["fla"] = None

import pytest  # noqa: E402
import torch  # noqa: E402

from meddecide.model.meddecide_model import LoraSettings, MedDecideModel  # noqa: E402
from meddecide.model.readout_checks import (  # noqa: E402
    causal_flag_max_abs_diff,
    export_max_abs_diff,
    generation_identity,
    init_max_abs_diff,
    padding_max_abs_diff,
    perturb_head,
    prompts_for,
    randomise_lora,
    synthetic_choice_item,
)

REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
HUB = Path(os.environ.get("HF_HOME", "/workspace/.hf_home")) / "hub"
SNAPSHOT = HUB / "models--Qwen--Qwen3.5-4B" / "snapshots" / REVISION
LORA = LoraSettings(r=4, alpha=8)


@pytest.fixture(scope="session")
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
    out = tmp_path_factory.mktemp("tiny_qwen35")
    AutoModelForCausalLM.from_config(cfg, dtype=torch.float32).save_pretrained(out)
    AutoTokenizer.from_pretrained(str(SNAPSHOT)).save_pretrained(out)
    return out


@pytest.fixture(scope="session")
def option_model(tiny_dir) -> MedDecideModel:
    model = MedDecideModel(str(tiny_dir), lora=LORA, device="cpu", dtype="float32",
                           readout="option_code", variant="bare")
    randomise_lora(model)
    return model


def test_initialised_option_code_readout_equals_the_letter_readout_with_mixed_option_counts(option_model) -> None:
    assert option_model.optcode_names[:3] == ["A", "B", "C"]
    expected_ids = [option_model._key_token(chr(ord("A") + i)) for i in range(26)]
    assert option_model.optcode_token_ids[:26] == expected_ids
    items = [synthetic_choice_item(2, 0), synthetic_choice_item(4, 1), synthetic_choice_item(9, 2)]
    diff = init_max_abs_diff(option_model, items)
    assert diff["logits_max_abs_diff"] < 1e-4
    assert diff["probs_max_abs_diff"] < 1e-4


def test_exported_causal_lm_reproduces_the_native_probabilities_with_the_adapter_separate(tiny_dir, tmp_path) -> None:
    trained = MedDecideModel(str(tiny_dir), lora=LORA, device="cpu", dtype="float32", readout="option_code")
    randomise_lora(trained)
    perturb_head(trained)
    items = [synthetic_choice_item(2, 3), synthetic_choice_item(4, 4), synthetic_choice_item(9, 5)]
    assert torch.equal(trained.exported_option_code_lm_head()[trained.optcode_token_ids],
                       trained.optcode.weight.detach())
    diff = export_max_abs_diff(trained, items, tmp_path, merged=False)
    assert diff["probs_max_abs_diff"] < 1e-3


def test_exported_causal_lm_reproduces_the_native_probabilities_with_the_adapter_merged(tiny_dir, tmp_path) -> None:
    trained = MedDecideModel(str(tiny_dir), lora=LORA, device="cpu", dtype="float32", readout="option_code")
    randomise_lora(trained, seed=7)
    perturb_head(trained, seed=8)
    items = [synthetic_choice_item(3, 6), synthetic_choice_item(5, 7)]
    diff = export_max_abs_diff(trained, items, tmp_path, merged=True)
    assert diff["probs_max_abs_diff"] < 1e-3


def test_non_causal_flag_changes_earlier_positions_only_when_it_is_on(tiny_dir) -> None:
    model = MedDecideModel(str(tiny_dir), lora=None, device="cpu", dtype="float32", readout="option_code")
    item = synthetic_choice_item(4, 8)
    causal_change = causal_flag_max_abs_diff(model, item, bidirectional=False)
    bidirectional_change = causal_flag_max_abs_diff(model, item, bidirectional=True)
    assert causal_change < 1e-5  # causal: the change cannot reach an earlier position
    assert bidirectional_change > 1e-4  # bidirectional: the change reaches earlier positions
    assert model.bidirectional_full_attention is False


@pytest.mark.parametrize("bidirectional", [False, True])
def test_option_code_padding_invariance_batch_one_against_padded(tiny_dir, bidirectional) -> None:
    model = MedDecideModel(str(tiny_dir), lora=None, device="cpu", dtype="float32", readout="option_code")
    diff = padding_max_abs_diff(model, synthetic_choice_item(4, 9), synthetic_choice_item(9, 10),
                                bidirectional=bidirectional)
    assert diff < 1e-3


def test_pointer_padding_invariance_batch_one_against_padded(tiny_dir) -> None:
    model = MedDecideModel(str(tiny_dir), lora=None, device="cpu", dtype="float32", readout="pointer")
    diff = padding_max_abs_diff(model, synthetic_choice_item(4, 11), synthetic_choice_item(7, 12),
                                bidirectional=False)
    assert diff < 1e-3


def test_generation_with_the_adapter_off_is_byte_identical_to_the_plain_base(option_model, tiny_dir) -> None:
    from transformers import AutoModelForCausalLM

    plain = AutoModelForCausalLM.from_pretrained(str(tiny_dir), dtype=torch.float32).eval()
    prompts = prompts_for(option_model, [synthetic_choice_item(4, 13), synthetic_choice_item(2, 14)])
    result = generation_identity(option_model, plain, prompts)
    assert result["byte_identical"] is True
    # the control: with the adapter on, the logits differ, so an adapter change would have been detected
    assert result["control_adapter_logits_max_abs_diff"] > 1e-4


def test_option_code_head_survives_save_and_load(tiny_dir, tmp_path) -> None:
    trained = MedDecideModel(str(tiny_dir), lora=LORA, device="cpu", dtype="float32", readout="option_code")
    randomise_lora(trained, seed=21)
    perturb_head(trained, seed=22)
    item = synthetic_choice_item(5, 15)
    batch = trained.collate([trained.encode_item(item)])
    _, before = trained.batch_logits(batch)
    trained.save(tmp_path / "model")
    loaded = MedDecideModel.load(tmp_path / "model", device="cpu", dtype="float32")
    loaded_batch = loaded.collate([loaded.encode_item(item)])
    _, after = loaded.batch_logits(loaded_batch)
    assert loaded.readout == "option_code" and loaded.optcode is not None
    assert torch.allclose(before, after, atol=1e-6)
