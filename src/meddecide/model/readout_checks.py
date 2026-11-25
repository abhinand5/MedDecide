"""O5 readout checks: the functions behind the CPU tiny-model tests and the GPU run on the real Qwen3.5-4B.

Each check returns the measured number it tests (a maximum absolute difference, or a boolean with its control), so the
caller records a measurement and the tolerance it was compared with, not only a verdict. The tests in
``tests/test_readouts_o5.py`` assert these numbers on a tiny random model; ``scripts/osler/o5_checks.py`` runs the same
functions on the pinned Qwen3.5-4B and writes ``outputs/osler_v0/O5/readout_checks.json``.
"""

from __future__ import annotations

import gc
from collections.abc import Sequence
from datetime import date
from pathlib import Path
from typing import Any

import torch
from torch import nn

from meddecide.bench.schema import Item, Option, QuestionType, Tier, make_item
from meddecide.eval.readout import option_letter, render_prompt
from meddecide.model.meddecide_model import MedDecideModel, letter_positions
from meddecide.model.meddecide_model import segment_softmax as segment_softmax_by_counts

PROMPT_TOKENS_GENERATED = 6


def synthetic_choice_item(n_options: int, seed: int) -> Item:
    """A validated choice item with ``n_options`` lettered options (a fixed clinical stem; only the options vary)."""
    options = [Option(key=option_letter(i), label=f"Candidate answer {i + 1}") for i in range(n_options)]
    return make_item(
        tier=Tier.FRESH,
        source="synthetic_o5",
        source_record_id=f"o5-{seed}-{n_options}",
        source_url="https://example.org/synthetic",
        source_license="synthetic",
        record_date=date(2026, 1, 1),
        split="dev",
        template_id="synthetic_o5_v1",
        skill="synthetic",
        qtype=QuestionType.CHOICE,
        state=(
            "A 61-year-old patient with hypertension presents with a two-day history of headache and "
            "blurred vision. Blood pressure is 178/104 mmHg."
        ),
        question=f"Which of the following is the most appropriate next step ({seed})?",
        options=options,
        gold=options[seed % n_options].key,
        option_order_seed=seed,
    )


def randomise_lora(model: MedDecideModel, scale: float = 0.05, seed: int = 1) -> None:
    """Give the adapter non-zero B matrices, so that it changes the output and the checks can see it."""
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, param in model.peft_model.named_parameters():
            if "lora_B" in name:
                param.copy_(torch.randn(param.shape, generator=gen) * scale)


def perturb_head(model: MedDecideModel, scale: float = 0.5, seed: int = 2) -> None:
    """Stand in for training: move the option-code rows away from their initialisation."""
    if model.optcode is None:
        raise ValueError("this model has no option-code head")
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        noise = torch.randn(model.optcode.weight.shape, generator=gen).to(model.optcode.weight.device)
    with torch.no_grad():
        model.optcode.weight.add_(noise * scale)


def letter_probabilities(model: MedDecideModel, batch: Any) -> tuple[torch.Tensor, torch.Tensor]:
    """The zero-shot letter readout: LM-head logits of the offered letters at the answer position, renormalised."""
    hidden = model.hidden_states(batch)
    rows = torch.arange(batch.batch_size, device=hidden.device)
    h = hidden[rows, batch.answer_positions.to(hidden.device)]
    causal = model.peft_model.get_base_model() if model.peft_model is not None else model.base
    vocab = causal.lm_head(h).float()
    n_options = [int(n) for n in batch.n_options.tolist()]
    letters = torch.tensor([model._key_token(chr(ord("A") + j)) for j in letter_positions(n_options)])
    flat = vocab[batch.marker_item.to(vocab.device), letters]
    return flat, segment_softmax_by_counts(flat, n_options)


def exported_causal_lm(base_model_id: str, exported_matrix: torch.Tensor, *, revision: str | None = None,
                       dtype: torch.dtype = torch.float32, device: str = "cpu") -> nn.Module:
    """A plain causal LM whose LM head is an untied copy that carries the exported matrix.

    The input embeddings are not touched: the head is replaced by a new module, so the tie to the embeddings is gone.
    """
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(base_model_id, revision=revision, dtype=dtype, device_map=device)
    head = nn.Linear(int(exported_matrix.shape[1]), int(exported_matrix.shape[0]), bias=False, device=device,
                     dtype=dtype)
    with torch.no_grad():
        head.weight.copy_(exported_matrix.to(device=device, dtype=dtype))
    model.lm_head = head
    return model.eval()


def causal_option_probabilities(causal_model: nn.Module, batch: Any, token_ids: Sequence[int]) -> torch.Tensor:
    """Offered-code probabilities from a causal LM at the answer position (the final prompt token)."""
    mask = batch.attention_mask
    position_ids = (mask.long().cumsum(-1) - 1).clamp(min=0)
    device = next(causal_model.parameters()).device
    with torch.no_grad():
        out = causal_model(input_ids=batch.input_ids.to(device), attention_mask=mask.to(device),
                           position_ids=position_ids.to(device), use_cache=False, logits_to_keep=1)
    last = out.logits[:, -1, :].float()
    n_options = [int(n) for n in batch.n_options.tolist()]
    codes = torch.tensor([int(token_ids[p]) for p in letter_positions(n_options)], dtype=torch.long,
                         device=last.device)
    flat = last[batch.marker_item.to(last.device).long(), codes]
    return segment_softmax_by_counts(flat, n_options)


CHUNK_ITEMS = 1  # items per batch in the checks: fp32 attention on long prompts needs one item at a time on the GPU


def chunks(items: Sequence[Item], size: int = CHUNK_ITEMS) -> list[list[Item]]:
    """Consecutive groups of ``size`` items (the checks batch each group, not the whole sample)."""
    if size < 1:
        raise ValueError("size must be >= 1")
    return [list(items[i : i + size]) for i in range(0, len(items), size)]


@torch.no_grad()
def init_max_abs_diff(model: MedDecideModel, items: Sequence[Item], chunk: int = CHUNK_ITEMS) -> dict[str, float]:
    """Initialised option-code readout against the zero-shot letter readout over the offered letters (chunked)."""
    logits_diff, probs_diff, options = 0.0, 0.0, 0
    for part in chunks(items, chunk):
        batch = model.collate([model.encode_item(item) for item in part])
        flat, probs = model.batch_logits(batch)
        letter_flat, letter_probs = letter_probabilities(model, batch)
        logits_diff = max(logits_diff, float((flat - letter_flat).abs().max()))
        probs_diff = max(probs_diff, float((probs - letter_probs).abs().max()))
        options += int(batch.marker_item.numel())
    return {"logits_max_abs_diff": logits_diff, "probs_max_abs_diff": probs_diff,
            "n_items": float(len(items)), "n_options_total": float(options)}


def export_plan(model: MedDecideModel, items: Sequence[Item], workdir: Path, *,
                chunk: int = CHUNK_ITEMS) -> dict[str, Any]:
    """Everything the export check needs, computed while the live model is on the GPU and held on the CPU.

    The native probabilities, the batches, the exported LM-head matrix and the adapter are kept, so the live model can be
    released before the exported causal LM is loaded (both fp32 copies do not fit on the GPU together with activations).
    """
    adapter_dir = workdir / "adapter"
    model.peft_model.save_pretrained(adapter_dir)
    batches, native, options = [], [], 0
    for part in chunks(items, chunk):
        batch = model.collate([model.encode_item(item) for item in part])
        with torch.no_grad():
            _, probs = model.batch_logits(batch)
        native.append(probs.detach().float().cpu())
        batches.append(batch)
        options += int(batch.marker_item.numel())
    return {"batches": batches, "native": native, "matrix": model.exported_option_code_lm_head().float().cpu(),
            "token_ids": list(model.optcode_token_ids), "adapter_dir": adapter_dir,
            "base_model_id": model.base_model_id, "revision": model.revision, "dtype": model.torch_dtype,
            "device": model.device, "n_items": len(items), "n_options_total": options}


def export_check(plan: dict[str, Any], *, merged: bool) -> dict[str, float]:
    """The exported causal LM (adapter kept separate, or merged) against the native probabilities of the plan."""
    from peft import PeftModel

    causal = exported_causal_lm(plan["base_model_id"], plan["matrix"], revision=plan["revision"],
                                dtype=plan["dtype"], device=plan["device"])
    causal = PeftModel.from_pretrained(causal, plan["adapter_dir"])
    if merged:
        causal = causal.merge_and_unload()
    diff = 0.0
    for batch, native in zip(plan["batches"], plan["native"], strict=True):
        probs = causal_option_probabilities(causal, batch, plan["token_ids"])
        diff = max(diff, float((native.to(probs.device) - probs).abs().max()))
    del causal
    gc.collect()
    torch.cuda.empty_cache()
    return {"probs_max_abs_diff": diff, "n_items": float(plan["n_items"]),
            "n_options_total": float(plan["n_options_total"])}


def export_max_abs_diff(model: MedDecideModel, items: Sequence[Item], workdir: Path, *,
                        merged: bool, chunk: int = CHUNK_ITEMS) -> dict[str, float]:
    """Convenience wrapper (the CPU tests): plan and check with the live model still in memory."""
    return export_check(export_plan(model, items, workdir, chunk=chunk), merged=merged)


def padding_max_abs_diff(model: MedDecideModel, first: Item, longer: Item, *, bidirectional: bool) -> float:
    """Probabilities of one item scored alone against the same item padded in a batch with a longer item."""
    previous = model.bidirectional_full_attention
    model.bidirectional_full_attention = bidirectional
    try:
        alone = model.collate([model.encode_item(first)])
        padded = model.collate([model.encode_item(first), model.encode_item(longer)])
        _, probs_alone = model.batch_logits(alone)
        _, probs_padded = model.batch_logits(padded)
        own = probs_padded[padded.marker_item.to(probs_padded.device) == 0]
        return float((probs_alone.to(own.device) - own).abs().max())
    finally:
        model.bidirectional_full_attention = previous


def causal_flag_max_abs_diff(model: MedDecideModel, item: Item, *, bidirectional: bool, back: int = 4) -> float:
    """Largest change in the hidden states before the last ``back`` tokens when one of those tokens changes.

    With the flag off the causal model must show no change (0 up to float noise); with it on, the change should reach
    earlier positions. The returned value is the largest absolute difference at positions before the changed token.
    """
    previous = model.bidirectional_full_attention
    model.bidirectional_full_attention = bidirectional
    try:
        batch = model.collate([model.encode_item(item)]).to(model.device)
        ids = batch.input_ids.clone()
        changed = ids.clone()
        position = ids.shape[1] - back
        changed[0, position] = (changed[0, position] + 17) % 1000 + 1
        other = dataclass_replace_input(batch, changed)
        before = model.hidden_states(batch)[0, : position].float()
        after = model.hidden_states(other)[0, : position].float()
        return float((before - after).abs().max()) if before.numel() else 0.0
    finally:
        model.bidirectional_full_attention = previous


def dataclass_replace_input(batch: Any, input_ids: torch.Tensor) -> Any:
    import dataclasses

    return dataclasses.replace(batch, input_ids=input_ids)


def generation_identity(model: MedDecideModel, plain_model: nn.Module, prompts: Sequence[str]) -> dict[str, Any]:
    """Greedy generation with the adapter off against an untouched base, plus a control: the adapter on must change
    the next-token logits, so the comparison can detect a changed adapter."""
    device = next(plain_model.parameters()).device
    with model.adapter_disabled():
        adapter_off = model.generate_greedy(list(prompts), max_new_tokens=PROMPT_TOKENS_GENERATED)
    plain_out: list[list[int]] = []
    for prompt in prompts:
        ids = model.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        with torch.no_grad():
            generated = plain_model.generate(ids, max_new_tokens=PROMPT_TOKENS_GENERATED, do_sample=False)
        plain_out.append(generated[0, ids.shape[1]:].tolist())
    ids = model.tokenizer(prompts[0], return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
    with torch.no_grad():
        on = model.peft_model(input_ids=ids, use_cache=False).logits[0, -1].float()
        with model.adapter_disabled():
            off = model.peft_model(input_ids=ids, use_cache=False).logits[0, -1].float()
    return {"byte_identical": adapter_off == plain_out, "n_prompts": len(prompts),
            "control_adapter_logits_max_abs_diff": float((on - off).abs().max())}


def prompts_for(model: MedDecideModel, items: Sequence[Item]) -> list[str]:
    return [render_prompt(item, model.tokenizer, model.variant) for item in items]
