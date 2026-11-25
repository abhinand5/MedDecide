"""Clef-Flash's text tower as a causal LM (distill_v0 D3/D5).

Clef-Flash (`Cloudflare/clef-flash`) is a Qwen3.5 vision-language checkpoint
(`Qwen3_5ForConditionalGeneration`) trained by Cloudflare with its own joint schema head. Osler-9B
needs only its **text** tower with our option-code readout, so this module exposes the tower as a
causal LM without writing a converted copy of the 18 GB checkpoint (the workspace is quota-bound and
the copy would only duplicate weights already in the HF cache):

- load the full conditional-generation model (device, dtype as requested),
- wrap ``model.language_model`` and ``lm_head`` in :class:`ClefTextBackbone`, which presents the
  interface ``MedDecideModel`` expects from ``AutoModelForCausalLM`` (``.model`` decoder,
  ``.lm_head``, ``.config``, a ``logits`` output, gradient-checkpointing delegation),
- drop the vision tower (counted and reported by the caller).

The readout is initialised from ``lm_head``'s rows exactly as on the Qwen3.5-9B fallback.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from transformers.modeling_outputs import CausalLMOutputWithPast

REPO = "Cloudflare/clef-flash"
REVISION = "fde727a287004204b7518dcc983fe64379776712"


class ClefTextBackbone(nn.Module):
    """The language tower of a Qwen3.5 conditional-generation model, exposed as a causal LM."""

    base_model_prefix = "model"  # PEFT's conversion mapping reads this
    supports_gradient_checkpointing = True
    is_parallelizable = False

    def __init__(self, language_model: nn.Module, lm_head: nn.Module) -> None:
        super().__init__()
        self.model = language_model
        self.lm_head = lm_head
        self.config = language_model.config

    def __getattr__(self, name: str) -> Any:
        """Fall through to the language tower for the PreTrainedModel surface PEFT and generation probe."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            model = self._modules.get("model") if "_modules" in self.__dict__ else None
            if model is not None and hasattr(model, name):
                return getattr(model, name)
            raise

    def forward(self, *args: Any, logits_to_keep: int | None = None, **kwargs: Any) -> CausalLMOutputWithPast:
        """A causal-LM forward over the text tower: LM-head logits (optionally only the last token) and the tower's hidden states.

        ``hidden_states`` is passed through unchanged, because ``MedDecideModel.hidden_states`` reads
        ``output.hidden_states[-1]`` for the generic Qwen path and the Clef tower uses the same convention.
        """
        want_hidden = bool(kwargs.pop("output_hidden_states", False))
        outputs = self.model(*args, output_hidden_states=want_hidden, **kwargs)
        hidden: Tensor = outputs.last_hidden_state
        logits: Tensor = self.lm_head(hidden if logits_to_keep is None else hidden[:, -int(logits_to_keep):])
        return CausalLMOutputWithPast(logits=logits,
                                      hidden_states=outputs.hidden_states if want_hidden else None)

    def prepare_inputs_for_generation(self, input_ids: Tensor, past_key_values: Any = None,
                                      attention_mask: Tensor | None = None, **kwargs: Any) -> dict[str, Any]:
        """PEFT reads this attribute when it wraps a CAUSAL_LM; the student never generates, so it is a pass-through."""
        inputs: dict[str, Any] = {"input_ids": input_ids, "past_key_values": past_key_values,
                                  "attention_mask": attention_mask}
        inputs.update({key: value for key, value in kwargs.items() if value is not None})
        return inputs

    def gradient_checkpointing_enable(self, **kwargs: Any) -> None:
        self.model.gradient_checkpointing_enable(**kwargs)

    def enable_input_require_grads(self) -> None:
        self.model.enable_input_require_grads()

    def resize_token_embeddings(self, *args: Any, **kwargs: Any) -> Any:  # PEFT probes this on some paths
        return self.model.resize_token_embeddings(*args, **kwargs)

    def get_input_embeddings(self) -> nn.Module:
        return self.model.get_input_embeddings()

    def get_output_embeddings(self) -> nn.Module:
        return self.lm_head


def load_clef_text_backbone(*, dtype: str = "bfloat16", device: str = "cuda:0",
                            repo: str = REPO, revision: str | None = REVISION,
                            local_files_only: bool = True) -> tuple[ClefTextBackbone, dict[str, Any]]:
    """Load the pinned Clef snapshot and return its text tower as a causal LM, with what was dropped."""
    from transformers import Qwen3_5ForConditionalGeneration

    torch_dtype = getattr(torch, dtype)
    full = Qwen3_5ForConditionalGeneration.from_pretrained(
        repo, revision=revision, dtype=torch_dtype, device_map=device,
        attn_implementation="sdpa", local_files_only=local_files_only)
    language_model = full.model.language_model
    lm_head = full.lm_head
    visual_parameters = sum(parameter.numel() for parameter in full.model.visual.parameters()) \
        if getattr(full.model, "visual", None) is not None else 0
    backbone = ClefTextBackbone(language_model, lm_head)
    info = {"repo": repo, "revision": revision, "dropped_visual_parameters": int(visual_parameters),
            "dropped_visual_tensors": len(list(full.model.visual.state_dict())) if getattr(full.model, "visual", None) else 0,
            "text_hidden_size": int(language_model.config.hidden_size),
            "text_layers": int(language_model.config.num_hidden_layers),
            "vocab_size": int(language_model.config.vocab_size)}
    if getattr(full.model, "visual", None) is not None:
        full.model.visual = None  # the vision tower is not part of the student; release it
    del full
    torch.cuda.empty_cache()
    return backbone, info


def checkpoint_files(repo: str = REPO, revision: str | None = REVISION) -> list[Path]:
    """The cached snapshot's weight files (for the provenance record); empty when not cached."""
    from huggingface_hub import snapshot_download

    try:
        snapshot = Path(snapshot_download(repo, revision=revision, local_files_only=True))
    except Exception:
        return []
    return sorted(snapshot.glob("*.safetensors"))
