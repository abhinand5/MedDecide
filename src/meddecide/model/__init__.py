"""MedDecide model package: frozen base + decision-path LoRA + pointer head (D3)."""

from meddecide.model.head import HeadSettings, PointerHead, segment_softmax
from meddecide.model.markers import PromptMarkers, locate_option_markers, options_block
from meddecide.model.meddecide_model import (
    DEFAULT_LORA,
    DEFAULT_LORA_TARGETS,
    Decision,
    EncodedItem,
    LoraSettings,
    MedDecideModel,
    ModelBatch,
    ScoredItems,
    render_item_prompt,
    reorder_options,
)

__all__ = [
    "DEFAULT_LORA",
    "DEFAULT_LORA_TARGETS",
    "Decision",
    "EncodedItem",
    "HeadSettings",
    "LoraSettings",
    "MedDecideModel",
    "ModelBatch",
    "PointerHead",
    "PromptMarkers",
    "ScoredItems",
    "locate_option_markers",
    "options_block",
    "render_item_prompt",
    "reorder_options",
    "segment_softmax",
]
