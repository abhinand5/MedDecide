"""Non-causal full attention on the decision path (arm N, D23).

Inside :func:`full_attention_bidirectional`, the full-attention layers of the base model attend in both directions, with
padding keys still excluded. The linear-attention layers are unchanged (their mask is created by the model as before),
and outside the context, or with ``enabled=False``, nothing is patched, so the causal path is the untouched original.

The patch replaces the mask factory that the Qwen3.5 text model calls for its full-attention layers. The model's
attention module then receives a mask and skips its causal rule (``is_causal`` applies only when no mask is given).
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import torch
from torch import Tensor


def bidirectional_mask(padding_mask: Tensor | None, batch: int, seq: int, device: torch.device) -> Tensor:
    """A ``(batch, 1, seq, seq)`` boolean mask: every query attends to every non-padding key.

    ``padding_mask`` is the 2-D attention mask (1 for real tokens); ``None`` means no padding.
    """
    if padding_mask is None:
        return torch.ones(batch, 1, seq, seq, dtype=torch.bool, device=device)
    keys = padding_mask.to(device=device, dtype=torch.bool)
    if keys.shape != (batch, seq):
        raise ValueError(f"padding mask {tuple(keys.shape)} does not match (batch, seq) = ({batch}, {seq})")
    return keys[:, None, None, :].expand(batch, 1, seq, seq)


@contextmanager
def full_attention_bidirectional(enabled: bool) -> Iterator[None]:
    """Make the full-attention layers bidirectional for the duration of the block (no-op when ``enabled`` is False)."""
    if not enabled:
        yield
        return
    # imported here, not at module load: importing the Qwen3.5 modelling file binds its kernel decorators
    import transformers.models.qwen3_5.modeling_qwen3_5 as qwen3_5

    original = qwen3_5.create_causal_mask

    def non_causal_mask(**kwargs: object) -> Tensor:
        embeds = kwargs["inputs_embeds"]
        if not isinstance(embeds, Tensor):
            raise TypeError("inputs_embeds must be a tensor")
        batch, seq = int(embeds.shape[0]), int(embeds.shape[1])
        padding = kwargs.get("attention_mask")
        if padding is not None and not isinstance(padding, Tensor):
            raise TypeError("attention_mask must be a tensor or None")
        return bidirectional_mask(padding, batch, seq, embeds.device)

    qwen3_5.create_causal_mask = non_causal_mask
    try:
        yield
    finally:
        qwen3_5.create_causal_mask = original
