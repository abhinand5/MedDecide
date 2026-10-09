"""Pointer head: score each offered option from its own hidden state (D3).

For each offered option the head reads hidden states produced by the frozen base (plus the
decision-path LoRA):

* the state at that option's **key token** inside the rendered prompt (found by
  :mod:`meddecide.model.markers`) — the option's marker;
* optionally the state at the **end of that option's line**, the first causal position whose
  hidden state contains the option's own label (see below); and
* the state at the **answer position** (the final prompt token, where the harness reads
  next-token logits), which carries the record and the question.

They are concatenated and passed through a small MLP to one logit per option, and the softmax
is taken over **exactly the offered options** of that item. There is no fixed output
vocabulary anywhere: option count is unbounded by construction (2-64 tested, 255 allowed by
the schema), and an option's score never depends on other items or on a letter table.

``HeadSettings.option_state`` selects which option states are concatenated:

``key_end`` (default)
    ``[key ; line-end ; answer]``. The base is causal: at the key token ``A`` of
    ``A. Ampicillin`` the hidden state has *not* read ``Ampicillin``. With only the key state
    the head cannot know what its own option says — measured on the S7 overfit test, training
    accuracy stalled at 0.36/64 items. Adding the line-end state is what makes the option's
    content visible to the head while keeping the key token as the option's anchor.
``key``
    ``[key ; answer]`` — the literal marker-only form, kept selectable so the difference is
    measurable rather than asserted.
``end``
    ``[line-end ; answer]`` — no key state at all.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import torch
from torch import Tensor, nn

OptionState = Literal["key", "key_end", "end"]
OPTION_STATES: tuple[str, ...] = ("key", "key_end", "end")


@dataclass(frozen=True)
class HeadSettings:
    """Pointer-head shape and which option hidden states it reads."""

    hidden: int = 256
    dropout: float = 0.0
    option_state: OptionState = "key_end"

    def __post_init__(self) -> None:
        if self.option_state not in OPTION_STATES:
            raise ValueError(
                f"option_state must be one of {OPTION_STATES}, got {self.option_state!r}"
            )
        if self.hidden < 1:
            raise ValueError("head hidden width must be >= 1")

    @property
    def n_option_states(self) -> int:
        return 2 if self.option_state == "key_end" else 1

    def input_width(self, d_model: int) -> int:
        """Width of the concatenated per-option vector."""
        return (self.n_option_states + 1) * d_model

    def to_dict(self) -> dict[str, Any]:
        return {
            "hidden": self.hidden,
            "dropout": self.dropout,
            "option_state": self.option_state,
        }


def segment_softmax(logits: Tensor, item_index: Tensor, n_items: int) -> Tensor:
    """Softmax of ``logits`` within each item's block of offered options.

    ``logits`` is flat ``(n_options_total,)`` and ``item_index`` maps each position to its
    item. The result is non-negative and sums to 1 **per item**; positions of different items
    never compete (that is the whole point of the pointer head).
    """
    if logits.ndim != 1:
        raise ValueError("logits must be 1-D (one entry per offered option)")
    if item_index.shape != logits.shape:
        raise ValueError("item_index must have one entry per option")
    if n_items < 1:
        raise ValueError("n_items must be >= 1")
    neg_inf = torch.full((n_items,), float("-inf"), device=logits.device, dtype=logits.dtype)
    # the max is a numerical-stability constant, not something we differentiate through
    max_per_item = neg_inf.scatter_reduce(0, item_index, logits.detach(), reduce="amax",
                                          include_self=False)
    shifted = logits - max_per_item[item_index]
    exp = torch.exp(shifted)
    denom = torch.zeros(n_items, device=logits.device, dtype=logits.dtype)
    denom = denom.index_add(0, item_index, exp)
    if not torch.all(denom > 0):
        raise ValueError("an item received no option logits")
    return exp / denom[item_index]


class PointerHead(nn.Module):
    """One logit per offered option from the option's own hidden state and the answer state.

    ``forward`` takes the hidden states of a whole padded batch plus flat index tensors, so a
    batch may mix items with different option counts (2 and 64 in the same batch is fine).
    """

    def __init__(self, d_model: int, settings: HeadSettings | None = None) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.settings = settings or HeadSettings()
        width = self.settings.input_width(self.d_model)
        self.norm = nn.LayerNorm(width)
        self.mlp = nn.Sequential(
            nn.Linear(width, self.settings.hidden),
            nn.GELU(),
            nn.Dropout(self.settings.dropout),
            nn.Linear(self.settings.hidden, 1),
        )

    def forward(
        self,
        hidden_states: Tensor,
        marker_positions: Tensor,
        marker_item: Tensor,
        answer_positions: Tensor,
        option_end_positions: Tensor | None = None,
    ) -> Tensor:
        """Return one logit per marker (flat, in the order of the marker tensors).

        ``hidden_states`` is ``(batch, seq, d_model)``; ``marker_positions``,
        ``marker_item`` and (when used) ``option_end_positions`` are ``(n_markers,)``;
        ``answer_positions`` is ``(batch,)``.
        """
        if hidden_states.ndim != 3:
            raise ValueError("hidden_states must be (batch, seq, d_model)")
        if marker_positions.ndim != 1 or marker_item.shape != marker_positions.shape:
            raise ValueError("marker_positions and marker_item must be flat and equal-length")
        parts = []
        if self.settings.option_state in ("key", "key_end"):
            parts.append(hidden_states[marker_item, marker_positions].float())
        if self.settings.option_state in ("end", "key_end"):
            if option_end_positions is None:
                raise ValueError(
                    f"option_state={self.settings.option_state!r} needs option_end_positions"
                )
            parts.append(hidden_states[marker_item, option_end_positions].float())
        parts.append(hidden_states[marker_item, answer_positions[marker_item]].float())
        # the head is trained in float32 regardless of the base dtype: it is well under a
        # million parameters and its logits are the model's output
        combined = torch.cat(parts, dim=-1)
        return self.mlp(self.norm(combined)).squeeze(-1)

    def option_logits(
        self,
        hidden_states: Tensor,
        marker_positions: Tensor,
        marker_item: Tensor,
        answer_positions: Tensor,
        n_items: int,
        option_end_positions: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Return ``(flat logits, flat probabilities)`` over exactly the offered options."""
        logits = self(
            hidden_states,
            marker_positions,
            marker_item,
            answer_positions,
            option_end_positions,
        )
        return logits, segment_softmax(logits, marker_item, n_items)
