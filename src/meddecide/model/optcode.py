"""Option-code readout (D23): one ``[K, d]`` matrix of code rows, initialised from the LM head.

The readout reads the final hidden state at the answer position, the state the letter readout reads, and returns one
logit per offered option: the dot product of that state with the row of the option's code. Rows start as copies of the
LM-head rows of their code tokens. At initialisation the readout therefore equals the zero-shot letter readout, and the
softmax over the offered codes equals the letter readout renormalised over the offered letters. After training the rows
diverge, and :func:`exported_lm_head` writes them back into a copy of the LM head, so a plain causal LM scores the codes.

Codes are A-Z, then the two-letter uppercase codes the tokenizer encodes as one token, in alphabetical order. K = 255
covers the largest allowed choice item.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch
from torch import Tensor, nn

from meddecide.model.head import segment_softmax

K = 255
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def code_names(k: int, single_token: Callable[[str], bool]) -> list[str]:
    """The first ``k`` codes: A-Z, then the two-letter codes that ``single_token`` accepts, in order.

    Raises if the tokenizer offers fewer than ``k`` codes; the O5 check reports the count it found.
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    names = list(LETTERS[: min(k, 26)])
    if not all(single_token(c) for c in names):
        raise ValueError("a single letter is not one token in this tokenizer")
    for first in LETTERS:
        for second in LETTERS:
            if len(names) >= k:
                break
            code = first + second
            if single_token(code):
                names.append(code)
    if len(names) < k:
        raise ValueError(f"only {len(names)} single-token codes are available, {k} requested")
    return names[:k]


def code_token_ids(names: Sequence[str], encode: Callable[[str], list[int]]) -> list[int]:
    """Token id of each code; every code must be exactly one token."""
    ids: list[int] = []
    for name in names:
        encoded = encode(name)
        if len(encoded) != 1:
            raise ValueError(f"code {name!r} is not one token ({encoded})")
        ids.append(int(encoded[0]))
    if len(set(ids)) != len(ids):
        raise ValueError("two codes map to the same token")
    return ids


class OptionCodeHead(nn.Module):
    """A trainable ``[K, d]`` readout over the option codes.

    ``weight`` is kept in float32 whatever the base dtype: it is small, and its logits are the model output.
    ``token_ids`` records, for each code row, the LM-head row it was copied from (and is written back to on export).
    """

    def __init__(self, d_model: int, k: int = K) -> None:
        super().__init__()
        if d_model < 1 or k < 1:
            raise ValueError("d_model and k must be >= 1")
        self.d_model = int(d_model)
        self.k = int(k)
        self.weight = nn.Parameter(torch.zeros(self.k, self.d_model, dtype=torch.float32))
        self.register_buffer("token_ids", torch.zeros(self.k, dtype=torch.long))

    @torch.no_grad()
    def init_from_lm_head(self, lm_head_weight: Tensor, token_ids: Sequence[int]) -> None:
        """Copy the LM-head rows of the code tokens into the head (one row per code)."""
        if len(token_ids) != self.k:
            raise ValueError(f"expected {self.k} token ids, got {len(token_ids)}")
        index = torch.as_tensor(list(token_ids), dtype=torch.long, device=lm_head_weight.device)
        rows = lm_head_weight.detach()[index].to(torch.float32)
        self.weight.copy_(rows.to(self.weight.device))
        self.token_ids.copy_(index.to(self.token_ids.device))

    def forward(self, answer_hidden: Tensor, marker_item: Tensor, code_index: Tensor) -> tuple[Tensor, Tensor]:
        """Flat logits and probabilities over exactly the offered options.

        ``answer_hidden`` is ``(batch, d)``, the final hidden state at each item's answer position. ``marker_item`` maps
        each flat option to its item; ``code_index`` gives each flat option's code row (its letter position).
        """
        if answer_hidden.ndim != 2 or answer_hidden.shape[1] != self.d_model:
            raise ValueError("answer_hidden must be (batch, d_model)")
        if marker_item.shape != code_index.shape or marker_item.ndim != 1:
            raise ValueError("marker_item and code_index must be flat and equal-length")
        if code_index.numel() and int(code_index.max()) >= self.k:
            raise ValueError("an item offers more options than the head has codes")
        h = answer_hidden.float()[marker_item.to(answer_hidden.device)]
        w = self.weight[code_index.to(self.weight.device)]
        flat = (h * w).sum(dim=-1)
        probs = segment_softmax(flat, marker_item.to(flat.device), int(answer_hidden.shape[0]))
        return flat, probs


@torch.no_grad()
def exported_lm_head(lm_head_weight: Tensor, head: OptionCodeHead) -> Tensor:
    """A copy of the LM-head matrix in which only the code-token rows are replaced by the trained head rows.

    Every other row is unchanged, so a causal LM with this matrix as its LM head reads the same codes as the native
    readout. The head rows are cast to the matrix's dtype, so a bfloat16 export rounds them.
    """
    out = lm_head_weight.detach().clone()
    index = head.token_ids.to(out.device)
    out[index] = head.weight.detach().to(device=out.device, dtype=out.dtype)
    return out
