"""Zero-shot verbalizer readout: score option letters from one prefill pass (T6).

The model renders an item through its own chat template with thinking disabled, and the
probabilities of the option-letter tokens at the answer position are read from the next
token distribution. No text is generated at eval time.

Design notes that matter for correctness (ADVISORY failure mode 3):

* every option letter must be a **single token** for the tokenizer, checked loudly;
* whether the model expects ``" A"`` (leading space) or ``"A"`` is *measured* per model
  and recorded, never assumed;
* thinking must be disabled (``enable_thinking=False``) and any think block the template
  forces must be closed before the answer position.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from meddecide.bench.schema import Item, QuestionType

SYSTEM_PROMPT = (
    "You are a medical decision model. Answer with the letter of the correct option only."
)

LetterVariant = Literal["space", "bare"]

# 255 options is the schema maximum; option keys are A..Z then AA..IV style only if a
# source ever needs it, so the letter table is generated rather than hardcoded.
def option_letter(index: int) -> str:
    """0-based option index → option letter (A..Z, then AA, AB, …)."""
    if index < 0:
        raise ValueError("option index must be >= 0")
    letters = ""
    n = index
    while True:
        n, rem = divmod(n, 26)
        letters = chr(ord("A") + rem) + letters
        if n == 0:
            return letters


@dataclass
class LabelTokenCheck:
    """Result of the single-token check for every option letter of a model."""

    model_id: str
    variant: LetterVariant
    letters: list[str]
    token_ids: list[int]
    all_single_token: bool
    offenders: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "variant": self.variant,
            "n_letters": len(self.letters),
            "all_single_token": self.all_single_token,
            "offenders": self.offenders,
        }


def check_label_tokens(
    tokenizer: Any,
    model_id: str,
    *,
    n_options: int = 32,
    variants: Sequence[LetterVariant] = ("space", "bare"),
) -> list[LabelTokenCheck]:
    """Assert each option letter is a single token; try ``" A"`` and ``"A"`` variants.

    Returns one report per variant. ``all_single_token`` must be true for the variant the
    harness uses — otherwise the readout is meaningless and the caller must fail loudly.
    """
    checks: list[LabelTokenCheck] = []
    letters = [option_letter(i) for i in range(n_options)]
    for variant in variants:
        token_ids: list[int] = []
        offenders: list[dict[str, Any]] = []
        for letter in letters:
            text = f" {letter}" if variant == "space" else letter
            ids = tokenizer.encode(text, add_special_tokens=False)
            if len(ids) != 1:
                offenders.append({"letter": letter, "text": text, "token_ids": ids,
                                  "pieces": tokenizer.convert_ids_to_tokens(ids)})
            token_ids.append(ids[0] if ids else -1)
        checks.append(
            LabelTokenCheck(
                model_id=model_id,
                variant=variant,
                letters=letters,
                token_ids=token_ids,
                all_single_token=not offenders,
                offenders=offenders,
            )
        )
    return checks


def pick_variant(checks: Sequence[LabelTokenCheck]) -> LabelTokenCheck:
    """Prefer the ``space`` variant (chat models are trained with a leading space)."""
    ok = [c for c in checks if c.all_single_token]
    if not ok:
        detail = "; ".join(
            f"{c.variant}: {len(c.offenders)} multi-token letters" for c in checks
        )
        raise ValueError(f"no option-letter variant is single-token for this tokenizer ({detail})")
    for c in ok:
        if c.variant == "space":
            return c
    return ok[0]


def render_prompt(item: Item, tokenizer: Any, variant: LetterVariant = "space") -> str:
    """Render one item through the model's chat template, thinking disabled.

    The prompt ends at the assistant's answer position, so the next-token distribution is
    the distribution over options. ``tokenizer.apply_chat_template`` is given
    ``add_generation_prompt=True`` and ``enable_thinking=False`` where the template
    accepts it; templates that do not accept it are called without it.
    """
    options_block = "\n".join(f"{opt.key}. {opt.label}" for opt in item.options)
    user = f"{item.state}\n\n{item.question}\n\nOptions:\n{options_block}\n\nAnswer:"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
    kwargs: dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        # Template does not take enable_thinking; ask again without it.
        return tokenizer.apply_chat_template(messages, **kwargs)


def option_token_ids(item: Item, tokenizer: Any, variant: LetterVariant = "space") -> list[int]:
    """Token ids of the option letters for this item, in option order."""
    ids: list[int] = []
    for opt in item.options:
        text = f" {opt.key}" if variant == "space" else opt.key
        encoded = tokenizer.encode(text, add_special_tokens=False)
        if len(encoded) != 1:
            raise ValueError(f"option key {opt.key!r} is not a single token ({encoded})")
        ids.append(encoded[0])
    return ids


def read_option_probabilities(
    logits: np.ndarray,
    token_ids: Sequence[int],
) -> dict[str, Any]:
    """Softmax over the option-letter logits, plus the full-vocabulary label mass.

    ``logits`` is the next-token logit vector (vocab,). Returns option probabilities
    (normalised over options), the raw per-option token probability (label mass), and the
    softmax denominator used, so the number can be recomputed.
    """
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim != 1:
        raise ValueError("logits must be 1-D (vocab,)")
    if not token_ids:
        raise ValueError("no option token ids given")
    option_logits = logits[np.asarray(token_ids, dtype=np.int64)]
    shifted = option_logits - option_logits.max()
    exp = np.exp(shifted)
    option_probs = exp / exp.sum()
    # full-vocabulary softmax, for the label-mass diagnostic
    full_shifted = logits - logits.max()
    full = np.exp(full_shifted)
    full /= full.sum()
    return {
        "option_probs": option_probs,
        "option_token_mass": full[np.asarray(token_ids, dtype=np.int64)],
        "label_mass": float(full[np.asarray(token_ids, dtype=np.int64)].sum()),
        "option_logits": option_logits,
    }


def expected_level(option_probs: Sequence[float]) -> float:
    """Expected score level (1-based) for a ``score`` item: ``sum_k k * p_k``."""
    probs = np.asarray(option_probs, dtype=np.float64)
    levels = np.arange(1, probs.size + 1, dtype=np.float64)
    return float(np.sum(levels * probs))


def is_score_item(item: Item) -> bool:
    return item.qtype is QuestionType.SCORE
