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
    "You are a medical decision model. Answer with a single letter only, no explanation."
)
# The user turn ends with an explicit instruction to emit one letter. The first version of
# this prompt ended with a bare "Answer:" and the model continued the question's *subject
# matter* instead of choosing ("Answer: A 67-year-old ..."), which put the argmax on 'A'
# regardless of content: 22 % accuracy with a 79 % 'A' rate on 20 MedQA items. With the
# instruction below the same model scored 55 % on those items with a spread argmax. The
# prompt is part of the fixed T6 protocol for every model, so it is recorded here and in
# each run's config.
SINGLE_LETTER_INSTRUCTION = (
    "Respond with a single letter: A, B, C, D, ... for the correct option. "
    "The correct option is:"
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
    """Pick a variant that is single-token for every option letter.

    This only guarantees the readout is *possible*; which variant the model actually emits is
    a separate, empirical question (:func:`detect_variant`). Preference order here is
    ``bare`` first because a chat model that answers with a letter usually writes the letter
    itself, not a space-prefixed continuation.
    """
    ok = [c for c in checks if c.all_single_token]
    if not ok:
        detail = "; ".join(
            f"{c.variant}: {len(c.offenders)} multi-token letters" for c in checks
        )
        raise ValueError(f"no option-letter variant is single-token for this tokenizer ({detail})")
    for c in ok:
        if c.variant == "bare":
            return c
    return ok[0]


PROBE_ITEM = {
    "state": (
        "A 54-year-old patient with type 2 diabetes and hypertension presents with polyuria "
        "and polydipsia."
    ),
    "question": "Which laboratory test is most appropriate first?",
    "options": [
        ("A", "Fasting plasma glucose"),
        ("B", "Serum sodium"),
        ("C", "Serum calcium"),
        ("D", "Thyroid stimulating hormone"),
    ],
}


def probe_prompt(tokenizer: Any) -> str:
    """A fixed synthetic multiple-choice prompt used only to detect the answer variant."""
    item = _probe_item()
    return render_prompt(item, tokenizer, "bare")


def _probe_item() -> Item:
    from meddecide.bench.schema import Tier, make_item

    return make_item(
        tier=Tier.FRESH,
        source="variant_probe",
        source_record_id="probe",
        source_url="https://example.org/probe",
        source_license="synthetic",
        record_date="2026-01-01",
        split="test",
        template_id="variant_probe_v1",
        skill="probe",
        qtype=QuestionType.CHOICE,
        state=PROBE_ITEM["state"],
        question=PROBE_ITEM["question"],
        options=[{"key": k, "label": v} for k, v in PROBE_ITEM["options"]],
        gold="A",
        option_order_seed=0,
    )


def detect_variant(model: Any, tokenizer: Any, checks: Sequence[LabelTokenCheck]) -> tuple[LetterVariant, dict]:
    """Measure which variant the model actually emits, by greedy generation on a fixed probe.

    Guessing is not acceptable here: on Qwen3.5-0.8B the ``" A"`` token (id 357) sits ~12
    logits below the bare ``A`` token (id 32) at the answer position, while both are valid
    single tokens — a harness that preferred ``" A"`` would read the wrong distribution
    entirely. The probe is generated greedily (1 token) and the variant whose letter token is
    produced wins; ties or non-letter output fall back to ``pick_variant`` and the fallback is
    recorded.
    """
    import torch

    usable = {c.variant: c for c in checks if c.all_single_token}
    if not usable:
        raise ValueError("no single-token variant to detect")
    prompt = probe_prompt(tokenizer)
    encoded = tokenizer(prompt, return_tensors="pt")
    device = getattr(model, "device", None)
    if device is not None:
        encoded = {k: v.to(device) for k, v in encoded.items()}
    with torch.inference_mode():
        generated = model.generate(**encoded, max_new_tokens=1, do_sample=False)
    new_tokens = generated[0, encoded["input_ids"].shape[1] :]
    token_id = int(new_tokens[0]) if len(new_tokens) else -1
    detail = {"probe_token_id": token_id, "probe_token": tokenizer.convert_ids_to_tokens([token_id])[0]
              if token_id >= 0 else None}
    for variant, check in usable.items():
        # the probe prompt lists options A-D first, so any of the first four letters counts
        if token_id in set(check.token_ids[:4]):
            detail["detected"] = variant
            detail["method"] = "greedy generation on a fixed synthetic probe prompt"
            return variant, detail
    fallback = pick_variant(checks).variant
    detail["detected"] = fallback
    detail["method"] = "fallback (probe produced a non-letter token)"
    detail["usable_variants"] = sorted(usable)
    return fallback, detail


def render_prompt(item: Item, tokenizer: Any, variant: LetterVariant = "space") -> str:
    """Render one item through the model's chat template, thinking disabled.

    The prompt ends at the assistant's answer position, so the next-token distribution is
    the distribution over options. ``tokenizer.apply_chat_template`` is given
    ``add_generation_prompt=True`` and ``enable_thinking=False`` where the template
    accepts it; templates that do not accept it are called without it.
    """
    options_block = "\n".join(f"{opt.key}. {opt.label}" for opt in item.options)
    user = (
        f"{item.state}\n\n{item.question}\n\nOptions:\n{options_block}\n\n"
        f"{SINGLE_LETTER_INSTRUCTION}"
    )
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
    return [key_token_id(opt.key, tokenizer, variant) for opt in item.options]


def key_token_id(key: str, tokenizer: Any, variant: LetterVariant = "space") -> int:
    """Token id of one option key, cheap enough to call per key (the tokenizer caches)."""
    text = f" {key}" if variant == "space" else key
    encoded = tokenizer.encode(text, add_special_tokens=False)
    if len(encoded) != 1:
        raise ValueError(f"option key {key!r} is not a single token ({encoded})")
    return encoded[0]


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
    option_ids = np.asarray(token_ids, dtype=np.int64)
    option_token_mass = full[option_ids]
    # ``label_mass`` is the raw full-vocabulary mass of the option tokens. On models whose
    # next-token distribution is spread over tens of thousands of near-tied tokens this is
    # tiny even when the argmax is a clean option letter (measured: 7e-07 for Qwen3.5-0.8B
    # on MedQA, while the argmax was the letter the model generates greedily). Two further
    # diagnostics are therefore reported: whether the full-vocabulary argmax *is* one of the
    # option tokens (`vocab_argmax_is_option`), and the probability of the top option within
    # the option softmax (`top1_over_option_mass`).
    option_id_set = {int(t) for t in option_ids}
    # How many *strictly larger* logits the best option token has above it. This is a
    # tie-robust rank: on Qwen3.5-0.8B the next-token logits are so tightly packed that dozens
    # of tokens share a rounded value, and `argsort` position (the obvious implementation)
    # varies run to run for the same prompt — it reported rank 125 for letters that a direct
    # top-6 inspection showed at positions 1-4. Counting strict comparisons is stable and is
    # what "is the option letter among the model's most likely next tokens" actually means.
    best_option_logit = float(option_logits.max())
    n_strictly_higher = int(np.sum(logits > best_option_logit))
    return {
        "option_probs": option_probs,
        "option_token_mass": option_token_mass,
        "label_mass": float(option_token_mass.sum()),
        "vocab_argmax_is_option": bool(int(full.argmax()) in option_id_set),
        "n_tokens_above_best_option": n_strictly_higher,
        "best_option_in_top5": bool(n_strictly_higher < 5),
        "best_option_in_top50": bool(n_strictly_higher < 50),
        "top1_over_option_mass": float(option_probs.max()),
        "option_logits": option_logits,
    }


def expected_level(option_probs: Sequence[float]) -> float:
    """Expected score level (1-based) for a ``score`` item: ``sum_k k * p_k``."""
    probs = np.asarray(option_probs, dtype=np.float64)
    levels = np.arange(1, probs.size + 1, dtype=np.float64)
    return float(np.sum(levels * probs))


def is_score_item(item: Item) -> bool:
    return item.qtype is QuestionType.SCORE
