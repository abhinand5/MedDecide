"""Locate each option's key token and the answer position inside a rendered prompt.

The prompt text itself comes from :func:`meddecide.eval.readout.render_prompt` — the same
renderer the zero-shot harness uses — so the pointer head reads the architecture's existing
rendering and no second prompt format exists to drift away from it. What this module adds is
the *position* of each option key token inside that prompt, plus the answer position (the
final prompt token, where the harness reads next-token logits).

Method. The options block is rebuilt exactly as ``render_prompt`` builds it
(``A. <label>`` lines), located in the prompt, and each option key's character offset is
mapped to the token that covers it through the fast tokenizer's offset mapping. No token
sequence is guessed and no token id is assumed: the located token id is compared with
:func:`meddecide.eval.readout.key_token_id` for the harness's letter variant, and a
disagreement is surfaced (a readout that scores a different token than the prompt shows is a
correctness bug, not a detail).

Two shapes occur in practice and both are recorded rather than hidden:

* the key is a single token for this tokenizer (``A``..``Z``; every real MedDecide item has at
  most 17 options) — the normal case;
* the key is multi-token or merged with the preceding newline (``option_letter`` emits ``BQ``,
  ``BZ``, ``CJ`` for some high option counts) — the marker is then the token covering the
  key's first character, and the anomaly is flagged.

Why an option's **line-end** position is also located: the base model is causal, so the hidden
state at ``A`` in ``A. Ampicillin`` has not read ``Ampicillin`` — it has only read the options
before it. A head that reads only the key-token state cannot know what its own option says
(measured: 0.36 accuracy when asked to overfit 64 items). ``end_positions`` is the last token
of each option's line, the first position whose hidden state contains that option's own label;
:class:`meddecide.model.head.HeadSettings` decides which of the two the head consumes.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass, field
from typing import Any

from meddecide.bench.schema import Item
from meddecide.eval.readout import (
    SINGLE_LETTER_INSTRUCTION,
    LetterVariant,
    key_token_id,
    render_prompt,
)

OPTIONS_HEADER = "Options:\n"
# the block is followed by a blank line and the fixed instruction; checking a prefix of it
# stops a state that happens to quote an options block from being mistaken for the real one
_BLOCK_SUFFIX = "\n\n" + SINGLE_LETTER_INSTRUCTION[:20]


def options_block(item: Item) -> str:
    """The option lines exactly as :func:`render_prompt` writes them."""
    return "\n".join(f"{opt.key}. {opt.label}" for opt in item.options)


@dataclass(frozen=True)
class PromptMarkers:
    """Token positions of one rendered prompt: per-option markers and the answer position.

    ``input_ids`` are the ids of the tokenization that produced the offsets, so callers use
    them directly instead of tokenizing again (a second call could in principle differ).
    """

    prompt: str
    input_ids: list[int]
    option_keys: list[str]
    positions: list[int]
    marker_token_ids: list[int]
    # position of the last token of each option's line. In a causal model this is the first
    # position whose hidden state has read *this option's own label*; the key token's state has
    # not (see the module docstring and ``HeadSettings.option_state``).
    end_positions: list[int] = field(default_factory=list)
    # the harness's own key token for this option, when the key is a single token for this
    # tokenizer; None when it is not
    harness_key_token_ids: list[int | None] = field(default_factory=list)
    single_token_key: list[bool] = field(default_factory=list)
    # the token covering the key's first character also covers text before it (a BPE merge
    # across the newline), so the marker is not exactly the key token
    merged_with_prefix: list[bool] = field(default_factory=list)

    @property
    def n_options(self) -> int:
        return len(self.positions)

    @property
    def answer_position(self) -> int:
        """Index of the final prompt token, where the harness reads next-token logits."""
        return len(self.input_ids) - 1

    @property
    def mismatches(self) -> list[dict[str, Any]]:
        """Options whose located marker token is not the harness's key token."""
        out: list[dict[str, Any]] = []
        for i, key in enumerate(self.option_keys):
            expected = self.harness_key_token_ids[i]
            if expected is not None and expected != self.marker_token_ids[i]:
                out.append(
                    {
                        "option_key": key,
                        "index": i,
                        "marker_token_id": self.marker_token_ids[i],
                        "harness_key_token_id": expected,
                        "merged_with_prefix": self.merged_with_prefix[i],
                    }
                )
        return out

    @property
    def matches_harness_key_tokens(self) -> bool:
        return not self.mismatches

    def to_dict(self) -> dict[str, Any]:
        """Marker bookkeeping without any prompt text (the repo is public)."""
        return {
            "n_options": self.n_options,
            "input_tokens": len(self.input_ids),
            "answer_position": self.answer_position,
            "single_token_key": all(self.single_token_key),
            "merged_with_prefix": sum(self.merged_with_prefix),
            "n_mismatches": len(self.mismatches),
            "mean_option_line_tokens": (
                float(sum(e - p + 1 for p, e in zip(self.positions, self.end_positions, strict=True)))
                / max(1, self.n_options)
            ),
        }


def locate_option_markers(
    item: Item,
    tokenizer: Any,
    *,
    prompt: str | None = None,
    variant: LetterVariant = "bare",
) -> PromptMarkers:
    """Tokenize ``item``'s prompt and locate every option marker and the answer position.

    ``item`` must already be canonicalised (:func:`meddecide.eval.readout.canonicalise_options`)
    so its keys are the letters the prompt shows.
    """
    if prompt is None:
        prompt = render_prompt(item, tokenizer, variant)
    block = options_block(item)
    needle = OPTIONS_HEADER + block
    if needle not in prompt:
        raise ValueError("prompt does not contain the options block built from the item")
    start = prompt.rindex(needle) + len(OPTIONS_HEADER)
    if not prompt[start + len(block) :].startswith(_BLOCK_SUFFIX):
        raise ValueError("the located options block is not followed by the fixed instruction")
    encoded = tokenizer(prompt, add_special_tokens=True, return_offsets_mapping=True)
    input_ids = [int(t) for t in encoded["input_ids"]]
    offsets = [(int(a), int(b)) for a, b in encoded["offset_mapping"]]
    if len(input_ids) != len(offsets):
        raise AssertionError("tokenizer returned a different number of ids and offsets")
    starts = [a for a, _ in offsets]

    positions: list[int] = []
    end_positions: list[int] = []
    marker_token_ids: list[int] = []
    harness_ids: list[int | None] = []
    single_token: list[bool] = []
    merged: list[bool] = []
    # Option line offsets come from the block's own construction, not from searching for the
    # key text: a label may itself contain "C. " (e.g. "C. difficile"), and a search for
    # "C. " then lands inside the *previous* option's label. Measured: the S7 smoke run scored
    # token " C" (351) instead of key token "C" (34) on an item whose label read
    # "… C. difficile …", and the marker check refused it. Accumulating lengths is exact.
    option_chars: list[int] = []
    line_ends: list[int] = []
    cursor = start
    for opt in item.options:
        line = f"{opt.key}. {opt.label}"
        if prompt[cursor : cursor + len(line)] != line:  # pragma: no cover - impossible
            raise AssertionError(
                f"options block mismatch at char {cursor} for key {opt.key!r}"
            )
        option_chars.append(cursor)
        line_ends.append(cursor + len(line))
        cursor += len(line) + 1  # the "\n" that joins the option lines
    if cursor - 1 != start + len(block):  # pragma: no cover - impossible
        raise AssertionError("options block length disagrees with its line count")

    for i, opt in enumerate(item.options):
        char = option_chars[i]
        token_pos = bisect_right(starts, char) - 1
        if token_pos < 0 or not (offsets[token_pos][0] <= char < offsets[token_pos][1]):
            raise ValueError(f"no token covers option key {opt.key!r} at char {char}")
        positions.append(token_pos)
        marker_token_ids.append(input_ids[token_pos])
        merged.append(offsets[token_pos][0] < char)
        try:
            harness_ids.append(key_token_id(opt.key, tokenizer, variant))
            single_token.append(True)
        except ValueError:
            harness_ids.append(None)
            single_token.append(False)
        # the last character of the option's line (never the newline that ends it)
        last_char = max(line_ends[i] - 1, char)
        token_pos = bisect_right(starts, last_char) - 1
        while token_pos > positions[i] and offsets[token_pos][0] >= line_ends[i]:
            token_pos -= 1
        if token_pos < positions[i]:
            token_pos = positions[i]
        end_positions.append(token_pos)
    return PromptMarkers(
        prompt=prompt,
        input_ids=input_ids,
        option_keys=[opt.key for opt in item.options],
        positions=positions,
        marker_token_ids=marker_token_ids,
        end_positions=end_positions,
        harness_key_token_ids=harness_ids,
        single_token_key=single_token,
        merged_with_prefix=merged,
    )
