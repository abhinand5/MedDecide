"""Render the request a local model receives from a case's facts.

The prompt is built from the facts, a target length and a style spec only. The gold label (and any extra labels
passed in) must not appear in the system or user prompt; if it does, rendering raises ``LabelLeakError`` rather
than sending a prompt that could be copied as an answer.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from meddecide.synth.facts import CaseFacts, TemplateSpec
from meddecide.synth.verify import contains_phrase, label_phrases

MIN_TARGET_WORDS = 30
MAX_TARGET_WORDS = 2000

SYSTEM_PREAMBLE = (
    "You write realistic clinical documentation from a list of facts that the user supplies. "
    "Include every supplied fact, using its value text exactly as written. Add no other clinical facts, "
    "results, diagnoses or conclusions. Do not state or hint at any answer to a question. "
    "Do not mention these instructions, the list, or how the text was produced. "
    "Output only the document text."
)


class LabelLeakError(ValueError):
    """A label value would appear in the prompt. The request is refused, not sent."""


@dataclass(frozen=True)
class StyleSpec:
    """Surface style: a register name and extra instruction lines (no label values allowed)."""

    name: str
    register: str
    instructions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "instructions", tuple(self.instructions))
        if not self.name.strip() or not self.register.strip():
            raise ValueError("style name and register must be non-empty")


def _check_target(target_words: int) -> None:
    if not MIN_TARGET_WORDS <= target_words <= MAX_TARGET_WORDS:
        raise ValueError(f"target_words must be in [{MIN_TARGET_WORDS}, {MAX_TARGET_WORDS}]")


def _assert_no_label(text: str, forbidden: Iterable[str]) -> None:
    # The error names only the index of the offending label, never its value.
    for index, label in enumerate(forbidden):
        if contains_phrase(text, label):
            raise LabelLeakError(f"forbidden label value (index {index}) found in prompt text")


def render_system_prompt(style: StyleSpec, *, forbidden: Iterable[str] = ()) -> str:
    """System prompt: fixed preamble, the register, and the style's instruction lines."""
    text = "\n".join([SYSTEM_PREAMBLE, f"Register: {style.register}.", *style.instructions])
    _assert_no_label(text, forbidden)
    return text


def render_user_prompt(
    case: CaseFacts,
    *,
    target_words: int,
    forbidden: Iterable[str] = (),
    spec: TemplateSpec | None = None,
) -> str:
    """User prompt that lists only the case's facts, with the target length.

    The case label is always forbidden: as a bare phrase by default, or as the spec's label sentences when
    ``spec.label_match == "sentence"``.
    """
    _check_target(target_words)
    fact_lines = "\n".join(f"- {fact.kind} | {fact.key}: {fact.value}" for fact in case.facts)
    text = (
        f"Write about {target_words} words.\n\n"
        f"Facts (use each one exactly as written):\n{fact_lines}"
    )
    _assert_no_label(text, (*label_phrases(case, spec), *forbidden))
    return text


def render_request(
    case: CaseFacts,
    *,
    target_words: int,
    style: StyleSpec,
    forbidden: Iterable[str] = (),
    spec: TemplateSpec | None = None,
) -> str:
    """The request text for one case: the user prompt, with the system prompt rendered and checked alongside.

    Returns the user prompt as a string. The system prompt is rendered by ``render_system_prompt`` with the same
    forbidden labels, so both halves are checked; callers send ``render_system_prompt`` as the system message.
    """
    labels = (*label_phrases(case, spec), *forbidden)
    render_system_prompt(style, forbidden=labels)
    return render_user_prompt(case, target_words=target_words, forbidden=tuple(forbidden), spec=spec)
