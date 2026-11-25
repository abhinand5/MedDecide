"""Structured facts, case identity and template specs for the grounded generator.

A case is a family name, a tuple of typed facts assembled by code from structured sources, and one gold label.
The case id is a stable hash of the family and the sorted fact list, so the same facts give the same id on any
machine and in any order. Nothing in this module reads data files or calls a model.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from meddecide.utils.hashing import normalize_text, stable_hash

FACT_KINDS: tuple[str, ...] = (
    "condition",
    "medication",
    "lab",
    "finding",
    "history",
    "demographic",
    "encounter",
)


def has_token(text: str) -> bool:
    """True when ``text`` has at least one letter or digit (the same rule ``verify.tokenize`` uses).

    A value without one tokenizes to an empty sequence, so it can never be matched or checked.
    """
    return any(ch.isalnum() for ch in unicodedata.normalize("NFKC", text))


def require_token(text: str, what: str) -> None:
    """Raise ``ValueError`` if ``text`` has no letter or digit. The message never echoes the value."""
    if not has_token(text):
        raise ValueError(f"{what} must contain at least one letter or digit")


@dataclass(frozen=True)
class Fact:
    """One typed fact. ``value`` is the exact text the generated item must contain."""

    kind: str
    key: str
    value: str

    def __post_init__(self) -> None:
        if self.kind not in FACT_KINDS:
            raise ValueError(f"unknown fact kind: {self.kind!r}")
        if not normalize_text(self.key):
            raise ValueError("fact key must be non-empty")
        if not normalize_text(self.value):
            raise ValueError("fact value must be non-empty")
        require_token(self.value, "fact value")


def fact_list_id(family: str, facts: Iterable[Fact], *, length: int = 16) -> str:
    """Stable id of a family and its fact list (order-independent, NFC-normalised)."""
    payload = {
        "family": family,
        "facts": sorted([fact.kind, fact.key, fact.value] for fact in facts),
    }
    return stable_hash(payload, length=length)


@dataclass(frozen=True)
class CaseFacts:
    """A case: its family, its facts and its gold label. The label is never part of the prompt."""

    family: str
    facts: tuple[Fact, ...]
    label: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "facts", tuple(self.facts))
        if not normalize_text(self.family):
            raise ValueError("family must be non-empty")
        if not self.facts:
            raise ValueError("a case needs at least one fact")
        if len(set(self.facts)) != len(self.facts):
            raise ValueError("duplicate facts in a case")
        if not normalize_text(self.label):
            raise ValueError("label must be non-empty")
        require_token(self.label, "label")

    @property
    def case_id(self) -> str:
        return fact_list_id(self.family, self.facts)


# "token": the bare label is banned as a whole-token phrase. "sentence": only the spec's label sentences are banned
# (for binary families whose labels, such as "yes" and "no", occur in ordinary clinical text).
LABEL_MATCH_MODES: tuple[str, ...] = ("token", "sentence")

OptionBuilder = Callable[[tuple[Fact, ...]], tuple[str, ...]]


@dataclass(frozen=True)
class TemplateSpec:
    """A question family: the allowed gold labels and the builder of the options offered for a case.

    ``label_match`` selects the leak rule used by ``verify_text`` and the render functions when given the spec:
    "token" bans each label as a phrase; "sentence" bans only ``label_sentences`` (exact phrases such as
    "the answer is yes"), never the bare label word.
    """

    family: str
    label_set: tuple[str, ...]
    option_builder: OptionBuilder
    label_match: str = "token"
    label_sentences: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "label_set", tuple(self.label_set))
        object.__setattr__(self, "label_sentences", tuple(self.label_sentences))
        if self.label_match not in LABEL_MATCH_MODES:
            raise ValueError(f"label_match must be one of {LABEL_MATCH_MODES}")
        if self.label_match == "sentence":
            if not self.label_sentences:
                raise ValueError("label_match='sentence' needs at least one label sentence")
            if any(len(normalize_text(sentence).split()) < 2 for sentence in self.label_sentences):
                raise ValueError("a label sentence must have at least two words")
        if len(self.label_set) < 2:
            raise ValueError("a template needs at least two labels")
        if len(set(self.label_set)) != len(self.label_set):
            raise ValueError("duplicate labels in a template")
        if any(not normalize_text(label) for label in self.label_set):
            raise ValueError("labels must be non-empty")
        for label in self.label_set:
            require_token(label, "label")
        for sentence in self.label_sentences:
            require_token(sentence, "label sentence")

    def options_for(self, case: CaseFacts) -> tuple[str, ...]:
        """Options offered for ``case``. Checks the family, option count, and that the label is offered."""
        if case.family != self.family:
            raise ValueError(f"case family {case.family!r} does not match template {self.family!r}")
        if case.label not in self.label_set:
            raise ValueError("case label is not in the template label set")
        options = tuple(self.option_builder(case.facts))
        if len(options) < 2 or len(set(options)) != len(options):
            raise ValueError("options must be at least two and unique")
        if case.label not in options:
            raise ValueError("case label is not among the offered options")
        return options
