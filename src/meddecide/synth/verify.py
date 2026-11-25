"""Verification of generated text against its facts and banned tokens.

Matching rules (documented, deliberately simple):

1. Text is NFKC-normalised and casefolded. Every character that is not a letter or digit is a separator.
2. The result is a token sequence. A phrase (a fact value, a label, a banned phrase) matches when its token
   sequence occurs contiguously in the text's token sequence. So "1.9 mg/dL" matches "1.9 mg/dl" and
   "1.9 mg / dL", but "15" does not match "5" and "1,9" matches "1.9".
3. No stemming, no numeric normalisation, and no synonyms, with one explicit exception (rule 4).
4. Year-unit synonym, applied to fact values only (not labels or banned phrases): a fact value that is exactly
   "<number> years" also matches the hyphenated and singular forms "<number>-year-old", "<number>-year" and
   "<number> year old" (all case-insensitive; they tokenize to "<number> year"). The number must be the same
   token: "67 years" does not match "76 years" or "6 years". No other unit and no other fact is affected.

Every rejection carries reason codes. ``primary_reason`` is the first code in ``REASON_PRIORITY`` that applies,
so per-family rejection counts by primary reason sum to the number of rejections.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass

from meddecide.synth.facts import CaseFacts, TemplateSpec, require_token

REASON_EMPTY = "empty_text"
REASON_BANNED_LABEL = "banned_label"
REASON_BANNED_META = "banned_meta_token"
REASON_MISSING_FACT = "missing_fact"
REASON_LENGTH = "length_out_of_range"

REASON_PRIORITY: tuple[str, ...] = (
    REASON_EMPTY,
    REASON_BANNED_LABEL,
    REASON_BANNED_META,
    REASON_MISSING_FACT,
    REASON_LENGTH,
)

# Purpose: catch generated text that talks about itself as a test item or a prompt (it answers a question, names a
# distractor, quotes the instructions, speaks as an AI). It must hold only clearly meta terms. Ordinary clinical
# words ("option", "gold standard", "label", "facts") do not belong here: they occur in real clinical text and
# would reject good items. Leaks of the gold label are handled by the label rule, not by this list.
META_BANNED_PHRASES: tuple[str, ...] = (
    "answer",
    "correct answer",
    "ground truth",
    "distractor",
    "distractors",
    "synthetic",
    "as an ai",
    "language model",
    "instructions",
)


def tokenize(text: str) -> tuple[str, ...]:
    """Token sequence used by every match in this module (see the module docstring for the rules)."""
    folded = unicodedata.normalize("NFKC", text).casefold()
    cleaned = "".join(ch if ch.isalnum() else " " for ch in folded)
    return tuple(cleaned.split())


def contains_phrase(text: str, phrase: str) -> bool:
    """True when the token sequence of ``phrase`` occurs contiguously in ``text``. Empty phrases never match."""
    needle = tokenize(phrase)
    if not needle:
        return False
    haystack = tokenize(text)
    width = len(needle)
    return any(haystack[i : i + width] == needle for i in range(len(haystack) - width + 1))


def _year_synonym(phrase: str) -> tuple[str, ...] | None:
    """Token sequence ("<number>", "year") for a phrase that is exactly "<number> years", else None."""
    tokens = tokenize(phrase)
    if len(tokens) == 2 and tokens[0].isdigit() and tokens[1] == "years":
        return (tokens[0], "year")
    return None


def contains_fact(text: str, value: str) -> bool:
    """Fact match: ``contains_phrase`` plus the year-unit synonym rule (module docstring, rule 4)."""
    if contains_phrase(text, value):
        return True
    alt = _year_synonym(value)
    return alt is not None and contains_phrase(text, " ".join(alt))


def label_phrases(case: CaseFacts, spec: TemplateSpec | None = None) -> tuple[str, ...]:
    """Phrases whose presence in generated text or a prompt leaks the label of ``case``.

    Without a spec, or with ``label_match == "token"``, this is the bare label. With "sentence" it is the spec's
    label sentences only (the bare label word is allowed in ordinary text, e.g. "no fever").
    """
    if spec is None or spec.label_match == "token":
        phrases: tuple[str, ...] = (case.label,)
    else:
        if spec.family != case.family:
            raise ValueError(f"case family {case.family!r} does not match template {spec.family!r}")
        phrases = spec.label_sentences
    for phrase in phrases:
        require_token(phrase, "label phrase")
    return phrases


@dataclass(frozen=True)
class Verdict:
    """Outcome for one generated text. ``reasons`` is empty if and only if the text is kept.

    ``missing_facts`` holds fact values and must never be written to a manifest.
    """

    family: str
    case_id: str
    reasons: tuple[str, ...]
    primary_reason: str | None
    missing_facts: tuple[str, ...]
    banned_hits: tuple[str, ...]
    word_count: int

    @property
    def kept(self) -> bool:
        return not self.reasons


def verify_text(
    text: str,
    case: CaseFacts,
    *,
    word_range: tuple[int, int] | None = None,
    banned_phrases: tuple[str, ...] = META_BANNED_PHRASES,
    spec: TemplateSpec | None = None,
) -> Verdict:
    """Check ``text`` against the facts and banned tokens of ``case``. Returns a verdict; never drops silently."""
    words = len(text.split())
    found: set[str] = set()

    if not tokenize(text):
        found.add(REASON_EMPTY)

    if any(contains_phrase(text, phrase) for phrase in label_phrases(case, spec)):
        found.add(REASON_BANNED_LABEL)

    banned_hits = tuple(phrase for phrase in banned_phrases if contains_phrase(text, phrase))
    if banned_hits:
        found.add(REASON_BANNED_META)

    missing = tuple(fact.value for fact in case.facts if not contains_fact(text, fact.value))
    if missing:
        found.add(REASON_MISSING_FACT)

    if word_range is not None:
        low, high = word_range
        if not low <= words <= high:
            found.add(REASON_LENGTH)

    reasons = tuple(code for code in REASON_PRIORITY if code in found)
    return Verdict(
        family=case.family,
        case_id=case.case_id,
        reasons=reasons,
        primary_reason=reasons[0] if reasons else None,
        missing_facts=missing,
        banned_hits=banned_hits,
        word_count=words,
    )
