"""Leakage probe: can a model recover a kept item's label from its text alone?

For each sampled item the local model gets the text and the offered options (lettered) and one question: which option
is the answer, given only this text. Reasoning is off and the reply is parsed to one letter. A rate well above chance
means the text reveals its label. Only counts and rates leave this module; the reply text is never stored.
"""

from __future__ import annotations

import random
import re
from collections.abc import Sequence
from dataclasses import dataclass

PROBE_SYSTEM = "You answer multiple-choice questions about clinical text. Reply with the option letter only."
PROBE_QUESTION = "Given only this text, which option is the answer?"
LETTERS = "ABCDEFGHIJ"
_LETTER = re.compile(r"(?<![A-Za-z])([A-J])(?![A-Za-z])")


def probe_user_prompt(text: str, displays: Sequence[str]) -> str:
    """The user turn: the item text, the question, and the lettered options."""
    if not 2 <= len(displays) <= len(LETTERS):
        raise ValueError("a probe needs between two and ten options")
    options = "\n".join(f"{LETTERS[i]}. {shown}" for i, shown in enumerate(displays))
    return f"{text}\n\n{PROBE_QUESTION}\n{options}\nReply with the letter only."


def parse_letter(reply: str, n_options: int) -> int | None:
    """Index of the option the reply names (its first standalone letter in range), or None when none is named."""
    for match in _LETTER.finditer(reply.strip()):
        index = LETTERS.index(match.group(1))
        if index < n_options:
            return index
    return None


@dataclass(frozen=True)
class ProbeOutcome:
    """One probed item. ``correct`` is None when the reply named no option (counted as unparsed, not as wrong)."""

    template: str
    n_options: int
    correct: bool | None


def sample_items(items: Sequence[dict], n: int, *, seed: int) -> list[dict]:
    """A seeded random sample without replacement (all items when fewer than ``n``)."""
    rng = random.Random(f"{seed}:probe")
    if len(items) <= n:
        return list(items)
    return rng.sample(list(items), n)


def summarize_probe(outcomes: Sequence[ProbeOutcome]) -> dict[str, float | int | None]:
    """Counts and rates for one template. ``chance`` is the mean of 1/(number of options) over the probed items."""
    n = len(outcomes)
    parsed = [o for o in outcomes if o.correct is not None]
    correct = sum(1 for o in parsed if o.correct)
    return {
        "probed": n,
        "unparsed": n - len(parsed),
        "correct": correct,
        "answer_recovered_rate": (correct / len(parsed)) if parsed else None,
        "chance": (sum(1 / o.n_options for o in outcomes) / n) if n else None,
    }
