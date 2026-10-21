"""Serving checks for the local teacher (osler_v4 Y0): synthetic prompts, throughput arithmetic and reasoning detection.

Nothing here reads MIMIC or any other credentialed text. Prompts are built from fixed word lists with a seed, so every
request is distinct (the server cannot answer a repeated prefix from its cache) and no real record is involved.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

SYMPTOMS = ("fatigue", "cough", "dyspnoea", "nausea", "abdominal pain", "dizziness", "fever", "oedema", "headache")
FINDINGS = ("a raised creatinine", "a low sodium", "an elevated white count", "a normal chest film", "a mild anaemia")
PLANS = ("was continued on the current regimen", "was started on oral therapy", "was reviewed by the ward team")
TEMPLATES = (
    "The patient, aged {age}, reported {symptom} for {days} days before admission.",
    "On review, the record notes {finding} and the plan {plan}.",
    "Observations were stable; the team documented {symptom} that had eased since day {days}.",
    "The second clinician recorded {finding}; the note states that treatment {plan}.",
)


def synthetic_record(n_chars: int, seed: int) -> str:
    """A synthetic clinical-style text of at least ``n_chars`` characters, deterministic for a seed."""
    rng = random.Random(seed)
    sentences: list[str] = []
    size = 0
    while size < n_chars:
        sentence = rng.choice(TEMPLATES).format(
            age=rng.randint(18, 95), symptom=rng.choice(SYMPTOMS), days=rng.randint(1, 14),
            finding=rng.choice(FINDINGS), plan=rng.choice(PLANS),
        )
        sentences.append(sentence)
        size += len(sentence) + 1
    return " ".join(sentences)[:n_chars]


@dataclass(frozen=True)
class RunStats:
    """Totals for one batch of requests and the rates derived from them."""

    requests: int
    failed: int
    wall_s: float
    prompt_tokens: int
    completion_tokens: int

    @property
    def output_tokens_per_s(self) -> float:
        return self.completion_tokens / self.wall_s if self.wall_s > 0 else 0.0

    @property
    def requests_per_hour(self) -> float:
        ok = self.requests - self.failed
        return ok * 3600.0 / self.wall_s if self.wall_s > 0 else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {"requests": self.requests, "failed": self.failed, "wall_s": round(self.wall_s, 2),
                "prompt_tokens": self.prompt_tokens, "completion_tokens": self.completion_tokens,
                "output_tokens_per_s": round(self.output_tokens_per_s, 1),
                "requests_per_hour": round(self.requests_per_hour, 1)}


def run_stats(usages: Sequence[Mapping[str, Any]], failed: int, wall_s: float) -> RunStats:
    """Sum the ``usage`` blocks of the successful responses; ``failed`` counts the rest."""
    return RunStats(
        requests=len(usages) + failed,
        failed=failed,
        wall_s=wall_s,
        prompt_tokens=sum(int(u.get("prompt_tokens") or 0) for u in usages),
        completion_tokens=sum(int(u.get("completion_tokens") or 0) for u in usages),
    )


def reasoning_text(message: Mapping[str, Any]) -> str:
    """The reasoning the server split out of a chat message (vLLM names the field ``reasoning_content`` or ``reasoning``)."""
    for key in ("reasoning_content", "reasoning"):
        value = message.get(key)
        if value:
            return str(value)
    return ""
