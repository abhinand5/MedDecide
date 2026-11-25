"""The 13-gram near-duplicate check (osler_v4 Y4, CG v2): synthetic text only."""

from __future__ import annotations

from meddecide.panel import ngram


def _row(state: str, question: str = "Which is right?", labels: tuple[str, ...] = ("yes", "no")) -> dict:
    return {"state": state, "question": question, "options": [{"key": str(i), "label": lab} for i, lab in enumerate(labels)]}


SHARED = "the patient reported a persistent dry cough with fever and fatigue over the past two weeks"


def test_a_shared_run_of_thirteen_tokens_is_a_hit() -> None:
    panel = [_row(SHARED)]
    index = ngram.build_index(panel)
    assert ngram.hits(_row(f"Note: {SHARED} and more."), index) > 0


def test_a_shared_run_shorter_than_thirteen_tokens_is_not_a_hit() -> None:
    short = "a dry cough with fever and fatigue"          # 7 tokens shared
    index = ngram.build_index([_row(f"{short} alpha beta gamma delta epsilon zeta eta theta")])
    assert ngram.hits(_row(f"{short} iota kappa lambda mu nu xi omicron pi"), index) == 0


def test_text_with_fewer_tokens_than_n_cannot_hit() -> None:
    index = ngram.build_index([_row(SHARED)])
    assert ngram.hits(_row("short text only"), index) == 0


def test_normalisation_ignores_case_and_punctuation() -> None:
    index = ngram.build_index([_row(SHARED.upper() + "!!!")])
    assert ngram.hits(_row(SHARED.replace(" ", ",  ")), index) > 0
