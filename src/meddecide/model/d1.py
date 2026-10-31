"""LiquidAI/d1-2B (LFM Open License v1.0) through its own System One API, as its model card documents (section "Inference").

The card's protocol, followed here without change:

- load with ``AutoModel.from_pretrained(REPO, trust_remote_code=True, dtype=bfloat16)``; the checkpoint's code and
  calibration come from the pinned revision (``REVISION``);
- each question is a dict in the card's schema: ``{"type": "noul", "instructions": ...}`` (P(yes)),
  ``{"type": "choice", "instructions": ..., "criteria": {name: description}}`` (a distribution over the names) or
  ``{"type": "score", "instructions": ..., "criteria": [levels, lowest first]}`` (a distribution over the levels);
- one state and one question per call, so no question shares a sequence with another (one forward pass per item);
- the answers are read from ``model.system_one(state, questions)["answers"]``; the checkpoint's ``calibration.json``
  is applied by the card's engine to every answer, so the probabilities returned here are the calibrated ones.

The helpers below are pure: they map an item's options to the card's question dict and the card's answer back to a
probability vector in the item's option order, and check that vector. ``D1Model`` is the only class that touches
torch and the checkpoint.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

REPO = "LiquidAI/d1-2B"
REVISION = "10e7671d44e4b37700020b1eed3f594c5f2af0b4"
LICENSE = "LFM Open License v1.0"
CARD_TRANSFORMERS = "5.19"  # the card's install line: transformers>=5.19
QTYPES = ("noul", "choice", "score")
SCORE_MAX_LEVELS = 10  # the card: a score has 2 to 10 ordered levels
PROB_SUM_TOL = 1e-4  # a probability vector must sum to one within this (float32 softmax after calibration)
_QUESTION = "q"  # the one question's name in a call


def question_for(qtype: str, question: str, options: Sequence[tuple[str, str]]) -> dict[str, Any]:
    """The card's question dict for one benchmark item.

    ``options`` is the item's ``(key, label)`` list in presentation order. A noul item must offer exactly ``yes`` and
    ``no``; the card returns P(yes) for it. A choice names its options by their keys (single letters stay letters, so the
    card's option codes are the keys) and describes them by their labels. A score lists its labels lowest first.
    """
    if qtype not in QTYPES:
        raise ValueError(f"qtype {qtype!r}: one of {QTYPES}")
    if not question or not question.strip():
        raise ValueError("an empty question text")
    keys = [str(k) for k, _ in options]
    if len(set(keys)) != len(keys):
        raise ValueError(f"option keys are not unique: {keys}")
    if qtype == "noul":
        if sorted(keys) != ["no", "yes"]:
            raise ValueError(f"a noul item offers yes and no, not {keys}")
        return {"type": "noul", "instructions": question}
    if qtype == "choice":
        if len(keys) < 2:
            raise ValueError("a choice has at least two options")
        return {"type": "choice", "instructions": question,
                "criteria": {str(k): str(label) for k, label in options}}
    if not 2 <= len(keys) <= SCORE_MAX_LEVELS:
        raise ValueError(f"a score has 2 to {SCORE_MAX_LEVELS} levels, not {len(keys)}")
    return {"type": "score", "instructions": question, "criteria": [str(label) for _, label in options]}


def probabilities_from_answer(qtype: str, options: Sequence[tuple[str, str]], answer: Mapping[str, Any]) -> list[float]:
    """The card's answer as probabilities in the item's option order.

    noul: ``answer["noul"]`` is P(yes); P(no) is its complement. choice: ``answer["probabilities"]`` is keyed by the
    criteria names in order. score: ``answer["probabilities"]`` is keyed by level index ``"0"``..``"n-1"``.
    """
    keys = [str(k) for k, _ in options]
    if qtype == "noul":
        if answer.get("type") != "noul" or "noul" not in answer:
            raise ValueError(f"expected a noul answer, got {answer.get('type')!r}")
        p_yes = float(answer["noul"])
        return [p_yes if k == "yes" else 1.0 - p_yes for k in keys]
    dist = answer.get("probabilities")
    if not isinstance(dist, Mapping):
        raise ValueError(f"a {qtype} answer has no probabilities: {sorted(answer)}")
    if qtype == "choice":
        if answer.get("type") != "choice" or list(dist) != keys:
            raise ValueError(f"choice answer keys {list(dist)} differ from the item's {keys}")
        return [float(dist[k]) for k in keys]
    if answer.get("type") != "score" or sorted(dist, key=int) != [str(i) for i in range(len(keys))]:
        raise ValueError(f"score answer levels {sorted(dist)} differ from the item's {len(keys)} levels")
    return [float(dist[str(i)]) for i in range(len(keys))]


def validate_probabilities(probs: Sequence[float]) -> None:
    """Raise unless every probability is finite and in [0, 1] and the vector sums to one within PROB_SUM_TOL."""
    if not probs:
        raise ValueError("an empty probability vector")
    for p in probs:
        if not math.isfinite(p) or p < 0.0 or p > 1.0:
            raise ValueError(f"probability {p!r} is not in [0, 1]")
    total = sum(probs)
    if abs(total - 1.0) > PROB_SUM_TOL:
        raise ValueError(f"probabilities sum to {total!r}, not 1")


class D1Model:
    """The loaded checkpoint and its System One call. Construct with ``D1Model.load()``."""

    def __init__(self, model: Any, revision: str) -> None:
        self.model = model
        self.revision = revision

    @classmethod
    def load(cls, revision: str = REVISION, device: str = "cuda", dtype: str = "bfloat16") -> D1Model:
        """Load the pinned revision with the card's call. Raises the loader's own error if the architecture does not load."""
        import torch
        from transformers import AutoModel

        model = AutoModel.from_pretrained(REPO, revision=revision, trust_remote_code=True,
                                          dtype=getattr(torch, dtype))
        return cls(model.to(device).eval(), revision)

    @property
    def calibrated(self) -> bool:
        """Whether the checkpoint's calibration.json is applied to the answers (the card's engine reads it at load)."""
        return self.model.engine.calibration is not None

    def answer(self, state: Any, qtype: str, question: str,
               options: Sequence[tuple[str, str]]) -> tuple[list[float], dict[str, Any]]:
        """One item: its probabilities in option order, and what the call reports (answer type, input tokens, the score's median)."""
        spec = question_for(qtype, question, options)
        out = self.model.system_one(state, {_QUESTION: spec})
        answer = out["answers"][_QUESTION]
        probs = probabilities_from_answer(qtype, options, answer)
        validate_probabilities(probs)
        meta = {"answer_type": answer["type"], "input_tokens": int(out["usage"]["input_tokens"]),
                "score_median": answer.get("score") if qtype == "score" else None}
        return probs, meta
