"""Hosted Liquid ``d1-ultra`` teacher client for distill_v0 (D1 teacher ceiling, D2 training labels).

The contract was measured before the loop (loops/distill_v0/ADVISORY.md §1.1) and re-checked on
2026-11-03: ``POST https://openrouter.ai/api/alpha/decisions`` with ``{model, state: [str],
questions: {name: {type, instructions, criteria?}}}``. A ``choice`` answer carries
``probabilities`` keyed by the criteria keys, a ``noul`` answer carries ``noul`` = P(yes), and a
``score`` answer carries ``probabilities`` keyed by the level index (``"0"``…``"n-1"``) with a
``legend``. ``usage.input_tokens`` and ``usage.cost`` are returned on every response.

This module is pure apart from hashing a file: request construction, answer validation, the
freeze-row schema, and the cost ledger. The HTTP driver is ``scripts/distill_v0/d1_ceiling.py``.

Rules encoded here (ADVISORY §1.2, GOAL §5):
- one permutation per item: the item's own option order is the only order sent;
- a response is valid only when its key set equals the offered key set, the probabilities are
  finite and non-negative, they sum to 1 within ``PROB_TOLERANCE``, and there is no refusal;
- the freeze row contains no item text (no state, question, option label or gold);
- **MIMIC-derived data never goes to the hosted teacher** (AGENTS.md data rules, D28/D60): the
  guard refuses any row whose set, source, licence or metadata names MIMIC/PhysioNet/credentialed
  data. MIMIC stays on a self-hosted model.
"""

from __future__ import annotations

import hashlib
import json
import math
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ENDPOINT = "https://openrouter.ai/api/alpha/decisions"
MODEL = "liquid/d1-ultra-20261015"  # pinned: the ~latest alias may move
PROVIDER = "Liquid"
QUESTION_ID = "q"
CAP_USD = 15.00  # hard cap for the loop; the labeler stops cleanly at it
PROB_TOLERANCE = 1e-3  # ADVISORY §1.2: probabilities sum to 1 within 1e-3
MAX_OPTIONS = 255
TEACHER = MODEL  # the value stored in every freeze row (the teacher id, never a nickname)

FORBIDDEN_LICENCE_MARKERS = ("physionet", "credentialed health data")
FORBIDDEN_DATA_MARKERS = ("mimic", "physionet", "eicu", "hirid")


class TeacherAnswerError(ValueError):
    """An answer that cannot be used; ``reason`` is a stable code written next to the item id."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


@dataclass(frozen=True)
class TeacherRequest:
    """One item's request, with the option order the answer must match."""

    item_id: str
    template_id: str
    qtype: str
    option_keys: tuple[str, ...]
    body: Mapping[str, Any]

    def body_bytes(self) -> bytes:
        return json.dumps(self.body, ensure_ascii=False).encode("utf-8")


def request_for_item(item: Mapping[str, Any]) -> tuple[TeacherRequest | None, str | None]:
    """The teacher request for one benchmark row, or ``(None, reason)`` when it cannot be sent exactly."""
    state = str(item.get("state") or "")
    question = str(item.get("question") or "")
    options = list(item.get("options") or [])
    qtype = str(item.get("qtype") or "")
    if not state.strip():
        return None, "empty_state"
    if not question.strip():
        return None, "empty_question"
    if not 2 <= len(options) <= MAX_OPTIONS:
        return None, "option_count"
    keys = tuple(str(option["key"]) for option in options)
    labels = tuple(str(option.get("label") or "") for option in options)
    if any(not key.strip() for key in keys):
        return None, "empty_key"
    if len(set(keys)) != len(keys):
        return None, "repeated_key"
    spec: dict[str, Any]
    if qtype == "choice":
        if any(not label.strip() for label in labels):
            return None, "empty_label"
        spec = {"type": "choice", "instructions": question, "criteria": dict(zip(keys, labels, strict=True))}
    elif qtype == "noul":
        if keys != ("yes", "no"):
            return None, "noul_keys"
        spec = {"type": "noul", "instructions": question}
    elif qtype == "score":
        if any(not label.strip() for label in labels):
            return None, "empty_label"
        spec = {"type": "score", "instructions": question, "criteria": list(labels)}
    else:
        return None, "qtype"
    body = {"model": MODEL, "state": [state], "questions": {QUESTION_ID: spec}}
    return TeacherRequest(str(item["item_id"]), str(item.get("template_id") or ""), qtype, keys, body), None


def _checked_distribution(values: Sequence[float], n: int) -> list[float]:
    if len(values) != n:
        raise TeacherAnswerError("length")
    if any(not isinstance(value, (int, float)) or isinstance(value, bool) for value in values):
        raise TeacherAnswerError("non_numeric")
    probs = [float(value) for value in values]
    if any(not math.isfinite(prob) for prob in probs):
        raise TeacherAnswerError("non_finite")
    if any(prob < 0.0 for prob in probs):
        raise TeacherAnswerError("negative")
    if abs(sum(probs) - 1.0) > PROB_TOLERANCE:
        raise TeacherAnswerError("prob_sum")
    return probs


def parse_response(payload: Mapping[str, Any], request: TeacherRequest) -> list[float]:
    """The teacher's distribution aligned to ``request.option_keys``; raises ``TeacherAnswerError``."""
    answers = payload.get("answers")
    if not isinstance(answers, Mapping):
        raise TeacherAnswerError("no_answers")
    answer = answers.get(QUESTION_ID)
    if not isinstance(answer, Mapping):
        raise TeacherAnswerError("missing_question")
    if answer.get("refusal") or answer.get("refused") or answer.get("error"):
        raise TeacherAnswerError("refusal")
    declared = answer.get("type")
    if declared is not None and str(declared) != request.qtype:
        raise TeacherAnswerError("qtype_mismatch")
    if request.qtype == "choice":
        probabilities = answer.get("probabilities")
        if not isinstance(probabilities, Mapping):
            raise TeacherAnswerError("no_probabilities")
        if set(map(str, probabilities)) != set(request.option_keys):
            raise TeacherAnswerError("key_set")
        return _checked_distribution([probabilities[key] for key in request.option_keys], len(request.option_keys))
    if request.qtype == "noul":
        p_yes = answer.get("noul")
        if not isinstance(p_yes, (int, float)) or isinstance(p_yes, bool):
            raise TeacherAnswerError("no_noul")
        p_yes = float(p_yes)
        if not math.isfinite(p_yes):
            raise TeacherAnswerError("non_finite")
        if not 0.0 <= p_yes <= 1.0:
            raise TeacherAnswerError("range")
        return [p_yes if key == "yes" else 1.0 - p_yes for key in request.option_keys]
    if request.qtype == "score":
        probabilities = answer.get("probabilities")
        if not isinstance(probabilities, Mapping):
            raise TeacherAnswerError("no_probabilities")
        expected = [str(index) for index in range(len(request.option_keys))]
        if set(map(str, probabilities)) != set(expected):
            raise TeacherAnswerError("key_set")
        return _checked_distribution([probabilities[index] for index in expected], len(expected))
    raise TeacherAnswerError("qtype")


def argmax_index(probs: Sequence[float]) -> int:
    """Index of the largest probability; a tie goes to the earlier option."""
    if not probs:
        raise ValueError("argmax_index needs at least one probability")
    return max(range(len(probs)), key=probs.__getitem__)


def predicted_key(request: TeacherRequest, probs: Sequence[float]) -> str:
    """The offered key with the largest probability (for ``score`` this is the modal level)."""
    return request.option_keys[argmax_index(probs)]


def is_correct(item: Mapping[str, Any], probs: Sequence[float]) -> bool:
    """Whether the teacher's modal key equals the item's gold key (the D1 ceiling definition)."""
    request, reason = request_for_item(item)
    if request is None:
        raise ValueError(f"item {item.get('item_id')!r} cannot be scored: {reason}")
    return predicted_key(request, probs) == str(item["gold"])


def freeze_row(
    request: TeacherRequest,
    probs: Sequence[float],
    *,
    cost: float,
    input_tokens: int,
    teacher: str = TEACHER,
    text_key: str | None = None,
) -> dict[str, Any]:
    """One frozen label row (ADVISORY §1.2). It contains no state, question, option label or gold.

    ``text_key`` is the normalised-text key of the row. It is added for the training basket because
    Winogrande twins share an item id while their state text differs: an item-id-keyed label file
    would collapse distinct rows. It is omitted where the id is unique (the D1 ceiling sets).
    """
    if len(probs) != len(request.option_keys):
        raise ValueError("the distribution does not match the request's option count")
    row = {
        "item_id": request.item_id,
        "template_id": request.template_id,
        "qtype": request.qtype,
        "option_keys": list(request.option_keys),
        "teacher_probs": [float(prob) for prob in probs],
        "teacher": teacher,
        "cost": float(cost),
        "input_tokens": int(input_tokens),
    }
    if text_key is not None:
        row["text_key"] = str(text_key)
    return row


def failure_row(request: TeacherRequest, reason: str, *, attempts: int) -> dict[str, Any]:
    """One failed row: the item id and the reason, never any text."""
    return {"item_id": request.item_id, "template_id": request.template_id, "qtype": request.qtype,
            "reason": reason, "attempts": int(attempts)}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_frozen(path: Path) -> dict[str, dict[str, Any]]:
    """Frozen rows keyed by item id; the caller checks the hash before trusting the file."""
    rows: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            rows[str(row["item_id"])] = row
    return rows


def hosted_guard(item: Mapping[str, Any], *, allow_protected: bool = False) -> tuple[bool, str | None]:
    """Whether this row may be sent to the hosted teacher, with the refusal reason when it may not.

    MIMIC/PhysioNet/credentialed data is refused by licence, source, set name and metadata markers,
    not by a hand-kept list of set names. ``allow_protected`` is set only under the operator's
    authorization: on 2026-11-03 the operator confirmed the provider's email confirmations and
    instructed that MIMIC-derived datasets may be used. The caller records the authorization next to
    every run that passes it.
    """
    if allow_protected:
        return True, None
    licence = str(item.get("source_license") or "").lower()
    if any(marker in licence for marker in FORBIDDEN_LICENCE_MARKERS):
        return False, "physionet_licence"
    haystack = " ".join(
        str(item.get(field) or "") for field in ("set_name", "source", "source_url", "benchmark")
    ).lower()
    haystack += " " + json.dumps(item.get("meta") or {}, ensure_ascii=False).lower()
    if any(marker in haystack for marker in FORBIDDEN_DATA_MARKERS):
        return False, "mimic_source"
    return True, None


@dataclass
class CostLedger:
    """The loop's API spend: a hard cap, per-call charges, and the counts the manifest reports."""

    cap_usd: float = CAP_USD
    spent_usd: float = 0.0
    calls: int = 0
    failures: int = 0
    input_tokens: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def seed(self, cost: float, input_tokens: int, calls: int) -> None:
        """Start from spend already booked elsewhere (a resumed run's frozen labels), never from a caller's estimate."""
        if cost < 0 or input_tokens < 0 or calls < 0:
            raise ValueError("a ledger cannot be seeded with negative values")
        with self._lock:
            self.spent_usd += float(cost)
            self.input_tokens += int(input_tokens)
            self.calls += int(calls)

    def charge(self, cost: float, input_tokens: int) -> None:
        if cost < 0 or input_tokens < 0:
            raise ValueError("a negative charge cannot be booked")
        with self._lock:
            self.spent_usd += float(cost)
            self.input_tokens += int(input_tokens)
            self.calls += 1

    def note_failure(self) -> None:
        with self._lock:
            self.failures += 1

    @property
    def exhausted(self) -> bool:
        with self._lock:
            return self.spent_usd >= self.cap_usd

    @property
    def remaining_usd(self) -> float:
        with self._lock:
            return max(0.0, self.cap_usd - self.spent_usd)

    def to_dict(self) -> dict[str, Any]:
        with self._lock:
            return {"cap_usd": self.cap_usd, "spent_usd": round(self.spent_usd, 6), "calls": self.calls,
                    "failures": self.failures, "input_tokens": self.input_tokens,
                    "remaining_usd": round(max(0.0, self.cap_usd - self.spent_usd), 6)}
