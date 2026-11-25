"""Soft-label targets from pplx-decider-v1.2-27b (osler_v4 Y5, D48): the request and answer mapping, the exclusion keys and the
order of the anchor pool. Pure functions; the GPU driver is scripts/osler_v4/y5_soft_labels.py.

The teacher is the authors' code and checkpoint, pinned to REVISION (Apache-2.0). A training row holds one question, so each
row is one request with that question under QUESTION_ID:
- choice: criteria keyed by the option labels (the flash runner's convention); labels must be unique, because the answer is
  keyed by label;
- noul: the native noul question with criteria Yes / No (the teacher's noul is P(true), so P(yes) is the answer);
- score: the levels as the criteria list, lowest first (the answer is keyed "0", "1", ... by position).

A row that cannot be sent exactly is skipped with a reason. It is never sent in a changed form, and no row drops silently.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from meddecide.panel.overlap import row_text_key
from meddecide.train.osler_v3 import is_anchor
from meddecide.utils.hashing import stable_hash

REPO = "perplexity-ai/pplx-decider-v1.2-27b"
REVISION = "cce0f3f80033e9d89d80831d5281fd72124aacce"  # the pinned Apache-2.0 snapshot (its README and LICENSE agree)
TEACHER = f"{REPO}@{REVISION}"
QUESTION_ID = "q"
NOUL_CRITERIA = {"true": "Yes", "false": "No"}
MAX_OPTIONS = 255  # the authors' limit (protocol.MAX_OPTIONS)
PROB_TOLERANCE = 1e-5

RecordKey = tuple[str, str]


def request_for_row(row: Mapping[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    """The teacher request for one training row, or ``(None, reason)`` when the row cannot be sent exactly."""
    options = row["options"]
    labels = [str(option["label"]) for option in options]
    if not str(row["state"]).strip():
        return None, "empty_state"
    if not str(row["question"]).strip():
        return None, "empty_question"
    if not 1 <= len(options) <= MAX_OPTIONS:
        return None, "option_count"
    question = str(row["question"])
    qtype = row["qtype"]
    spec: dict[str, Any]
    if qtype == "choice":
        if any(not label.strip() for label in labels):
            return None, "empty_label"
        if len(set(labels)) != len(labels):
            return None, "repeated_label"
        spec = {"type": "choice", "instructions": question, "criteria": dict.fromkeys(labels)}
    elif qtype == "noul":
        if [str(option["key"]) for option in options] != ["yes", "no"]:
            return None, "noul_keys"
        spec = {"type": "noul", "instructions": question, "criteria": dict(NOUL_CRITERIA)}
    elif qtype == "score":
        if len(labels) < 2 or any(not label.strip() for label in labels):
            return None, "score_levels"
        spec = {"type": "score", "instructions": question, "criteria": labels}
    else:
        return None, "qtype"
    return {"state": str(row["state"]), "questions": {QUESTION_ID: spec}}, None


def teacher_probabilities(row: Mapping[str, Any], answer: Mapping[str, Any]) -> list[float]:
    """The teacher's probabilities aligned to the row's options, in the row's own order."""
    options = row["options"]
    if row["qtype"] == "choice":
        by_label = answer["probabilities"]
        aligned = [float(by_label[str(option["label"])]) for option in options]
    elif row["qtype"] == "noul":
        p_yes = float(answer["noul"])
        aligned = [p_yes if str(option["key"]) == "yes" else 1.0 - p_yes for option in options]
    elif row["qtype"] == "score":
        by_level = answer["probabilities"]
        aligned = [float(by_level[str(index)]) for index in range(len(options))]
    else:
        raise ValueError(f"no teacher mapping for qtype {row['qtype']!r}")
    if any(not math.isfinite(p) or p < 0 for p in aligned) or abs(sum(aligned) - 1.0) > PROB_TOLERANCE:
        raise ValueError("teacher probabilities are not a distribution over the row's options")
    return aligned


def argmax_index(values: Sequence[float]) -> int:
    """Index of the largest value; a tie goes to the earlier option."""
    return max(range(len(values)), key=values.__getitem__)


def argmax_correct(row: Mapping[str, Any], probs: Sequence[float]) -> bool:
    return str(row["options"][argmax_index(probs)]["key"]) == str(row["gold"])


def compare_scorings(reference: Sequence[Sequence[float]], other: Sequence[Sequence[float]]) -> dict[str, Any]:
    """Argmax agreement and probability differences between two scorings of the same rows, in the same option order."""
    if len(reference) != len(other):
        raise ValueError("the two scorings have different numbers of rows")
    agree = 0
    max_diff = 0.0
    diff_sum = 0.0
    values = 0
    for ref, got in zip(reference, other, strict=True):
        if len(ref) != len(got):
            raise ValueError("the two scorings disagree on the option count of a row")
        agree += argmax_index(ref) == argmax_index(got)
        for a, b in zip(ref, got, strict=True):
            max_diff = max(max_diff, abs(a - b))
            diff_sum += abs(a - b)
            values += 1
    return {"rows": len(reference), "argmax_agree": agree, "max_abs_diff": max_diff,
            "mean_abs_diff": diff_sum / values if values else 0.0}


def accuracy_by_qtype(scored: Iterable[tuple[Mapping[str, Any], Sequence[float]]]) -> dict[str, dict[str, float | int]]:
    """Teacher argmax accuracy against the gold key, per qtype and over all rows (the denominator is stated)."""
    counts: dict[str, list[int]] = {"all": [0, 0]}
    for row, probs in scored:
        correct = int(argmax_correct(row, probs))
        for bucket in (counts["all"], counts.setdefault(str(row["qtype"]), [0, 0])):
            bucket[0] += 1
            bucket[1] += correct
    return {name: {"items": items, "correct": correct, "accuracy": correct / items}
            for name, (items, correct) in counts.items()}


def overlap_keys(row: Mapping[str, Any]) -> tuple[str, RecordKey | None]:
    """The normalised-text key and the (source, record id) key of a row; the record key is None without both fields."""
    source = str(row.get("source") or "")
    record = str(row.get("source_record_id") or "")
    return row_text_key(row), ((source, record) if source and record else None)


def overlap_reason(row: Mapping[str, Any], text_keys: set[str], record_keys: set[RecordKey]) -> str | None:
    """Why a row overlaps the exclusion sets (``text``, ``record``, ``text and record``), or None."""
    text_key, record_key = overlap_keys(row)
    text = text_key in text_keys
    record = record_key is not None and record_key in record_keys
    if text and record:
        return "text and record"
    if text:
        return "text"
    if record:
        return "record"
    return None


def pool_order_key(key: str, seed: int = 0) -> str:
    """Stable order of the anchor pool, by the stable hash of a unique key (the normalised-text key of a pool row).

    The item id is not a usable key: its hash leaves the state text out, so Winogrande twins (one record, two sentences)
    share an id. The pool is keyed by text, and the scored records are keyed by pool position.
    """
    return stable_hash({"seed": seed, "tag": "y5_anchor_order", "key": str(key)}, length=16)


def soft_label_row(row: Mapping[str, Any], probs: Sequence[float]) -> dict[str, Any]:
    """The training row with the teacher's distribution over its options (in the row's option order)."""
    return {**row, "teacher_probs": [float(p) for p in probs], "teacher": TEACHER}


def attach_teacher(
    row: Mapping[str, Any], soft: Mapping[str, tuple[list[str], list[float]]]
) -> tuple[dict[str, Any], str]:
    """The row with its teacher distribution when it qualifies, and the status either way (the caller counts statuses).

    Only anchor templates are distilled (the osler_v3 domain prefixes are not), the match is by normalised text, and the
    option keys must be the same and in the same order. ``soft`` maps a text key to (option keys, teacher probabilities).
    """
    anchor = is_anchor(str(row["template_id"]))
    match = soft.get(overlap_keys(row)[0])
    if not anchor:
        return dict(row), "domain: text has a soft label (not distilled)" if match else "domain: no soft label"
    if match is None:
        return dict(row), "anchor: no soft label"
    keys, probs = match
    if keys != [str(option["key"]) for option in row["options"]]:
        return dict(row), "anchor: soft label has other option keys (not attached)"
    return soft_label_row(row, probs), "anchor: attached"
