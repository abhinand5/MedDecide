"""Unsloth decision answers as prediction rows in the run_student format (student_v1 V5, V8).

An Unsloth answer carries probabilities over the offered options. This module orders them by the
item's option keys, takes the argmax from those probabilities, and writes the same fields as
``run_student.py`` (``Prediction.to_json``), so ``scripts/bench/g1.py`` reads both kinds of file.
Readout-only fields (label mass, vocabulary argmax, top-5 rank) do not apply to a decision head and
are written as ``None``; the readout-health gate therefore reads these cells as non-letter readouts.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

RAW_KEYS = {"noul": {"yes": "true", "no": "false"}}


def raw_key(qtype: str, key: str) -> str:
    """The key Unsloth uses for one of our option keys (score levels are numbered from 0 there)."""
    if qtype == "noul":
        return RAW_KEYS["noul"][key]
    if qtype == "score":
        return str(int(key) - 1)
    return key


def option_probabilities(qtype: str, raw_probs: Mapping[str, float], option_keys: Sequence[str]) -> list[float]:
    """Probabilities over ``option_keys`` (in that order) from an Unsloth answer's probability dict."""
    values = []
    for key in option_keys:
        mapped = raw_key(qtype, key)
        if mapped not in raw_probs:
            raise KeyError(f"no probability for option {key!r} (raw key {mapped!r}) in {sorted(raw_probs)}")
        values.append(float(raw_probs[mapped]))
    total = float(sum(values))
    if not np.isfinite(total) or total <= 0:
        raise ValueError("option probabilities must be finite and positive in total")
    return [v / total for v in values]


def prediction_row(
    item: Mapping[str, Any],
    probs: Sequence[float],
    *,
    model_id: str,
    run_id: str,
    split: str,
    latency_s: float | None = None,
    prompt_tokens: int | None = None,
) -> dict[str, Any]:
    """One run_student-format prediction for an item and its option probabilities."""
    option_keys = [str(o["key"]) for o in item["options"]]
    if len(probs) != len(option_keys):
        raise ValueError("probabilities do not match the item's options")
    p = np.asarray(probs, dtype=np.float64)
    argmax_key = option_keys[int(np.argmax(p))]
    gold = str(item["gold"])
    qtype = str(item["qtype"])
    expected_level = None
    if qtype == "score":
        expected_level = float(sum(float(option_keys[i]) * p[i] for i in range(len(option_keys))))
    return {
        "item_id": str(item["item_id"]),
        "model_id": model_id,
        "run_id": run_id,
        "source": str(item.get("source", "")),
        "template_id": str(item["template_id"]),
        "split": split,
        "qtype": qtype,
        "option_keys": option_keys,
        "original_option_keys": option_keys,
        "option_probs": [float(v) for v in p],
        "label_mass": None,
        "vocab_argmax_is_option": None,
        "n_tokens_above_best_option": None,
        "best_option_in_top5": None,
        "top1_over_option_mass": None,
        "argmax_key": argmax_key,
        "gold_key": gold,
        "correct": bool(argmax_key == gold),
        "expected_level": expected_level,
        "latency_s": latency_s,
        "variant": "unsloth-decision-head",
        "prompt_tokens": prompt_tokens,
        "transform": {"readout": "unsloth-decision-head", "letter_readout_fields": "not applicable"},
    }
