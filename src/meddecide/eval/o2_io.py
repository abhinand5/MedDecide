"""Shared I/O for the osler_v0 O2 scoreboard: the common item file and per-model prediction files.

A prediction row is written for every item a model is asked about: scored (with probabilities aligned to the
item's option keys) or skipped (with a reason). Nothing is dropped silently, and a restarted run resumes from
the rows already written.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

PANEL_AND_ROBUSTNESS = ("ext_panel", "robustness")


@dataclass(frozen=True)
class Scope:
    """Which benchmarks a model is scored on."""

    name: str
    benchmarks: tuple[str, ...]


ALL_SETS = Scope("all", ("v0.2", "ext_panel", "robustness"))
PANEL_ROBUSTNESS = Scope("panel_robustness", PANEL_AND_ROBUSTNESS)


def iter_items(path: Path, scope: Scope) -> Iterator[dict[str, Any]]:
    """Common-set items in file order, restricted to the scope's benchmarks."""
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            if row["benchmark"] in scope.benchmarks:
                yield row


def completed_ids(out_path: Path) -> set[str]:
    """Item ids already written to a prediction file (for resuming)."""
    if not out_path.exists():
        return set()
    done: set[str] = set()
    with out_path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                done.add(json.loads(line)["item_id"])
    return done


def open_predictions(out_path: Path) -> TextIO:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return out_path.open("a", encoding="utf-8")


def prediction_row(item: dict[str, Any], *, status: str, reason: str | None = None,
                   probs: list[float] | None = None, latency_s: float | None = None,
                   prompt_tokens: int | None = None, extras: dict[str, Any] | None = None) -> dict[str, Any]:
    """The one shape every runner writes. ``probs`` is aligned with the item's option keys."""
    keys = [o["key"] for o in item["options"]]
    if status == "scored" and (probs is None or len(probs) != len(keys)):
        raise ValueError(f"{item['item_id']}: probabilities must align with the {len(keys)} option keys")
    if status not in ("scored", "skipped"):
        raise ValueError(f"unknown status {status!r}")
    return {
        "item_id": item["item_id"],
        "benchmark": item["benchmark"],
        "set_name": item["set_name"],
        "template_id": item["template_id"],
        "qtype": item["qtype"],
        "option_keys": keys,
        "gold": item["gold"],
        "status": status,
        "reason": reason,
        "probs": probs,
        "latency_s": latency_s,
        "prompt_tokens": prompt_tokens,
        "extras": extras or {},
    }


def write_row(fh: TextIO, row: dict[str, Any]) -> None:
    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    fh.flush()
