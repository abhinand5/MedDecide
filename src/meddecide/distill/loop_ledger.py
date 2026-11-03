"""Loop-level API cost ledger for distill_v0 (D1, D2).

One append-only JSONL at ``outputs/distill_v0/cost_ledger.jsonl``; every completed (or resumed-away)
label run appends an entry with its spend, so the $15 hard cap is enforced over the loop, not per
process. Entries are aggregates only: task, timestamps, calls, failures, tokens, cost.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from meddecide.utils.provenance import utcnow


def read_entries(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    entries: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                entries.append(json.loads(line))
    return entries


def prior_cost(entries: list[dict[str, Any]]) -> float:
    """The loop's spend before the current run, summed from the ledger entries."""
    return float(sum(float(entry.get("cost_usd") or 0.0) for entry in entries))


def append_entry(path: Path, entry: dict[str, Any]) -> dict[str, Any]:
    """Append one entry (with an appended timestamp) and return it as written."""
    record = {**entry, "recorded_utc": utcnow()}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    return record
