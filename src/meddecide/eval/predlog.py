"""Reading append-only prediction logs (bench_v0_fix0 P7).

Prediction JSONL files are **append-only across runs**: the F6/F7 files accumulated rows on
every re-run, so a row count can exceed the item count (the JEV-9B MedQA file held 5,092 rows
for 1,273 items). Every consumer therefore goes through this reader, which

* deduplicates by ``(run_id, item_id)``, keeping the **last** row for a key (a later run
  supersedes an earlier one),
* treats a missing ``run_id`` as the empty string, so files written before loop 1 still read
  as one run per model,
* reports the raw row count, the unique count, and the distinct-item count side by side, so a
  reader never has to guess which number they are looking at.

Nothing here rewrites a file: raw outputs are immutable (R9).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class PredictionLog:
    """Deduplicated view of one prediction file, with the raw counts kept beside it."""

    path: str
    rows: list[dict[str, Any]] = field(default_factory=list)
    n_raw: int = 0
    n_without_run_id: int = 0
    unreadable_lines: int = 0

    @property
    def n_unique(self) -> int:
        """Rows after deduplication by ``(run_id, item_id)``."""
        return len(self.rows)

    @property
    def n_duplicate_rows(self) -> int:
        return self.n_raw - self.n_unique - self.unreadable_lines

    @property
    def n_distinct_items(self) -> int:
        return len({str(row.get("item_id")) for row in self.rows})

    @property
    def run_ids(self) -> list[str]:
        return sorted({str(row.get("run_id") or "") for row in self.rows})

    def rows_for_run(self, run_id: str) -> list[dict[str, Any]]:
        """Exactly the rows of one run — one row per item, the unit a metric is computed on.

        This is the call a consumer should make: the whole file is a *log* of every run that
        ever appended to it, so counting all its rows counts the same item once per run.
        """
        return [row for row in self.rows if str(row.get("run_id") or "") == run_id]

    def per_run(self) -> dict[str, dict[str, int]]:
        counts: dict[str, dict[str, int]] = {}
        for run_id in self.run_ids:
            rows = self.rows_for_run(run_id)
            counts[run_id] = {
                "n_rows": len(rows),
                "n_distinct_items": len({str(row.get("item_id")) for row in rows}),
            }
        return counts

    def report(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "n_raw_rows": self.n_raw,
            "n_unique_rows": self.n_unique,
            "n_duplicate_rows": self.n_duplicate_rows,
            "n_distinct_items": self.n_distinct_items,
            "n_rows_without_run_id": self.n_without_run_id,
            "run_ids": self.run_ids,
            "per_run": self.per_run(),
            "unreadable_lines": self.unreadable_lines,
            "dedupe_key": "(run_id, item_id), last row wins",
            "note": (
                "rows from different runs are kept: filter to one run_id (rows_for_run) "
                "before computing a metric"
            ),
        }


def read_prediction_log(path: Path) -> PredictionLog:
    """Read ``path`` and return the deduplicated view (last row per ``(run_id, item_id)``)."""
    log = PredictionLog(path=str(path))
    if not path.is_file():
        return log
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            log.n_raw += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                log.unreadable_lines += 1
                continue
            run_id = str(row.get("run_id") or "")
            if not run_id:
                log.n_without_run_id += 1
            by_key[(run_id, str(row.get("item_id")))] = row
    log.rows = list(by_key.values())
    return log
