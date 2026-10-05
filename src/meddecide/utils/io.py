"""I/O helpers: JSONL read/write with validation counts, and manifest building.

"No silent drops" (AGENTS.md data rules): every reader/writer that can skip a record
returns the counts and the reasons.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError

from meddecide.utils.hashing import file_sha256


def write_json(path: Path, payload: Any) -> Path:
    """Write JSON with stable formatting, creating parent directories."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=False) + "\n")
    return path


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text())


def write_jsonl(path: Path, rows: Iterable[BaseModel | dict[str, Any]]) -> Path:
    """Write one JSON object per line, in the order given."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            payload = row.model_dump(mode="json") if isinstance(row, BaseModel) else row
            fh.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
    return path


@dataclass
class LoadReport:
    """Outcome of a validated JSONL load: what was kept and what was dropped, with reasons."""

    n_lines: int = 0
    n_kept: int = 0
    dropped: Counter[str] = field(default_factory=Counter)
    errors: list[dict[str, Any]] = field(default_factory=list)

    @property
    def n_dropped(self) -> int:
        return self.n_lines - self.n_kept

    def drop(self, reason: str, line_no: int, detail: str = "") -> None:
        self.dropped[reason] += 1
        if len(self.errors) < 50:  # keep the report bounded
            self.errors.append({"line": line_no, "reason": reason, "detail": detail})

    def check_closes(self) -> None:
        """R5: parts must sum to the total."""
        if self.n_kept + self.n_dropped != self.n_lines:
            raise AssertionError(
                f"load report does not close: {self.n_kept} kept + {self.n_dropped} dropped "
                f"!= {self.n_lines} lines"
            )

    def to_dict(self) -> dict[str, Any]:
        self.check_closes()
        return {
            "n_lines": self.n_lines,
            "n_kept": self.n_kept,
            "n_dropped": self.n_dropped,
            "dropped_by_reason": dict(sorted(self.dropped.items())),
            "errors": self.errors,
        }


def read_jsonl[T: BaseModel](
    path: Path,
    model: type[T] | None = None,
    *,
    allow_blank: bool = True,
) -> tuple[list[T | dict[str, Any]], LoadReport]:
    """Read JSONL, optionally validating each line into ``model``.

    Returns ``(rows, report)``; the report accounts for every line, including JSON
    syntax errors and schema-validation failures.
    """
    path = Path(path)
    report = LoadReport()
    rows: list[T | dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            if not line.strip():
                if allow_blank:
                    continue
                report.n_lines += 1
                report.drop("blank_line", line_no)
                continue
            report.n_lines += 1
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                report.drop("invalid_json", line_no, str(exc))
                continue
            if model is None:
                rows.append(payload)
                report.n_kept += 1
                continue
            try:
                rows.append(model.model_validate(payload))
                report.n_kept += 1
            except ValidationError as exc:
                report.drop("schema_validation", line_no, str(exc)[:300])
    report.check_closes()
    return rows, report


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Stream JSONL rows without materialising the file (for large prediction files)."""
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def count_by(rows: Sequence[Any], key: Callable[[Any], Any]) -> dict[str, int]:
    """Count rows by a key function, returning a sorted plain dict (JSON-friendly)."""
    counts: Counter[str] = Counter()
    for row in rows:
        counts[str(key(row))] += 1
    return dict(sorted(counts.items()))


def build_manifest(
    files: dict[str, Path],
    rows_by_tier: dict[str, Sequence[Any]],
    *,
    keys: Sequence[str] = ("source", "split", "qtype", "template_id"),
) -> dict[str, Any]:
    """Manifest of a bench build: per-file sha256 + counts per source x split x qtype x template.

    ``files`` maps a label to the JSONL path; ``rows_by_tier`` maps the same label to the
    validated item rows in that file. Counts are nested dicts so that every part can be
    summed back to the file total (R5).
    """
    manifest: dict[str, Any] = {"files": {}, "counts": {}, "totals": {}}
    for label, path in sorted(files.items()):
        path = Path(path)
        rows = rows_by_tier.get(label, [])
        entry: dict[str, Any] = {
            "path": str(path),
            "exists": path.is_file(),
            "sha256": file_sha256(path) if path.is_file() else None,
            "n_rows": len(rows),
        }
        counts: dict[str, Any] = {}
        for row in rows:
            node = counts
            for i, key in enumerate(keys):
                value = str(getattr(row, key, None) if not isinstance(row, dict) else row.get(key))
                if i == len(keys) - 1:
                    node[value] = node.get(value, 0) + 1
                else:
                    node = node.setdefault(value, {})
        entry["counts"] = counts
        entry["count_by_source"] = count_by(rows, lambda r: r.source)
        entry["count_by_split"] = count_by(rows, lambda r: r.split)
        entry["count_by_qtype"] = count_by(rows, lambda r: r.qtype)
        if sum(entry["count_by_source"].values()) != len(rows):
            raise AssertionError(f"manifest counts do not close for {label}")
        manifest["files"][label] = entry
        manifest["counts"][label] = counts
        manifest["totals"][label] = len(rows)
    return manifest
