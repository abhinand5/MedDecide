"""Provenance helpers (AGENTS.md R7).

Every run records: git commit, command, config, seed, model ids + revisions, dataset ids
+ revisions, hardware, and wall-clock. These helpers collect the parts that can be
derived automatically; callers add the rest (model ids, seeds) explicitly.
"""

from __future__ import annotations

import json
import os
import platform
import socket
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

__version__ = "0.0.1"


def utcnow() -> str:
    """Current UTC time as an ISO-8601 string with a ``Z`` suffix."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def git_commit(repo: Path | None = None) -> str:
    """Full commit hash of HEAD, or ``"UNKNOWN"`` outside a git checkout."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo) if repo else None,
            capture_output=True,
            text=True,
            check=True,
            timeout=20,
        )
        return out.stdout.strip()
    except (subprocess.SubprocessError, OSError):  # pragma: no cover - env dependent
        return "UNKNOWN"


def git_dirty(repo: Path | None = None) -> bool:
    """True when the working tree has uncommitted changes."""
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(repo) if repo else None,
            capture_output=True,
            text=True,
            check=True,
            timeout=20,
        )
        return bool(out.stdout.strip())
    except (subprocess.SubprocessError, OSError):  # pragma: no cover - env dependent
        return False


@dataclass(slots=True)
class Provenance:
    """Machine-readable provenance for one run."""

    run_name: str
    command: str
    started_at: str = field(default_factory=utcnow)
    finished_at: str | None = None
    wall_clock_s: float | None = None
    git_commit: str = field(default_factory=git_commit)
    git_dirty: bool = field(default_factory=git_dirty)
    package_version: str = __version__
    hostname: str = field(default_factory=socket.gethostname)
    platform: str = field(default_factory=platform.platform)
    python: str = field(default_factory=lambda: sys.version.split()[0])
    gpu: str | None = None
    seed: int | None = None
    config: dict[str, Any] = field(default_factory=dict)
    models: list[dict[str, Any]] = field(default_factory=list)
    datasets: list[dict[str, Any]] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    notes: str = ""

    def finish(self) -> Provenance:
        """Stamp the end time and wall-clock duration."""
        self.finished_at = utcnow()
        try:
            start = datetime.strptime(self.started_at, "%Y-%m-%dT%H:%M:%SZ")
            end = datetime.strptime(self.finished_at, "%Y-%m-%dT%H:%M:%SZ")
            self.wall_clock_s = (end - start).total_seconds()
        except ValueError:  # pragma: no cover - only if started_at was overridden
            self.wall_clock_s = None
        return self

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def write(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        return path


def gpu_name() -> str | None:
    """First GPU's name via ``nvidia-smi``, or ``None`` when unavailable."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
            timeout=20,
        )
        first = out.stdout.strip().splitlines()[0]
        return first.strip()
    except (subprocess.SubprocessError, OSError, IndexError):  # pragma: no cover
        return None


def cache_paths() -> dict[str, str]:
    """Cache locations that must live on the large volume (AGENTS.md environment rules)."""
    keys = ("HF_HOME", "UV_CACHE_DIR", "TMPDIR", "TORCH_HOME")
    return {k: os.environ.get(k, "") for k in keys}


class Timer:
    """Context manager returning elapsed seconds (used for throughput measurements)."""

    def __enter__(self) -> Timer:
        self._t0 = time.perf_counter()
        self.elapsed_s = 0.0
        return self

    def __exit__(self, *exc: object) -> None:
        self.elapsed_s = time.perf_counter() - self._t0
