#!/usr/bin/env python
"""Record the code state a run used: HEAD, dirty files, and a hash of the tracked diff plus every
untracked file (R7). Usage: ``python scripts/student/v2_code_state.py <out.json>``."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout


def main() -> None:
    untracked = sorted(git("ls-files", "--others", "--exclude-standard").splitlines())
    digest = hashlib.sha256(git("diff").encode())
    for name in untracked:
        digest.update(name.encode() + b"\0" + (ROOT / name).read_bytes())
    state = {
        "head": git("rev-parse", "HEAD").strip(),
        "dirty_files": sorted(git("status", "--porcelain").splitlines()),
        "code_sha256": digest.hexdigest(),
        "recorded_at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    out = Path(sys.argv[1])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"head": state["head"], "code_sha256": state["code_sha256"]}))


if __name__ == "__main__":
    main()
