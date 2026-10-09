#!/usr/bin/env python
"""V0 snapshot for student_v1: git commit, benchmark and training-mix hashes, test count, env versions.

Run from the repo root with the main environment for ``snapshot.json`` and with the Unsloth
environment for ``--env unsloth``; each call writes one file under ``outputs/student_v1/V0/``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

OUT_DIR = Path("outputs/student_v1/V0")
PACKAGES = ("torch", "transformers", "unsloth", "unsloth_zoo", "peft", "accelerate")
HASHED = {
    "bench_v0_2_manifest": "data/bench/v0.2/manifest.json",
    "student_v0_training_manifest": "data/train/student_v0/manifest.json",
    "student_v0_train_jsonl": "data/train/student_v0/train.jsonl",
    "student_v0_dev_jsonl": "data/train/student_v0/dev.jsonl",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_versions() -> dict[str, str | None]:
    out: dict[str, str | None] = {"python": sys.version.split()[0]}
    for name in PACKAGES:
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            out[name] = None
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", choices=["main", "unsloth"], default="main")
    parser.add_argument("--tests", default=None, help="pytest summary line, recorded verbatim")
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.env == "unsloth":
        payload = {"env": "envs/unsloth", "versions": package_versions()}
        (OUT_DIR / "envs_unsloth.json").write_text(json.dumps(payload, indent=2) + "\n")
        print(json.dumps(payload, indent=2))
        return 0

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
    ).stdout.strip()
    snapshot = {
        "git_commit": commit,
        "git_worktree_clean": dirty == "",
        "branch": "loop/student_v1",
        "sha256": {name: sha256(Path(p)) for name, p in HASHED.items()},
        "paths": HASHED,
        "pytest": args.tests,
        "env_main": {"env": "project .venv", "versions": package_versions()},
    }
    (OUT_DIR / "snapshot.json").write_text(json.dumps(snapshot, indent=2) + "\n")
    unsloth_env = OUT_DIR / "envs_unsloth.json"
    envs = {
        "main": snapshot["env_main"],
        "unsloth": json.loads(unsloth_env.read_text()) if unsloth_env.exists() else None,
    }
    (OUT_DIR / "envs.json").write_text(json.dumps(envs, indent=2) + "\n")
    print(json.dumps(snapshot, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
