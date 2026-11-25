"""osler_v3 X1: fetch the real sources at pinned revisions and record their SHA-256 (ADVISORY X1; D42; R7 provenance).

- RAGTruth (MIT): the original release, github.com/ParticleMedia/RAGTruth at a pinned commit (response.jsonl, source_info.jsonl,
  LICENSE) into data/raw/ragtruth/<commit>/. The processed mirror (wandb/RAGTruth-processed) is not used: it has no licence.
- DrugProt (CC BY 4.0): OpenMed/drugprot-parquet at a pinned revision (train and validation) into the HF cache.
- MultiCaRe (CC BY 4.0): OpenMed/multicare-cases at a pinned revision (cases.parquet) into the HF cache.

Writes outputs/osler_v3/X1/fetch.json (paths relative to the repo, revisions, sizes, SHA-256). Idempotent: a file already present
with the recorded SHA-256 is not downloaded again. Never reads MIMIC or any panel dataset (D37, D42).

Usage:  source scripts/pod_env.sh && uv run --frozen python scripts/osler_v3/x1_fetch.py
"""

from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download

from meddecide.utils.io import write_json
from meddecide.utils.provenance import git_commit, utcnow

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "outputs" / "osler_v3" / "X1" / "fetch.json"
RAGTRUTH_COMMIT = "c103204b9ce28d6bbad859304bf30de72b8ed8fe"
RAGTRUTH_DIR = REPO / "data" / "raw" / "ragtruth" / RAGTRUTH_COMMIT
RAGTRUTH_FILES = {  # file -> SHA-256 recorded when the release was first fetched (2026-10-18)
    "response.jsonl": "e4c2e4ac24fff676d8984cc61c35d791612fadc58015335d97dd632375e18073",
    "source_info.jsonl": "0dffc26ea9f3c1c3d7c7e8336b56ef1646e3cec876edffcca3c9c624d12d578b",
    "LICENSE": "b7fd7d6bdfe0cbba63c63a310914beb4a4acb8bf08da73849219f45385f5b244",
}
DRUGPROT = ("OpenMed/drugprot-parquet", "baba427cb0a238e70dd383bf1ec2a70f4ccef078",
            ("data/train.parquet", "data/validation.parquet"))
MULTICARE = ("OpenMed/multicare-cases", "e5b0a5418dbb6edce6b731c47664cb4f269833bb", ("cases.parquet",))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ragtruth_url(name: str) -> str:
    base = f"https://raw.githubusercontent.com/ParticleMedia/RAGTruth/{RAGTRUTH_COMMIT}"
    return f"{base}/LICENSE" if name == "LICENSE" else f"{base}/dataset/{name}"


def fetch_ragtruth() -> list[dict[str, Any]]:
    RAGTRUTH_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, expected in RAGTRUTH_FILES.items():
        path = RAGTRUTH_DIR / name
        if not (path.exists() and sha256(path) == expected):
            urllib.request.urlretrieve(ragtruth_url(name), path)
        got = sha256(path)
        if got != expected:
            raise RuntimeError(f"{name}: SHA-256 {got} does not match the recorded {expected}")
        rows.append({"source": "RAGTruth", "file": str(path.relative_to(REPO)), "bytes": path.stat().st_size, "sha256": got,
                     "revision": RAGTRUTH_COMMIT, "licence": "MIT"})
    return rows


def fetch_hf(name: str, repo: str, revision: str, files: tuple[str, ...]) -> list[dict[str, Any]]:
    rows = []
    for filename in files:
        path = Path(hf_hub_download(repo, filename, repo_type="dataset", revision=revision))
        rows.append({"source": name, "repo": repo, "file": filename, "revision": revision,
                     "bytes": path.stat().st_size, "sha256": sha256(path), "cache_path_is_outside_repo": True})
    return rows


def main() -> None:
    files = fetch_ragtruth()
    files += fetch_hf("DrugProt", *DRUGPROT)
    files += fetch_hf("MultiCaRe", *MULTICARE)
    write_json(OUT, {"fetched_utc": utcnow(), "git_commit": git_commit(REPO), "files": files,
                     "command": "scripts/osler_v3/x1_fetch.py"})
    for row in files:
        print(f"{row['source']}: {row['file']} {row['bytes']} bytes sha256 {row['sha256'][:12]}", flush=True)


if __name__ == "__main__":
    main()
