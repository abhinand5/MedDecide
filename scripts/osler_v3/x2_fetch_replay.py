"""osler_v3 X2: fetch the broad general replay sources at pinned revisions and record their SHA-256 (ADVISORY X2; D36).

Human-labelled general multiple-choice and yes/no sets, each with a licence verified on its Hub card or its README (recorded
in the table below). Train splits only; no test, dev or panel split is read into training. Files:

- StrategyQA (MIT), QuaRTz (CC BY 4.0), AQuA-RAT (Apache-2.0), MathQA (Apache-2.0): Hub parquet or archives at pinned revisions;
- BIG-bench subtasks (Apache-2.0; tasksource/bigbench): five human-written subtasks, train files;
- SocialIQA (CC BY 4.0, README): the AI2 archive (its loader script's URL), extracted into its own directory;
- CosmosQA (CC BY 4.0, card): the training CSV at the GitHub commit that last changed it.

Writes outputs/osler_v3/X2/fetch.json. Idempotent: a file already present with its recorded SHA-256 is not fetched again.
Never reads MIMIC, the external panel or any test split (D37, D42).

Usage:  source scripts/pod_env.sh && uv run --frozen python scripts/osler_v3/x2_fetch_replay.py
"""

from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download

from meddecide.utils.io import write_json
from meddecide.utils.provenance import git_commit, utcnow

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "outputs" / "osler_v3" / "X2" / "fetch.json"
RAW = REPO / "data" / "raw" / "replay"
HF_FILES: list[tuple[str, str, str, str, str]] = [
    # (source, repo, revision, file, licence as recorded on the card)
    ("strategyqa", "ChilleD/StrategyQA", "705562638fe1d8ca6bb98c66fc8f94d45fda8c83", "data/train-00000-of-00001-506370352f622815.parquet", "MIT"),
    ("quartz", "allenai/quartz", "28c1dbb56caf81799296cb17892fa73402e23464", "data/train-00000-of-00001.parquet", "CC BY 4.0"),
    ("aqua_rat", "deepmind/aqua_rat", "33301c6a050c96af81f63cad5562cb5363e88971", "raw/train-00000-of-00001.parquet", "Apache-2.0"),
    ("bigbench/snarks", "tasksource/bigbench", "b8ee34693560a13a9feac6aa5540449e18b7eddb", "snarks/train-00000-of-00001.parquet", "Apache-2.0"),
    ("bigbench/ruin_names", "tasksource/bigbench", "b8ee34693560a13a9feac6aa5540449e18b7eddb", "ruin_names/train-00000-of-00001.parquet", "Apache-2.0"),
    ("bigbench/sports_understanding", "tasksource/bigbench", "b8ee34693560a13a9feac6aa5540449e18b7eddb",
     "sports_understanding/train-00000-of-00001.parquet", "Apache-2.0"),
    ("bigbench/causal_judgment", "tasksource/bigbench", "b8ee34693560a13a9feac6aa5540449e18b7eddb", "causal_judgment/train-00000-of-00001.parquet", "Apache-2.0"),
    ("bigbench/date_understanding", "tasksource/bigbench", "b8ee34693560a13a9feac6aa5540449e18b7eddb", "date_understanding/train-00000-of-00001.parquet",
     "Apache-2.0"),
]
SOCIAL_IQA_URL = "https://storage.googleapis.com/ai2-mosaic/public/socialiqa/socialiqa-train-dev.zip"
COSMOS_REPO = "wilburOne/cosmosqa"
COSMOS_PATH = "data/train.csv"
MATHQA_URL = "https://math-qa.github.io/math-QA/data/MathQA.zip"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, path: Path, *, expected: str | None = None) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or (expected is not None and sha256(path) != expected):
        urllib.request.urlretrieve(url, path)
    digest = sha256(path)
    if expected is not None and digest != expected:
        raise RuntimeError(f"{path.name}: SHA-256 {digest} does not match {expected}")
    return digest


def cosmos_commit() -> str:
    url = f"https://api.github.com/repos/{COSMOS_REPO}/commits?path={COSMOS_PATH}&per_page=1"
    with urllib.request.urlopen(url, timeout=60) as resp:
        return str(json.loads(resp.read())[0]["sha"])


def main() -> None:
    files: list[dict[str, Any]] = []
    for source, repo, revision, filename, licence in HF_FILES:
        path = Path(hf_hub_download(repo, filename, repo_type="dataset", revision=revision))
        files.append({"source": source, "repo": repo, "revision": revision, "file": filename, "licence": licence,
                      "bytes": path.stat().st_size, "sha256": sha256(path), "cache_path_is_outside_repo": True})
    social = RAW / "social_iqa" / "socialiqa-train-dev.zip"
    digest = download(SOCIAL_IQA_URL, social)
    files.append({"source": "social_iqa", "url": SOCIAL_IQA_URL, "file": str(social.relative_to(REPO)),
                  "licence": "CC BY 4.0 (README)", "bytes": social.stat().st_size, "sha256": digest})
    commit = cosmos_commit()
    cosmos = RAW / "cosmos_qa" / commit / "train.csv"
    digest = download(f"https://raw.githubusercontent.com/{COSMOS_REPO}/{commit}/{COSMOS_PATH}", cosmos)
    files.append({"source": "cosmos_qa", "repo": COSMOS_REPO, "revision": commit, "file": str(cosmos.relative_to(REPO)),
                  "licence": "CC BY 4.0 (card)", "bytes": cosmos.stat().st_size, "sha256": digest})
    mathqa = RAW / "math_qa" / "MathQA.zip"
    digest = download(MATHQA_URL, mathqa)
    files.append({"source": "math_qa", "url": MATHQA_URL, "file": str(mathqa.relative_to(REPO)), "licence": "Apache-2.0 (card)",
                  "bytes": mathqa.stat().st_size, "sha256": digest})
    write_json(OUT, {"fetched_utc": utcnow(), "git_commit": git_commit(REPO), "files": files,
                     "command": "scripts/osler_v3/x2_fetch_replay.py"})
    for row in files:
        print(f"{row['source']}: {row['bytes']} bytes sha256 {row['sha256'][:12]}", flush=True)


if __name__ == "__main__":
    main()
