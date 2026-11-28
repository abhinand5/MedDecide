#!/usr/bin/env python
"""Deploy the static leaderboard build (`site/dist/`) to Hugging Face Static Space
`abhinand/MedDecide-Leaderboard`, add it to the Osler HF Collection, and link the landing page
and Space in the Osler-9B, Osler-4B, and MedDecide-Bench cards.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from huggingface_hub import HfApi

REPO = Path(__file__).resolve().parents[2]
SITE_DIST = REPO / "site/dist"
STAGE_DIR = REPO / "outputs/osler_rc0/R8/hf_space_staging"
COLLECTION_SLUG = "abhinand/osler-calibrated-medical-decision-models-6b0af4d7e2e7a5eed75e3619"

SPACE_README = """---
title: MedDecide & Osler Leaderboard
emoji: 🩺
colorFrom: green
colorTo: blue
sdk: static
pinned: false
license: apache-2.0
short_description: Pareto frontier & medical decision model leaderboard
models:
- abhinand/Osler-9B
- abhinand/Osler-4B
datasets:
- abhinand/MedDecide-Bench
---

# MedDecide & Osler Leaderboard

Interactive two-tier medical decision model leaderboard comparing 12 models across General Medical QA (`panel`, `knowledge`, fresh post-cutoff `confirm_general`) and EHR & Clinical Documentation understanding (`mr_verdict`, `mimic_suite`).

- **Full Project Page:** [abhinand5.github.io/MedDecide](https://abhinand5.github.io/MedDecide)
- **Models:** [`abhinand/Osler-9B`](https://huggingface.co/abhinand/Osler-9B) · [`abhinand/Osler-4B`](https://huggingface.co/abhinand/Osler-4B)
- **Public Benchmark:** [`abhinand/MedDecide-Bench`](https://huggingface.co/datasets/abhinand/MedDecide-Bench)
- **Code & Submissions:** [github.com/abhinand5/MedDecide](https://github.com/abhinand5/MedDecide)
"""


def load_hf_token2() -> str:
    if os.environ.get("HF_TOKEN2"):
        return os.environ["HF_TOKEN2"].strip()
    for candidate in (Path("/workspace/secrets.env"), Path("/workspace/.secrets.env")):
        if not candidate.is_file():
            continue
        for line in candidate.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            if line.startswith("export "):
                line = line[len("export ") :].strip()
            k, v = line.split("=", 1)
            if k.strip() == "HF_TOKEN2":
                return v.strip().strip('"').strip("'")
    raise RuntimeError("HF_TOKEN2 not found")


def stage_space() -> Path:
    if not (SITE_DIST / "leaderboard.html").is_file():
        raise FileNotFoundError(f"{SITE_DIST / 'leaderboard.html'} missing — run `npm run build` in site/ first")
    if STAGE_DIR.exists():
        shutil.rmtree(STAGE_DIR)
    STAGE_DIR.mkdir(parents=True)

    # Serve leaderboard-first view as index.html on the HF Static Space
    shutil.copy2(SITE_DIST / "leaderboard.html", STAGE_DIR / "index.html")
    shutil.copy2(SITE_DIST / "index.html", STAGE_DIR / "landing.html")
    shutil.copytree(SITE_DIST / "assets", STAGE_DIR / "assets")
    (STAGE_DIR / "README.md").write_text(SPACE_README, encoding="utf-8")
    return STAGE_DIR


def main() -> None:
    stage_dir = stage_space()
    api = HfApi(token=load_hf_token2())
    who = api.whoami()["name"]
    assert who == "abhinand", f"Expected abhinand, got {who}"

    space_id = "abhinand/MedDecide-Leaderboard"
    api.create_repo(
        repo_id=space_id,
        repo_type="space",
        space_sdk="static",
        exist_ok=True,
        private=False,
    )
    info = api.upload_folder(
        repo_id=space_id,
        repo_type="space",
        folder_path=str(stage_dir),
        commit_message="Deploy MedDecide & Osler interactive static leaderboard",
        delete_patterns="*",
    )
    print(f"Space live -> https://huggingface.co/spaces/{space_id} ({getattr(info, 'commit_url', info)})")

    api.add_collection_item(
        collection_slug=COLLECTION_SLUG,
        item_id=space_id,
        item_type="space",
        note="Interactive two-tier medical decision model leaderboard, 2-axis Pareto frontier & calibration analytics",
        exists_ok=True,
    )
    print(f"Added Space to collection -> https://huggingface.co/collections/{COLLECTION_SLUG}")


if __name__ == "__main__":
    main()
