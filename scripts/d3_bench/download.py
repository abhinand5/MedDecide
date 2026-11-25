"""Download one pinned Hub snapshot into HF_HOME and print its local path (d3_bench step 1).

Usage:
    HF_HOME=/workspace/.hf_home uv run --frozen python scripts/d3_bench/download.py \
        --repo vllm-sr/d3-ultra --revision ccdcfa611649a59e095e0b5e1732a15cedafadd6
"""

from __future__ import annotations

import argparse

from huggingface_hub import snapshot_download


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--repo", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    path = snapshot_download(args.repo, revision=args.revision, max_workers=args.workers)
    print("DONE", path, flush=True)


if __name__ == "__main__":
    main()
