"""Y7 (osler_v4 ADVISORY Y7): the replay of a Stage B block's training file (CPU), written under outputs/osler_v4/Y7/replays.

scripts/osler_v3/x7_replay.py writes its replay to outputs/osler_v3/replays, an earlier loop's output directory that this loop does
not edit (GOAL: never edit earlier outputs). This wrapper runs that script's own main() and redirects only its output directory.
The replay is the batches the trainer will draw for the block (13,992 steps, the chosen recipe); the block's driver refuses a
replay made from another file.

Usage (CPU; after scripts/osler_v4/y7_build_block.py):
    uv run --frozen python scripts/osler_v4/y7_replay.py --name B1
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "outputs" / "osler_v4" / "Y7" / "replays"
STEPS = 13_992


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--name", required=True, help="the block name, e.g. B1")
    args = ap.parse_args()
    spec = importlib.util.spec_from_file_location("x7_replay", REPO / "scripts" / "osler_v3" / "x7_replay.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.OUT = OUT
    OUT.mkdir(parents=True, exist_ok=True)
    train = REPO / "data" / "train" / "osler_v4" / f"block_{args.name}.jsonl"
    sys.argv = [sys.argv[0], "--train", str(train), "--name", args.name, "--steps", str(STEPS)]
    return module.main()


if __name__ == "__main__":
    raise SystemExit(main())
