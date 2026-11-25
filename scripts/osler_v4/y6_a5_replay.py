"""Y6 (osler_v4 ADVISORY Y6): the A5 replay of data/train/osler_v4/y6_a5_mix.jsonl (CPU), written under outputs/osler_v4/Y6/replays.

scripts/osler_v3/x7_replay.py writes its replay to outputs/osler_v3/replays, an earlier loop's output directory that this loop does
not edit (GOAL: never edit earlier outputs). This wrapper runs that script's own main() and redirects only its output directory.
The replay is the batches the trainer will draw for A5 (13,992 steps, the osler_v3 recipe); y6_stage_a.py train refuses a replay
made from another file.

Usage (CPU; after scripts/osler_v4/y6_a5_mix.py):
    uv run --frozen python scripts/osler_v4/y6_a5_replay.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MIX = REPO / "data" / "train" / "osler_v4" / "y6_a5_mix.jsonl"
OUT = REPO / "outputs" / "osler_v4" / "Y6" / "replays"
STEPS = 13_992


def main() -> int:
    spec = importlib.util.spec_from_file_location("x7_replay", REPO / "scripts" / "osler_v3" / "x7_replay.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.OUT = OUT
    OUT.mkdir(parents=True, exist_ok=True)
    sys.argv = [sys.argv[0], "--train", str(MIX), "--name", "A5", "--steps", str(STEPS)]
    return module.main()


if __name__ == "__main__":
    raise SystemExit(main())
