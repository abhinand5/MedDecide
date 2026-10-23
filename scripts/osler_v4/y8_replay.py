"""osler_v4 Y8 (ADVISORY Y8): the replay of the final mix's training file (CPU), written under outputs/osler_v4/Y8/replays.

The replay is the batches the trainer draws for the final run (63,304 steps on the chosen recipe). It is the same generator as the Stage B
replays (scripts/osler_v3/x7_replay.py, run through scripts/osler_v4/y7_replay.py's pattern); only the output directory, the training
file and the step count differ. The Y8 driver refuses a replay made from another file.

Usage (CPU; after scripts/osler_v4/y8_final_mix.py):
    uv run --frozen python scripts/osler_v4/y8_replay.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "outputs" / "osler_v4" / "Y8" / "replays"
MIX = REPO / "data" / "train" / "osler_v4" / "final_mix.jsonl"
STEPS = 63_304


def main() -> int:
    spec = importlib.util.spec_from_file_location("x7_replay", REPO / "scripts" / "osler_v3" / "x7_replay.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.OUT = OUT
    OUT.mkdir(parents=True, exist_ok=True)
    sys.argv = [sys.argv[0], "--train", str(MIX), "--name", "final_mix", "--steps", str(STEPS)]
    return module.main()


if __name__ == "__main__":
    raise SystemExit(main())
