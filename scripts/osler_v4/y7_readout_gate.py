"""Y7 (osler_v4 ADVISORY Y7; D12, D21): the readout gate of one Stage B arm, template by template, on fast dev v4 and CG v2 (CPU).

scripts/osler_v4/y6_readout_gate.py writes to outputs/osler_v4/Y6, the Stage A output directory that this loop does not edit
(GOAL: never edit earlier outputs). This wrapper runs that script's own main() and redirects only its output directory, as
y7_replay.py does for osler_v3's replay script. The gate itself is unchanged.

Usage (CPU; after the arm's predictions exist):
    uv run --frozen python scripts/osler_v4/y7_readout_gate.py --arm B1
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "outputs" / "osler_v4" / "Y7"


def main() -> None:
    spec = importlib.util.spec_from_file_location("y6_readout_gate", REPO / "scripts" / "osler_v4" / "y6_readout_gate.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.OUT = OUT
    module.main()


if __name__ == "__main__":
    main()
