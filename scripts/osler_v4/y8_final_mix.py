"""osler_v4 final mix (operator ruling, Stage B, 2026-10-23, items 1 and 3; ADVISORY Y8): the kept mix for the final run (CPU).

Entering blocks. Item 1: each MIMIC-family block (B1, B2, B3) enters the final mix if its own lower bound is above +0.02 and its
knowledge lower bound is above -0.02; its keep rule v2 verdict is reported unchanged. B4 enters only if keep rule v2 kept it (it is
judged on B0). The base is osler_v2's C1 (A0's mix). The build is meddecide.train.final_mix.build_final_mix, which checks every cap
before it writes. All four keep decisions must exist; a missing decision stops the script.

Writes data/train/osler_v4/final_mix.jsonl (training rows, gitignored) and outputs/osler_v4/Y8/final_mix.json (counts, shares, the
entering rule of each block, file hashes).

Usage (CPU; after the keep decisions of B1 to B4 exist):
    uv run --frozen python scripts/osler_v4/y8_final_mix.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from meddecide.utils.io import read_json

REPO = Path(__file__).resolve().parents[2]
Y7 = REPO / "outputs" / "osler_v4" / "Y7"
Y8 = REPO / "outputs" / "osler_v4" / "Y8"
BASE = REPO / "data" / "train" / "osler_v2" / "block_C1.jsonl"
BROAD = REPO / "data" / "train" / "osler_v3" / "replay_broad_train.jsonl"
OUT = REPO / "data" / "train" / "osler_v4" / "final_mix.jsonl"
FAMILY = ("B1", "B2", "B3")
OWN_LIMIT = 0.02  # ruling item 1: the own lower bound must be above this
KNOWLEDGE_LIMIT = -0.02  # ruling item 1: the knowledge lower bound must be above this


def resolve(path: str) -> Path:
    return Path(path) if Path(path).is_absolute() else REPO / path


def entering(name: str) -> tuple[bool, str, dict[str, Any]]:
    """Whether a block enters the final mix, the reason, and the bounds the decision used."""
    keep = read_json(Y7 / f"keep_{name}.json")
    lower = {k: float(v["lower"]) for k, v in keep["statistics"].items()}
    kept = bool(keep["verdict"]["kept"])
    if name in FAMILY:
        enters = lower["own"] > OWN_LIMIT and lower["knowledge"] > KNOWLEDGE_LIMIT
        reason = (f"operator ruling item 1: own lower {lower['own']:+.4f} against +0.02; knowledge lower "
                  f"{lower['knowledge']:+.4f} against -0.02; keep rule v2 kept={kept} (reported unchanged)")
    else:
        enters = kept
        reason = "keep rule v2 as written (judged on B0)"
    return enters, reason, {"lower": lower, "keep_rule_v2_kept": kept}


def add_files(name: str) -> list[Path]:
    return [resolve(p) for p in read_json(Y7 / f"block_{name}.json")["add"]]


def main() -> None:
    # The final mix is composed in memory by scripts/osler_v4/y8_train.py: the disk quota cannot hold the ~9.5 GB file
    # (CLAIMS C164). The composition and its counts are printed by scripts/osler_v4/y8_compose_check.py.
    raise SystemExit("y8_final_mix.py no longer writes the mix file; see y8_compose_check.py and y8_train.py")

if __name__ == "__main__":
    main()
