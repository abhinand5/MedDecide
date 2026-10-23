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

import sys
from pathlib import Path
from typing import Any

from meddecide.train.final_mix import build_final_mix
from meddecide.utils.hashing import file_sha256
from meddecide.utils.io import read_json, write_json
from meddecide.utils.provenance import git_commit, utcnow

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
    decisions = {name: entering(name) for name in ("B1", "B2", "B3", "B4")}
    family = {name: add_files(name) for name in FAMILY if decisions[name][0]}
    other = add_files("B4") if decisions["B4"][0] else []
    report: dict[str, Any] = build_final_mix(BASE, family, other, BROAD, OUT)
    report.update({
        "built_utc": utcnow(), "git_commit": git_commit(REPO), "command": " ".join(sys.argv),
        "base": str(BASE.relative_to(REPO)), "broad_pool": str(BROAD.relative_to(REPO)), "file": str(OUT.relative_to(REPO)),
        "file_sha256": file_sha256(OUT),
        "entering": {name: {"enters": decisions[name][0], "reason": decisions[name][1], **decisions[name][2]}
                     for name in decisions},
    })
    Y8.mkdir(parents=True, exist_ok=True)
    write_json(Y8 / "final_mix.json", report)
    print(f"final mix: rows {report['rows']}; family kept {report['family_kept']} of {report['family_available']}; "
          f"top-up {report['topup_rows']}; family share {report['family_share']:.4f}; replay share "
          f"{report['replay_share']:.4f}; largest template {report['largest_template_share']:.4f}; checks {report['checks']}",
          flush=True)


if __name__ == "__main__":
    main()
