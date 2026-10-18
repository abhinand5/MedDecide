"""osler_v2 W10: the five-claim spot-check (ADVISORY W10; AGENTS.md R1, R6).

Re-derives five headline numbers from the raw artifacts in a fresh process and compares each with the value recorded in
loops/osler_v2/CLAIMS.md. A mismatch is reported, never edited away. Writes outputs/osler_v2/W10/spot_check.json.

Usage: uv run --frozen python scripts/osler_v2/w10_spot_check.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from meddecide.utils.io import write_json

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "outputs" / "osler_v2" / "W10" / "spot_check.json"


def load(path: str) -> Any:
    return json.loads((REPO / path).read_text(encoding="utf-8"))


def main() -> None:
    checks = []

    # 1. the final model's fast dev v2 accuracy (CLAIMS W052)
    r = load("outputs/osler_v2/W7/osler_4b/block_result.json")
    checks.append({"claim": "W052 final Osler-4B fast dev v2 accuracy", "recorded": 0.8648,
                   "recomputed": round(r["full_dev_metrics"]["accuracy"], 4), "artifact": "outputs/osler_v2/W7/osler_4b/block_result.json"})

    # 2. the base choice's MR lower bound (CLAIMS W045)
    b = load("outputs/osler_v2/W6/keep/base_choice.json")
    checks.append({"claim": "W045 base choice: MR lower bound (C0p minus C0q)", "recorded": 0.0283,
                   "recomputed": round(b["lower_bounds"]["mr"], 4), "artifact": "outputs/osler_v2/W6/keep/base_choice.json"})

    # 3. the gate's panel comparison against zero-shot pplx (CLAIMS W056)
    g = load("outputs/osler_v2/W8/gate_o3.json")
    checks.append({"claim": "W056 panel vs zero-shot pplx: accuracy lower bound", "recorded": -0.0344,
                   "recomputed": round(g["primary"]["external_panel"]["vs"]["zero-shot pplx-decider-flash-v1.2"]["accuracy"]["lo"], 4),
                   "artifact": "outputs/osler_v2/W8/gate_o3.json"})

    # 4. the replay share of C0 from the replay manifest and B1's count (CLAIMS W019; W035)
    m = load("outputs/osler_v2/W2/replay_manifest.json")
    c0_rows = m["c0_rows"]
    replay_rows = m["b1_replay_rows_v1_templates"] + m["new_replay_rows"]["total"]
    checks.append({"claim": "W035 replay share of C0 (v1 + v2 replay rows over C0 rows)", "recorded": 0.2265,
                   "recomputed": round(replay_rows / c0_rows, 4), "artifact": "outputs/osler_v2/W2/replay_manifest.json"})

    # 5. the final run's realised items and its validation (CLAIMS W052)
    checks.append({"claim": "W052 final run realised items (target 250,000) and log mismatches", "recorded": "250044 / 0",
                   "recomputed": f"{r['realised_items']} / {r['validation']['logged_steps']['mismatches']}",
                   "artifact": "outputs/osler_v2/W7/osler_4b/block_result.json"})

    for c in checks:
        c["match"] = (c["recorded"] == c["recomputed"]) if isinstance(c["recorded"], str) else abs(
            float(c["recorded"]) - float(c["recomputed"])) < 1e-4
    report = {"checks": checks, "all_match": all(c["match"] for c in checks)}
    write_json(OUT, report)
    for c in checks:
        print(f"{c['match']!s:5} {c['claim']}: recorded {c['recorded']} recomputed {c['recomputed']}", flush=True)
    print(f"all match: {report['all_match']}", flush=True)


if __name__ == "__main__":
    main()
