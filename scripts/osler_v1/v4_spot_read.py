"""V4 spot read (osler_v1 ADVISORY V4, step 3): 20 verified teacher pages per generator, printed to the terminal and nothing else.

The sample is drawn with a fixed seed from every verified row of each generator (the rows are already verified by the build).
Nothing is written to disk: the teacher text is private (data/gen is gitignored) and is read for defect types only. The defect
types found are recorded in SELF_AUDIT, never the text.

Usage (a fresh process, after the closure):
    uv run --frozen python scripts/osler_v1/v4_spot_read.py --n 20 --seed 0
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TEACHER = REPO / "data" / "gen" / "osler_v1" / "teacher"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    for name in ("indicator", "narrative"):
        rows = [json.loads(line) for line in (TEACHER / f"{name}.jsonl").open(encoding="utf-8") if line.strip()]
        sample = random.Random(args.seed).sample(rows, min(args.n, len(rows)))
        print(f"===== {name}: {len(sample)} of {len(rows)} verified rows (seed {args.seed}) =====", flush=True)
        for i, row in enumerate(sample, 1):
            print(f"--- {name} {i}/{len(sample)} | split {row['split']} | generator {row['generator']} | "
                  f"role {row.get('role', '')} | endpoint {row.get('teacher_endpoint', 'not recorded')}", flush=True)
            print("STATE:", row["state"], flush=True)
            print("QUESTION:", row["question"], flush=True)
            print("OPTIONS:", [(o["key"], o["label"]) for o in row["options"]], "GOLD:", row["gold"], flush=True)


if __name__ == "__main__":
    main()
