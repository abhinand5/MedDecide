"""osler_v5 Z2 paired statistics (CPU): each arm against R0 at step 13,992, with the statistics of the choice rule: knowledge dev (item micro),
MR and MIMIC partitions (template macro), paired within templates, 1,000 resamples, seed 0 (meddecide.train.blocks). Writes
outputs/osler_v5/Z2/pairs.json. Usage (CPU):  uv run --frozen python scripts/osler_v5/z2_pairs.py R0p R2 R3"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from meddecide.eval.composite import mimic_partition, mr_partition
from meddecide.train.blocks import macro_difference, micro_difference, pair_by_item
from meddecide.utils.io import write_json

REPO = Path(__file__).resolve().parents[2]
Z2 = REPO / "outputs" / "osler_v5" / "Z2"
STEP = 13992


def rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load(arm: str) -> dict:
    folder = Z2 / arm / "evals" / f"step_{STEP:06d}"
    return {"fast": rows(folder / "fast_dev.jsonl"), "know": rows(folder / "knowledge_dev.jsonl")}


def main() -> None:
    arms = sys.argv[1:] or ["R0p", "R2", "R3"]
    control = load("R0")
    out: dict = {}
    for arm in arms:
        if not (Z2 / arm / "evals" / f"step_{STEP:06d}" / "fast_dev.jsonl").exists():
            out[arm] = "NOT MEASURED — no evaluation at 13,992 yet"
            continue
        x = load(arm)
        k = micro_difference(pair_by_item(x["know"], control["know"]))
        m = macro_difference(pair_by_item(mr_partition(x["fast"]), mr_partition(control["fast"])))
        mi = macro_difference(pair_by_item(mimic_partition(x["fast"]), mimic_partition(control["fast"])))
        out[arm] = {"knowledge_micro_vs_R0": k.as_dict(), "mr_macro_vs_R0": m.as_dict(), "mimic_macro_vs_R0": mi.as_dict()}
        print(f"{arm} minus R0 at 13,992: knowledge {k.point:+.4f} [{k.lo:+.4f}, {k.hi:+.4f}]; MR {m.point:+.4f} [{m.lo:+.4f}, {m.hi:+.4f}]; "
              f"MIMIC {mi.point:+.4f} [{mi.lo:+.4f}, {mi.hi:+.4f}]", flush=True)
    write_json(Z2 / "pairs.json", out)


if __name__ == "__main__":
    main()
