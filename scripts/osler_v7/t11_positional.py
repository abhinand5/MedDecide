#!/usr/bin/env python
"""osler_v7 T11 — all-orders positional health of the 63,304-step 9B final (GPU, dev only).

Scores the DDXPlus (49-option) dev items under the original order and two seeded permutations with the final checkpoint
and writes `positional.json` (ADVISORY §9; the per-evaluation files only carry the original order).

Run (GPU): ``uv run --frozen --with pillow==12.3.0 --with "torchvision==0.29.0" python scripts/osler_v7/t11_positional.py``
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
from transformers import AutoModel

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from meddecide.distill.d3_train import D3Student, prepare_rows  # noqa: E402
from meddecide.eval.composite import permutation_for, positional_health  # noqa: E402
from meddecide.utils.io import write_json  # noqa: E402
from meddecide.utils.provenance import git_commit, utcnow  # noqa: E402

SNAP = Path(os.environ.get("HF_HOME", "/workspace/.hf_home")) / "hub" / "models--vllm-sr--d3-flash" / "snapshots" / \
    "67e03e1bc157961c582ee7c6a35d9b15fada3dec"
DEV = ROOT / "data" / "train" / "osler_v7" / "d3_dev.jsonl"
OUT = ROOT / "outputs" / "osler_v7" / "T11"
DDX_PREFIX = "cg_v2_ddxplus"
SEEDS = (1, 2)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=SNAP)
    parser.add_argument("--checkpoint", type=Path, default=OUT / "final" / "checkpoints" / "step_063304.pt")
    parser.add_argument("--dev", type=Path, default=DEV)
    parser.add_argument("--out", type=Path, default=OUT / "positional.json")
    parser.add_argument("--batch", type=int, default=32)
    args = parser.parse_args()
    rows = [row for row in read_jsonl(args.dev) if str(row["template_id"]).startswith(DDX_PREFIX)]
    model = AutoModel.from_pretrained(str(args.snapshot), trust_remote_code=True, dtype=torch.bfloat16, device="cuda:0")
    runtime = model.runtime
    student = D3Student(runtime, lora_r=32, lora_alpha=32).to(runtime.device)
    student.load_checkpoint_state(torch.load(args.checkpoint, map_location=student.device, weights_only=False)["model"])
    student.eval()
    result: dict[str, Any] = {}
    for seed in (0, *SEEDS):
        if seed == 0:
            permuted = rows
        else:
            permuted = []
            for index, row in enumerate(rows):
                options = list(row["options"])
                order = permutation_for(len(options), seed=seed, index=index)
                permuted.append(dict(row, options=[options[i] for i in order]))
        prepared = prepare_rows(runtime, permuted)
        probs = student.probabilities([list(row.sequence) for row in prepared], [row.count for row in prepared],
                                      batch_size=args.batch)
        preds = []
        for index, (row, values) in enumerate(zip(prepared, probs, strict=True)):
            best = max(range(len(values)), key=values.__getitem__)
            preds.append({"item_id": row.row_id, "template_id": str(permuted[index]["template_id"]),
                          "n_options": row.count, "gold_index": row.gold, "predicted_index": best,
                          "correct": best == row.gold})
        result[str(seed)] = positional_health(preds).to_dict()
        print(f"[t11pos] seed {seed}: passes {result[str(seed)]['passes']}", flush=True)
    write_json(args.out, {"generated_utc": utcnow(), "git_commit": git_commit(ROOT),
                          "command": "uv run --frozen python scripts/osler_v7/t11_positional.py",
                          "checkpoint": str(args.checkpoint.relative_to(ROOT)), "items": len(rows), "orders": result,
                          "passes_all_orders": all(result[str(seed)]["passes"] for seed in (0, *SEEDS))})


if __name__ == "__main__":
    main()
