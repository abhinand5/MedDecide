#!/usr/bin/env python
"""V6 gate: are exact-length batches the same computation as batch size 1 (both unpadded)?

Scores 300 seen-template dev items (the first 300 of student_v0's selection sample, by item id) with the
step-1,000 checkpoint in bf16, twice: one item per batch, and exact-length batches of up to 16. Both
are padding-free, so they should agree up to batch-dimension noise. Writes
``outputs/student_v1/V6/bucket_check.json`` with the largest logit difference, argmax disagreements and
timings. Arm A's dev evaluation uses the exact-length route only if this gate passes.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from meddecide.model.meddecide_model import MedDecideModel  # noqa: E402
from meddecide.train.data import read_items  # noqa: E402

CKPT = ROOT / "outputs" / "student_v0" / "S9_run3" / "checkpoints" / "step_1000"
OUT = ROOT / "outputs" / "student_v1" / "V6" / "bucket_check.json"
BENCH = ROOT / "data" / "bench" / "v0.2"


def main() -> None:
    ids = json.loads((ROOT / "outputs" / "student_v0" / "S9_run3" / "dev_eval_ids.json").read_text())["item_ids"]
    want = set(ids[:300])
    found = {}
    for folder in ("tier1", "fresh"):
        for path in sorted((BENCH / folder).glob("*.jsonl")):
            for item in read_items(path):
                if item.item_id in want:
                    found[item.item_id] = item
    items = [found[i] for i in ids[:300] if i in found]
    model = MedDecideModel.load(CKPT, device="cuda:0", dtype="bfloat16")

    t0 = time.time()
    one = model.score_items(items, batch_size=1, round_to_chunk=False)
    t_one = time.time() - t0
    t0 = time.time()
    buckets = model.score_items(items, batch_size=16, round_to_chunk=False, length_buckets=True)
    t_buckets = time.time() - t0

    deltas = [float(np.max(np.abs(a - b))) for a, b in zip(one.logits, buckets.logits, strict=True)]
    flips = sum(int(np.argmax(a) != np.argmax(b)) for a, b in zip(one.probs, buckets.probs, strict=True))
    distinct_lengths = len(set(one.prompt_tokens))
    result = {
        "kind": "v6_bucket_check",
        "checkpoint": str(CKPT.relative_to(ROOT)),
        "dtype": "bfloat16",
        "items": len(items),
        "distinct_prompt_lengths": distinct_lengths,
        "max_abs_logit_delta": max(deltas),
        "mean_abs_logit_delta": float(np.mean(deltas)),
        "argmax_disagreements": flips,
        "batches_length_buckets": len(buckets.batch_sizes),
        "seconds_batch_one": round(t_one, 1),
        "seconds_length_buckets": round(t_buckets, 1),
        "gate": "PASS" if flips <= 0.01 * len(items) else "FAIL",
        "gate_rule": "argmax disagreements <= 1 % of items (set before the run: bf16 rounding alone "
                     "moves logits by more than 1e-2, so a logit threshold would test noise)",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("items", "max_abs_logit_delta", "argmax_disagreements",
                                             "seconds_batch_one", "seconds_length_buckets", "gate")}))


if __name__ == "__main__":
    main()
