"""Y6 (osler_v4): is a trained arm's score batch-dependent? Run when the GPU is free (one GPU job at a time).

The targets are the CG v2 items on which A3's re-score changed the decision against osler_v3's recorded A3 predictions. Each target
is scored alone (batch size 1) and inside a batch of eight CG v2 items (the target and the seven items after it in the file), on
the same code path (Trainer.score), in float32 and in bfloat16. The difference between the two batchings is the batch dependence.
The differences from the GPU re-score and from osler_v3's record are printed for reference only. The model's forward pass calls a
CUDA-only causal-conv kernel, so this cannot run on the CPU in this environment.

Usage (GPU; the checkpoint and the predictions must exist):
    uv run --frozen python scripts/osler_v4/y6_batch_check.py
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

from meddecide.model.pplx import load_any
from meddecide.train.config import load_config
from meddecide.train.mix_items import mix_row_to_item
from meddecide.train.trainer import Trainer
from meddecide.utils.io import iter_jsonl, write_json

RECIPE = Path("configs/osler_v3/block_arm.yaml")
CG = Path("data/bench/osler_v4/cg_v2.jsonl")
CHECKPOINT = Path("outputs/osler_v3/stage_a/A3/checkpoints/best")
RECORD = Path("outputs/osler_v3/stage_a/A3/cg_predictions.jsonl")
REESCORE = Path("outputs/osler_v4/Y6/A3/cg_v2_predictions.jsonl")
OUT = Path("outputs/osler_v4/Y6/batch_check_A3.json")
CONTEXT = 7


def max_difference(a: list[float], b: list[float]) -> float:
    return max(abs(x - y) for x, y in zip(a, b, strict=True))


def main() -> None:
    record = {str(r["item_id"]): r for r in iter_jsonl(RECORD)}
    rescore = {str(r["item_id"]): r for r in iter_jsonl(REESCORE)}
    flips = [k for k in record if k in rescore and record[k]["predicted_key"] != rescore[k]["predicted_key"]]
    rows = list(iter_jsonl(CG))
    items = [mix_row_to_item(row) for row in rows]
    position = {item.item_id: i for i, item in enumerate(items)}
    targets = sorted(flips, key=lambda k: position[k])
    print(f"targets (decision changed against osler_v3): {len(targets)}", flush=True)
    base = load_config(RECIPE)
    results: dict[str, object] = {"checkpoint": str(CHECKPOINT), "context_items": CONTEXT, "targets": len(targets),
                                  "dtypes": {}}
    for dtype in ("float32", "bfloat16"):
        config = dataclasses.replace(base, device="cuda:0", dtype=dtype)
        model = load_any(CHECKPOINT, device="cuda:0", dtype=dtype)
        trainer = Trainer(model, config)
        per_target = []
        for k in targets:
            i = position[k]
            batch = items[i:i + 1 + CONTEXT]
            alone = trainer.score([items[i]], batch_size=1)
            batched = trainer.score(batch, batch_size=len(batch))
            index = {item.item_id: j for j, item in enumerate(batched.items)}
            p_alone = list(map(float, alone.probs[0]))
            p_batched = list(map(float, batched.probs[index[k]]))
            per_target.append({
                "item_id": k,
                "n_options": len(p_alone),
                "max_abs_diff_alone_vs_batched": round(max_difference(p_alone, p_batched), 6),
                "argmax_alone": int(max(range(len(p_alone)), key=lambda j: p_alone[j])),
                "argmax_batched": int(max(range(len(p_batched)), key=lambda j: p_batched[j])),
                "max_abs_diff_alone_vs_gpu_rescore": round(max_difference(p_alone, list(map(float, rescore[k]["probs"]))), 6),
                "max_abs_diff_alone_vs_osler_v3_record": round(max_difference(p_alone, list(map(float, record[k]["probs"]))), 6),
                "gpu_rescore_correct": bool(rescore[k]["correct"]),
                "osler_v3_correct": bool(record[k]["correct"]),
            })
            print(f"{dtype}: target batch dependence {per_target[-1]['max_abs_diff_alone_vs_batched']:.6f}; "
                  f"alone vs GPU re-score {per_target[-1]['max_abs_diff_alone_vs_gpu_rescore']:.6f}; "
                  f"alone vs osler_v3 record {per_target[-1]['max_abs_diff_alone_vs_osler_v3_record']:.6f}", flush=True)
        results["dtypes"][dtype] = per_target  # type: ignore[index]
        del model, trainer
    write_json(OUT, results)
    print(f"wrote {OUT}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
