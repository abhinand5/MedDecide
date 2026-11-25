#!/usr/bin/env python
"""osler_v7 T7 — positional health of the 4B blend candidates in all three orders (GPU, dev only).

The main selection (`t7_select.py`) scores the dev set once per blend scale and reports the original-order positional
check from those rows. This addendum scores the DDXPlus (49-option) items under the original order and the two seeded
permutations for every scale (0 = base, 0.25, 0.5, 0.75, 1.0) with the trained checkpoint, so the blend table can report
the all-orders positional health that ADVISORY §9 names as a blocking check. Writes `positional.json` under
`outputs/osler_v7/T7/`.

Run (GPU; one job): ``uv run --frozen python scripts/osler_v7/t7_positional.py --arm outputs/osler_v7/T6/arm``
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from meddecide.eval.composite import permutation_for, positional_health  # noqa: E402
from meddecide.model.blend import Blend  # noqa: E402
from meddecide.model.meddecide_model import reorder_options  # noqa: E402
from meddecide.model.pplx import PplxDecisionModel  # noqa: E402
from meddecide.train.mix_items import mix_row_to_item  # noqa: E402
from meddecide.train.osler_arm import dev_prediction_rows  # noqa: E402
from meddecide.utils.io import write_json  # noqa: E402
from meddecide.utils.provenance import git_commit, utcnow  # noqa: E402

DEV = ROOT / "data" / "bench" / "osler_v7" / "dev_format_matched.jsonl"
SCALES = (0.0, 0.25, 0.5, 0.75, 1.0)
SEEDS = (1, 2)
DDX_PREFIX = "cg_v2_ddxplus"
OUT = ROOT / "outputs" / "osler_v7" / "T7"


def apply_blend(model: PplxDecisionModel, blend: Blend, scale: float) -> None:
    with torch.no_grad():
        readout = (1.0 - scale) * blend.base_readout + scale * blend.trained_readout
        model.optcode.weight.copy_(readout.to(device=model.optcode.weight.device, dtype=model.optcode.weight.dtype))
        for module, scales in blend.base_scaling:
            for name, value in scales.items():
                module.scaling[name] = value * scale


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", type=Path, required=True)
    parser.add_argument("--step", type=int, default=13_992)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-batch-tokens", type=int, default=16384)
    args = parser.parse_args()
    args.arm = args.arm if args.arm.is_absolute() else ROOT / args.arm
    args.out = args.out if args.out.is_absolute() else ROOT / args.out
    with DEV.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    items = [mix_row_to_item(row) for row in rows]
    ddx = [(i, item) for i, item in enumerate(items) if str(item.template_id).startswith(DDX_PREFIX)]
    if not ddx or any(item.n_options <= 26 for _, item in ddx):
        raise SystemExit("the positional check needs the DDXPlus items with more than 26 options")
    started = time.time()
    model = PplxDecisionModel.load(args.arm / "checkpoints" / f"step_{args.step:06d}", device=args.device,
                                   dtype=args.dtype)
    model.eval_mode()
    blend = Blend(model)
    results: dict[str, Any] = {}
    for scale in SCALES:
        apply_blend(model, blend, scale)
        per_order: dict[str, Any] = {}
        for seed in (0, *SEEDS):
            if seed == 0:
                ordered = [item for _, item in ddx]
            else:
                ordered = [reorder_options(item, permutation_for(item.n_options, seed=seed, index=index))
                           for index, item in ddx]
            scored = model.score_items(ordered, batch_size=args.batch_size, max_batch_tokens=args.max_batch_tokens)
            health = positional_health(dev_prediction_rows(scored)).to_dict()
            per_order[str(seed)] = health
        results[str(scale)] = {"orders": per_order,
                               "passes_all_orders": all(per_order[str(seed)]["passes"] for seed in (0, *SEEDS))}
        print(f"[t7pos] scale {scale:g}: passes all orders {results[str(scale)]['passes_all_orders']}",
              flush=True)
    write_json(args.out / "positional.json", {
        "generated_utc": utcnow(),
        "git_commit": git_commit(ROOT),
        "command": "uv run --frozen python scripts/osler_v7/t7_positional.py",
        "checkpoint": str((args.arm / "checkpoints" / f"step_{args.step:06d}").relative_to(ROOT)),
        "items": len(ddx),
        "scales": results,
        "wall_seconds": round(time.time() - started, 1),
    })
    print(json.dumps({scale: results[scale]["passes_all_orders"] for scale in results}, indent=1))


if __name__ == "__main__":
    main()
