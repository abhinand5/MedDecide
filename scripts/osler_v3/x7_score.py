"""osler_v3 X7/X8/X9 scoring of a selected checkpoint on the clinical-general (CG) partition, with the mean KL to the base (ADVISORY
X7: "every run's MR, CG, knowledge and general partitions, and their mean KL to the base on CG items").

- CG accuracy is the item accuracy (micro) on the CG partition (data/bench/osler_v3/cg_v3.jsonl, built by x6_partitions.py).
- The mean KL to the base is KL(base || model) averaged over the CG items, the base being the same checkpoint with its adapter off
  and its pre-training readout (MedDecideModel.reference_option_probs), exactly the quantity the trainer's KL term uses.
- The MR, knowledge and general partitions come from the run's own dev predictions (fast dev v3), with blocks.partition_pairs.

Writes outputs/osler_v3/<stage>/<arm>/cg_result.json (aggregates) and cg_predictions.jsonl (raw predictions; gitignored).

Usage (after the run's training, one GPU job at a time):
    source scripts/pod_env.sh && uv run --frozen python scripts/osler_v3/x7_score.py --stage stage_a --arm A1
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path
from typing import Any

import torch

from meddecide.model.pplx import load_any
from meddecide.train.config import load_config
from meddecide.train.data import iter_batches
from meddecide.train.mix_items import mix_row_to_item
from meddecide.train.osler_arm import dev_prediction_rows
from meddecide.train.trainer import Trainer
from meddecide.utils.io import write_json, write_jsonl
from meddecide.utils.provenance import git_commit, utcnow

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "configs" / "osler_v3" / "block_arm.yaml"
CG = REPO / "data" / "bench" / "osler_v3" / "cg_v3.jsonl"
OUT_ROOT = REPO / "outputs" / "osler_v3"
KL_BATCH = 16


def read_items(path: Path) -> list[Any]:
    with path.open(encoding="utf-8") as fh:
        return [mix_row_to_item(json.loads(line)) for line in fh if line.strip()]


@torch.no_grad()
def mean_kl_to_base(model: Any, items: list[Any], batch_size: int, max_batch_tokens: int, max_prompt_tokens: int) -> dict[str, float]:
    """Mean over items of KL(base || model) across each item's offered options, with the base from the adapter-off readout."""
    from meddecide.train.losses import kl_to_reference

    model.eval_mode()
    total = 0.0
    n = 0
    for _, batch in iter_batches(model, items, batch_size=batch_size, max_batch_tokens=max_batch_tokens,
                                 max_prompt_tokens=max_prompt_tokens, augment_options=False):
        mb = model.collate(batch)
        ref = model.reference_option_probs(mb)
        _logits, probs = model.batch_logits(mb)
        anchor = torch.ones(mb.batch_size, dtype=torch.bool)
        kl = kl_to_reference(probs, ref.to(probs.device), mb.marker_item.to(probs.device), mb.batch_size, anchor)
        total += float(kl.detach()) * mb.batch_size
        n += mb.batch_size
    if n == 0:
        raise ValueError("no CG items to score")
    return {"mean_kl_to_base": total / n, "items": n}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stage", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--cg", type=Path, default=CG)
    args = ap.parse_args()
    started = time.time()
    run_dir = OUT_ROOT / args.stage / args.arm
    best = run_dir / "checkpoints" / "best"
    if not best.exists():
        raise SystemExit(f"{args.arm}: no best checkpoint at {best} (the run has not finished or kept none)")
    config = dataclasses.replace(load_config(CONFIG), kl_to_base=0.0)
    model = load_any(best, device=config.device, dtype=config.dtype)
    items = read_items(args.cg)
    scored = Trainer(model, config).score(items)
    metrics = scored.metrics()
    per_template = scored.per_qtype_metrics() if hasattr(scored, "per_qtype_metrics") else {}
    kl = mean_kl_to_base(model, items, config.batch_size, config.max_batch_tokens, config.max_prompt_tokens)
    rows = dev_prediction_rows(scored)
    write_jsonl(run_dir / "cg_predictions.jsonl", rows)
    result = {
        "stage": args.stage, "arm": args.arm, "cg_file": str(args.cg.relative_to(REPO)), "cg_items": len(items),
        "cg_accuracy_micro": metrics["accuracy"], "cg_brier": metrics.get("brier"), "cg_metrics": {
            k: v for k, v in metrics.items() if isinstance(v, (int, float))},
        "cg_per_qtype": per_template, "mean_kl_to_base_cg": kl["mean_kl_to_base"], "kl_items": kl["items"],
        "scored_utc": utcnow(), "git_commit": git_commit(REPO), "wall_clock_s": round(time.time() - started, 1),
    }
    write_json(run_dir / "cg_result.json", result)
    print(f"{args.arm}: CG accuracy {metrics['accuracy']:.4f} over {len(items)} items; mean KL to base "
          f"{kl['mean_kl_to_base']:.4f}", flush=True)


if __name__ == "__main__":
    main()
