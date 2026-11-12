#!/usr/bin/env python
"""osler_v7 T4 — Lever-3 grid on the osler_v6 final (inference only, dev; chooses nothing).

ADVISORY §5: on the v6 final, grid the LoRA scale {0.5, 0.75, 1} against the readout scale {0, 0.5, 1} on the
format-matched dev set. If "base readout + trained LoRA" (readout 0, LoRA 1) recovers the two-option and two-letter
accuracy, the damage sits in the readout and T6-T8 anchor it; the grid itself selects no model and feeds no gate.

The scored items are the frozen dev file from T2 (`data/bench/osler_v7/dev_format_matched.jsonl`, 22,934 items). For
each of the nine combinations the script sets the LoRA scaling multiplier and the readout blend
``(1 - r) * R_base + r * R_trained`` (the osler_v5 `Blend` semantics, split into its two parts), scores the whole dev
set once on the GPU, and writes aggregate metrics plus per-item prediction rows (gitignored; no probabilities are
kept). Metrics: item and template-macro accuracy overall; the MR verdict family; the MIMIC templates; the CG-origin
and knowledge-origin partitions; per qtype; per option-count group; and the D52b positional check on the >26-option
templates.

Run (GPU; one job, detached)::

    source scripts/pod_env.sh && setsid nohup uv run --frozen python scripts/osler_v7/t4_grid.py \
        > outputs/osler_v7/T4/logs/t4_grid.log 2>&1 < /dev/null &
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from meddecide.eval.composite import positional_health, template_macro_accuracy  # noqa: E402
from meddecide.model.blend import Blend  # noqa: E402
from meddecide.model.pplx import PplxDecisionModel  # noqa: E402
from meddecide.train.mix_items import mix_row_to_item  # noqa: E402
from meddecide.train.osler_arm import dev_prediction_rows  # noqa: E402
from meddecide.utils.io import write_json, write_jsonl  # noqa: E402
from meddecide.utils.provenance import git_commit, utcnow  # noqa: E402

DEV = ROOT / "data" / "bench" / "osler_v7" / "dev_format_matched.jsonl"
CHECKPOINT = ROOT / "outputs" / "osler_v6" / "Z6" / "final" / "checkpoints" / "step_063304"
OUT = ROOT / "outputs" / "osler_v7" / "T4"
LORA_SCALES = (0.5, 0.75, 1.0)
READOUT_SCALES = (0.0, 0.5, 1.0)
MR_FAMILY = {
    "mr_icd_support_template_v2", "mr_indicator_evidence_v1", "mr_page_category_tabular_v1",
    "mtsamples_page_category_v1", "pmc_similar_patient_v1", "codiesp_icd_support_v1", "mkurman_icd_support_v1",
}


def option_group(n_options: int) -> str:
    """The option-count bucket a row belongs to (the grid's format slices)."""
    if n_options <= 5:
        return str(n_options)
    if n_options <= 9:
        return "6_9"
    if n_options == 10:
        return "10"
    return "11_49"


def summarise(rows: list[dict[str, Any]], meta: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Aggregate metrics of one grid combination from its per-item prediction rows."""
    if not rows:
        raise ValueError("no prediction rows")

    def block(subset: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "items": len(subset),
            "accuracy_item": round(sum(bool(r["correct"]) for r in subset) / len(subset), 8),
            "accuracy_template_macro": round(template_macro_accuracy(subset), 8),
        }

    by_qtype: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_qtype[str(row["qtype"])].append(row)
        by_group[option_group(int(row["n_options"]))].append(row)
    wide = [row for row in rows if int(row["n_options"]) > 26]
    mr = [row for row in rows if str(row["template_id"]) in MR_FAMILY]
    mimic = [row for row in rows if str(row["template_id"]).startswith("mimic_")]
    cg = [row for row in rows if meta[str(row["item_id"])]["dev_origin"] == "cg_v2"]
    knowledge = [row for row in rows if meta[str(row["item_id"])]["dev_origin"] == "knowledge_dev_v2"]
    positional = positional_health([r for r in rows if str(r["template_id"]) in
                                    {str(w["template_id"]) for w in wide}]).to_dict() if wide else {
        "status": "NOT MEASURED — no >26-option dev row"}
    return {
        "overall": block(rows),
        "mr_family": block(mr) if mr else {"items": 0},
        "mimic_templates": block(mimic) if mimic else {"items": 0},
        "cg_origin_micro": {"items": len(cg), "accuracy_item": round(sum(bool(r["correct"]) for r in cg) / len(cg), 8)}
        if cg else {"items": 0},
        "knowledge_origin": {"items": len(knowledge),
                             "accuracy_item": round(sum(bool(r["correct"]) for r in knowledge) / len(knowledge), 8)}
        if knowledge else {"items": 0},
        "by_qtype": {qtype: {"items": len(subset),
                             "accuracy_item": round(sum(bool(r["correct"]) for r in subset) / len(subset), 8)}
                     for qtype, subset in sorted(by_qtype.items())},
        "by_option_group": {group: {"items": len(subset),
                                    "accuracy_item": round(sum(bool(r["correct"]) for r in subset) / len(subset), 8)}
                            for group, subset in sorted(by_group.items())},
        "two_letter": {"items": len(wide),
                       "accuracy_item": round(sum(bool(r["correct"]) for r in wide) / len(wide), 8)} if wide else {
            "items": 0},
        "positional": positional,
    }


def apply_scales(model: PplxDecisionModel, blend: Blend, lora_scale: float, readout_scale: float) -> None:
    """Set the LoRA multiplier and the readout blend for one grid cell (Blend.apply with the two parts separated)."""
    if not 0.0 <= lora_scale <= 1.0 or not 0.0 <= readout_scale <= 1.0:
        raise ValueError("scales must be in [0, 1]")
    with torch.no_grad():
        readout = (1.0 - readout_scale) * blend.base_readout + readout_scale * blend.trained_readout
        model.optcode.weight.copy_(readout.to(device=model.optcode.weight.device, dtype=model.optcode.weight.dtype))
        for module, scales in blend.base_scaling:
            for name, scale in scales.items():
                module.scaling[name] = scale * lora_scale


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--dev", type=Path, default=DEV)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-batch-tokens", type=int, default=8192)
    args = parser.parse_args()
    started = time.time()
    (args.out_dir / "predictions").mkdir(parents=True, exist_ok=True)
    (args.out_dir / "logs").mkdir(exist_ok=True)
    with args.dev.open(encoding="utf-8") as handle:
        dev_rows = [json.loads(line) for line in handle if line.strip()]
    meta = {str(row["item_id"]): {"dev_origin": str(row["dev_origin"])} for row in dev_rows}
    items = [mix_row_to_item(row) for row in dev_rows]
    if len({item.item_id for item in items}) != len(items):
        raise SystemExit("the dev file has duplicate item ids")
    # Batch padding dominates otherwise: the dev file mixes 30-token MCQ rows with multi-thousand-token MIMIC
    # notes. Sorting by prompt-length proxy before scoring keeps each batch's rows similar in length; the metrics
    # join on item_id, so scoring order does not matter.
    items.sort(key=lambda item: len(item.state) + len(item.question))
    print(f"[data] {len(items)} dev items", file=sys.stderr, flush=True)
    model = PplxDecisionModel.load(args.checkpoint, device=args.device, dtype=args.dtype)
    model.eval_mode()
    blend = Blend(model)
    print(f"[model] loaded {args.checkpoint}", file=sys.stderr, flush=True)

    grid = []
    for lora_scale in LORA_SCALES:
        for readout_scale in READOUT_SCALES:
            apply_scales(model, blend, lora_scale, readout_scale)
            t0 = time.time()
            scored = model.score_items(items, batch_size=args.batch_size, max_batch_tokens=args.max_batch_tokens)
            rows = dev_prediction_rows(scored)
            label = f"lora{lora_scale:g}_readout{readout_scale:g}"
            keep = [{k: row[k] for k in ("item_id", "source", "template_id", "qtype", "n_options",
                                         "gold_index", "predicted_index", "correct")} for row in rows]
            write_jsonl(args.out_dir / "predictions" / f"{label}.jsonl", keep)
            metrics = summarise(keep, meta)
            metrics.update({"lora_scale": lora_scale, "readout_scale": readout_scale, "label": label,
                            "seconds": round(time.time() - t0, 1)})
            grid.append(metrics)
            print(f"[grid] {label}: item {metrics['overall']['accuracy_item']:.4f}, "
                  f"macro {metrics['overall']['accuracy_template_macro']:.4f}, "
                  f"two-letter {metrics['two_letter']['accuracy_item']:.4f}, {metrics['seconds']}s",
                  file=sys.stderr, flush=True)
    result = {
        "generated_utc": utcnow(),
        "git_commit": git_commit(ROOT),
        "command": "uv run --frozen python scripts/osler_v7/t4_grid.py",
        "checkpoint": str(args.checkpoint.relative_to(ROOT)),
        "dev": {"path": str(args.dev.relative_to(ROOT)), "items": len(items)},
        "grid": grid,
        "design": {"lora_scales": list(LORA_SCALES), "readout_scales": list(READOUT_SCALES),
                   "note": "readout 1 is the trained readout, 0 the base readout (reference_readout); the LoRA "
                           "multiplier scales the adapter scaling. Diagnostic only: no selection, no gate."},
        "env": {"device": args.device, "dtype": args.dtype, "torch": torch.__version__,
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"},
        "wall_seconds": round(time.time() - started, 1),
    }
    write_json(args.out_dir / "grid.json", result)
    print(json.dumps({"cells": len(grid), "wall_seconds": result["wall_seconds"]}, indent=1), flush=True)


if __name__ == "__main__":
    main()
