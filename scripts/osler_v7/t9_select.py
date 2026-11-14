#!/usr/bin/env python
"""osler_v7 T9 — blend selection and dev evaluations for the 9B arm (same ADVISORY §4/§7 rules as T7).

Loads the d3-flash arm checkpoint into `D3Student`, and for s in {0, 0.25, 0.5, 0.75, 1} sets the LoRA adapter scaling to
`base_scaling * s` and the readout to `(1-s)*base_readout + s*trained_readout`. Each candidate is scored on the frozen
22,934-item dev set; osler_v5's S, the partition accuracies, knowledge-dev accuracy and the D52b positional check (all
three orders) are computed. The paired template-macro difference against the s=0 base is bootstrapped (1,000 resamples,
seed 0) and §7 decides: eligible when the lower bound is > -0.01; among eligible candidates with all-orders positional
health the highest S is chosen; if none qualifies the arm is reported BLOCKED.

No test, panel, MIMIC audit or held-out item is read. Predictions stay in the gitignored outputs.

Run (GPU; one job)::

    uv run --frozen --with pillow==12.3.0 --with "torchvision==0.29.0" python scripts/osler_v7/t9_select.py \
        --checkpoint outputs/osler_v7/T8/arm/checkpoints/final_step_013992.pt --out outputs/osler_v7/T9
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
from transformers import AutoModel

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from meddecide.distill.d3_train import D3Student, prepare_rows  # noqa: E402
from meddecide.eval.composite import composite, permutation_for, positional_health  # noqa: E402
from meddecide.train.blocks import macro_difference, pair_by_item  # noqa: E402
from meddecide.utils.io import write_json  # noqa: E402
from meddecide.utils.provenance import git_commit, utcnow  # noqa: E402

SNAP = Path(os.environ.get("HF_HOME", "/workspace/.hf_home")) / "hub" / "models--vllm-sr--d3-flash" / "snapshots" / \
    "67e03e1bc157961c582ee7c6a35d9b15fada3dec"
DEV = ROOT / "data" / "train" / "osler_v7" / "d3_dev.jsonl"
SCALES = (0.0, 0.25, 0.5, 0.75, 1.0)
SEEDS = (1, 2)
DDX_PREFIX = "cg_v2_ddxplus"
NON_INFERIORITY = -0.01


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def apply_blend(student: D3Student, base_readout: torch.Tensor, base_scaling: dict[str, float], scale: float) -> None:
    with torch.no_grad():
        student.readout.data.copy_(((1.0 - scale) * base_readout + scale * student.trained_readout).to(student.readout.device))
        for name, adapter in student.adapters.items():
            adapter.scaling = base_scaling[name] * scale


def predictions(student: D3Student, rows: list[Any], templates: list[str], *, batch_size: int) -> list[dict[str, Any]]:
    probs = student.probabilities([list(r.sequence) for r in rows], [r.count for r in rows], batch_size=batch_size)
    out = []
    for row, template, values in zip(rows, templates, probs, strict=True):
        best = max(range(len(values)), key=values.__getitem__)
        out.append({"item_id": row.row_id, "template_id": template, "source": row.source, "n_options": row.count,
                    "gold_index": row.gold, "predicted_index": best, "correct": best == row.gold})
    return out


def metrics(rows: list[dict[str, Any]], base_rows: list[dict[str, Any]], origins: dict[str, str]) -> dict[str, Any]:
    fast = [row for row in rows if origins[row["item_id"]] == "fastdev_v4"]
    cg = [row for row in rows if origins[row["item_id"]] == "cg_v2"]
    know = [row for row in rows if origins[row["item_id"]] == "knowledge_dev_v2"]
    parts = composite(fast, cg)
    diff = macro_difference(pair_by_item(rows, base_rows))
    return {"S": parts.to_dict()["S"], "composite": parts.to_dict(),
            "knowledge": sum(bool(row["correct"]) for row in know) / len(know),
            "paired_template_macro_difference": diff.as_dict(), "eligible": bool(diff.lo > NON_INFERIORITY)}


def choose(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = [c for c in candidates if c["eligible"] and c["positional_passes_all_orders"]]
    if not eligible:
        return {"chosen_scale": None, "reason": "no candidate passes eligibility and all-orders positional health"}
    best = max(eligible, key=lambda c: c["S"])
    return {"chosen_scale": best["scale"], "reason": "highest dev S among eligible candidates with positional health",
            "S": best["S"]}


def positional_all_orders(student: D3Student, ddx_rows: list[dict[str, Any]], *, batch_size: int) -> dict[str, Any]:
    out: dict[str, Any] = {}
    source = ddx_rows
    for seed in (0, *SEEDS):
        if seed == 0:
            permuted = source
        else:
            permuted = []
            for index, row in enumerate(source):
                options = list(row["options"])
                order = permutation_for(len(options), seed=seed, index=index)
                permuted.append(dict(row, options=[options[i] for i in order]))
        prepared = prepare_rows(student.runtime, permuted)
        batch = predictions(student, prepared, [str(row["template_id"]) for row in permuted], batch_size=batch_size)
        out[str(seed)] = positional_health(batch).to_dict()
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=SNAP)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dev", type=Path, default=DEV)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--reuse", action="store_true",
                        help="reuse prediction files already written; recompute metrics and positional health only")
    args = parser.parse_args()
    args.checkpoint = args.checkpoint if args.checkpoint.is_absolute() else ROOT / args.checkpoint
    args.dev = args.dev if args.dev.is_absolute() else ROOT / args.dev
    out = args.out if args.out.is_absolute() else ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(exist_ok=True)
    if (out / "selection.json").exists() and not args.resume:
        raise SystemExit(f"{out}/selection.json exists: run once")
    dev_rows = read_jsonl(args.dev)
    origins = {row["row_id"]: row["dev_origin"] for row in dev_rows}
    model = AutoModel.from_pretrained(str(args.snapshot), trust_remote_code=True, dtype=torch.bfloat16, device="cuda:0")
    runtime = model.runtime
    student = D3Student(runtime, lora_r=args.lora_r, lora_alpha=args.lora_alpha).to(runtime.device)
    payload = torch.load(args.checkpoint, map_location=student.device, weights_only=False)
    student.load_checkpoint_state(payload["model"])
    student.eval()
    base_readout = runtime.readout.detach().clone()
    student.trained_readout = student.readout.detach().clone()
    base_scaling = {name: float(adapter.scaling) for name, adapter in student.adapters.items()}
    prepared = prepare_rows(runtime, dev_rows)
    templates = [str(row["template_id"]) for row in dev_rows]
    ddx_rows = [row for row in dev_rows if str(row["template_id"]).startswith(DDX_PREFIX)]
    started = time.time()
    predictions_dir = out / "predictions"
    predictions_dir.mkdir(exist_ok=True)
    base_rows: list[dict[str, Any]] | None = None
    candidates: list[dict[str, Any]] = []
    for scale in SCALES:
        apply_blend(student, base_readout, base_scaling, scale)
        path = predictions_dir / f"scale_{scale:g}.jsonl"
        if args.reuse and path.exists():
            rows = read_jsonl(path)
        else:
            rows = predictions(student, prepared, templates, batch_size=args.batch)
            with path.open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row) + "\n")
        if base_rows is None:
            base_rows = rows
        pos = positional_all_orders(student, ddx_rows, batch_size=args.batch)
        candidate = {"scale": scale, **metrics(rows, base_rows, origins),
                     "positional": pos,
                     "positional_passes_all_orders": all(pos[str(seed)]["passes"] for seed in (0, *SEEDS))}
        candidates.append(candidate)
        print(f"[t9] scale {scale:g}: S {candidate['S']:.6f} eligible {candidate['eligible']} "
              f"positional {candidate['positional_passes_all_orders']} ({time.time() - started:.0f}s)", flush=True)
    result = {
        "generated_utc": utcnow(), "git_commit": git_commit(ROOT),
        "command": "uv run --frozen python scripts/osler_v7/t9_select.py",
        "checkpoint": str(args.checkpoint.relative_to(ROOT)),
        "dev": {"path": str(args.dev.relative_to(ROOT)), "items": len(dev_rows)},
        "rule": {"scales": list(SCALES), "non_inferiority_lower_bound": NON_INFERIORITY,
                 "positional_requirement": "all three orders pass (ADVISORY §9)",
                 "choice": "highest dev S among eligible candidates with positional health"},
        "candidates": candidates,
        "choice": choose(candidates),
        "wall_seconds": round(time.time() - started, 1),
    }
    write_json(out / "selection.json", result)
    print(json.dumps(result["choice"], indent=1))


if __name__ == "__main__":
    main()
