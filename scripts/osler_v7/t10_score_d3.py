#!/usr/bin/env python
"""osler_v7 T10 — score one item file with the d3-flash arm or its zero-shot base (GPU, one job at a time).

The model is the T8 9B checkpoint loaded into `D3Student` (candidate) or the same runtime with the adapters disabled and
the checkpoint's base readout (zero-shot base). Rows are prepared with the runtime's own `D3.prepare`; item ids are the
trainer's recomputed ids (`mix_row_to_item`), so candidate and base predictions pair row by row. Writes a prediction
file and a manifest under the gitignored outputs.

Run (GPU; example)::

    uv run --frozen --with pillow==12.3.0 --with "torchvision==0.29.0" python scripts/osler_v7/t10_score_d3.py \
        --checkpoint outputs/osler_v7/T8/arm/checkpoints/step_013992.pt --items data/bench/osler_v7/confirm/general.jsonl \
        --name 9b_cand_general --out outputs/osler_v7/T10
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
from meddecide.train.mix_items import mix_row_to_item  # noqa: E402
from meddecide.utils.io import write_json, write_jsonl  # noqa: E402
from meddecide.utils.provenance import git_commit, utcnow  # noqa: E402

SNAP = Path(os.environ.get("HF_HOME", "/workspace/.hf_home")) / "hub" / "models--vllm-sr--d3-flash" / "snapshots" / \
    "67e03e1bc157961c582ee7c6a35d9b15fada3dec"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def to_d3_row(row: dict[str, Any]) -> dict[str, Any]:
    options = row["options"] if isinstance(row["options"], list) else json.loads(row["options"])
    return {"row_id": mix_row_to_item(row).item_id, "source": str(row["source"]),
            "template_id": str(row["template_id"]), "qtype": str(row["qtype"]), "state": str(row["state"]),
            "question": str(row["question"]), "gold": str(row["gold"]), "split": "eval",
            "options": [{"key": str(o["key"]), "label": str(o["label"])} for o in options]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=SNAP)
    parser.add_argument("--checkpoint", type=Path, default=None, help="the arm checkpoint; omitted means zero-shot base")
    parser.add_argument("--items", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    args.items = args.items if args.items.is_absolute() else ROOT / args.items
    args.out = args.out if args.out.is_absolute() else ROOT / args.out
    args.checkpoint = None if args.checkpoint is None else (
        args.checkpoint if args.checkpoint.is_absolute() else ROOT / args.checkpoint)
    rows = read_jsonl(args.items)
    if args.limit:
        rows = rows[: args.limit]
    prepared_items = [mix_row_to_item(row) for row in rows]
    if len({item.item_id for item in prepared_items}) != len(prepared_items):
        raise SystemExit("duplicate item ids after conversion: stop")
    started = time.time()
    model = AutoModel.from_pretrained(str(args.snapshot), trust_remote_code=True, dtype=torch.bfloat16, device="cuda:0")
    runtime = model.runtime
    student = D3Student(runtime, lora_r=args.lora_r, lora_alpha=args.lora_alpha).to(runtime.device)
    base_readout = runtime.readout.detach().clone()
    if args.checkpoint is None:
        student.readout.data.copy_(base_readout)
        for adapter in student.adapters.values():
            adapter.scaling = 0.0
    else:
        payload = torch.load(args.checkpoint, map_location=student.device, weights_only=False)
        student.load_checkpoint_state(payload["model"])
    student.eval()
    converted = [to_d3_row(row) for row in rows]
    prepared = prepare_rows(runtime, converted)
    probs = student.probabilities([list(row.sequence) for row in prepared], [row.count for row in prepared],
                                  batch_size=args.batch, max_tokens=args.max_tokens)
    out_rows = []
    for index, (row, values) in enumerate(zip(prepared, probs, strict=True)):
        best = max(range(len(values)), key=values.__getitem__)
        out_rows.append({"item_id": row.row_id, "template_id": converted[index]["template_id"],
                         "source": row.source, "n_options": row.count, "gold_index": row.gold,
                         "predicted_index": best, "correct": best == row.gold})
    out_dir = args.out / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "predictions.jsonl", out_rows)
    write_json(out_dir / "manifest.json", {
        "generated_utc": utcnow(), "git_commit": git_commit(ROOT),
        "command": "uv run --frozen python scripts/osler_v7/t10_score_d3.py",
        "model": "zero-shot d3-flash" if args.checkpoint is None else str(args.checkpoint.relative_to(ROOT)),
        "items": str(args.items.relative_to(ROOT)), "rows": len(out_rows), "items_read": len(rows),
        "wall_seconds": round(time.time() - started, 1),
    })
    print(json.dumps({"name": args.name, "rows": len(out_rows), "wall_s": round(time.time() - started, 1)}, indent=1))


if __name__ == "__main__":
    main()
