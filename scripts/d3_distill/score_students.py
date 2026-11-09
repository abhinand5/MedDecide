"""d3_distill step 3 (GPU): score a student (base or a trained checkpoint) on an item set, through its own runtime.

The student is loaded with the card's ``AutoModel`` path, the checkpoint's LoRA/readout are loaded through
``D3Student``, and every item is scored as one one-question request by ``D3Student.probabilities`` (the runtime's own
``D3.probabilities`` with the adapter and trained readout in place). Writes per-item predictions (gitignored) and an
aggregate with accuracy by template and source. Evaluation only: no gradient, no training.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch
from transformers import AutoModel

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from meddecide.distill.d3_train import D3Student, TrainRow  # noqa: E402

SETS = {
    "medmcqa_val": REPO / "data" / "bench" / "d3_distill" / "medmcqa_val.jsonl",
    "medqa_test": REPO / "data" / "bench" / "d3_distill" / "medqa_test.jsonl",
}


def utcnow() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def prepare_item(runtime, item: dict, index: int) -> TrainRow:
    options = [(o["key"], o["label"]) for o in item["options"]]
    if item["qtype"] == "choice":
        request = {"type": "choice", "instructions": item["question"],
                   "criteria": {o["key"]: o["label"] for o in item["options"]}}
    elif item["qtype"] == "noul":
        request = {"type": "noul", "instructions": item["question"]}
    else:
        request = {"type": "score", "instructions": item["question"], "criteria": [o["label"] for o in item["options"]]}
    ready = runtime.prepare(item["state"], {"q": request})
    if "q" in ready.errors:
        raise SystemExit(f"{item.get('eval_id', index)}: {ready.errors['q']['message']}")
    gold = next(i for i, (key, _) in enumerate(options) if key == item["gold"])
    return TrainRow(row_id=str(item.get("eval_id", index)), sequence=tuple(ready.sequences["q"]),
                    count=len(options), gold=gold, teacher=None, split="eval", source=str(item.get("source", "")))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--snapshot", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, default=None)
    ap.add_argument("--sets", default="medmcqa_val")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max-rows", type=int, default=0)
    args = ap.parse_args()

    out = args.out if args.out.is_absolute() else REPO / args.out
    out.mkdir(parents=True, exist_ok=True)
    model = AutoModel.from_pretrained(
        str(args.snapshot.resolve()), trust_remote_code=True, dtype=torch.bfloat16, device="cuda:0"
    )
    runtime = model.runtime
    student = D3Student(runtime, lora_r=32, lora_alpha=32).to(runtime.device)
    if args.checkpoint is not None:
        payload = torch.load(args.checkpoint, map_location=student.device, weights_only=False)
        student.load_checkpoint_state(payload["model"])
        print(f"[{utcnow()}] loaded checkpoint step {payload['step']} from {args.checkpoint}", flush=True)
    runtime.readout = student.readout.detach()

    aggregate = {"snapshot": str(args.snapshot), "revision": args.snapshot.name,
                 "checkpoint": None if args.checkpoint is None else str(args.checkpoint),
                 "checkpoint_step": None if args.checkpoint is None else int(payload["step"]),
                 "finished_utc": utcnow(), "sets": {}}
    for name in [s.strip() for s in args.sets.split(",") if s.strip()]:
        path = SETS.get(name, Path(name))
        items = read_jsonl(path)
        if args.max_rows:
            items = items[: args.max_rows]
        started = time.time()
        rows = [prepare_item(runtime, item, i) for i, item in enumerate(items)]
        probs = student.probabilities([r.sequence for r in rows], [r.count for r in rows], batch_size=args.batch)
        predictions = []
        for item, row, value in zip(items, rows, probs, strict=True):
            best = max(range(len(value)), key=value.__getitem__)
            keys = [o["key"] for o in item["options"]]
            predictions.append(
                {
                    "eval_id": item.get("eval_id"), "source": item.get("source"),
                    "template_id": item.get("template_id"), "gold": item["gold"],
                    "option_keys": keys, "probs": value, "answer": keys[best],
                    "correct": int(best == row.gold),
                }
            )
        pred_path = out / f"{name}.jsonl"
        with pred_path.open("w", encoding="utf-8") as fh:
            for row in predictions:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        correct = sum(r["correct"] for r in predictions)
        by_template: Counter[str] = Counter()
        by_template_total: Counter[str] = Counter()
        for r in predictions:
            by_template[str(r["template_id"])] += r["correct"]
            by_template_total[str(r["template_id"])] += 1
        aggregate["sets"][name] = {
            "file": str(path),
            "items": len(predictions),
            "correct": correct,
            "accuracy": correct / len(predictions),
            "per_template": {
                t: {"items": by_template_total[t], "correct": by_template[t],
                    "accuracy": by_template[t] / by_template_total[t]}
                for t in sorted(by_template_total)
            },
            "predictions": str(pred_path.relative_to(REPO)),
            "wall_s": round(time.time() - started, 1),
        }
        print(f"[{utcnow()}] {name}: {correct}/{len(predictions)} = {correct / len(predictions):.4f}", flush=True)
    aggregate["peak_gpu_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    (out / "aggregate.json").write_text(json.dumps(aggregate, indent=1), encoding="utf-8")
    print(json.dumps({s: v["accuracy"] for s, v in aggregate["sets"].items()}, indent=1), flush=True)


if __name__ == "__main__":
    main()
