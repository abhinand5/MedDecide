"""d3_distill step 3 (GPU): score a student (base or a trained checkpoint) on an item set, through its own runtime.

The student is loaded with the card's ``AutoModel`` path, the checkpoint's LoRA/readout are loaded through
``D3Student``, and every item is scored as one one-question request by ``D3Student.probabilities`` (the runtime's own
``D3.probabilities`` with the adapter and trained readout in place).

Prediction rows use the Gate O4 common shape (``item_id``, ``template_id``, ``gold_key``, ``predicted_key``,
``correct``, ``probs``), with probabilities in the item's own option order. For the frozen Y9 sets the ``item_id`` is the
trainer's recomputed id (``mix_row_to_item(row).item_id``), which is what the pplx and trainer-based runners write and
what the gate pairs on.

Evaluation only: no gradient, no training. Writes per-item predictions (gitignored) and an aggregate.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch
from transformers import AutoModel

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from meddecide.distill.d3_train import D3Student, TrainRow, render_order, render_targets  # noqa: E402
from meddecide.eval.gate_o4 import readout_failures  # noqa: E402
from meddecide.train.mix_items import mix_row_to_item  # noqa: E402

Y9 = REPO / "data" / "bench" / "osler_v4" / "y9"
SETS = {
    "medmcqa_val": REPO / "data" / "bench" / "d3_distill" / "medmcqa_val.jsonl",
    "medqa_test": REPO / "data" / "bench" / "d3_distill" / "medqa_test.jsonl",
    "knowledge_subsample": REPO / "data" / "bench" / "d3_distill" / "knowledge_subsample.jsonl",
    "knowledge": Y9 / "knowledge.jsonl",
    "panel": Y9 / "panel.jsonl",
    "mr_verdict": Y9 / "mr_verdict.jsonl",
}
Y9_SETS = ("knowledge", "panel", "mr_verdict")


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
        raise SystemExit(f"{item.get('item_id', index)}: {ready.errors['q']['message']}")
    gold, _ = render_targets(item["qtype"], [key for key, _ in options], item["gold"], None)
    return TrainRow(row_id=str(item.get("item_id", index)), sequence=tuple(ready.sequences["q"]),
                    count=len(options), gold=gold, teacher=None, split="eval", source=str(item.get("source", "")))


def pairing_id(item: dict, is_y9: bool, index: int) -> str:
    if is_y9:
        return mix_row_to_item(item).item_id
    return str(item.get("eval_id", item.get("item_id", index)))


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
    step = None
    if args.checkpoint is not None:
        payload = torch.load(args.checkpoint, map_location=student.device, weights_only=False)
        student.load_checkpoint_state(payload["model"])
        step = int(payload["step"])
        print(f"[{utcnow()}] loaded checkpoint step {step} from {args.checkpoint}", flush=True)
    runtime.readout = student.readout.detach()

    aggregate = {"snapshot": str(args.snapshot), "revision": args.snapshot.name,
                 "checkpoint": None if args.checkpoint is None else str(args.checkpoint),
                 "checkpoint_step": step, "finished_utc": utcnow(), "sets": {}}
    for name in [s.strip() for s in args.sets.split(",") if s.strip()]:
        path = SETS.get(name, Path(name))
        is_y9 = name in Y9_SETS
        items = read_jsonl(path)
        if args.max_rows:
            items = items[: args.max_rows]
        started = time.time()
        rows = [prepare_item(runtime, item, i) for i, item in enumerate(items)]
        probs = student.probabilities([r.sequence for r in rows], [r.count for r in rows], batch_size=args.batch)
        predictions = []
        for index, (item, value) in enumerate(zip(items, probs, strict=True)):
            order = render_order(item["qtype"], [o["key"] for o in item["options"]])
            # rendered position pos holds item option order[pos]; put the probabilities back in item order
            item_probs = [value[order.index(i)] for i in range(len(order))]
            keys = [o["key"] for o in item["options"]]
            best = max(range(len(item_probs)), key=item_probs.__getitem__)
            predictions.append(
                {
                    "item_id": pairing_id(item, is_y9, index),
                    "raw_id": str(item.get("item_id", item.get("eval_id", index))),
                    "source": item.get("source"),
                    "template_id": item.get("template_id"),
                    "qtype": item["qtype"],
                    "gold_key": item["gold"],
                    "predicted_key": keys[best],
                    "correct": int(keys[best] == item["gold"]),
                    "probs": item_probs,
                    "status": "scored",
                }
            )
        pred_path = out / f"{name}.jsonl"
        with pred_path.open("w", encoding="utf-8") as fh:
            for row in predictions:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        by_template: dict[str, list[int]] = {}
        by_source: dict[str, list[int]] = {}
        for r in predictions:
            by_template.setdefault(str(r["template_id"]), []).append(r["correct"])
            by_source.setdefault(str(r["source"]), []).append(r["correct"])
        failing = readout_failures(predictions)
        aggregate["sets"][name] = {
            "file": str(path),
            "items": len(predictions),
            "correct": sum(r["correct"] for r in predictions),
            "accuracy": sum(r["correct"] for r in predictions) / len(predictions),
            "template_macro_accuracy": sum(sum(v) / len(v) for v in by_template.values()) / len(by_template),
            "per_template": {t: {"items": len(v), "correct": sum(v), "accuracy": sum(v) / len(v)}
                             for t, v in sorted(by_template.items())},
            "per_source": {s: {"items": len(v), "correct": sum(v), "accuracy": sum(v) / len(v)}
                           for s, v in sorted(by_source.items())},
            "readout_failures": failing,
            "predictions": str(pred_path.relative_to(REPO)),
            "wall_s": round(time.time() - started, 1),
        }
        print(f"[{utcnow()}] {name}: {sum(r['correct'] for r in predictions)}/{len(predictions)} = "
              f"{sum(r['correct'] for r in predictions) / len(predictions):.4f}"
              f"{' READOUT_FAIL ' + str(failing) if failing else ''}", flush=True)
    aggregate["peak_gpu_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    (out / "aggregate.json").write_text(json.dumps(aggregate, indent=1), encoding="utf-8")
    print(json.dumps({s: v["accuracy"] for s, v in aggregate["sets"].items()}, indent=1), flush=True)


if __name__ == "__main__":
    main()
