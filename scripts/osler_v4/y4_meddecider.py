"""Y4 baseline (osler_v4 ADVISORY Y4 §4): MedDecider-4B (lion-ai, pinned revision 570b3709…) through its authors' code, on our items.

The authors' class (meddecider_infer.MedDecider, in the snapshot) scores each item in two option orders and averages them; the
vector form below is the O2 runner's decide_vector (scripts/osler/o2_run_meddecider.py), which reproduces decide(orders=2) at
4 dp (the O2 runner checks 50 items; --verify does the same here). Item ids are the trainer's (mix_row_to_item), so the
predictions pair by item id with the other arms (meddecide.train.blocks.pair_by_item).

Writes outputs/osler_v4/Y4/baselines/<name>/predictions.jsonl and manifest.json (and verify.json with --verify).

Usage (one GPU job at a time; detached):
    source scripts/pod_env.sh && setsid nohup uv run --frozen --with pillow==12.3.0 python scripts/osler_v4/y4_meddecider.py \
        --items data/bench/osler_v4/fastdev_v4.jsonl --name fastdev_v4_meddecider4b --verify 50 \
        > outputs/osler_v4/Y4/logs/meddecider_fastdev.log 2>&1 < /dev/null &
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import snapshot_download

from meddecide.distill import soft_labels as sl
from meddecide.train.mix_items import mix_row_to_item
from meddecide.utils.io import iter_jsonl, write_json
from meddecide.utils.provenance import gpu_name, utcnow

MODEL = "lion-ai/MedDecider-4B"
REVISION = "570b37090c379ad722186a79a3432a31be463fc9"
MAX_OPTIONS = 10  # the authors' LETTERS = "ABCDEFGHIJ" (meddecider_infer.py); the O2 runner uses the same cap
OUT = Path("outputs/osler_v4/Y4/baselines")


def decide_vector(mj: Any, state: str, question: str, labels: list[str], qtype: str) -> list[float]:
    """decide(orders=2) of the authors' class, as a vector aligned with ``labels`` (the O2 runner's form)."""
    n = len(labels)
    temperature = mj.cfg["temperature"].get(qtype, 1.0)
    probs = torch.zeros(n)
    orders = [list(range(n)), list(range(n))[::-1]]
    with torch.no_grad():
        for order in orders:
            text = mj.prompt(state, question, [labels[i] for i in order])
            ids = mj.tok(text, return_tensors="pt", add_special_tokens=False).to(mj.model.device)
            z = mj.model(**ids, logits_to_keep=1).logits[0, -1, mj.letter_ids[:n]].float().cpu() / temperature
            p = torch.softmax(z, -1)
            for pos, orig in enumerate(order):
                probs[orig] += p[pos] / len(orders)
    return [float(x) for x in probs]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--items", required=True, type=Path)
    parser.add_argument("--name", required=True)
    parser.add_argument("--out-root", type=Path, default=OUT, help="the output directory (Y9 writes under outputs/osler_v4/Y9)")
    parser.add_argument("--limit", type=int, default=0, help="score only the first N items (smoke runs)")
    parser.add_argument("--verify", type=int, default=0, help="items checked against the authors' decide() dict")
    parser.add_argument("--model", default=MODEL, help="the Hub repo (MedDecider-4B by default; MedDecider-9B is an extra row)")
    parser.add_argument("--revision", default=REVISION, help="the pinned commit of --model")
    args = parser.parse_args()
    started = time.time()
    local = Path(snapshot_download(args.model, revision=args.revision))
    sys.path.insert(0, str(local))
    from meddecider_infer import MedDecider  # the authors' module, from the checkpoint

    mj = MedDecider(str(local))
    out_dir = args.out_root / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    verify_diffs: list[float] = []
    counts: Counter[str] = Counter()
    scored = 0
    with (out_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in iter_jsonl(args.items):
            if args.limit and counts["rows read"] >= args.limit:
                break
            counts["rows read"] += 1
            labels = [str(o["label"]) for o in row["options"]]
            if len(labels) > MAX_OPTIONS:
                counts[f"skipped: more options than the authors' prompt letters ({MAX_OPTIONS})"] += 1
                continue
            if len(set(labels)) != len(labels):
                counts["skipped: repeated labels (the authors' prompt keys options by label)"] += 1
                continue
            _, why = sl.request_for_row(row)
            if why is not None:
                counts[f"skipped: cannot be sent exactly ({why})"] += 1
                continue
            probs = decide_vector(mj, str(row["state"]), str(row["question"]), labels, str(row["qtype"]))
            if len(verify_diffs) < args.verify:
                authors = mj.decide(row["state"], row["question"], labels, row["qtype"], orders=2)
                verify_diffs.append(max(abs(probs[i] - authors[labels[i]]) for i in range(len(labels))))
            item = mix_row_to_item(row)
            predicted = sl.argmax_index(probs)
            handle.write(json.dumps({
                "item_id": item.item_id, "source": item.source, "template_id": row["template_id"],
                "qtype": row["qtype"], "gold_key": str(row["gold"]),
                "predicted_key": str(row["options"][predicted]["key"]), "correct": sl.argmax_correct(row, probs),
                "probs": probs, "status": "scored",
            }) + "\n")
            scored += 1
    gc.collect()
    if args.verify:
        write_json(out_dir / "verify.json", {"items": len(verify_diffs),
                                             "max_abs_diff": max(verify_diffs) if verify_diffs else None,
                                             "note": "vector form vs the authors' decide(orders=2) dict (rounded to 4 dp)"})
    write_json(out_dir / "manifest.json", {
        "model": args.model, "revision": args.revision, "items_file": str(args.items), "scored": scored,
        "counts": dict(sorted(counts.items())), "gpu": gpu_name(), "wall_clock_s": round(time.time() - started, 1),
        "generated_utc": utcnow(),
    })
    print(f"[{utcnow()}] {args.name}: scored {scored}; {dict(counts)}", flush=True)


if __name__ == "__main__":
    main()
