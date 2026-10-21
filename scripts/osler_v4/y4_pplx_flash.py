"""Y4 baseline (osler_v4 ADVISORY Y4 §4; Y6's CG v2 point estimate): zero-shot pplx-decider-flash-v1.2 through its authors' code.

The pinned revision is 6d771ad7…, the base revision of osler_v2's final adapter (outputs/osler_v2/W7/osler_4b/checkpoints/best/
adapter/adapter_config.json). Requests and answers use the mapping of meddecide.distill.soft_labels (the same as the Y5 teacher),
and each item's id is the id the trainer-based arms use (meddecide.train.mix_items.mix_row_to_item), so every model's
predictions pair by item id (meddecide.train.blocks.pair_by_item).

Writes outputs/osler_v4/Y4/baselines/<name>/predictions.jsonl (one row per scored item, status "scored") and manifest.json
(counts, skips by reason, the revision and the temperature). Predictions are private (MIMIC items when the items are MIMIC).

Usage (one GPU job at a time; detached):
    source scripts/pod_env.sh && setsid nohup uv run --frozen --with pillow==12.3.0 --with "torchvision==0.29.*" \
        python scripts/osler_v4/y4_pplx_flash.py --items data/bench/osler_v4/fastdev_v4.jsonl --name fastdev_v4_flash \
        > outputs/osler_v4/Y4/logs/flash_fastdev.log 2>&1 < /dev/null &
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch

from meddecide.distill import soft_labels as sl
from meddecide.train.mix_items import mix_row_to_item
from meddecide.utils.io import iter_jsonl, write_json
from meddecide.utils.provenance import gpu_name, utcnow

REVISION = "6d771ad7b5ecff5e93b8fc7b90d4c5ccc341954d"
CHECKPOINT = Path(os.environ["HF_HOME"]) / "hub" / "models--perplexity-ai--pplx-decider-flash-v1.2" / "snapshots" / REVISION
OUT = Path("outputs/osler_v4/Y4/baselines")
CHUNK = 512


def teacher_package() -> Any:
    """The authors' package of the pinned flash snapshot (imported after its source directory is on the path)."""
    source = CHECKPOINT / "source" / "src"
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    import pplx_decider

    return pplx_decider


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--items", required=True, type=Path)
    parser.add_argument("--name", required=True)
    parser.add_argument("--limit", type=int, default=0, help="score only the first N items (smoke runs)")
    args = parser.parse_args()
    started = time.time()
    package = teacher_package()
    decider = package.Decider.from_pretrained(CHECKPOINT, device="cuda", fast=False)
    out_dir = OUT / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter()
    scored_items = 0
    with (out_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        batch: list[dict[str, Any]] = []

        def flush() -> None:
            nonlocal scored_items
            requests = [sl.request_for_row(row)[0] for row in batch]
            try:
                responses = decider.decide(requests)
            except (package.RequestError, package.LengthError):
                responses = []
                for request in requests:
                    try:
                        responses.append(decider.decide(request))
                    except (package.RequestError, package.LengthError) as exc:
                        responses.append({"error": type(exc).__name__})
            for row, response in zip(batch, responses, strict=True):
                if "error" in response:
                    counts[f"error: {response['error']}"] += 1
                    continue
                probs = sl.teacher_probabilities(row, response["answers"][sl.QUESTION_ID])
                item = mix_row_to_item(row)
                predicted = sl.argmax_index(probs)
                handle.write(json.dumps({
                    "item_id": item.item_id, "source": item.source, "template_id": row["template_id"],
                    "qtype": row["qtype"], "gold_key": str(row["gold"]),
                    "predicted_key": str(row["options"][predicted]["key"]), "correct": sl.argmax_correct(row, probs),
                    "probs": probs, "input_tokens": int(response["usage"]["input_tokens"]), "status": "scored",
                }) + "\n")
                scored_items += 1
            batch.clear()

        limit = args.limit or None
        for row in iter_jsonl(args.items):
            if limit is not None and counts["rows read"] >= limit:
                break
            counts["rows read"] += 1
            _, why = sl.request_for_row(row)
            if why is not None:
                counts[f"skipped: cannot be sent exactly ({why})"] += 1
                continue
            batch.append(row)
            if len(batch) == CHUNK:
                flush()
        if batch:
            flush()
    gc.collect()  # the process ends after the manifest: the card is free for the next job then
    torch.cuda.empty_cache()
    write_json(out_dir / "manifest.json", {
        "model": "pplx-decider-flash-v1.2 (zero-shot, authors' code)", "revision": REVISION,
        "checkpoint": str(CHECKPOINT), "items_file": str(args.items), "scored": scored_items,
        "counts": dict(sorted(counts.items())), "gpu": gpu_name(), "wall_clock_s": round(time.time() - started, 1),
        "generated_utc": utcnow(),
    })
    print(f"[{utcnow()}] {args.name}: scored {scored_items}; {dict(counts)}", flush=True)


if __name__ == "__main__":
    main()
