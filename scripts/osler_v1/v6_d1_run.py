"""V6 additional row: LiquidAI/d1-3B through its own code (ADVISORY V6 step 2; an additional row only, D31), on the MR
evaluation file in the O2 prediction format.

The model ships its own code (system_one): a noul item gives p(yes); a choice item gives a distribution over named criteria.
Choice items are scored in both option orders and averaged per option, as the O2 protocol does for the competitors. Items
the model cannot read (an error in its own code, for example a state longer than its context) are written as skipped rows
with the reason; nothing is dropped silently. Resumes from its predictions file.

Writes outputs/osler_v1/V6/baselines/d1-3b/predictions.jsonl and run_meta.json (the revision and the counts).

Usage (one GPU job at a time; the GPU must be free):
    source scripts/pod_env.sh && setsid nohup uv run --frozen python scripts/osler_v1/v6_d1_run.py \
        > outputs/osler_v1/V6/logs/run_d1.log 2>&1 < /dev/null &
    smoke test: ... --limit 30 --out-root outputs/osler_v1/V6/smoke_d1
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModel

from meddecide.eval.o2_io import completed_ids, open_predictions, prediction_row, write_row
from meddecide.utils.io import write_json
from meddecide.utils.provenance import gpu_name, utcnow

REPO = Path(__file__).resolve().parents[2]
MODEL_ID = "LiquidAI/d1-3B"
REVISION = "051bcc464b01b9f92942b364d9586b0ef5912432"
ITEMS = REPO / "data" / "bench" / "osler_mr_v0" / "mr_eval.jsonl"
OUT_ROOT = REPO / "outputs" / "osler_v1" / "V6" / "baselines"
BATCH = 16


def request_for(item: dict[str, Any], *, reverse: bool) -> tuple[Any, dict[str, Any]]:
    """The (state, named questions) request of one item; ``reverse`` lists a choice's options in the opposite order."""
    keys = [o["key"] for o in item["options"]]
    if item["qtype"] == "noul":
        return item["state"], {"q": {"type": "noul", "instructions": item["question"]}}
    if item["qtype"] != "choice":
        raise ValueError(f"item {item['item_id']} has question type {item['qtype']!r}, which this runner does not score")
    labels = {o["key"]: o["label"] for o in item["options"]}
    order = list(reversed(keys)) if reverse else keys
    criteria = {k: labels[k] for k in order}
    return item["state"], {"q": {"type": "choice", "instructions": item["question"], "criteria": criteria}}


def probs_from(item: dict[str, Any], answer: dict[str, Any]) -> list[float]:
    """The item's probabilities aligned to its option keys, from one d1 answer."""
    keys = [o["key"] for o in item["options"]]
    if item["qtype"] == "noul":
        p_yes = float(answer["noul"])
        return [p_yes, 1.0 - p_yes] if keys == ["yes", "no"] else _raise(item, "noul options are not yes then no")
    by_name = answer["probabilities"]
    return [float(by_name[k]) for k in keys]


def _raise(item: dict[str, Any], why: str) -> list[float]:
    raise ValueError(f"item {item['item_id']}: {why}")


def score_batch(model, items: list[dict[str, Any]]) -> list[tuple[dict[str, Any], list[float]]]:
    """Two requests per choice item (both orders), one per noul item; returns (item, averaged probabilities)."""
    requests: list[tuple[Any, dict[str, Any]]] = []
    owners: list[tuple[int, bool]] = []  # (index into items, reversed order?)
    for i, item in enumerate(items):
        requests.append(request_for(item, reverse=False))
        owners.append((i, False))
        if item["qtype"] == "choice":
            requests.append(request_for(item, reverse=True))
            owners.append((i, True))
    results = model.system_one_batch(requests)
    per_item: dict[int, list[list[float]]] = {}
    for (i, _), result in zip(owners, results, strict=True):
        answer = result["answers"]["q"]
        per_item.setdefault(i, []).append(probs_from(items[i], answer))
    out = []
    for i, item in enumerate(items):
        vectors = per_item[i]
        averaged = [sum(v[j] for v in vectors) / len(vectors) for j in range(len(vectors[0]))]
        out.append((item, averaged))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--limit", type=int, default=0, help="score only the first N new rows (smoke tests)")
    parser.add_argument("--out-root", default=str(OUT_ROOT))
    args = parser.parse_args()
    out_dir = Path(args.out_root) / "d1-3b"
    out_path = out_dir / "predictions.jsonl"
    started = time.time()
    model = AutoModel.from_pretrained(MODEL_ID, revision=REVISION, trust_remote_code=True, dtype=torch.bfloat16)
    model = model.to("cuda:0").eval()
    load_s = round(time.time() - started, 1)
    print(f"[{utcnow()}] d1-3B loaded on {gpu_name()} in {load_s} s (revision {REVISION})", flush=True)
    with ITEMS.open(encoding="utf-8") as fh_items:
        rows = [json.loads(line) for line in fh_items if line.strip()]
    rows = [r for r in rows if r["benchmark"] == "mr_v0"]
    done = completed_ids(out_path)
    todo = [r for r in rows if r["item_id"] not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"resume_rows={len(done)} to_do={len(todo)} -> {out_path}", flush=True)
    counts = {"scored": 0, "skipped": 0}
    t0 = time.time()
    with open_predictions(out_path) as fh:
        for start in range(0, len(todo), BATCH):
            chunk = todo[start:start + BATCH]
            try:
                scored = score_batch(model, chunk)
            except Exception:  # a failed batch is retried one item at a time, so one bad item is isolated
                scored = []
                for item in chunk:
                    try:
                        scored.extend(score_batch(model, [item]))
                    except Exception as item_error:
                        write_row(fh, prediction_row(item, status="skipped",
                                                     reason=f"d1 error: {type(item_error).__name__}: {str(item_error)[:160]}"))
                        counts["skipped"] += 1
            for item, probs in scored:
                write_row(fh, prediction_row(item, status="scored", probs=probs,
                                             extras={"orders": 2 if item["qtype"] == "choice" else 1,
                                                     "authors_code": "LiquidAI/d1-3B system_one (its own code)"}))
                counts["scored"] += 1
            if (start // BATCH) % 50 == 0:
                rate = (counts["scored"] + counts["skipped"]) / max(time.time() - t0, 1e-6)
                print(f"[{utcnow()}] {counts} ({rate:.2f} rows/s)", flush=True)
    write_json(out_dir / "run_meta.json", {"model": MODEL_ID, "revision": REVISION, "gpu": gpu_name(), "utc": utcnow(),
                                           "load_s": load_s, "counts_this_run": counts,
                                           "wall_clock_s": round(time.time() - started, 1), "batch": BATCH})
    print(f"[{utcnow()}] done: this run {counts}; total rows {len(completed_ids(out_path))}", flush=True)


if __name__ == "__main__":
    main()
