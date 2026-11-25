"""O2: perplexity-ai/pplx-decider-v1.1-27b through its authors' code (source/src/autojev/model.py).

Runs in envs/pplx27b (the authors' uv.lock). This script imports nothing from meddecide: the env deliberately
does not install the project, so the reader and the row writer below use the same schema as meddecide.eval.o2_io.

Protocol (the authors'): DecisionModel(checkpoint).predict([row]) with one row per call, as the README shows.
The checkpoint's own temperature (1.0087) and its saved non-causal full-attention mode apply. The authors'
code does not average option orders, so single order is the authors' protocol. Choice criteria are passed
with our option keys in our order; noul uses criteria false/true, and the output is re-ordered to (yes, no).
The authors' prepare() refuses prompts over 8,192 tokens; such items would be skipped with that reason.

Usage (from the repo root):
    envs/pplx27b/.venv/bin/python -I scripts/osler/o2_run_pplx.py --checkpoint <snapshot> --scope all
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ITEMS = REPO / "data/bench/v0.3_ext/o2_items.jsonl"
OUT_ROOT = REPO / "outputs/osler_v0/O2"
SCOPES = {"all": ("v0.2", "ext_panel", "robustness"), "panel_robustness": ("ext_panel", "robustness")}


def utcnow() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def iter_items(path: Path, benchmarks: tuple[str, ...]):
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                row = json.loads(line)
                if row["benchmark"] in benchmarks:
                    yield row


def completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open(encoding="utf-8") as fh:
        return {json.loads(line)["item_id"] for line in fh if line.strip()}


def prediction_row(item, *, status, reason=None, probs=None, latency_s=None, extras=None):
    return {
        "item_id": item["item_id"], "benchmark": item["benchmark"], "set_name": item["set_name"],
        "template_id": item["template_id"], "qtype": item["qtype"],
        "option_keys": [o["key"] for o in item["options"]], "gold": item["gold"],
        "status": status, "reason": reason, "probs": probs, "latency_s": latency_s,
        "prompt_tokens": None, "extras": extras or {},
    }


def authors_question(item):
    """The authors' question dict for our item, with options in our order."""
    labels = [o["label"] for o in item["options"]]
    if item["qtype"] == "choice":
        return {"type": "choice", "instructions": item["question"],
                "criteria": {o["key"]: o["label"] for o in item["options"]}}
    if item["qtype"] == "noul":
        return {"type": "noul", "instructions": item["question"], "criteria": {"false": "No", "true": "Yes"}}
    return {"type": "score", "instructions": item["question"], "criteria": labels}


def to_our_order(item, probs_authors):
    """The authors' probability vector, in the order their options() returns, mapped to our option order."""
    if item["qtype"] == "noul":
        # authors' order is (false, true); ours is (yes, no)
        return [probs_authors[1], probs_authors[0]]
    return list(probs_authors)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--scope", choices=list(SCOPES), default="all")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    import torch

    sys.path.insert(0, str(args.checkpoint / "source" / "src"))
    from autojev.model import DecisionModel  # the authors' code, from the checkpoint

    out_path = OUT_ROOT / "pplx-decider-v1.1-27b" / "predictions.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    model = DecisionModel(args.checkpoint, device="cuda")
    print(f"[{utcnow()}] pplx loaded in {time.time() - t0:.1f}s; temperature={model.temperature} "
          f"attention={model.attention_mode} gpu={torch.cuda.get_device_name(0)}", flush=True)
    done = completed_ids(out_path)
    print(f"resume_rows={len(done)} -> {out_path}", flush=True)
    n_new = 0
    with out_path.open("a", encoding="utf-8") as fh:
        for item in iter_items(ITEMS, SCOPES[args.scope]):
            if item["item_id"] in done:
                continue
            if args.limit and n_new >= args.limit:
                break
            t1 = time.time()
            try:
                probs = model.predict([{"state": item["state"], "question": authors_question(item)}],
                                      batch_size=1)[0]
            except ValueError as exc:
                if "token limit" in str(exc) or "exceeds" in str(exc):
                    row = prediction_row(item, status="skipped", reason=f"authors' 8,192-token limit: {exc}"[:200])
                else:
                    raise
            else:
                ours = to_our_order(item, probs)
                if not all(math.isfinite(p) for p in ours):
                    raise RuntimeError(f"non-finite probabilities for {item['item_id']}")
                row = prediction_row(item, status="scored", probs=[float(p) for p in ours],
                                     latency_s=round(time.time() - t1, 4),
                                     extras={"temperature": model.temperature,
                                             "attention_mode": model.attention_mode})
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            n_new += 1
            if n_new % 500 == 0:
                print(f"[{utcnow()}] {n_new} new rows", flush=True)
    print(f"[{utcnow()}] done: new rows={n_new}", flush=True)


if __name__ == "__main__":
    main()
