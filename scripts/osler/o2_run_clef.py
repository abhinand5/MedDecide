"""O2: Cloudflare Clef and Clef-Flash through the authors' joint_schema_model.py (systemone endpoint function).

Runs in envs/clef (pins follow the card's tested stack). Imports nothing from meddecide; the reader and the row
writer use the same schema as meddecide.eval.o2_io.

Protocol (the authors'): load_release_model(checkpoint) then systemone(model, processor, request) with one
question per request (the common set's items are single-question). The authors' encoder sorts choice keys, so
option order cannot change their output: single pass is their protocol. The endpoint's own limit is 16,384
tokens; the common set is already capped at 8,192, so no item should reach it. noul answers come back as P(true)
(rounded to 4 dp, as the endpoint returns them): our (yes, no) probabilities are (P(true), 1 - P(true)).

Usage (from the repo root):
    envs/clef/.venv/bin/python -I scripts/osler/o2_run_clef.py --path <snapshot> --id clef --scope all
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
    if item["qtype"] == "choice":
        return {"type": "choice", "instructions": item["question"],
                "criteria": {o["key"]: o["label"] for o in item["options"]}}
    if item["qtype"] == "noul":
        return {"type": "noul", "instructions": item["question"]}
    return {"type": "score", "instructions": item["question"], "criteria": [o["label"] for o in item["options"]]}


def our_probs(item, answer):
    """The authors' answer for one question, as a vector in our option order."""
    if item["qtype"] == "noul":
        p_true = float(answer["noul"])
        return [p_true, 1.0 - p_true]  # (yes, no)
    if item["qtype"] == "choice":
        return [float(answer["probabilities"][o["key"]]) for o in item["options"]]
    return [float(answer["probabilities"][str(i)]) for i in range(len(item["options"]))]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--path", type=Path, required=True)
    ap.add_argument("--id", required=True, help="clef or clef_flash (output directory name)")
    ap.add_argument("--scope", choices=list(SCOPES), default="all")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    import torch

    sys.path.insert(0, str(args.path))
    import joint_schema_model as authors  # the authors' module, from the checkpoint

    out_path = OUT_ROOT / args.id / "predictions.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    model, processor = authors.load_release_model(args.path, device="cuda")
    print(f"[{utcnow()}] {args.id} loaded in {time.time() - t0:.1f}s on {torch.cuda.get_device_name(0)}", flush=True)
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
            request = {"model": args.id, "state": item["state"], "questions": {"q": authors_question(item)}}
            try:
                response = authors.systemone(model, processor, request)
            except ValueError as exc:
                if "exceed" in str(exc) or "limit" in str(exc):
                    row = prediction_row(item, status="skipped", reason=f"authors' limit: {exc}"[:200])
                else:
                    raise
            else:
                probs = our_probs(item, response["answers"]["q"])
                if not all(math.isfinite(p) for p in probs):
                    raise RuntimeError(f"non-finite probabilities for {item['item_id']}")
                row = prediction_row(item, status="scored", probs=probs, latency_s=round(time.time() - t1, 4),
                                     extras={"answer_type": response["answers"]["q"]["type"]})
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            n_new += 1
            if n_new % 500 == 0:
                print(f"[{utcnow()}] {n_new} new rows", flush=True)
    print(f"[{utcnow()}] done: new rows={n_new}", flush=True)


if __name__ == "__main__":
    main()
