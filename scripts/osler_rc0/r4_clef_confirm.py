#!/usr/bin/env python
"""osler_rc0 R4 — Cloudflare Clef-Flash on the fresh tier-2 `confirm/general.jsonl` set (authors' stack).

Run under the authors' isolated venv (no meddecide import)::

    envs/clef/.venv/bin/python -I scripts/osler_rc0/r4_clef_confirm.py \
        --path /workspace/.hf_home/hub/models--Cloudflare--clef-flash/snapshots/fde727a287004204b7518dcc983fe64379776712 \
        --items data/bench/osler_v7/confirm/general.jsonl \
        --out outputs/osler_rc0/R4/preds/clef_confirm_general/predictions.jsonl

Same protocol as `scripts/osler_v5/z10_clef.py` (whose local-only gate is re-checked here): `load_release_model`, then
`systemone(model, processor, request)` one question per request; the answer is mapped to our option order. Items the
authors' limit rejects are written as skipped rows with the reason; nothing is dropped silently. The output is schema C
(`item_id` raw ids, `option_keys`, `gold`, `status`, `probs`).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

REPO = Path(__file__).resolve().parents[2]
CHECK = REPO / "outputs" / "osler_v5" / "Z10" / "clef_local_check.json"


def utcnow() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def check_gate(snapshot: Path) -> None:
    if not CHECK.exists():
        raise SystemExit(f"refused: {CHECK.relative_to(REPO)} does not exist (run z10_clef_local_check.py first)")
    record = json.loads(CHECK.read_text(encoding="utf-8"))
    if not (record.get("local_only") is True and record.get("revision_matches") is True):
        raise SystemExit("refused: the local-only check did not pass (Clef cells are NOT MEASURED)")
    if Path(record["snapshot"]).resolve() != snapshot:
        raise SystemExit("refused: the snapshot checked is not the snapshot given")


def iter_rows(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def authors_question(item: dict) -> dict:
    if item["qtype"] == "choice":
        return {"type": "choice", "instructions": item["question"],
                "criteria": {option["key"]: option["label"] for option in item["options"]}}
    if item["qtype"] == "noul":
        return {"type": "noul", "instructions": item["question"]}
    return {"type": "score", "instructions": item["question"],
            "criteria": [option["label"] for option in item["options"]]}


def our_probs(item: dict, answer: dict) -> list[float]:
    if item["qtype"] == "noul":
        p_true = float(answer["noul"])
        return [p_true, 1.0 - p_true]
    if item["qtype"] == "choice":
        return [float(answer["probabilities"][option["key"]]) for option in item["options"]]
    return [float(answer["probabilities"][str(i)]) for i in range(len(item["options"]))]


def prediction_row(item: dict, *, status: str, reason: str | None = None, probs: list[float] | None = None,
                   latency_s: float | None = None, extras: dict | None = None) -> dict:
    return {
        "item_id": item["item_id"], "benchmark": item.get("benchmark", "v0.2"),
        "set_name": item.get("set_name", "confirm_general"),
        "template_id": item["template_id"], "qtype": item["qtype"],
        "option_keys": [option["key"] for option in item["options"]], "gold": item["gold"],
        "status": status, "reason": reason, "probs": probs, "latency_s": latency_s,
        "prompt_tokens": None, "extras": extras or {},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--path", type=Path, required=True, help="the local snapshot directory of Cloudflare/clef-flash")
    parser.add_argument("--items", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    snapshot = args.path.resolve()
    check_gate(snapshot)
    import torch

    sys.path.insert(0, str(snapshot))
    import joint_schema_model as authors  # the authors' module, from the snapshot

    args.out.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    model, processor = authors.load_release_model(snapshot, device="cuda")
    print(f"[{utcnow()}] clef_flash loaded in {time.time() - started:.1f}s on {torch.cuda.get_device_name(0)}", flush=True)
    n_scored = n_skipped = 0
    with args.out.open("w", encoding="utf-8") as handle:
        for item in iter_rows(args.items):
            request = {"model": "clef_flash", "state": item["state"], "questions": {"q": authors_question(item)}}
            t0 = time.time()
            try:
                response = authors.systemone(model, processor, request)
            except ValueError as error:
                if "exceed" not in str(error) and "limit" not in str(error):
                    raise
                row = prediction_row(item, status="skipped", reason=f"authors' limit: {error}"[:200])
                n_skipped += 1
            else:
                probs = our_probs(item, response["answers"]["q"])
                if not all(math.isfinite(p) for p in probs):
                    raise RuntimeError(f"non-finite probabilities for {item['item_id']}")
                row = prediction_row(item, status="scored", probs=probs, latency_s=round(time.time() - t0, 4),
                                     extras={"answer_type": response["answers"]["q"]["type"],
                                             "z10_set": "confirm_general"})
                n_scored += 1
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            if (n_scored + n_skipped) % 250 == 0:
                print(f"[{utcnow()}] {n_scored + n_skipped} items, {n_scored} scored, {n_skipped} skipped", flush=True)
    print(json.dumps({"utc": utcnow(), "items": n_scored + n_skipped, "scored": n_scored, "skipped": n_skipped},
                     indent=1), flush=True)


if __name__ == "__main__":
    main()
