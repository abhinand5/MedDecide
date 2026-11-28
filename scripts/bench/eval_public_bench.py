#!/usr/bin/env python
"""Run any Hugging Face System-One decision model (`system_one` API) on `abhinand/MedDecide-Bench`
(`confirm_general`, 5,988 public-domain items) and write verifiable `predictions.jsonl`.

Example:
    python scripts/bench/eval_public_bench.py \
        --model abhinand/Osler-9B \
        --out predictions_osler9b.jsonl
    python scripts/bench/verify_submission.py \
        --predictions predictions_osler9b.jsonl \
        --model-name MyModel-7B --hf-repo myorg/MyModel-7B --params-b 7.0 \
        --out-submission site/submissions/myorg__MyModel-7B.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModel


def to_system_one_question(item: dict[str, Any], *, pplx_label_keys: bool = False) -> dict[str, Any]:
    qtype = str(item["qtype"])
    if qtype == "noul":
        return {"type": "noul", "instructions": item["question"]}
    if qtype == "choice":
        if pplx_label_keys:
            criteria = {str(o["label"]): None for o in item["options"]}
        else:
            criteria = {str(o["key"]): str(o["label"]) for o in item["options"]}
        return {"type": "choice", "instructions": item["question"], "criteria": criteria}
    return {
        "type": "score",
        "instructions": item["question"],
        "criteria": [str(o["label"]) for o in item["options"]],
    }


def extract_probs(item: dict[str, Any], answer: dict[str, Any], *, pplx_label_keys: bool = False) -> dict[str, float]:
    qtype = str(item["qtype"])
    opts = item["options"]
    if qtype == "noul":
        p_yes = float(answer["noul"])
        return {str(o["key"]): (p_yes if str(o["key"]) == "yes" else 1.0 - p_yes) for o in opts}
    if qtype == "choice":
        pmap = answer["probabilities"]
        if pplx_label_keys:
            return {str(o["key"]): float(pmap[str(o["label"])]) for o in opts}
        return {str(o["key"]): float(pmap[str(o["key"])]) for o in opts}
    pmap = answer["probabilities"]
    if isinstance(pmap, dict):
        return {str(o["key"]): float(pmap[str(i)]) for i, o in enumerate(opts)}
    return {str(o["key"]): float(pmap[i]) for i, o in enumerate(opts)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="HF model repo ID or local path supporting system_one()")
    parser.add_argument("--out", type=Path, required=True, help="Output predictions JSONL path")
    parser.add_argument("--limit", type=int, default=None, help="Optional item cap for smoke tests")
    parser.add_argument(
        "--pplx-label-keys",
        action="store_true",
        help="Pass choice criteria as {label: None} (for Osler-4B / pplx-decider checkpoints)",
    )
    args = parser.parse_args()

    ds = load_dataset("abhinand/MedDecide-Bench", "confirm_general", split="test")
    if args.limit is not None:
        ds = ds.select(range(min(args.limit, len(ds))))

    model = AutoModel.from_pretrained(args.model, trust_remote_code=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for row in tqdm(ds, desc=f"Evaluating {args.model}"):
            item = dict(row)
            q = to_system_one_question(item, pplx_label_keys=args.pplx_label_keys)
            ans = model.system_one(state=item["state"], questions={"q": q})["answers"]["q"]
            probs = extract_probs(item, ans, pplx_label_keys=args.pplx_label_keys)
            fh.write(json.dumps({"item_id": item["item_id"], "probs": probs}) + "\n")
    print(f"Saved {len(ds)} predictions -> {args.out}")


if __name__ == "__main__":
    main()
