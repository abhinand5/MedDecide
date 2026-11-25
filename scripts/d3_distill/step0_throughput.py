"""d3_distill step 0: teacher throughput on 1,000 MedMCQA training items (vllm-sr/d3-ultra, local, one state per item).

Measures items/s through the card's public path (system_one, one request per item) and checks each answer is a valid
distribution. Gate (PLAN step 0): >= 25 items/s. Writes aggregates only to the given output JSON.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "d3_bench"))
from run_d3 import ask, load  # noqa: E402  (the benchmark runner's own load and call path)


def medmcqa_items(n: int) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("openlifescienceai/medmcqa", split="train")
    items = []
    for i in range(n):
        r = ds[i]
        opts = {"A": r["opa"], "B": r["opb"], "C": r["opc"], "D": r["opd"]}
        items.append({"id": f"mcq{i}", "stem": r["question"], "options": opts, "gold": "ABCD"[r["cop"]]})
    return items


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", type=Path, required=True)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    items = medmcqa_items(args.n)
    model, load_s = load(args.snapshot)
    bad, correct, t0 = 0, 0, time.time()
    for it in items:
        probs, _ = ask(model, it["stem"], "choice", "Which option is the correct answer?", it["options"])
        keys = list(it["options"])
        ok = probs is not None and abs(sum(probs) - 1.0) <= 1e-3 and all(math.isfinite(p) and p >= 0 for p in probs) \
            and len({round(p, 6) for p in probs}) > 1
        bad += 0 if ok else 1
        if ok and keys[max(range(len(keys)), key=lambda j: probs[j])] == it["gold"]:
            correct += 1
    elapsed = time.time() - t0
    rate = args.n / elapsed
    result = {"n": args.n, "elapsed_s": round(elapsed, 2), "items_per_s": round(rate, 3), "load_s": round(load_s, 1),
              "invalid_distributions": bad, "dev_accuracy_diagnostic": round(correct / args.n, 4),
              "gate_25_items_per_s": "PASS" if rate >= 25 and bad == 0 else "FAIL", "path": "system_one, one request per item"}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
