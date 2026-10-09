"""O2: MedDecider (lion-ai) scored through its authors' code (meddecider_infer.py, in the checkpoint).

Protocol (the authors' own, from meddecider_infer.py and the card):
  - the authors' MedDecider class loads the base model named in meddecider_config.json and merges the LoRA;
  - decide(state, question, options, qtype, orders=2): both option orders scored and averaged, one forward pass
    per order, temperature per qtype from meddecider_config.json (choice 1.33 / 1.43 / 1.23 / 1.49 by size;
    noul and score 1.0);
  - the authors' letter set is A to J, so questions with more than 10 options are not scored (reason recorded).

One difference from the authors' call: decide() returns a dict keyed by label, so two options with the same
label would collapse (the replace-gold robustness items do this). This runner reproduces decide()'s arithmetic
exactly on the option vector instead. --verify N checks the vector form against the authors' decide() on N
items with unique labels and records the largest difference (the authors' dict is rounded to 4 dp).

Items: the common set (o2_items.jsonl), all three benchmarks. Items over the common cap are not scored.

Usage:
    source scripts/pod_env.sh && uv run --frozen python scripts/osler/o2_run_meddecider.py \
        --model lion-ai/MedDecider-4B --revision 570b37090c379ad722186a79a3432a31be463fc9 --verify 50
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from huggingface_hub import snapshot_download

from meddecide.eval.o2_io import (
    ALL_SETS,
    completed_ids,
    iter_items,
    open_predictions,
    prediction_row,
    write_row,
)
from meddecide.utils.io import write_json
from meddecide.utils.provenance import gpu_name, utcnow

REPO = Path(__file__).resolve().parents[2]
ITEMS = REPO / "data/bench/v0.3_ext/o2_items.jsonl"
OUT_ROOT = REPO / "outputs/osler_v0/O2"
MAX_OPTIONS = 10  # the authors' LETTERS = "ABCDEFGHIJ"


def decide_vector(mj, state: str, question: str, labels: list[str], qtype: str) -> list[float]:
    """decide(orders=2) of the authors' class, returned as a vector aligned with ``labels``."""
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
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True, help="Hub repo id of the MedDecider adapter")
    ap.add_argument("--revision", required=True, help="pinned commit sha (configs/osler_v0/competitors.json)")
    ap.add_argument("--verify", type=int, default=50, help="items checked against the authors' decide()")
    ap.add_argument("--limit", type=int, default=0, help="score only the first N new items (smoke tests)")
    args = ap.parse_args()
    slug = "meddecider-" + args.model.split("MedDecider-")[-1].lower()
    out_dir = OUT_ROOT / slug
    out_path = out_dir / "predictions.jsonl"

    local = Path(snapshot_download(args.model, revision=args.revision))
    sys.path.insert(0, str(local))
    from meddecider_infer import MedDecider  # the authors' module, from the checkpoint

    t0 = time.time()
    mj = MedDecider(str(local))
    load_s = round(time.time() - t0, 1)
    print(f"[{utcnow()}] {args.model}@{args.revision[:12]} on {gpu_name()}: loaded in {load_s}s; "
          f"temperature={mj.cfg['temperature']} base={mj.cfg['base_model']}", flush=True)

    verify = {"n": 0, "max_abs_diff": None, "note": "vector form vs authors' decide() dict (4 dp)"}
    checked = 0
    diffs: list[float] = []
    if args.verify:
        rows = [r for r in iter_items(ITEMS, ALL_SETS) if len({o["label"] for o in r["options"]}) == len(r["options"])
                and len(r["options"]) <= MAX_OPTIONS][: args.verify * 4]
        for row in rows:
            if checked >= args.verify:
                break
            labels = [o["label"] for o in row["options"]]
            authors = mj.decide(row["state"], row["question"], labels, row["qtype"], orders=2)
            vec = decide_vector(mj, row["state"], row["question"], labels, row["qtype"])
            diffs.append(max(abs(vec[i] - authors[labels[i]]) for i in range(len(labels))))
            checked += 1
        verify = {"n": checked, "max_abs_diff": max(diffs) if diffs else None,
                  "mean_abs_diff": sum(diffs) / len(diffs) if diffs else None,
                  "note": "vector form vs authors' decide() dict (dict rounded to 4 dp)"}
        write_json(out_dir / "verify_decide.json", verify)
        print(f"verify vs authors' decide(): n={checked} max|diff|={verify['max_abs_diff']}", flush=True)

    done = completed_ids(out_path)
    print(f"resume_rows={len(done)} -> {out_path}", flush=True)
    n_new = 0
    t_start = time.time()
    with open_predictions(out_path) as fh:
        for row in iter_items(ITEMS, ALL_SETS):
            if row["item_id"] in done:
                continue
            if args.limit and n_new >= args.limit:
                break
            labels = [o["label"] for o in row["options"]]
            if len(labels) > MAX_OPTIONS:
                write_row(fh, prediction_row(
                    row, status="skipped",
                    reason=f"more than {MAX_OPTIONS} options: the authors' letter set is A to J",
                ))
                n_new += 1
                continue
            t0 = time.time()
            probs = decide_vector(mj, row["state"], row["question"], labels, row["qtype"])
            latency = time.time() - t0
            write_row(fh, prediction_row(
                row, status="scored", probs=probs, latency_s=round(latency, 4),
                extras={"orders": 2, "temperature": mj.cfg["temperature"].get(row["qtype"], 1.0),
                        "authors_code": "meddecider_infer.py MedDecider.prompt + logits (decide arithmetic)"},
            ))
            n_new += 1
            if n_new % 500 == 0:
                print(f"[{utcnow()}] {n_new} new rows ({n_new / (time.time() - t_start):.2f} items/s)", flush=True)
    write_json(out_dir / "run_meta.json", {"model": args.model, "revision": args.revision, "local": str(local),
                                           "load_s": load_s, "gpu": gpu_name(), "utc": utcnow(),
                                           "verify": verify, "config": mj.cfg})
    print(f"[{utcnow()}] done: new rows={n_new}; total rows={len(completed_ids(out_path))}", flush=True)


if __name__ == "__main__":
    main()
