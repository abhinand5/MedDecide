"""O2: autotrust JEV checkpoints (JEV-27B, JEV-9B) through the card's decision-head protocol.

The protocol is the one the card documents and the F7 baseline implements (scripts/bench/run_jev_baseline.py,
imported here so the scoreboard and the Gate baseline share one implementation): the prompt ends in
"[decision]:", one prefill pass, the full-vocabulary log-softmax at the last position, the head's per-slot bias,
the per-kind temperature from calibration.json, and a softmax over the kind's verbalizer tokens. The card's
vLLM route is not run (see the deviation recorded for O0).

Limits of the head, from the card: 16 choice slots, 2 noul slots, 6 score slots. Items beyond a kind's slots
are skipped with the reason recorded. Single option order (the card does not average orders).

noul: the head reads "false" and "true"; our noul options are yes (true) and no (false), so the output is
re-ordered to the item's order (yes, no).

Usage:
    source scripts/pod_env.sh && uv run --frozen python scripts/osler/o2_run_jev.py \
        --repo autotrust/JEV-27B --scope all
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from meddecide.eval.o2_io import (
    ALL_SETS,
    PANEL_ROBUSTNESS,
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
F7_PATH = REPO / "scripts/bench/run_jev_baseline.py"
SLOTS = {"choice": 16, "noul": 2, "score": 6}


def load_f7():
    spec = importlib.util.spec_from_file_location("run_jev_baseline", F7_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--repo", required=True)
    ap.add_argument("--scope", choices=["all", "panel_robustness"], default="all")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--verify", type=int, default=20,
                    help="items where logits_to_keep=1 is checked against the full-sequence logits")
    args = ap.parse_args()
    scope = ALL_SETS if args.scope == "all" else PANEL_ROBUSTNESS
    slug = args.repo.split("/")[-1].lower()
    out_dir = OUT_ROOT / slug
    out_path = out_dir / "predictions.jsonl"

    f7 = load_f7()
    head, temperatures = f7.load_head(args.repo)
    tok = AutoTokenizer.from_pretrained(args.repo)
    t0 = time.time()
    base = AutoModelForCausalLM.from_pretrained(args.repo, dtype=torch.bfloat16, device_map="cuda")
    model = PeftModel.from_pretrained(base, args.repo, subfolder="adapter_vllm").eval()
    load_s = round(time.time() - t0, 1)
    write_json(out_dir / "run_meta.json", {"repo": args.repo, "protocol": "card decision head via F7 helpers",
                                           "load_s": load_s, "temperatures": temperatures, "gpu": gpu_name(),
                                           "utc": utcnow(), "slots": SLOTS})
    print(f"[{utcnow()}] {args.repo} on {gpu_name()}: loaded in {load_s}s; temperatures={temperatures}", flush=True)

    verify_diffs: list[float] = []
    if args.verify:
        sample = [r for r in iter_items(ITEMS, scope)
                  if len(r["options"]) <= SLOTS[r["qtype"]]][: args.verify]
        with torch.inference_mode():
            for r in sample:
                labels = [o["label"] for o in r["options"]]
                prompt = f7.build_prompt(r["qtype"], r["state"], r["question"], labels)
                ids = f7.verbalizer_ids(head, r["qtype"], len(labels))
                enc = tok(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
                full = f7.head_probabilities(model(**enc).logits[0, -1, :], ids, head, r["qtype"],
                                             temperatures[r["qtype"]], len(labels))
                kept = f7.head_probabilities(model(**enc, logits_to_keep=1).logits[0, -1, :], ids, head,
                                             r["qtype"], temperatures[r["qtype"]], len(labels))
                verify_diffs.append(max(abs(a - b) for a, b in zip(full, kept, strict=True)))
        meta = json.loads((out_dir / "run_meta.json").read_text())
        meta["verify_logits_to_keep"] = {"n": len(verify_diffs), "max_abs_diff": max(verify_diffs) if verify_diffs else None}
        write_json(out_dir / "run_meta.json", meta)
        print(f"verify logits_to_keep=1 vs full: n={len(verify_diffs)} max|diff|={max(verify_diffs) if verify_diffs else None}", flush=True)
    done = completed_ids(out_path)
    print(f"resume_rows={len(done)} -> {out_path}", flush=True)
    n_new = 0
    with open_predictions(out_path) as fh, torch.inference_mode():
        for row in iter_items(ITEMS, scope):
            if row["item_id"] in done:
                continue
            if args.limit and n_new >= args.limit:
                break
            qtype = row["qtype"]
            n = len(row["options"])
            if n > SLOTS[qtype]:
                write_row(fh, prediction_row(
                    row, status="skipped",
                    reason=f"{n} options exceed the head's {SLOTS[qtype]} {qtype} slots"))
                n_new += 1
                continue
            t0 = time.time()
            labels = [o["label"] for o in row["options"]]
            kind = qtype
            prompt = f7.build_prompt(kind, row["state"], row["question"], labels)
            ids = f7.verbalizer_ids(head, kind, n)
            enc = tok(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
            logits = model(**enc, logits_to_keep=1).logits[0, -1, :]
            probs = f7.head_probabilities(logits, ids, head, kind, temperatures[kind], n)
            if kind == "noul":
                # head order is (false, true); our options are (yes, no)
                probs = [probs[1], probs[0]]
            latency = time.time() - t0
            write_row(fh, prediction_row(
                row, status="scored", probs=[float(p) for p in probs], latency_s=round(latency, 4),
                prompt_tokens=int(enc["input_ids"].shape[1]),
                extras={"kind": kind, "temperature": temperatures[kind], "verbalizer_ids": ids[:n]},
            ))
            n_new += 1
            if n_new % 500 == 0:
                print(f"[{utcnow()}] {n_new} new rows", flush=True)
    print(f"[{utcnow()}] done: new rows={n_new}; total rows={len(completed_ids(out_path))}", flush=True)


if __name__ == "__main__":
    main()
