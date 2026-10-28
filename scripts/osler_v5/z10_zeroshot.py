"""osler_v5 Z10 (ADVISORY §10 Z10; GPU; additional, labelled; no choice): zero-shot Qwen3.5 base models on the Z10 sets.

The readout is osler_v4's Y4 zero-shot readout, not a copy: scripts/osler_v4/y4_zeroshot.py is loaded by path, and its to_o2_row
and the W5 zero-shot runner (scripts/osler/o2_run_zeroshot.py, canonical_item, the prompt and the letter probabilities) are used
unchanged. One item at a time, single order, the model's own chat template, next-token logits at the answer position. Each row is
in the O2 schema with the full-vocabulary label mass and the vocabulary-argmax flag that the D12 gate reads (meddecide.eval.o2_metrics).

Sets (osler_v4's Y9 copies; data/bench/osler_v4/y9/): mr_verdict, mr_reported (MR suite v1), mimic_suite and mimic_heldout (MIMIC
audit suite v1), mednli_test, panel (7 sets) and knowledge (5,456 items; 5,435 are paired with osler_v4's zero-shot pplx in the report).
Rows go to outputs/osler_v5/Z10/zeroshot/<slug>/<set>.jsonl (gitignored; resumable per set). variant.json records the readout variant.
MIMIC items go only to this local model (AGENTS.md data rules).

Usage (GPU; one job; detached):
    source scripts/pod_env.sh && setsid nohup uv run --frozen python scripts/osler_v5/z10_zeroshot.py \
        --model Qwen/Qwen3.5-4B --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a --slug qwen35_4b \
        > outputs/osler_v5/Z10/logs/zeroshot_qwen35_4b.log 2>&1 < /dev/null &
A smoke run passes --limit N (N new items per set, written to the same folder; delete the smoke files before the full run).
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from meddecide.eval.o2_io import completed_ids, open_predictions, prediction_row, write_row
from meddecide.eval.readout import (
    check_label_tokens,
    detect_variant,
    option_token_ids,
    read_option_probabilities,
    render_prompt,
)
from meddecide.utils.io import iter_jsonl, write_json
from meddecide.utils.provenance import gpu_name, utcnow

REPO = Path(__file__).resolve().parents[2]
Y4_PATH = REPO / "scripts" / "osler_v4" / "y4_zeroshot.py"
Y9_DIR = REPO / "data" / "bench" / "osler_v4" / "y9"
OUT = REPO / "outputs" / "osler_v5" / "Z10" / "zeroshot"
SETS = {
    "mr_verdict": "mr_verdict.jsonl",
    "mr_reported": "mr_reported.jsonl",
    "mimic_suite": "mimic_suite.jsonl",
    "mimic_heldout": "mimic_heldout.jsonl",
    "mednli_test": "mednli_test.jsonl",
    "panel": "panel.jsonl",
    "knowledge": "knowledge.jsonl",
}


def _y4_module() -> Any:
    """osler_v4's Y4 zero-shot runner, loaded by path (its to_o2_row and its W5 readout module are used, not copied)."""
    spec = importlib.util.spec_from_file_location("y4_zeroshot", Y4_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["y4_zeroshot"] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", required=True, help="a base model id, e.g. Qwen/Qwen3.5-4B")
    parser.add_argument("--revision", required=True, help="the pinned commit of --model (configs/osler_v0/competitors.json)")
    parser.add_argument("--slug", required=True, help="output folder name, e.g. qwen35_4b")
    parser.add_argument("--limit", type=int, default=0, help="score only the first N new items of each set (smoke runs)")
    args = parser.parse_args()
    y4 = _y4_module()
    zs = y4._zeroshot_module()
    out_dir = OUT / args.slug
    out_dir.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    model = AutoModelForCausalLM.from_pretrained(args.model, revision=args.revision, dtype=torch.bfloat16,
                                                 device_map="cuda").eval()
    checks = check_label_tokens(tok, args.model)
    variant, variant_detail = detect_variant(model, tok, checks)
    write_json(out_dir / "variant.json", {"model": args.model, "revision": args.revision, "variant": variant,
                                          "detail": variant_detail, "checks": [c.to_dict() for c in checks],
                                          "utc": utcnow(), "gpu": gpu_name(), "sets": SETS})
    print(f"[{utcnow()}] {args.model} @ {args.revision[:8]} on {gpu_name()}: variant={variant}", flush=True)
    with torch.inference_mode():
        for set_name, file_name in SETS.items():
            out_path = out_dir / f"{set_name}.jsonl"
            done = completed_ids(out_path)
            print(f"[{utcnow()}] {set_name}: resume_rows={len(done)} -> {out_path}", flush=True)
            n_new = 0
            started = time.time()
            with open_predictions(out_path) as fh:
                for raw in iter_jsonl(Y9_DIR / file_name):
                    row = y4.to_o2_row(raw)
                    if row["item_id"] in done:
                        continue
                    if args.limit and n_new >= args.limit:
                        break
                    t0 = time.time()
                    item, original = zs.canonical_item(row)
                    text = render_prompt(item, tok)
                    enc = tok(text, return_tensors="pt", add_special_tokens=False).to("cuda")
                    n_tokens = int(enc["input_ids"].shape[1])
                    logits = model(**enc, logits_to_keep=1).logits[0, -1].float().cpu().numpy()
                    token_ids = option_token_ids(item, tok, variant=variant)
                    readout = read_option_probabilities(logits, token_ids)
                    write_row(fh, prediction_row(
                        row, status="scored", probs=[float(p) for p in readout["option_probs"]],
                        latency_s=round(time.time() - t0, 4), prompt_tokens=n_tokens,
                        extras={"variant": variant, "z10_set": set_name, "canonical_letters": [o.key for o in item.options],
                                "original_keys": original, "label_mass": readout["label_mass"],
                                "vocab_argmax_is_option": readout["vocab_argmax_is_option"],
                                "best_option_in_top5": readout["best_option_in_top5"],
                                "top1_over_option_mass": readout["top1_over_option_mass"]}))
                    n_new += 1
                    if n_new % 1000 == 0:
                        print(f"[{utcnow()}] {set_name}: {n_new} new rows, {n_new / (time.time() - started):.1f} items/s", flush=True)
            print(f"[{utcnow()}] {set_name}: done, new rows={n_new}; total rows={len(completed_ids(out_path))}", flush=True)
    print(f"[{utcnow()}] all sets done", flush=True)


if __name__ == "__main__":
    main()
