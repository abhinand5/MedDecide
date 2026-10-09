"""O2: zero-shot Qwen3.5 base scored with OUR harness (not an authors' code path; the D12 gate applies).

Readout, as the bench harness does it (meddecide.eval.readout):
  - the answer letter variant ("bare" or "space") is detected once per model by greedy generation on a
    fixed probe (detect_variant), never guessed; the detected variant is recorded in variant.json;
  - options are canonicalised to letters in display order (choice unchanged; noul rendered "A. Yes / B. No";
    score levels lowest first) as canonicalise_options does, and the probabilities map back to the item's
    original keys in the same order;
  - one forward pass at the answer position; read_option_probabilities gives the restricted probabilities
    (the scored numbers) and the full-vocabulary label mass and argmax diagnostics that the D12 gate uses.

Items are scored in file order, single order. Items over the common cap are not in o2_items.jsonl and are
never scored.

Usage:
    source scripts/pod_env.sh && uv run --frozen python scripts/osler/o2_run_zeroshot.py \
        --model Qwen/Qwen3.5-4B --scope panel_robustness --limit 0
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
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
from meddecide.eval.readout import (
    check_label_tokens,
    detect_variant,
    option_token_ids,
    read_option_probabilities,
    render_prompt,
)
from meddecide.utils.io import write_json
from meddecide.utils.provenance import gpu_name, utcnow

REPO = Path(__file__).resolve().parents[2]
ITEMS = REPO / "data/bench/v0.3_ext/o2_items.jsonl"
OUT_ROOT = REPO / "outputs/osler_v0/O2"
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def canonical_item(row: dict[str, Any]) -> tuple[SimpleNamespace, list[str]]:
    """The bench's canonicalise_options for a dict item: letters in display order.

    Returns the item (letters as keys, labels in display order) and the original keys in the same order.
    noul items with keys yes/no render as "A. Yes", "B. No" (the bench's fixed order), so their display order
    equals the stored order; choice is already lettered; score levels keep their stored (lowest-first) order.
    """
    original = [o["key"] for o in row["options"]]
    labels = [o["label"] for o in row["options"]]
    if row["qtype"] == "noul" and original == ["yes", "no"]:
        labels = ["Yes", "No"]
    options = [SimpleNamespace(key=LETTERS[i], label=label) for i, label in enumerate(labels)]
    return SimpleNamespace(state=row["state"], question=row["question"], options=options), original


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--scope", choices=["all", "panel_robustness"], default="panel_robustness")
    ap.add_argument("--limit", type=int, default=0, help="score only the first N new items (smoke tests)")
    args = ap.parse_args()
    scope = ALL_SETS if args.scope == "all" else PANEL_ROBUSTNESS
    slug = args.model.split("/")[-1].lower().replace(".", "")
    out_dir = OUT_ROOT / slug
    out_path = out_dir / "predictions.jsonl"

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda").eval()
    checks = check_label_tokens(tok, args.model)
    variant, variant_detail = detect_variant(model, tok, checks)
    write_json(out_dir / "variant.json", {"model": args.model, "variant": variant, "detail": variant_detail,
                                          "checks": [c.to_dict() for c in checks], "utc": utcnow()})
    print(f"[{utcnow()}] {args.model} on {gpu_name()}: variant={variant} scope={scope.name}", flush=True)

    done = completed_ids(out_path)
    print(f"resume_rows={len(done)} -> {out_path}", flush=True)
    n_new = 0
    t_start = time.time()
    with open_predictions(out_path) as fh, torch.inference_mode():
        for row in iter_items(ITEMS, scope):
            if row["item_id"] in done:
                continue
            if args.limit and n_new >= args.limit:
                break
            t0 = time.time()
            item, original = canonical_item(row)
            text = render_prompt(item, tok)
            enc = tok(text, return_tensors="pt", add_special_tokens=False).to("cuda")
            n_tokens = int(enc["input_ids"].shape[1])
            logits = model(**enc, logits_to_keep=1).logits[0, -1].float().cpu().numpy()
            token_ids = option_token_ids(item, tok, variant=variant)
            readout = read_option_probabilities(logits, token_ids)
            latency = time.time() - t0
            write_row(fh, prediction_row(
                row, status="scored", probs=[float(p) for p in readout["option_probs"]],
                latency_s=round(latency, 4), prompt_tokens=n_tokens,
                extras={
                    "variant": variant,
                    "canonical_letters": [o.key for o in item.options],
                    "original_keys": original,
                    "label_mass": readout["label_mass"],
                    "vocab_argmax_is_option": readout["vocab_argmax_is_option"],
                    "best_option_in_top5": readout["best_option_in_top5"],
                    "top1_over_option_mass": readout["top1_over_option_mass"],
                },
            ))
            n_new += 1
            if n_new % 500 == 0:
                rate = n_new / (time.time() - t_start)
                print(f"[{utcnow()}] scored {n_new} new rows ({rate:.1f} items/s)", flush=True)
    print(f"[{utcnow()}] done: new rows={n_new}; total rows={len(completed_ids(out_path))}", flush=True)


if __name__ == "__main__":
    main()
