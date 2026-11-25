"""Y4 zero-shot scoring of a base model with the bench readout (osler_v4): the D35 ceiling screen and the Qwen-base baselines.

Our items (MIMIC dev and suite candidates, fast dev v4, CG v2) are converted to the O2 row shape the readout reads; nothing
is invented: every field is copied from the item. The readout is the one of scripts/osler/o2_run_zeroshot.py, loaded by
path and not changed (canonical_item, the variant check, the prompt, the letter probabilities). Single order, resumable.

Writes outputs/osler_v4/Y4/zeroshot/<name>/{predictions.jsonl, variant.json}. The predictions hold option probabilities and
gold keys: private to this machine (MIMIC items), never committed.

Usage (one GPU job at a time; detached):
    source scripts/pod_env.sh && setsid nohup uv run --frozen python scripts/osler_v4/y4_zeroshot.py \
        --model Qwen/Qwen3.5-4B --items data/bench/osler_v4/mimic_dev_all.jsonl --name qwen35_4b_mimic_dev \
        > outputs/osler_v4/Y4/logs/zeroshot_qwen35_4b.log 2>&1 < /dev/null &
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
OUT = REPO / "outputs" / "osler_v4" / "Y4" / "zeroshot"
BASE_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"  # Qwen/Qwen3.5-4B, the osler base revision (cached)
ZEROSHOT_SCRIPT = REPO / "scripts" / "osler" / "o2_run_zeroshot.py"


def _zeroshot_module() -> Any:
    """The W5 zero-shot runner, loaded by path: its canonical_item is the readout's own (not copied here)."""
    spec = importlib.util.spec_from_file_location("o2_run_zeroshot", ZEROSHOT_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["o2_run_zeroshot"] = module
    spec.loader.exec_module(module)
    return module


def to_o2_row(row: dict[str, Any]) -> dict[str, Any]:
    """Our item (make_row or CG v2 schema) as the O2 row shape the readout reads. Each field is copied from the item."""
    return {
        "item_id": str(row["item_id"]),
        "benchmark": str(row.get("benchmark") or row.get("tier") or "osler_v4"),
        "set_name": str(row.get("set_name") or row.get("cg_partition") or row.get("source") or "osler_v4"),
        "template_id": str(row["template_id"]),
        "qtype": str(row["qtype"]),
        "state": row["state"],
        "question": row["question"],
        "options": [{"key": str(o["key"]), "label": str(o["label"])} for o in row["options"]],
        "gold": str(row["gold"]),
        "split": row.get("split"),
        "source_record_id": row.get("source_record_id"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="a base model id, e.g. Qwen/Qwen3.5-4B")
    parser.add_argument("--revision", default=BASE_REVISION, help="the pinned commit of --model (the osler base by default)")
    parser.add_argument("--items", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--limit", type=int, default=0, help="score only the first N new items (smoke runs)")
    args = parser.parse_args()
    zs = _zeroshot_module()
    out_dir = OUT / args.name
    out_path = out_dir / "predictions.jsonl"
    tok = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    model = AutoModelForCausalLM.from_pretrained(args.model, revision=args.revision, dtype=torch.bfloat16,
                                                 device_map="cuda").eval()
    checks = check_label_tokens(tok, args.model)
    variant, variant_detail = detect_variant(model, tok, checks)
    write_json(out_dir / "variant.json", {"model": args.model, "variant": variant, "detail": variant_detail,
                                          "checks": [c.to_dict() for c in checks], "utc": utcnow(),
                                          "items": str(args.items), "gpu": gpu_name()})
    done = completed_ids(out_path)
    print(f"[{utcnow()}] {args.model} on {gpu_name()}: variant={variant}; resume_rows={len(done)}", flush=True)
    n_new = 0
    started = time.time()
    with open_predictions(out_path) as fh, torch.inference_mode():
        for raw in iter_jsonl(args.items):
            row = to_o2_row(raw)
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
                extras={"variant": variant, "canonical_letters": [o.key for o in item.options],
                        "original_keys": original, "label_mass": readout["label_mass"],
                        "vocab_argmax_is_option": readout["vocab_argmax_is_option"],
                        "best_option_in_top5": readout["best_option_in_top5"],
                        "top1_over_option_mass": readout["top1_over_option_mass"]}))
            n_new += 1
            if n_new % 1000 == 0:
                print(f"[{utcnow()}] {n_new} new rows, {n_new / (time.time() - started):.1f} items/s", flush=True)
    print(f"[{utcnow()}] done: new rows={n_new}; total rows={len(completed_ids(out_path))}", flush=True)


if __name__ == "__main__":
    main()
