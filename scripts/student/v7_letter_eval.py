#!/usr/bin/env python
"""Evaluate arm B (letter readout) on the v0.2 test items, with the D12 readout measurements.

Unlike the pointer head, the letter readout's option scores live in the full vocabulary, so this evaluator
measures what run_student writes by construction for a pointer head: the full-vocabulary label mass of the
offered letters, whether the vocabulary argmax is an offered letter, and the rank of the best letter. It
also runs the greedy-agreement sample (the model's own greedy first token must equal the readout's argmax
letter) on 50 items per template, then applies ``meddecide.eval.health.evaluate_cell`` per template.

Every item is scored alone, with no padding (the V2 evaluation path). Writes:
  outputs/student_v1/V7/<tag>/preds_<model>__test.jsonl   run_student-format rows (measured readout fields)
  outputs/student_v1/V7/<tag>/d12_cells.json              the gate verdict per template
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "bench"))

from meddecide.bench.schema import Item  # noqa: E402
from meddecide.eval.health import evaluate_cell  # noqa: E402
from meddecide.eval.paired import item_brier  # noqa: E402
from meddecide.eval.readout import canonicalise_options  # noqa: E402
from meddecide.model.meddecide_model import (  # noqa: E402
    MedDecideModel,
    render_item_prompt,
    segment_softmax,
)

GREEDY_PER_TEMPLATE = 50


def letter_tokens(model: MedDecideModel, n: int) -> list[int]:
    return [model._key_token(chr(ord("A") + j)) for j in range(n)]


@torch.no_grad()
def score_item(model: MedDecideModel, item: Item) -> dict:
    """Letter probabilities and the full-vocabulary readout measurements for one item (no padding)."""
    encoded = model.encode_item(item)
    batch = model.collate([encoded], round_to_chunk=False).to(model.device)
    hidden = model.hidden_states(batch)
    h = hidden[torch.arange(1, device=hidden.device), batch.answer_positions.to(hidden.device)]
    causal = model.peft_model.get_base_model() if model.peft_model is not None else model.base
    vocab_logits = causal.lm_head(h).float()[0]
    vocab_probs = torch.softmax(vocab_logits, dim=-1)
    n = encoded.n_options
    tokens = torch.tensor(letter_tokens(model, n), device=vocab_probs.device, dtype=torch.long)
    letter_logits = vocab_logits[tokens]
    option_probs = segment_softmax(letter_logits, [n])
    label_mass = float(vocab_probs[tokens].sum())
    top_token = int(torch.argmax(vocab_probs))
    best = int(torch.argmax(option_probs))
    best_token_prob = float(vocab_probs[tokens[best]])
    rank = int((vocab_probs > best_token_prob).sum())
    return {
        "option_probs": option_probs.double().cpu().numpy(),
        "label_mass": label_mass,
        "vocab_argmax_is_option": bool((tokens == top_token).any()),
        "n_tokens_above_best_option": rank,
        "best_option_in_top5": rank < 5,
        "top1_over_option_mass": float(vocab_probs.max()) / label_mass if label_mass > 0 else None,
        "latency_s": None,
        "prompt_tokens": encoded.n_tokens,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--items", type=Path, default=ROOT / "outputs" / "student_v1" / "V9" / "test_items.jsonl")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model-id", default="meddecide-v1-arm-b")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    items = [Item.model_validate_json(line) for line in args.items.open(encoding="utf-8")]
    if args.limit:
        items = items[: args.limit]
    model = MedDecideModel.load(args.checkpoint, device="cuda:0", dtype="bfloat16")
    if model.readout != "letter":
        raise SystemExit(f"checkpoint readout is {model.readout!r}, expected 'letter'")
    model.eval_mode()
    run_id = f"{args.model_id}:{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}"
    rows, per_template = [], defaultdict(list)
    started = time.time()
    for index, item in enumerate(items):
        t0 = time.time()
        measured = score_item(model, item)
        latency = time.time() - t0
        canonical, original = canonicalise_options(item)
        keys = [o.key for o in canonical.options]
        probs = measured["option_probs"]
        argmax_letter = keys[int(np.argmax(probs))]
        gold_letter = canonical.gold
        correct = argmax_letter == gold_letter
        rows.append({
            "item_id": item.item_id, "model_id": args.model_id, "run_id": run_id,
            "source": item.source, "template_id": str(item.template_id), "split": "test",
            "qtype": str(item.qtype), "option_keys": keys, "original_option_keys": original,
            "option_probs": [float(p) for p in probs],
            "label_mass": measured["label_mass"],
            "vocab_argmax_is_option": measured["vocab_argmax_is_option"],
            "n_tokens_above_best_option": measured["n_tokens_above_best_option"],
            "best_option_in_top5": measured["best_option_in_top5"],
            "top1_over_option_mass": measured["top1_over_option_mass"],
            "argmax_key": argmax_letter, "gold_key": gold_letter, "correct": bool(correct),
            "expected_level": None, "latency_s": latency, "variant": "letter-readout",
            "prompt_tokens": measured["prompt_tokens"],
            "transform": {"readout": "letter", "letter_readout_fields": "measured on the full vocabulary"},
        })
        per_template[str(item.template_id)].append({
            "item": item, "argmax": argmax_letter, "gold": gold_letter, "correct": bool(correct),
            "label_mass": measured["label_mass"], "n": len(keys),
            "probs": probs, "gold_index": keys.index(gold_letter),
        })
        if index and index % 5000 == 0:
            print(f"scored {index}/{len(items)} in {time.time() - started:.0f}s", flush=True)

    # greedy agreement: the model's own greedy first token must be the readout's argmax letter
    cells = {}
    for template_id in sorted(per_template):
        group = sorted(per_template[template_id], key=lambda r: r["item"].item_id)
        sample = group[:GREEDY_PER_TEMPLATE]
        prompts = [render_item_prompt(model, r["item"]) for r in sample]
        generated = model.generate_greedy(prompts, max_new_tokens=4)
        matches = []
        for record, tokens in zip(sample, generated, strict=True):
            first = model.decode(tokens[:1]).strip()
            matches.append(first == record["argmax"])
        report = evaluate_cell(
            model_id=args.model_id,
            template_id=template_id,
            label_masses=[r["label_mass"] for r in group],
            correct=[r["correct"] for r in group],
            greedy_matches=matches,
            n_options=group[0]["n"],
            predicted_labels=[r["argmax"] for r in group],
        )
        cells[template_id] = {
            "n": len(group),
            "greedy_sample": len(sample),
            "greedy_agreement": float(np.mean(matches)) if matches else None,
            "median_label_mass": float(np.median([r["label_mass"] for r in group])),
            "accuracy": float(np.mean([r["correct"] for r in group])),
            "brier": float(np.mean([item_brier(r["probs"], r["gold_index"]) for r in group])),
            "gate": report.to_dict(),
        }

    preds_path = args.out_dir / f"preds_{args.model_id}__test.jsonl"
    with preds_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (args.out_dir / "d12_cells.json").write_text(json.dumps({"model_id": args.model_id, "cells": cells},
                                                            indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"items": len(rows), "cells": len(cells), "preds": str(preds_path.relative_to(ROOT)),
                      "seconds": round(time.time() - started, 1)}), flush=True)


if __name__ == "__main__":
    main()
