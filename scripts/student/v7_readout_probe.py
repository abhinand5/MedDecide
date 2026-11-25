#!/usr/bin/env python
"""Where the probability of arm B's letter readout sits: a labelled additional analysis of the D12 failures (V9).

For the first ``--per-template`` test items of every template (file order, deterministic), this scores the
letter-readout checkpoint once and reads the full next-token distribution at the answer position:

  * the mass on the offered letters in the readout's own form (``variant=bare``: "A", "B", ...), and on the
    leading-space form (" A", " B", ...), which the trained readout never reads;
  * the mass on letters that are not offered (for a 4-option item, E..Z), which the restricted softmax
    never sees during training;
  * whether the top token is an offered bare letter;
  * whether the scoring prompt is the generation prompt (same text), and whether the restricted argmax equals
    the first greedy token (stripped).

It describes the D12 failures; it does not change any verdict. Writes ``--out`` (aggregate per template and
per item, no item text). Runs on one GPU (one job at a time).

Run: ``uv run python scripts/student/v7_readout_probe.py --checkpoint outputs/student_v1/V7/arm_b/checkpoints/step_5000``
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from meddecide.bench.schema import Item  # noqa: E402
from meddecide.eval.readout import canonicalise_options, key_token_id  # noqa: E402
from meddecide.model.meddecide_model import MedDecideModel, render_item_prompt  # noqa: E402

ALPHABET = [chr(ord("A") + k) for k in range(26)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--items", type=Path, default=ROOT / "outputs" / "student_v1" / "V9" / "test_items.jsonl")
    parser.add_argument("--per-template", type=int, default=8)
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "student_v1" / "V9" / "arm_b" / "readout_probe.json")
    args = parser.parse_args()

    model = MedDecideModel.load(args.checkpoint, device="cuda:0", dtype="bfloat16")
    if model.readout != "letter":
        raise SystemExit(f"checkpoint readout is {model.readout!r}, expected 'letter'")
    model.eval_mode()
    tok = model.tokenizer
    bare_all = {k: key_token_id(k, tok, "bare") for k in ALPHABET}
    space_all = {k: key_token_id(k, tok, "space") for k in ALPHABET}

    chosen: dict[str, list[Item]] = defaultdict(list)
    with args.items.open(encoding="utf-8") as handle:
        for line in handle:
            item = Item.model_validate_json(line)
            template = str(item.template_id)
            if len(chosen[template]) < args.per_template:
                chosen[template].append(item)

    per_item: list[dict] = []
    for template in sorted(chosen):
        for item in chosen[template]:
            canonical, _ = canonicalise_options(item)
            n = len(canonical.options)
            offered = ALPHABET[:n]
            encoded = model.encode_item(item)
            batch = model.collate([encoded], round_to_chunk=False).to(model.device)
            with torch.no_grad():
                hidden = model.hidden_states(batch)
                h = hidden[torch.arange(1, device=hidden.device), batch.answer_positions.to(hidden.device)]
                causal = model.peft_model.get_base_model() if model.peft_model is not None else model.base
                probs = torch.softmax(causal.lm_head(h).float()[0], dim=-1).double().cpu().numpy()
            bare = np.array([probs[bare_all[k]] for k in ALPHABET])
            space = np.array([probs[space_all[k]] for k in ALPHABET])
            offered_idx = np.arange(n)
            other_idx = np.arange(n, 26)
            prompt = render_item_prompt(model, item)
            same_text = tok.decode(encoded.input_ids) == prompt
            offered_tokens = {bare_all[k] for k in offered}
            letter_tokens = set(bare_all.values())
            top5 = [int(i) for i in np.argsort(probs)[::-1][:5]]
            restricted = offered[int(np.argmax(bare[:n]))]
            generated = model.generate_greedy([prompt], max_new_tokens=4)[0]
            first = model.decode(generated[:1]).strip()
            per_item.append({
                "template_id": template,
                "qtype": item.qtype.value,
                "options": n,
                "mass_offered_bare": float(bare[offered_idx].sum()),
                "mass_offered_space": float(space[offered_idx].sum()),
                "mass_letters_not_offered": float(bare[other_idx].sum()) if n < 26 else 0.0,
                "top1_is_offered_bare": int(np.argmax(probs)) in offered_tokens,
                "top5_has_unoffered_letter": any(t in letter_tokens and t not in offered_tokens for t in top5),
                "scoring_prompt_is_generation_prompt": same_text,
                "restricted_argmax_equals_greedy_first": restricted == first,
            })

    summary: dict[str, dict] = {}
    for template in sorted({r["template_id"] for r in per_item}):
        rows = [r for r in per_item if r["template_id"] == template]
        summary[template] = {
            "items": len(rows),
            "median_mass_offered_bare": float(np.median([r["mass_offered_bare"] for r in rows])),
            "median_mass_offered_space": float(np.median([r["mass_offered_space"] for r in rows])),
            "median_mass_letters_not_offered": float(np.median([r["mass_letters_not_offered"] for r in rows])),
            "share_top1_offered_bare": float(np.mean([r["top1_is_offered_bare"] for r in rows])),
            "share_top5_has_unoffered_letter": float(np.mean([r["top5_has_unoffered_letter"] for r in rows])),
            "share_scoring_prompt_is_generation_prompt": float(
                np.mean([r["scoring_prompt_is_generation_prompt"] for r in rows])),
            "share_restricted_argmax_equals_greedy": float(
                np.mean([r["restricted_argmax_equals_greedy_first"] for r in rows])),
        }
    result = {
        "kind": "arm_b_readout_probe",
        "label": "additional analysis of the D12 failures; no verdict is changed by this file",
        "checkpoint": str(args.checkpoint),
        "items_per_template": args.per_template,
        "items": len(per_item),
        "templates": summary,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"items": len(per_item), "templates": len(summary), "out": str(args.out)}), flush=True)


if __name__ == "__main__":
    main()
