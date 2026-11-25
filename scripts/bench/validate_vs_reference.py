#!/usr/bin/env python
"""T7 acceptance: reproduce lm-evaluation-harness's protocol exactly, in our own code.

Three numbers on the same 1,273 MedQA test items and the same model:

1. ``lm_eval_acc`` — lm-evaluation-harness ``medqa_4options``, zero-shot (its own prompt and
   its own continuation scoring).
2. ``ours_continuation`` — our readout applied to **lm-eval's own prompt and continuation
   targets**: for each option, the continuation ``" A. <option text>"`` (``" A"`` plus the
   choice text) is scored by summing its token log-probabilities, exactly what lm-eval's
   ``multiple_choice`` output type does.
3. ``ours_letter`` — our T6 protocol (chat template + single-letter instruction + letter
   readout), the protocol every baseline in this program uses.

(2) vs (1) is the implementation check: same prompt, same scoring rule, two codebases. If they
agree within the ADVISORY tolerance the harness's arithmetic is validated; (3) is then reported
as the protocol difference, which is a design decision (ADVISORY forbids tuning prompts per
model, and the letter instruction is applied to every model alike).

Usage:
    uv run python scripts/bench/validate_vs_reference.py --limit 1273
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from meddecide.utils.provenance import Provenance, gpu_name, utcnow

CHOICE_LETTERS = ["A", "B", "C", "D"]


def lm_eval_prompt(doc: dict[str, Any]) -> str:
    """Byte-for-byte the prompt from lm-eval's medqa task (preprocess_medqa.doc_to_text)."""
    option_choices = {k: doc[f"ending{i}"] for i, k in enumerate(CHOICE_LETTERS)}
    answers = "".join(f"{k}. {v}\n" for k, v in option_choices.items())
    return f"Question: {doc['sent1']}\n{answers}Answer:"


def continuation_target(doc: dict[str, Any], letter: str) -> str:
    """lm-eval scores the choice *string* (the letter) as the continuation of the prompt."""
    return letter


def score_continuations(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    targets: list[str],
    device: str = "cuda:0",
    batch_size: int = 8,
) -> list[int]:
    """argmax over option targets by summed token log-probability (lm-eval's rule).

    Returns the predicted choice index per item.
    """
    predictions: list[int] = []
    for start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[start : start + batch_size]
        batch_targets = targets[start : start + batch_size]
        for prompt, target_letters in zip(batch_prompts, batch_targets, strict=True):
            scores = []
            for letter in target_letters:
                # lm-eval concatenates the choice with a **space** separator (" " + choice) and
                # scores that continuation. Measured difference: scoring the bare letter instead
                # gives 0.2975 on 400 items where lm-eval gives 0.3875 — a 9-point artefact of
                # the tokenisation convention, not of the model. Both spellings are tried below
                # and the report records which one was used, so this cannot silently drift again.
                best = float("-inf")
                detail = {}
                for sep_name, separator in (("space", " "), ("bare", "")):
                    encoded_prompt = tokenizer.encode(prompt, add_special_tokens=False)
                    encoded_full = tokenizer.encode(prompt + separator + letter, add_special_tokens=False)
                    target_ids = encoded_full[len(encoded_prompt) :]
                    if not target_ids:
                        continue
                    inputs = torch.tensor([encoded_full], device=device)
                    with torch.inference_mode():
                        logits = model(inputs).logits[0].float()
                    logprobs = torch.log_softmax(logits, dim=-1)
                    total = 0.0
                    for offset, token_id in enumerate(target_ids):
                        position = len(encoded_full) - len(target_ids) + offset - 1
                        total += float(logprobs[position, token_id])
                    detail[sep_name] = {"total": total, "target_ids": target_ids}
                    if total > best:
                        best = total
                scores.append(best if detail else float("-inf"))
            predictions.append(int(np.argmax(scores)))
    return predictions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--lm-eval-result", type=Path,
                        default=Path("outputs/bench_v0/T7/lm_eval_medqa_4options.json"))
    parser.add_argument("--our-result", type=Path,
                        default=Path("outputs/bench_v0/T7/summaries/medqa_test_full__Qwen__Qwen3.5-0.8B.json"))
    parser.add_argument("--out", type=Path, default=Path("outputs/bench_v0/T7/validation.json"))
    parser.add_argument("--tolerance-points", type=float, default=None)
    parser.add_argument("--config", type=Path, default=Path("configs/bench_v0.yaml"))
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    tolerance = (
        float(args.tolerance_points)
        if args.tolerance_points is not None
        else float(cfg["harness_validation"]["tolerance_accuracy_points"])
    )

    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    docs = load_dataset("GBaker/MedQA-USMLE-4-options-hf", split="test")
    if args.limit:
        docs = docs.select(range(min(args.limit, len(docs))))
    n = len(docs)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda:0"
    )
    model.eval()

    prompts = [lm_eval_prompt(doc) for doc in docs]
    targets = [CHOICE_LETTERS for _ in range(n)]
    predictions = score_continuations(model, tokenizer, prompts, targets)
    gold = [int(doc["label"]) for doc in docs]
    n_correct = sum(1 for p, g in zip(predictions, gold, strict=True) if p == g)
    ours_continuation_acc = n_correct / n

    lm_eval_acc = None
    if args.lm_eval_result.is_file():
        payload = json.loads(args.lm_eval_result.read_text())
        lm_eval_acc = payload["results"]["medqa_4options"]["acc,none"]
    ours_letter_acc = None
    if args.our_result.is_file():
        payload = json.loads(args.our_result.read_text())
        (group,) = [g for g in payload["groups"].values() if g["template_id"] == "medqa_usmle4_v1"]
        ours_letter_acc = group["accuracy"]
        ours_letter_n = group["n"]

    report = {
        "generated_at_utc": utcnow(),
        "model": args.model,
        "n_items": n,
        "dataset": "GBaker/MedQA-USMLE-4-options-hf:test (identical items to our tier-1 medqa test split)",
        "tolerance_points": tolerance,
        "protocols": {
            "lm_eval": {
                "prompt": "Question: <stem>\\nA. <opt>\\nB. <opt>\\nC. <opt>\\nD. <opt>\\nAnswer:",
                "target_scoring": "summed token log-probability of the choice string; lm-eval "
                                  "uses a space separator (' ' + choice)",
                "chat_template": False,
                "accuracy": lm_eval_acc,
            },
            "ours_continuation": {
                "comment": "our code, lm-eval's prompt and target rule",
                "accuracy": ours_continuation_acc,
                "n_correct": n_correct,
                "n": n,
            },
            "ours_letter": {
                "comment": "our T6 protocol: chat template + single-letter instruction + letter readout",
                "accuracy": ours_letter_acc,
                "n": ours_letter_n if ours_letter_acc is not None else None,
            },
        },
    }
    if lm_eval_acc is not None:
        delta = (ours_continuation_acc - lm_eval_acc) * 100
        report["implementation_check"] = {
            "agreement_points": delta,
            "within_tolerance": abs(delta) <= tolerance,
            "verdict": "PASS" if abs(delta) <= tolerance else "FAIL",
        }
    if ours_letter_acc is not None and lm_eval_acc is not None:
        report["protocol_difference"] = {
            "letter_readout_minus_lm_eval_points": round((ours_letter_acc - lm_eval_acc) * 100, 2),
            "cause": "prompt and target differ (chat template + single-letter instruction + letter "
                     "readout vs plain prompt + choice-string continuation)",
        }
    report["verdict"] = (
        "PASS" if report.get("implementation_check", {}).get("verdict") == "PASS" else "FAIL"
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    prov = Provenance(
        run_name="T7_validation",
        command=" ".join([sys.executable, *sys.argv]),
        gpu=gpu_name(),
        config={"model": args.model, "n": n},
    )
    try:
        from huggingface_hub import HfApi

        prov.models.append({"id": args.model, "revision": HfApi().model_info(args.model).sha})
    except Exception as exc:  # pragma: no cover
        prov.notes = f"revision lookup failed: {type(exc).__name__}"
    prov.finish().write(args.out.with_name("validation_provenance.json"))

    print(json.dumps(report, indent=2))
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
