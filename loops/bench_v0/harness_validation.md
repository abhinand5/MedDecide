# Harness validation against lm-evaluation-harness (T7)

**Task:** T7 — validate the eval harness against an independent reference implementation
**Date:** 2026-10-05  **Model:** `Qwen/Qwen3.5-0.8B`  **Task:** `medqa_4options` (zero-shot)
**Verdict: PASS** — agreement within the ±2.0 accuracy-point tolerance (measured gap **−0.50 pts**).

## What was compared

The same 400 MedQA test items, the same model, and the same scoring rule, run through two
independent codebases:

| | prompt | target scored | accuracy |
|---|---|---|---|
| **lm-evaluation-harness** (`medqa_4options`, zero-shot) | `Question: <stem>\nA. <opt>\n…\nAnswer:` | summed log-prob of `" " + choice` | **0.3875** (155/400) |
| **our harness, reference protocol** (`scripts/bench/validate_vs_reference.py`) | byte-identical prompt | summed log-prob of `" " + choice` | **0.3825** (153/400) |
| **our harness, T6 protocol** (letter readout, full test split) | chat template + "respond with a single letter" | option-letter token at the answer position | 0.4093 (521/1,273) |

**Agreement (implementation check): −0.50 points → PASS** (tolerance 2.0 points,
`configs/bench_v0.yaml:harness_validation.tolerance_accuracy_points`).

The item sets are the same items: our tier-1 `medqa` test split (from
`GBaker/MedQA-USMLE-4-options:test`, 1,273 items) and lm-eval's dataset
(`GBaker/MedQA-USMLE-4-options-hf:test`, 1,273 items) have an **identical 1,273-item
intersection** — 0 items unique to either side (checked by question text + option-label set).

## The protocol difference, reported separately

On the full test split our T6 protocol scores **0.4093** versus lm-eval's **0.3723**, i.e.
**+2.18 points** — *outside* the ±2-point tolerance, and expected to be: the protocols differ in
two documented ways.

1. **Prompt.** Ours renders the model's chat template with a system message and the instruction
   "Respond with a single letter: A, B, C, D, … The correct option is:"; lm-eval uses a plain
   `Question: … Answer:` completion prompt with no chat template. Measured effect of lm-eval's own
   chat-template mode on the same task: **0.3723 → 0.2734** (−9.9 points), which is what an
   instruction-tuned model does when handed a bare continuation prompt it was not tuned for.
2. **Target.** Ours reads the single option-letter token; lm-eval scores the continuation
   `" A"`. Measured effect in our own code on the same 400 items: scoring the bare letter `"A"`
   gives **0.2975** where the space-prefixed `" A"` gives **0.3825** — a 9-point artefact of the
   tokenisation convention alone. That was a real bug in the first version of this validation
   script (it scored the bare letter), found because the two implementations disagreed.

Because ADVISORY forbids tuning prompts per model, the T6 protocol is applied unchanged to every
model in the ladder; the number to quote for comparability with published lm-eval results is the
reference-protocol number, not the T6 number.

## Commands

```bash
# reference implementation
uv run lm_eval --model hf --model_args pretrained=Qwen/Qwen3.5-0.8B,dtype=bfloat16 \
    --tasks medqa_4options --batch_size 8 --num_fewshot 0 --limit 400 \
    --output_path outputs/bench_v0/T7/lm_eval_plain_400.json
# ours, same prompt and scoring rule
uv run python scripts/bench/validate_vs_reference.py --limit 400 \
    --lm-eval-result outputs/bench_v0/T7/lm_eval_plain_400_2026-10-05T22-01-29.922345.json
# ours, T6 protocol, full split
uv run python scripts/bench/run_eval.py --model Qwen/Qwen3.5-0.8B \
    --items data/bench/tier1/medqa.jsonl --splits test --task T7 --tag medqa_test_full
```

Raw artifacts: `outputs/bench_v0/T7/{validation.json,lm_eval_plain_400_*.json,summaries/}` (gitignored).
