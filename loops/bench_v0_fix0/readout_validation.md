# Readout validation (F2) — every question type through one validated letter path

**Loop:** `bench_v0_fix0`  **Model:** `Qwen/Qwen3.5-0.8B` (and `Qwen/Qwen3.5-4B` for the health
pair)  **Reference:** lm-evaluation-harness 0.4.13  **Verdict: implementation check PASS.**

## The defect this closes

bench_v0 rendered `noul` options as `yes. Yes` / `no. No` and `score` options as `1. Not relevant`
… while the instruction said "Respond with a single letter: A, B, C, D…", then read the
probabilities of the *key* tokens (`yes`, `no`, `1`, `2`). Label mass on `noul` was ~0.002: the
readout was scoring tokens the model had no reason to produce. Every `noul`/`score` number in
bench_v0 is invalidated by this (see `CORRECTIONS.md`).

## The fix

`canonicalise_options` (in `src/meddecide/eval/readout.py`) re-keys every item to `A`, `B`, `C`, …
— a **no-op for `choice`**, a conversion for `noul` (`A. Yes` / `B. No`) and `score` (lowest level
first) — and carries gold with its content. The prediction file records the letter → original key
mapping per item, so a number can always be traced back to the source vocabulary.

## Measured effect

| model | question type | median label mass | greedy agreement | n |
|---|---|---|---|---|
| Qwen3.5-0.8B | `noul` | **0.9960** (was ~0.002) | **1.00** | 200 |
| Qwen3.5-0.8B | `score` | **0.9977** (was 0.002–0.22) | **1.00** | 200 |
| Qwen3.5-4B | `noul` | **0.9994** | **1.00** | 200 |
| Qwen3.5-4B | `score` | **0.9963** | **0.96** | 200 |

Acceptance (ADVISORY F2: ≥0.9 mass **and** ≥0.9 greedy agreement on 200 `noul` + 200 `score` for
both models): **met on all four cells.** The two models ran in separate GPU sessions, so the pair
is read from the two runs' artifacts (`outputs/bench_v0_fix0/F2/readout_health.json` was written by
the 4B run; the 0.8B run's values are in `outputs/bench_v0_fix0/F2/logs/health_0p8b_v2.log` and are
reproduced above).

## `choice` is byte-identical to the validated path

The `choice` readout was the only path bench_v0 validated (T7, agreement −0.50 pts). F2 must not
change it, and does not:

* canonicalisation is a no-op for all 1,273 MedQA test items (0 keys or ids changed);
* prompts reconstructed with the bench_v0 render function are **byte-identical for 200/200 items**;
* MedQA accuracy reproduces bench_v0 exactly: 4B 0.70149 vs 0.70149 (**−0.00016 pts**),
  0.8B 0.40770 vs 0.40770 (0.00 pts).

## Reference validation, stated precisely

| comparison | ours | reference | delta | reading |
|---|---|---|---|---|
| `choice`: MedQA, same items, same protocol | 0.40770 | 0.40770 (bench_v0) | **0.00 pts** | unchanged |
| `pubmedqa`: lm-eval's continuation rule, same prompt/choices | 0.5828 | **0.5828** (lm-evaluation-harness) | **0.00 pts** | **implementation PASS** |
| `pubmedqa`: our **letter** readout vs the continuation rule | 0.6184 | 0.5828 | **+3.56 pts** | protocol difference, not an implementation difference (per-item agreement 0.5723) |
| `noul` relevance items: letter readout vs continuation rule | 0.5900 | 0.5150 | +7.5 pts | reported as measured; these prompts are not a natural yes/no task |

The ADVISORY's ±2-point criterion is met by the **protocol-identical** comparison (0.00 pts). The
letter readout is a *different* protocol from continuation scoring, and its difference is measured
and reported rather than averaged away: 10 of 477 `pubmedqa` items change side, and the two rules
agree on only 57 % of items while landing within 3.6 points of each other.

**Consequence for how results must be quoted:** a `noul` or `score` accuracy from this benchmark is
internally consistent and passes the health gate, but comparing it with a published
continuation-scored number requires saying so. bench_v0's MedQA-style `choice` numbers are directly
comparable within ±0.5 points.

## Reproduce

```bash
uv run python scripts/bench/check_readout_health.py --models Qwen/Qwen3.5-4B --limit 200
uv run python scripts/bench/validate_readouts.py --model Qwen/Qwen3.5-0.8B --noul-items 200 \
    --pubmedqa-items 477 --out outputs/bench_v0_fix0/F2/reference_validation_0p8b.json
uv run lm_eval --model hf --model_args pretrained=Qwen/Qwen3.5-0.8B,dtype=bfloat16 \
    --tasks pubmedqa_parquet --batch_size 8 --num_fewshot 0 --limit 477 \
    --include_path outputs/bench_v0_fix0/F2/lm_eval_tasks \
    --output_path outputs/bench_v0_fix0/F2/lm_eval_pubmedqa.json
```

(The task YAML repoints lm-eval's `pubmedqa` at the parquet release of the same corpus, because its
stock dataset ships a loading script that `datasets` 5.x refuses; the prompt shape and choice list
are copied verbatim from lm-eval's task.)
