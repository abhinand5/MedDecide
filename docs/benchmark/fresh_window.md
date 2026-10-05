# Fresh window (tier 2) — bench_v0

**Authored:** 2026-10-05, task T5.
**Why this file exists:** the fresh tier's validity rests on every item post-dating the
training data of every model in the ladder *and* the teacher. This file records, per model,
the date used and where it came from. If a date is not documented anywhere, the model's
Hugging Face repository creation date is used as a conservative upper bound (ADVISORY T5.1).

## Window

| | value |
|---|---|
| **window start (inclusive)** | `2026-09-10` |
| **window end (inclusive)** | build date, passed as `--window-end` (this build: `2026-10-05`) |
| **deciding model** | `deepseek-ai/DeepSeek-V4.1-Flash` (teacher) |

Window start = the **latest** of the dates below. Any record whose structured filter date
(first posted / first effective / Entrez date) is earlier than the start is excluded by
construction, and the build's acceptance check re-proves that zero records are older.

## Dates and their sources

| model | documented cutoff | source of the claim | date used | why |
|---|---|---|---|---|
| `LiquidAI/LFM2.5-350M` | **Mid-2024** | model card, "Knowledge cutoff: Mid-2024" | 2024-07-01 (midpoint of Mid-2024, used only as a lower bound) | documented; not the constraint |
| `Qwen/Qwen3.5-0.8B` | not documented | model card + Qwen3.5 blog citation (Feb 2026) contain no cutoff statement | **2026-02-28** (HF repo creation) | conservative upper bound |
| `Qwen/Qwen3.5-0.8B-Base` | not documented | model card contains no cutoff statement | **2026-02-28** (HF repo creation) | conservative upper bound |
| `Qwen/Qwen3.5-4B` | not documented | model card contains no cutoff statement | **2026-02-27** (HF repo creation) | conservative upper bound |
| `Qwen/Qwen3.5-9B` | not documented | model card contains no cutoff statement | **2026-02-27** (HF repo creation) | conservative upper bound |
| `google/medgemma-1.5-4b-it` | not documented | model card states "Model created: 4B multimodal: Jan 13, 2026"; no cutoff | **2026-01-07** (HF repo creation) | conservative upper bound |
| `deepseek-ai/DeepSeek-V4.1-Flash` (teacher) | not documented | model card contains no cutoff statement | **2026-09-10** (HF repo creation) | conservative upper bound; **the deciding date** |

Exact values read from the Hub on 2026-10-05 (repository creation timestamps, UTC):

```
LiquidAI/LFM2.5-350M           created 2026-03-31   sha 9e6c6ccf47
Qwen/Qwen3.5-0.8B              created 2026-02-28   sha 2fc0636471
Qwen/Qwen3.5-0.8B-Base         created 2026-02-28   sha dc7cdfe2ee
Qwen/Qwen3.5-4B                created 2026-02-27   sha 851bf6e806
Qwen/Qwen3.5-9B                created 2026-02-27   sha c202236235
google/medgemma-1.5-4b-it      created 2026-01-07   sha 91850547d9
deepseek-ai/DeepSeek-V4.1-Flash created 2026-09-10  sha 2cba9e42aa
```

The model cards themselves are cached at `outputs/bench_v0/T5/cards/` (gitignored).

## What this window implies, honestly

* It is **25 days wide** (2026-09-10 → 2026-10-05). That is narrow, and it is narrow because
  the teacher has no documented cutoff and its repository appeared on 2026-09-10. A shorter
  window means fewer fresh items per template; the build reports the counts it actually
  reached rather than padding them.
* The fallback is **conservative, not exact**: a repo creation date is an upper bound on the
  cutoff, so the true cutoff may be earlier and the window could legitimately be widened.
  Widening it is an operator decision, not an agent one — see STATE.md questions.
* Nothing in the window can have been seen by any ladder model *if* the creation-date bound
  holds. The contamination probe (T4) covers the established tier; the fresh tier's
  protection is this window plus the structured-field gold rule.

## Rebuilding for a later date

`scripts/bench/build_fresh.py --window-end YYYY-MM-DD` rebuilds the whole tier for a new end
date; the window start is read from `configs/bench_v0.yaml` (`fresh_window.start`). Two builds
with the same arguments produce identical files (checked by the acceptance check).
