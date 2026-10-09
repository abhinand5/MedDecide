# Fresh window (tier 2) — bench_v0 and v0.1

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


---

# v0.1 window (loop `bench_v0_fix0`, decision D11)

**Supersedes the v0 section above for all new work.** v0's window (2026-09-10 → build date) was
bounded by the *teacher's* repository date, which left openFDA's templates with 16–94 test items.
Decision **D11** changes the rule: the window is bounded by the models whose scores this program
reports as its own or as its starting points — the **ladder** — not by the teacher.

| | value |
|---|---|
| **v0.1 window start** | `2026-03-01` |
| **v0.1 window end** | build date (`2026-10-05` for the F4 build) |
| **strict slice start** | `2026-09-10` (the teacher's repo date) |
| **deciding model for the start** | `Qwen/Qwen3.5-0.8B` and friends — latest ladder repo date `2026-02-28`, so the window starts the next day |
| **rationale** | the teacher never labels benchmark items, so its training data cannot leak gold into them; later baselines (4B/9B, decision models) seeing in-window records can only *advantage them*, which is conservative for us |

## The strict slice

Every v0.1 fresh item carries `meta.strict_post_teacher` — `true` when the item's own filter date
is on/after **2026-09-10**. That date is after the ladder *and* after the 4B/9B and decision-model
baselines, so the strict slice is the subset no baseline could have seen. It is reported separately
(as bench_v0 did for the whole tier) and comprises **2,896 of 23,582 items** in the F4 build:
ClinicalTrials.gov 955, openFDA 30, PubMed 1,911.

The main v0.1 analysis uses the full 2026-03-01 window; the strict slice exists so a reviewer can
ask "and what does the number look like on records nobody could have seen?" without rebuilding.

## What changed structurally in v0.1

* **Class balancing per (template, split)** — up to `K` items per gold class, with `K` recorded per
  group. In the F4 build, templates with a dominant class are capped: `pubmed_observational_noul_v1`
  K=103 (206 test items, was 98.5 % one class), `pubmed_pubtype_choice_v1` K=19, `fda_route_choice_v1`
  K=7, `ct_intervention_type_choice_v1` K=9. Micro accuracy can now be read against a majority
  baseline that is not ~1.0.
* **Single-class splits are dropped with a reason** (`single_gold_class_in_split`) rather than
  exported; the F4 build dropped none.
* **Templates whose smallest class is below 200 test items are flagged** (`below_min_class_size`) so
  the F3 screen can apply its own rule. In F4 that is `ct_intervention_type_choice_v1`,
  `ct_primary_purpose_choice_v1`, `fda_route_choice_v1`, `pubmed_observational_noul_v1`,
  `pubmed_pubtype_choice_v1` (test) and several dev groups.
* **openFDA is no longer thin**: 7,297 items across three templates (was 222 across the 25-day
  window).

## Reproduce

```bash
uv run python scripts/bench/build_fresh.py --config configs/bench_v0_1.yaml \
    --window-start 2026-03-01 --window-end 2026-10-05 --pubmed-files 60 \
    --out data/bench/v0.1/fresh --manifest data/bench/v0.1/fresh/manifest.json
```
