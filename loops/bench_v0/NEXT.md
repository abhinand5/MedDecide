# NEXT — proposals for loop 1 (`student_v0`)

**These are proposals, not decisions.** The operator + advisor review picks what loop 1 does. Each
item says what it would cost and what would invalidate it, and every one traces to a measurement
in this loop.

---

## 1. Fix the two templates that flatter the model (do this first, it is cheap)

| template | problem | measured | proposal |
|---|---|---|---|
| `pubmed_mesh_major_choice_v1` | the gold MeSH topic appears in the abstract the model reads | 0.980–0.991 accuracy vs 0.259 majority, every model (C047) | drop it, or rebuild the state so the topic string cannot appear (e.g. index-only state) |
| `medquad_routing_v1` | the gold label's stem is in the stated question for 63 % of items; a stem-matching rule alone gets 0.539 | 0.926–0.987 vs 0.255 majority (C047) | drop from the headline set, or replace with a routing task whose answer is not a word in the question |

Neither is a harness defect: both are template-design defects that the T8 screen did not catch
because BoW macro accuracy stayed ≤ 0.56. **Proposal: add "gold string appears in the state" as a
first-class screen check** — it is one line and would have caught the MeSH case before T9.

## 2. Run the tier-1 contamination probe (T4) before any tier-1 claim

T4 was not run; every tier-1 number currently carries an unmeasured contamination caveat. It needs
the harness (ready) and a few GPU hours for Min-K%-Prob on ≤500 items per (model, source). **Cost:
~3–5 h.** Without it, tier-1 numbers are for comparability only and must be labelled so.

## 3. Redesign MedMCQA's use in the benchmark

The official test has **no "D" answers** and Qwen3.5-4B/9B answer "D" ~30 % of the time, so the
template measures position for them (C048, C051). Options: (a) rebuild the template with seeded
option shuffling as the *fixed* order (the permutation becomes part of the item id); (b) drop
MedMCQA test and use the official validation split as test with shuffling; (c) report MedMCQA only
as a position-bias probe. **Proposal: (a)**, because it keeps the comparability and the shuffle
already exists and is deterministic.

## 4. Decide the teacher question with the operator (T10 is blocked, not answered)

The teacher gate never ran: no endpoint was provided. The program's recorded risk is that
DeepSeek-V4.1-Flash was chosen without a bake-off. **Proposal: bring the endpoint up for one
window and run the T10 protocol as written** (≥2,000 dev items non-thinking, temperature per qtype
fitted on half, ECE on the other half, ≥200 items thinking mode, throughput both). Until then,
loop 2's bulk labelling has no measured basis.

## 5. Loop 1's training data: use dev, keep test clean

Tier-1 dev is 8,103 items and fresh dev is 10,632 (excluding the two dropped templates). Loop 1
trains on **train/dev gold only** and fits per-question-type temperature on dev. **Proposal:**
start with tier-1 dev (larger, cleaner) and use fresh dev only for a held-out sanity check; do not
touch fresh test until the G1 comparison.

## 6. Baselines to add before the SoTA claim

The decision models (Laya, Laya-typed, Julia-1, GLiNER2.5-Decide, open-jev-deberta, JEV-9B) are
`NOT MEASURED`. The headline claim is "≤4B MedDecide beats 27B-class general decision models", so
loop 3 needs them and loop 1 needs at least JEV-9B and one Laya for the G1 comparison. **Cost:
~2 h per model, timeboxed.** Proposal: do this early in loop 1 rather than late.

## 7. Harness throughput numbers for planning (measured, C053)

Median per-item serve time recorded in the prediction files, one RTX PRO 6000, batch size and
token budget as logged, on the same 43,913 test items per model:

| model | median latency per item | items/s | median prompt tokens |
|---|---|---|---|
| LFM2.5-350M | 0.0016 s | 641 | 392 |
| Qwen3.5-0.8B-Base | 0.0081 s | 123 | 398 |
| Qwen3.5-0.8B (tier-1 prompts) | 0.0033 s | 303 | 119 |
| Qwen3.5-0.8B (fresh prompts) | 0.0096 s | 104 | 475 |
| MedGemma-1.5-4b-it | 0.0117 s | 85 | 379 |
| Qwen3.5-4B | 0.0253 s | 40 | 398 |
| Qwen3.5-9B | 0.0320 s | 31 | 398 |

The same model is 3× slower on fresh prompts than on tier-1 prompts (they are ~4× longer), so
throughput must be quoted per tier, not per model. **Proposal:** for loop 3's 27B baselines, plan
on the 4-GPU window, not one card — at 31 items/s for a 9B, a 27B on one card would make the
fresh tier a multi-day run.

## 8. Open decisions for the operator (unchanged from the loop's questions)

1. **Teacher endpoint** (blocks T10 and loop 2's data volume).
2. **MedQuAD licence** is `UNKNOWN` on its mirror; it is one of 10 tier-1 templates and can be
   dropped without touching the harness.
3. **Fresh window width**: 25 days because the teacher documents no cutoff. A documented cutoff
   would widen it and thicken the openFDA templates.
4. **The audit**: 150 items are waiting; the loop cannot validate its own templates.
5. **The fixed T6 prompt** is part of the baseline protocol; if the operator wants a
   continuation-scoring protocol for comparability with published numbers, that is a decision, and
   T7 has already measured what it costs (+2.18 points on MedQA).
