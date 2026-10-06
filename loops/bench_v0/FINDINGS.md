# FINDINGS — MedDecide loop `bench_v0`

**Loop:** bench_v0 (loop 0 of the MedDecide program)
**Branch:** `loop/bench_v0`  **Closed:** 2026-10-06
**Status:** complete for what was measurable; two tasks blocked on inputs only the operator can
supply (T10 teacher endpoint, T11 human audit).

---

## Summary

**The question.** Before any model is trained, is there a trustworthy way to measure a medical
decision model — a benchmark whose answers cannot have been memorised, and a harness whose numbers
can be believed? This loop built that foundation and stopped for review.

**What we did.** Built MedDecide-Bench v0 with two tiers: **21,202 items** from eight established
public test sets (tier 1) and **41,502 items** in 12 templates built from records first posted
between **2026-09-10 and 2026-10-05** (tier 2, the "fresh" tier), where every label is a lookup of
a structured source field — no language model ever produced a gold label. Wrote an eval harness
that reads option probabilities from one prefill pass, validated it against
lm-evaluation-harness, screened the templates against regex and bag-of-words baselines, and ran
six zero-shot baselines over the model ladder: **264,478 predictions**.

**What we found.**

1. **The harness agrees with an independent implementation.** On the same 400 MedQA items, with
   the same prompt and the same scoring rule, our code and lm-evaluation-harness differ by
   **0.5 accuracy points** (ours 0.3825, lm-eval 0.3875; tolerance ±2.0) — C041. The harness is
   therefore fit to report numbers.

2. **The ladder scales on medical knowledge, and only there.** MedQA: 0.290 (350M) → 0.377
   (0.8B-Base) → 0.415 (0.8B) → 0.480 (MedGemma-4b) → **0.701 (4B)** → **0.754 (9B)**, majority
   baseline 0.277. MMLU medical: 0.241 → 0.860. That is the bar MedDecide must clear — C050.

3. **Relevance judging is at chance for every model we ran.** NFCorpus and SciFact `noul`
   templates sit at 0.500–0.641 against 0.500 majorities; the best model is the 4B at 0.632 on
   SciFact — C050.

4. **Two templates look good for the wrong reason, and both were caught by measurement rather
   than argument.** `pubmed_mesh_major_choice_v1` scores 0.98–0.99 because the MeSH topic appears
   in the abstract the model is reading (leakage); `medquad_routing_v1` scores 0.93–0.99 and its
   gold label's stem is present in the stated question for 63 % of items (keyword-heavy). Neither
   may be quoted as skill — C047.

5. **MedMCQA's collapse to 0.12 for the 4B/9B models is a positional-bias artefact, and shuffling
   proved it.** MedMCQA's official test contains **no "D" answers at all**; the 4B model answers
   "D" 31 % of the time. A seeded option shuffle leaves the "D" preference intact (29.8 % on a
   balanced gold), so the model has a position bias and that template measures position, not
   medicine, for those models — C048, C051.

**What it means.** Loop 1 can start: the harness is validated, the fresh tier is built and
reproducible, and the bar is measured. But three things must be fixed first: the MeSH and MedQuAD
templates need redesign (or exclusion), and every baseline number must be read beside its majority
baseline because two templates are dominated by one class.

**What did not work.**

- **T10 (teacher gate) is `BLOCKED`**: no `TEACHER_BASE_URL` was ever provided, so the teacher's
  calibration is unmeasured — the loop's teacher decision is deferred, not answered.
- **T11 (operator audit) is `BLOCKED — awaiting operator`**: the page and the 150-item sample are
  built and tested; the human review has not happened, so no item has been human-validated.
- **Decision-model baselines (Laya, Julia-1, GLiNER2.5-Decide, JEV-9B, open-jev) are
  `NOT MEASURED`**: each needs its own inference path (2 h timebox each) and the session time went
  to the ladder instead.
- **The fresh window is only 25 days wide**, because the teacher's repository date (2026-09-10)
  is the only defensible bound; the consequence is a thin openFDA tier (122 labels) and one
  template with a single gold class.

---

## Body

### 1. The benchmark

| | tier 1 (established) | tier 2 (fresh) |
|---|---|---|
| items | 21,202 | 41,502 |
| sources | MedQA, MedMCQA, PubMedQA, MMLU (6 medical subsets), MedQuAD, TREC-COVID, NFCorpus, SciFact | ClinicalTrials.gov, openFDA drug labels, PubMed update files |
| templates | 10 | 12 |
| test / dev | 13,099 / 8,103 | 30,870 / 10,632 |
| gold | the source's own answer field | a structured field (phase, allocation, publication type, check tag, MeSH topic, …) |

Integrity, re-derived from the raw JSONL in a fresh process: **0 split leaks, 0 duplicate item
ids, 0 question texts on both sides of the dev/test boundary, all file hashes matching the
manifest** (C014, C021). The fresh build is byte-identical when re-run with the same arguments
(C022). Three real defects were found and fixed on the way: item ids that collided across splits,
MedQuAD `question_id`s that are not unique (9,662 repeats), and content repeated across official
splits in MMLU/NFCorpus/MedQuAD.

The fresh window and the reason it is narrow: `docs/benchmark/fresh_window.md`. Only LFM2.5-350M
documents a cutoff ("Mid-2024"); Qwen3.5, MedGemma and DeepSeek-V4.1-Flash document none, so each
falls back to its repository creation date, and the teacher's (2026-09-10) bounds the window.

### 2. The harness

Protocol, fixed for every model and recorded in every run config: the model's own chat template
with thinking disabled, a system prompt plus "respond with a single letter … the correct option
is:", and the next-token distribution read over the option-letter tokens in one prefill pass. No
text is generated at eval time; no prompt is tuned per model.

Four defects were found by cross-checking, not by assuming — each would have silently poisoned
every downstream number:

| defect | symptom | detector |
|---|---|---|
| bare `Answer:` prompt | model continued the clinical vignette; 79 % mass on "A", 4/20 correct | greedy generation on the same prompt |
| left-padding index | read the logits *of* the final token, not the next-token distribution | batched run vs single-item run |
| wrong letter variant (`" A"` vs `"A"`) | label mass ~1e-06; argmax rarely an option; ECE 0.31 | readout diagnostics; fixed to 0.996 mass, ECE 0.226 |
| batch planner overflow | a 239-sequence batch despite `--batch-size 16`; every OOM in the ladder runs | per-batch debug line; fixed and regression-tested |

Validation: **C041** (0.5-point agreement with lm-evaluation-harness) plus the observation that the
T6 protocol sits +2.18 points above lm-eval's plain prompt on the full MedQA split, with both
causes measured separately (prompt: −9.9 points when lm-eval uses a chat template; target: −8.5
points for the bare-letter spelling). `loops/bench_v0/harness_validation.md`.

### 3. The baselines (264,478 predictions, six models)

Generated by `scripts/bench/report_baselines.py` from the raw prediction files; the committed
table is `loops/bench_v0/baselines.md` and the machine-readable form is
`outputs/bench_v0/T9/results.json` (C052). Highlights:

| template | best model | accuracy | majority | note |
|---|---|---|---|---|
| `medqa_usmle4_v1` | Qwen3.5-9B | 0.754 | 0.277 | clean ladder scaling |
| `mmlu_mc_v1` | Qwen3.5-9B | 0.860 | 0.325 | clean ladder scaling |
| `pubmedqa_ynm_v1` | Qwen3.5-4B | 0.736 | 0.551 | 3-way yes/no/maybe |
| `scifact_relevant_noul_v1` | Qwen3.5-4B | 0.632 | 0.500 | relevance is hard for all models |
| `nfcorpus_relevant_noul_v1` | several | 0.500–0.567 | 0.501 | at chance |
| `medquad_routing_v1` | Qwen3.5-9B | 0.987 | 0.255 | keyword-heavy (C047) |
| `pubmed_mesh_major_choice_v1` | all | 0.980–0.991 | 0.259 | leakage (C047) |
| `medmcqa_4opt_v1` | LFM2.5-350M | 0.368 | 0.383 | position bias for 4B/9B (C051) |

Two dropped fresh templates: `ct_healthy_volunteers_noul_v1` (all 2,900 test items share one gold
class) and `pubmed_observational_noul_v1` (BoW baseline 0.985 ≥ 0.95) — C033. The screen also
showed that **no** bag-of-words baseline exceeds 0.564 macro accuracy, so the micro numbers in
the table are not evidence of template difficulty (C034).

### 4. What the loop did not measure, and why that is reported rather than hidden

- **Teacher calibration: `BLOCKED — teacher endpoint not provided`.** The gate's question ("is the
  teacher's ECE ≤ 0.05 after temperature fitting?") is unanswered. No number was invented for it.
- **Human audit: `BLOCKED — awaiting operator`.** 150 fresh test items with the structured field
  behind each gold, and a self-contained audit page whose export logic is tested end-to-end
  (C038–C040).
- **Decision-model baselines: `NOT MEASURED — integration not started`.** The ladder came first.
- **Shuffle flip rate and abstention** are measured for one model on 30 items (T6 probes) plus the
  MedMCQA shuffle for the 4B; all other cells read `NOT MEASURED`.
- **Tier-1 contamination probe (T4)** was not run: it depends on the harness (now ready) but the
  session time went to the ladder. Tier-1 numbers therefore carry an unmeasured contamination
  caveat.

### 5. Five-claim spot-check (re-run in fresh processes at closure)

| claim | value re-derived | command | matches |
|---|---|---|---|
| C041 harness vs lm-eval | ours 0.3825, lm-eval 0.3875, gap −0.50 pts, PASS | `jq '{ours: .protocols.ours_continuation.accuracy, lm: .protocols.lm_eval.accuracy, gap: .implementation_check.agreement_points}' outputs/bench_v0/T7/validation.json` | yes |
| C014 tier-1 integrity | 6/6 checks true, 21,202 rows recounted | `jq '.checks, .totals.n_items_recounted' data/bench/tier1/audit.json` | yes |
| C020 fresh freshness | 0 items before window start, 41,502 recounted | `jq '.checks.zero_items_before_window_start, .totals.n_items_recounted' data/bench/fresh/audit.json` | yes |
| C052 ladder rows | 264,478 rows, 132 groups, 6 models | `jq '{n_prediction_rows, n_groups}' outputs/bench_v0/T9/results.json` | yes |
| C051 MedMCQA position bias | source order acc 0.1217 / D-share 0.3139; shuffled acc 0.1524 / D-share 0.2977 | recomputed from the two prediction files | yes |

---

## Pointers

- `loops/bench_v0/CLAIMS.md` — 52 rows; every number above has one.
- `loops/bench_v0/baselines.md`, `template_screen.md`, `harness_validation.md` — committed reports.
- `loops/bench_v0/NEXT.md` — proposals for loop 1 (proposals, not decisions).
- `outputs/bench_v0/<task>/SELF_AUDIT.md` — per-task audits, including the defects found.
- `docs/benchmark/{schema.md,fresh_window.md}` — the item contract and the window provenance.
