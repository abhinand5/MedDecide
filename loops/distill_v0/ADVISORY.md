# ADVISORY — distill_v0: two-student distillation from d1-ultra (autonomous)

Operator decisions in chat on 2026-11-04 (IST):

1. Train two models: `pplx-decider-flash-v1.2` (4B) and `clef-flash` (9B).
2. Target the best open model **under 10B overall** — a medical ≤10B Pareto frontier, not one metric.
3. Frozen test items are not used for training (the operator agreed).
4. No new teacher-only synthetic task templates; the existing MR/audit data is what we keep.
5. `d1-ultra` is the primary teacher; the local 27B is the backup if the API fails.
6. API budget: the balance is $23.44; the hard cap is **$15.00**; nothing above $50 in any case.

This file is the pre-registration. It is fixed before any training run; changes after the first dev result are recorded in STATE.md "Deviations" and are not allowed to change a verdict.

---

## 1. Teacher: Liquid `d1-ultra` through OpenRouter

### 1.1 Measured contract (2026-11-03/04, before the loop)

| property | measurement |
|---|---|
| endpoint | `POST https://openrouter.ai/api/alpha/decisions` |
| model id | `liquid/d1-ultra-20261015` (provider Liquid) |
| request | `{"model", "state": [str], "questions": {name: {type, instructions, criteria?}}}` |
| response | `answers.<q>.probabilities` over the offered criteria; `noul` gives `noul`; `score` gives a distribution + `score` mean + `legend` |
| option limit | 2–255 tested |
| state length | 23.7k tokens tested |
| rate | 23–32 calls/s at 24–32 threads; 64 threads is slower |
| price | **$0.042 per 1M input tokens** (measured over 1.4M tokens) |
| determinism | **not deterministic** (identical calls differ in the 4th decimal); `temperature`/`seed` accepted but do not fix it |
| option order | 2/30 argmax flips under reversal; mean max-prob shift 0.087; 3-permutation averaging gave no accuracy gain (0.767 → 0.767) |

Consequences for the design: one permutation per item; labels frozen with a SHA-256; no claim of byte-for-byte reproducibility; the model id is pinned, not the `~latest` alias.

### 1.2 Labeling protocol

- Basket built from `split == "train"` rows only. Before any call, every `item_id` and the state/question/option text hash is checked against every frozen set (`data/bench/osler_v4/y9/*.jsonl`, `data/bench/osler_mimic_v1/*.jsonl`, `data/bench/osler_v2/knowledge_dev_v2.jsonl`, `data/bench/osler_v4/fastdev_v4.jsonl`, `data/bench/osler_v4/cg_v2.jsonl`). A match is dropped with the reason `frozen_overlap`.
- Request construction: `choice` → criteria keyed by the item's option keys; `noul` → `type: noul`; `score` → criteria as the ordered level labels. The question text is the item's `question`; the state is the item's `state`.
- Validation per row: HTTP 200; the returned key set equals the offered key set; probabilities sum to 1 within 1e-3; no refusal; finite values. Up to 3 retries with 1.5 s backoff. A row that still fails is written as a failure with its reason and is not used for training.
- Freeze: one JSONL with `item_id, template_id, qtype, option_keys, teacher_probs, teacher, cost, input_tokens`; SHA-256 recorded in the manifest. The raw API responses are not kept (privacy by construction: no text is written back).
- Cost ledger: `usage.cost` summed per call; the labeler stops at **$15.00**, writes the partial manifest and exits 0 with `stop_reason: cap_reached`.
- Priority order so the cap buys the most valuable rows first: P0 knowledge/general (cheap), P1 PMC, P2 CT capped at 10k/template, P3 optional panel-like train splits and PubMedQA unlabelled.

### 1.3 Fallback

If the API is unavailable (5 consecutive failures over 10 minutes, or HTTP 401/402), the labeler stops and the loop may label the remaining T basket with the local `perplexity-ai/pplx-decider-v1.2-27b` at `cce0f3f80033e9d89d80831d5281fd72124aacce` through `scripts/osler_v4/y5_*` (the osler_v4 Y5 path; free, self-hosted). The fallback is recorded as a deviation, and the teacher id is written per row (`teacher` field), so the two label sets are never confused.

---

## 2. Basket

### 2.1 T — teacher-supervised (KL + gold CE)

Counts are the corrected-mix rows (`outputs/synth_v1/train/composition.json`, per-template, after the `fda_route_choice_v1` drop). Estimated cost is one permutation at $0.042/M tokens with the token estimates from the state lengths.

| template | rows | est. cost |
|---|---:|---:|
| `medmcqa_4opt_v1` | 42,370 | $0.11 |
| `medqa_usmle4_v1` | 9,084 | $0.11 |
| `mednli_v1` | 10,626 | $0.26 |
| `broad_aqua_rat_v1` | 72,469 | $0.28 |
| `snli_nli_choice_v1` | 61,680 | $0.20 |
| `winogrande_blank_choice_v1` | 40,398 | $0.14 |
| `hellaswag_ending_choice_v1` | 39,905 | $0.18 |
| `boolq_passage_noul_v1` | 9,427 | $0.09 |
| `csqa_choice_v1` | 10,765 | $0.03 |
| `qasc_science_choice_v1` | 9,064 | $0.02 |
| `arc_science_choice_v1` | 3,370 | $0.01 |
| `broad_social_iqa_v1` | 24,782 | $0.06 |
| `broad_cosmos_qa_v1` | 18,727 | $0.12 |
| `broad_strategyqa_v1` + `broad_quartz_v1` + `broad_math_qa_v1` + `broad_bigbench_*` | 4,973 | $0.02 |
| `pmc_similar_patient_v1` | 52,549 | $1.29 |
| `pubhealth_claim_true_noul_v1` | 8,300 | $0.65 |
| 9 `ct_*` templates, 10,000 each | 90,000 | $3.40 |
| **total** | **508,516** | **$7.0** |

Optional, only if the cap allows and time remains: MMLU-Pro health train, MedExpQA/MedExQA train, HeadQA (small), PubMedQA unlabelled (~61k rows, teacher-only labels, no gold). They are added to the manifest with their own counts and hashes.

### 2.2 G — gold-only (CE), no API calls

- MIMIC audit family: the 13 templates in `data/train/osler_v4/block_B1.jsonl` + `block_B2.jsonl` + `block_B3.jsonl`, 500,801 rows in the corrected mix. Patient splits are already enforced by the builders; held-out ICD families stay out.
- MR real-text family: `mr_*`, `codiesp_icd_support_v1`, `mtsamples_page_category_v1`, `synthea_*`, `ehr_rel_relatedness_v1` (~48k rows).
- The CT rows beyond the 10k/template teacher cap (~110k) and every other corrected-mix template not in T. They keep their gold CE; they are not teacher-labelled.
- The `pairs_test` set is evaluation-only; its training counterpart (`mimic_doc_support_v1`) is in the MIMIC family.

### 2.3 Dropped from training (187,623 rows, recorded with counts and reasons)

`fda_route_claim_noul_v1` (16,612), `fda_class_choice_v1` (13,452), `fda_class_choice_v2` (9,799), `fda_application_family_choice_v1` (18,628), `pubmed_pubtype_choice_v1` (42,370) and the `pubmed_check_*` family (86,762). These are literal metadata lookups with 14k–30k-character states; the decision model should not spend capacity on them (operator principle, 2026-11-04), and they were not teacher-labelled. They remain available as a sensitivity arm if the 4B dev curve shows a loss on CG v2 or fast dev; that is an additional arm, not a change to the gate.

### 2.4 Training pool

- **Pool = corrected mix (1,456,438) − 187,623 = 1,268,815 rows.** T 508,516 (40.1 %); MIMIC 500,801 (39.5 %); MR/real-text ~48k (3.8 %); CT remainder and other templates the rest. The pool composition, per-template counts, shares and SHA-256 go to `outputs/distill_v0/D2/composition.json`.
- Realised target: teacher rows 38–45 % of item slots; MIMIC 37–42 %; MR/real-text 3–6 %. The step log records the realised shares every 50 steps; the tripwire stops the run if teacher is outside 38–45 % or MIMIC outside 37–42 % over the first 1,000 steps.

---

## 3. Loss and dosing

- Per row: `ce_plus_brier` on gold (unchanged from osler_v4: CE + 1.0 × Brier).
- Teacher rows add `λ · KL(teacher ‖ model)` over the offered options, exactly `kl_to_reference` in `src/meddecide/train/losses.py`.
- Rows without `teacher_probs` contribute no KL.
- Batch composition: the existing seeded length-bucketed order over the pool; no forced 5+3. The per-batch teacher count is logged every step; the distribution over the first 1,000 steps is reported (mean, share with <3). Target mean 3.2 teacher items per batch of 8 (band 2.8–3.8); the band is a tripwire, not a sampler.
- **λ calibration (pre-registered):** on the first 200 steps, with λ = 1.0, measure the ratio of the KL gradient norm to the total gradient norm with a second backward pass on each of the 200 steps; set λ to the value whose predicted mean share is 0.40 (inside 0.30–0.50). The calibration table is written to `outputs/distill_v0/D4/lambda_calibration.json`. λ is frozen after that; no dev result may change it.
- No temperature on the teacher distribution; no label smoothing; no gold blending into the teacher.

---

## 4. Training

| setting | 4B | 9B |
|---|---|---|
| base | `perplexity-ai/pplx-decider-flash-v1.2` @ `6d771ad7…` | `Cloudflare/clef-flash` @ `fde727a2…` |
| backbone | `pplx` (`Qwen3_5Model` from the checkpoint) | Clef text tower via `Qwen3_5ForConditionalGeneration.model`, or `Qwen/Qwen3.5-9B` @ `c2022362…` if the port fails |
| LoRA | r=32, alpha=32, dropout=0, q/k/v/o | same |
| readout | option_code, rows from `lm_head` | same |
| steps | 63,304 | 63,304 |
| batch | 8, max 8,192 tokens (existing constraint); the realised teacher count per batch is logged | same, measured for memory |
| evals | 6,330, 12,661, 18,991, 25,322, 31,652, 37,982, 44,313, 50,643, 56,974, 63,304 | same |
| dev metric | fast dev v4 template macro, CG v2 micro, knowledge dev v2, MIMIC dev | same |
| checkpoints | full every 2,500, newest two | same |
| temperature | per q-type on dev, fitted at the end | same |

The 4B starts first. The 9B starts only after the 4B finishes (one GPU job at a time), and only if the 4B's step-13,992 dev curve does not show a collapse (MIMIC dev > 0.70, knowledge dev > 0.74, positional health passing). If it does, the 4B is stopped and the recipe is fixed first; the 9B is not started on a broken recipe.

---

## 5. Evaluation and Gate O7

- Sets: `knowledge`, `panel`, `mr_verdict`, `mimic_suite`, `mednli_test`, `pairs_test`, `mimic_heldout`, `mr_reported`, `mr_heldout_test` (frozen; the same files the osler_v4 gate used).
- Scoring: `scripts/osler_v5/z5_score.py`, calibrated per q-type with the dev-fitted temperatures; readout gate D12/D21 per cell.
- Statistics: item-paired bootstrap 95 % CI (1,000 resamples, within templates), exactly Gate O4's estimator (`meddecide.eval.gate_o4`).
- Checks: i panel vs pplx-flash lower > 0; ii knowledge vs pplx-flash lower > −0.02; iii MR vs osler_v4 final lower > −0.02; iv MIMIC vs osler_v4 final lower > −0.02. Gate O7 PASS iff i–iv for at least one student.
- Additional: per-set table for both students; the ≤10B Pareto table; pplx-27B ceiling; `pairs_test`, `mimic_heldout`, `mr_reported`, `mr_heldout_test`.
- No model is selected on a test set. There is no arm choice in this loop: both students are trained to the full budget and reported. The only stop rule is the step-13,992 dev tripwire in §4.

---

## 6. Claims and reporting

- Every number in FINDINGS.md has a CLAIMS.md row with an artifact and a recompute command.
- The panel and knowledge numbers are quoted with the paired interval, both point estimates, the item count and the template macro definition.
- Any cell failing the readout gate reads `READOUT_FAIL — <check>` and is not used in a verdict.
- The teacher's own ceiling on each set is reported as an evaluation row (D1) and is not a baseline in Gate O7.
- No MIMIC text, item text, prediction or teacher label is committed.

---

## 7. Deviations

Recorded in STATE.md as they happen. Pre-registered allowed deviations: the Clef port fallback (Qwen3.5-9B), the pplx-27B teacher fallback, and a smaller labelled basket if the $15 cap is reached. Nothing else changes the gate.
