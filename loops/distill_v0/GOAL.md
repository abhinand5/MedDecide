# distill_v0 — goal (harness objective)

|  |  |
|---|---|
| **Loop** | `distill_v0` (distillation from Liquid `d1-ultra`; two students, one label set) |
| **Branch** | `loop/distill_v0` — never commit to `main` or `dev` |
| **Base** | `loop/osler_v5` at `23a0362` |
| **Artifacts** | `loops/distill_v0/` (GOAL.md, ADVISORY.md, STATE.md, CLAIMS.md, FINDINGS.md) |
| **Run state** | `loops/distill_v0/STATE.md` |
| **Claims** | `loops/distill_v0/CLAIMS.md` |
| **Outputs** | `outputs/distill_v0/<task-id>/` (gitignored) |
| **Started** | `2026-11-03T19:30:00Z` |
| **Style** | autonomous; hard stop after the Gate O7 report |

> **For the harness.** Treat this file as the loop objective. Read `AGENTS.md`, then this file, then `ADVISORY.md` and `STATE.md`. Take the first `PENDING` task in `STATE.md` whose dependencies are `DONE`, mark it `IN_PROGRESS` with a UTC timestamp before working, and run its acceptance check before marking it `DONE`. Where `STATE.md` and memory disagree, `STATE.md` wins. Stop at §8 and do not start another loop.

---

## 1. Mission

Train two Osler decision models by distilling the decision distribution of Liquid's `d1-ultra` teacher:

- **Osler-4B** on `perplexity-ai/pplx-decider-flash-v1.2`, with the existing architecture;
- **Osler-9B** on `Cloudflare/clef-flash`, with our option-code readout on its text backbone.

Both students are measured on the frozen MedDecide sets and placed on the measured **≤ 10B medical Pareto frontier**. The headline claim: at least one student beats zero-shot `pplx-decider-flash-v1.2` on the external clinical panel (paired lower bound > 0) and holds knowledge (paired lower bound > −0.02), with the MIMIC audit suite and the MR verdict suite within −0.02 of the osler_v4 final.

The loop is judged by Gate O7 (§6). It ends with a FINDINGS.md and a hard stop.

---

## 2. Starting point (measured before the loop; samples, not gate results)

| set | d1-ultra | osler_v4 final | pplx-flash | pplx-27B | Clef-Flash |
|---|---:|---:|---:|---:|---:|
| external panel | **0.750** | 0.597 | 0.639 | 0.720 | 0.650 |
| knowledge | **0.792** | 0.702 | 0.710 | 0.817 | 0.711 |
| MedNLI | **0.905** | 0.876 | 0.872 | 0.875 | 0.864 |
| MIMIC held-out (110) | **0.794** | 0.638 | 0.720 | 0.772 | — |
| MR verdict | 0.707 | **0.843** | 0.704 | 0.775 | 0.683 |
| MIMIC suite | 0.683 | **0.800** | 0.634 | 0.682 | 0.639 |
| MIMIC doc pairs | 0.917 | **0.966** | 0.874 | 0.940 | — |

Teacher sample sizes: panel and MR, 60 items per template (n=420); knowledge, 60 per template (n=120); MedNLI, n=200; MIMIC suite, 40 per template (n=360); held-out, n=110; doc pairs, n=60. The full-set teacher numbers are task D1.

**Reading.** The teacher leads on the panel (+15 points), knowledge (+9), MedNLI (+3) and the held-out slice (+16); it trails the tuned student on the MIMIC suite (−12), the MR verdict (−14) and doc pairs (−5). Therefore distillation is applied to general and clinical-reasoning rows only. The audit family, the MR family and the rule-lookup templates stay gold-only. This is data curation by measurement, not by assumption.

---

## 3. Design

### 3.1 Students

| student | backbone | revision | head |
|---|---|---|---|
| Osler-4B | `perplexity-ai/pplx-decider-flash-v1.2` | `6d771ad7b5ecff5e93b8fc7b90d4c5ccc341954d` | option-code readout + attention-only LoRA (existing path) |
| Osler-9B | `Cloudflare/clef-flash` | `fde727a287004204b7518dcc983fe64379776712` | our option-code readout on the text backbone (not Clef's joint head) |

- The 9B port is task D3: load `Qwen3_5ForConditionalGeneration` from the Clef snapshot, take `.model` as the text backbone, initialise the readout rows from `lm_head`, and prove a forward pass and a 20-item smoke. **Timebox: 3 hours.** If the port is not clean, fall back to `Qwen/Qwen3.5-9B` at revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a` and record the fallback as a deviation.
- Both students: frozen backbone, LoRA r=32 / alpha=32 / dropout=0 on `q_proj,k_proj,v_proj,o_proj`, option-code readout, `lambda_brier: 1.0`, per-item loss weighting, option augmentation as in `configs/osler_v5/arm_F2.yaml`, seed 0, 63,304 steps, ten evaluations at the osler_v4 Y8 grid, positional health at every evaluation, full checkpoints every 2,500 steps (keep the newest two), temperature fitted per q-type at the end, watchdog as in `scripts/synth/v1_watchdog_d.sh`.
- Training order: 4B first, then 9B. One GPU job at a time.

### 3.2 Teacher

- Endpoint: `https://openrouter.ai/api/alpha/decisions`. Model: `liquid/d1-ultra-20261015` (pinned; the `~latest` alias may move).
- Key: `OPENROUTER_API_KEY`, sourced from `/workspace/.secrets.env`. Never written to any file.
- Output: the per-question distribution over the offered criteria, exactly our readout format. Multi-question per state is supported but not needed.
- **Cost control: hard cap of $15.00.** The labeler reads `usage.cost` on every response, keeps a ledger, and stops cleanly at the cap. It never truncates silently: whatever is labelled is listed in the manifest.
- One permutation per item. Probability sums, key sets and refusals are validated on every row; up to 3 retries; failures are counted and reported.
- The teacher is non-deterministic. Labels are frozen once to a JSONL file whose SHA-256 is recorded; re-running the API is not expected to reproduce them byte-for-byte.
- Fallback if the API fails or the cap is reached early: the local `perplexity-ai/pplx-decider-v1.2-27b` at `cce0f3f80033e9d89d80831d5281fd72124aacce` through the osler_v4 Y5 scoring path (free, self-hosted). Any use of the fallback is a deviation recorded in STATE.md.
- The backward-compatibility rule of `GOAL.md` §5 applies: **no frozen test item is ever teacher-labelled for training.**

### 3.3 Data buckets

**T — teacher-supervised (KL + gold CE), ~508k rows, one permutation, ~$7–8:**

| family | templates (rows) |
|---|---|
| knowledge | `medmcqa_4opt_v1` (42,370), `medqa_usmle4_v1` (9,084), `mednli_v1` (10,626) |
| general reasoning / replay | `broad_aqua_rat_v1` (72,469), `snli_nli_choice_v1` (61,680), `winogrande_blank_choice_v1` (40,398), `hellaswag_ending_choice_v1` (39,905), `boolq_passage_noul_v1` (9,427), `csqa_choice_v1` (10,765), `qasc_science_choice_v1` (9,064), `arc_science_choice_v1` (3,370), `broad_social_iqa_v1` (24,782), `broad_cosmos_qa_v1` (18,727), `broad_strategyqa_v1` (1,179), `broad_quartz_v1` (1,987), `broad_math_qa_v1` (521), `broad_bigbench_*` (1,286) |
| clinical real text | `pmc_similar_patient_v1` (52,549), `pubhealth_claim_true_noul_v1` (8,300) |
| clinical trials | 9 CT templates capped at 10,000 each (90,000) |

Optional, only if the cap allows: MMLU-Pro health train, MedExpQA/MedExQA train, HeadQA, and the PubMedQA unlabelled split (teacher-only labels, no gold). They enter only as labelled T rows and are recorded in the manifest.

**G — gold-only (CE), 760,299 rows:** every corrected-mix row not in T. The MIMIC audit family (500,801 rows, 13 templates), the MR real-text family (`mr_*`, `codiesp_*`, `mtsamples_*`, `synthea_*`, `ehr_rel_*`), the CT rows beyond the 10k/template cap (~110k) and the remaining templates. No teacher labels; no API calls; MIMIC content stays local.

**Dropped from training in v0 (187,623 rows, counted and recorded):** the literal field lookups — `fda_route_claim_noul_v1` (16,612), `fda_class_choice_v1` (13,452), `fda_class_choice_v2` (9,799), `fda_application_family_choice_v1` (18,628), `pubmed_pubtype_choice_v1` (42,370) and the `pubmed_check_*` family (86,762). Rationale: these are metadata lookups; the decision model should not spend capacity on them (operator principle, 2026-11-04). They were not teacher-labelled either (budget), so nothing is lost by dropping them. They remain available as a sensitivity arm if the dev curves show a loss.

**Hard rule:** the T basket is built only from `split == "train"` rows whose item ids do not occur in any frozen set (`data/bench/osler_v4/y9/*.jsonl`, `data/bench/osler_mimic_v1/*.jsonl`). This is checked by content hash before any label call.

### 3.4 Loss and dosing

- Loss on every row: `CE + λ · KL(teacher ‖ model)`; rows without a teacher distribution contribute CE only.
- The pool is ~40 % T / ~40 % MIMIC / ~20 % other G by design. Batching is the existing seeded, length-bucketed order; the realised number of teacher-labelled items per batch is logged every step and its distribution is reported. Target: mean 3.2 teacher items per batch of 8 (band 2.8–3.8).
- **λ calibration:** over the first 200 steps, λ is set so the distillation term's share of the applied gradient is 0.30–0.50; λ is then frozen for the rest of the run. The share is measured with a second backward pass on the KL term and logged (method and numbers in ADVISORY §5).
- No per-batch minimum is imposed: the D57 per-batch rule was a draft for a different loop and is not in force here. The realised dose is the pre-registered object.
- The teacher distribution is used as-is (no temperature rescaling) and is never blended with gold; gold stays in the CE term.
- **No test-set fitting of any kind.**

### 3.5 Training pool composition

The pool is the corrected mix (D56a, 1,456,438 rows) minus the rule-lookup templates (FDA and the PubMed publication-type/check family, 187,623 rows): **1,268,815 rows**. The T basket is 508,516 of them (40.1 %); the MIMIC family is 500,801 (39.5 %); the MR/real-text family ~48k (3.8 %); the remaining CT rows and other templates are gold-only. The pool composition (counts, shares, SHA-256, exclusions with reasons) is written to `outputs/distill_v0/D2/composition.json` and the hash is checked by the driver before any step. The realised teacher share of item slots must be 38–45 % and the MIMIC share 37–42 %; both are checked from the step log and reported. If either falls outside, the run is stopped by the tripwire and the sampler is fixed before relaunch (no silent change).

### 3.6 Evaluation and Gate O7 (pre-registered, before any run)

Frozen sets, never trained on, never teacher-labelled for training: knowledge (5,456), panel (4,353), MR verdict (13,103), MIMIC suite (4,253), MedNLI (1,422), `pairs_test` (500), `mimic_heldout` (110), `mr_reported` (6,191), `mr_heldout_test` (584).

Both students are scored calibrated (per-q-type temperature fitted on dev only) with the D12/D21 readout gate.

**Checks (paired bootstrap 95 % CI, as Gate O4):**

| id | check | limit |
|---|---|---|
| i | panel accuracy, student − zero-shot `pplx-decider-flash-v1.2` | lower bound > 0 |
| ii | knowledge accuracy, student − zero-shot `pplx-decider-flash-v1.2` | lower bound > −0.02 |
| iii | MR verdict accuracy, student − osler_v4 final | lower bound > −0.02 |
| iv | MIMIC suite accuracy, student − osler_v4 final | lower bound > −0.02 |

**Gate O7 PASS** iff checks i–iv hold for **at least one** student. A second student that misses a check is reported as a row, not a verdict.

**Pareto frontier (claim, not a gate).** Both students enter the ≤10B table with pplx-flash, Clef-Flash, MedDecider-4B/9B, JEV-9B and the osler_v4 final, on every set. A student is "on the frontier" if no measured ≤10B model is ≥ it on all sets and > it on at least one. pplx-27B is the ceiling reference, outside the class.

### 3.7 Budget and time

| item | estimate |
|---|---|
| API labels | $7–8 core, $15 hard cap |
| teacher ceiling (D1) | ~$1.5 |
| GitHub/OpenRouter balance | $23.44 at start (measured 2026-11-03) |
| 4B run | ~14 h GPU |
| 9B run | ~25 h GPU |
| wall clock | ~2.5 days |

---

## 4. Task board (detail and checklists in STATE.md)

| id | task | GPU | deps | acceptance |
|---|---|---|---|---|
| D0 | pre-registration: GOAL, ADVISORY, STATE, CLAIMS; AGENTS pointer | no | — | files committed; AGENTS current-loop updated |
| D1 | teacher ceiling on all frozen sets; labeler + ledger + freeze schema | no | D0 | 9 sets scored; cost, model id, date, token counts recorded; parity test |
| D2 | basket manifest (counts, hashes, exclusions) + label run + QA + freeze | no | D1 | every T row labelled or dropped with a reason; SHA-256; ledger ≤ cap; no test item |
| D3 | Clef-Flash port check (3 h timebox) + 20-item smoke | yes | D0 | forward + readout init works, or fallback recorded |
| D4 | distillation driver, batch logging, λ calibration, 4B 63,304-step run | yes | D2 | run completes; λ log; realised dose; positional health; dev curve |
| D5 | 9B 63,304-step run on the same labels and recipe | yes | D3, D4 | same as D4 |
| D6 | Gate O7 on both; scoreboard; SELF_AUDIT | yes | D4, D5 | checks i–iv; frontier table; audit |
| D7 | FINDINGS.md; closure; hard stop | no | D6 | aggregates only; claims rows; commit + push |

---

## 5. Hard rules

1. **No frozen test item is used for training.** No teacher label is attached to a test item. A separate "oracle" diagnostic is out of scope.
2. Every reported number has a `CLAIMS.md` row with a recompute command (R1); unmeasured is reported as NOT MEASURED (R3).
3. No silent drops: every filter returns counts and reasons (R5).
4. Raw outputs are immutable (R9); derived artifacts are generated by scripts.
5. Never commit MIMIC text, item text, predictions, teacher labels or secrets. Aggregates only.
6. One GPU job at a time; long jobs under `nohup`/`setsid` with logs in `outputs/distill_v0/<task>/logs/`.
7. Secrets come from the environment; the key is never written to a file.
8. Never edit a check, threshold or gold label to pass it (R8). That is BLOCKED.
9. The API hard cap is $15.00. If it is reached before the basket is complete, stop labelling, record what is labelled, and continue with it; do not raise the cap.
10. Do not delete files outside `outputs/distill_v0/` and `/workspace/tmp`; do not touch pods, volumes, HF repos or remote services.

---

## 6. Outcomes

- **Success:** Gate O7 PASS for at least one student, and both students placed on the measured ≤10B frontier. Headline: the panel gap is closed and knowledge is held.
- **Partial:** one student passes i–ii but misses iii or iv (audit/MR regression); or the panel wins without the knowledge guard. Reported with the per-set table; no spin.
- **Failure:** neither student passes; or the arm cannot be trained (port/API/step failure). Report the evidence; do not relabel.
- **Unmeasured:** any train that cannot run is NOT MEASURED with the reason.

---

## 7. What this loop cannot claim

- That distillation is the only cause of any gain: there is no no-distill control on the identical pool for the 9B; the 4B comparison is against the osler_v4 final (different mix) and is labelled additional. Any causal sentence is a hypothesis (R10).
- That the teacher's labels are medical ground truth: they are `d1-ultra`'s judgments.
- That the ≤10B frontier is complete: only the models we measured are in it.

---

## 8. Hard stop

After D7, write `loops/distill_v0/FINDINGS.md` (aggregates only), append the CLAIMS rows, update STATE.md, commit, push, and **stop**. Do not start another loop, do not plan the next one, do not train a third model.

---

## 9. Out of scope

- Osler-0.8B; further heads; RL; new benchmark items; release/upload of weights; the memorisation/PHI audit (a release gate, not a training gate).
- New synthetic task templates or teacher-only audit questions (operator decision 2026-11-04: the existing MR/audit data is what we keep).
- Test-set training of any kind.
- Reopening settled decisions: base ladder (D22/D39), architecture (D23/D26), readout-health gate (D12), release policy.
