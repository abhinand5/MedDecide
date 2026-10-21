# ADVISORY — osler_v4: MIMIC at scale, Nemotron-crafted minimal pairs, the panel lever, a longer final

Planned 2026-10-21 by the advisor (Claude Opus) with the operator, after the osler_v3 review. Read in full at the start of
every iteration. New decisions: D45–D50 in `docs/plans/PROGRAM.md` §3 (D26–D44 still hold).

## 1. Where things stand

osler_v2's Osler-4B (pplx-flash base, kept mix C1, 42,200 steps) is the model to beat:

| check (Gate O3) | result |
|---|---|
| MR suite v1 vs MedDecider-4B | +0.263 [+0.251, +0.276] (passes) |
| MR suite v1 vs zero-shot pplx-flash | +0.158 [+0.142, +0.173] (passes) |
| External panel vs MedDecider-4B | −0.004 [−0.020, +0.011] (passes) |
| **External panel vs zero-shot pplx-flash** | **−0.020 [−0.034, −0.006] (fails: lower bound −0.034, limit −0.02)** |
| Knowledge vs zero-shot pplx-flash | +0.002 on paired items |

What osler_v3 learned (`loops/osler_v3/FINDINGS.md`):
- **No Stage A recipe beat the control on the clinical-general (CG) dev proxy**, and the CG proxy did not separate the runs
  (all within 0.003). The control was chosen. Additional test rows (not used for any choice) suggest KL 0.1 and broad replay
  narrow the panel gap at no MR cost, but their intervals reach zero.
- **No Stage B block was kept**, because the keep rule demands an MR gain. B1 (RAGTruth + DrugProt) gained +0.151 on its own
  templates with MR −0.004 [−0.014, +0.006]. The rule had no path for a new skill (D47 fixes this).
- **Run length matters.** Every 13,992-step run is 0.018–0.025 below osler_v2's 42,200-step model on the MR verdict, the
  control included. osler_v2's final also selected its **last** checkpoint, so it had not plateaued.

What is new for this loop:
- **MIMIC is here** (official PhysioNet files only, D28, D37):
  - MIMIC-IV-Note 2.2 (discharge and radiology notes) is complete;
  - MIMIC-IV 3.1 `hosp` core is on disk: admissions, patients, diagnoses and procedures ICD with dictionaries, DRG codes,
    HCPCS, eMAR and prescriptions; labevents is in progress;
  - MIMIC-IV-ED 2.2 and MedNLI 1.0.0 are downloading now; the remaining MIMIC-IV tables and MIMIC-III 1.4 follow (about two
    days).
  - The download runs from `/workspace/download-manager/` (`./dm status`). **Never stop, restart or edit it**: that is
    the operator's.
- **The local teacher is now Nemotron 3.5 Super** (`nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4`, OpenMDW-1.1,
  which places no restriction on outputs; D45). Gemma 4 31B is retired.
  - It serves from `envs/vllm/` (vLLM 0.31.0, torch 2.13 cu130), the version vLLM's recipe requires.
  - The recipe lists one RTX Pro 6000 at a 16,384-token context with MTP speculative decoding (5 tokens).
  - The weights (75 GB) were downloaded by the advisor into `HF_HOME`.

## 2. The questions this loop answers

1. **How much does real MIMIC record-audit data add**: coding, DRG, clinical validation, medication and imaging audit, with
   labels from structured fields? (Stage B, B1.)
2. **Do teacher-edited minimal pairs on real notes teach "is this code supported by the documentation"** beyond what the
   coded labels teach? (B2.)
3. **Does soft-label distillation from pplx's 27B decision model on general items close the panel gap**, where KL-to-base
   did not show on dev? (Stage A, A5; needs the operator's yes on D48.)
4. **Does a longer final run (1.5× osler_v2's) keep improving?** (Y8, with the CG trend at every evaluation.)
5. **Does the final Osler-4B pass Gate O4** (D49), including the new MIMIC audit suite, and is it the best open model of its
   size on every scope?

Outcomes (write the ones that apply into FINDINGS):
- **(1) Gate O4 passes:** the recipe goes to the 9B next.
- **(2) MIMIC helps MIMIC but not the MR suite v1:** report both; the MIMIC suite is the closer proxy for real audit work.
- **(3) No Stage A arm beats the control on CG v2:** the panel gap is not visible on dev. Report the panel per dataset
  (additional), and propose nothing that uses the panel for a choice.
- **(4) A block is not kept:** report its dev numbers; it is dropped, not tuned.
- **(5) The final run's CG falls as MR rises:** report the trend at each evaluation; the selection rule stands.

## 3. Ground rules specific to this loop

1. **Everything from osler_v3 §3 still holds:**
   - base pplx-flash (D39); attention-only LoRA (D33); the D40 speed recipe;
   - no unofficial credentialed data (D37); labels by code or structured fields;
   - selection on dev only; held-out lists unchanged.
2. **KL-anchor domain prefixes (D41)**, fixed for every run of this loop:
   ```
   mr_, synthea_, chart_, codiesp_, mtsamples_, mimic_, ehr_rel_, pmc_, ragtruth_, drugprot_, multicare_, clinclaim_, restyle_, mednli_, mdace_
   ```
   Every MIMIC template id starts with `mimic_`; MedNLI's with `mednli_`.
3. **MIMIC handling (D28, D37, D46):**
   - **Local only.** MIMIC text, values and identifiers go only to local models. The teacher client must **refuse** a
     request that carries MIMIC content unless its base URL is loopback (127.0.0.1 or localhost). Y0 adds this check and its
     test. `OPENAI_*` in the secrets file point to a **hosted** endpoint; never use them for MIMIC work.
   - **Splits by patient:** `mimic.subject_split(subject_id, "mimic4")`. Every item from an admission, note or edited note
     inherits its patient's split.
   - **Test rows come only from late-anchor patients** (`mimic.late_anchor`, anchor_year_group ≥ 2014), so a MIMIC-IV test
     patient cannot also appear in MIMIC-III training.
   - **MIMIC-III training excludes every admission that MDACE annotates**, because MDACE is evaluation-only.
   - **Nothing MIMIC-derived is committed**: no text, values, ids, item counts per patient or quoted snippets. Commit
     aggregate counts only.
   - Copyrighted descriptors stay out:
     - no CPT descriptors (`d_hcpcs` long descriptions for CPT codes are AMA text; HCPCS Level II codes only, or numbers
       without descriptors);
     - no 3M APR-DRG descriptions (APR severity and mortality **levels** only);
     - CMS MS-DRG titles and ICD-10-CM/PCS titles are public-domain US government text and are allowed.
4. **Held-out stays held out:**
   - the ICD family `heart_failure` (`generators.HELD_OUT_ICD_FAMILIES`): every MIMIC item whose target code is in it goes
     to a held-out test file only;
   - the indicator `aki_creatinine_rise`: no MIMIC clinical-validation item may target acute kidney injury (N17.x) in
     training; AKI items go to the held-out file only.
5. **Teacher-crafted items (D43, D45):**
   - only over real text;
   - the label is the code's choice;
   - an independent verification call (different prompt, temperature 0, reasoning off) must agree;
   - string rules must hold.

   Never keep an item that fails verification, and never relabel one.
6. **Soft labels (D48, only if the operator confirmed it in STATE §2):**
   - pplx's 27B option distributions may be used only as a distillation target on **general anchor items** (never a
     domain template);
   - never on an item from any test split, the knowledge dev v2, the CG partitions, the external panel or held-out files;
   - always alongside the gold cross-entropy, never instead of it.
7. **The MR suite v1 stays frozen.** The MIMIC audit suite (Y4) is new and is frozen before any training run of this loop.
8. **Long notes:**
   - the prompt cap is 8,192 tokens (pplx tokenizer);
   - longer notes are cut by **section priority** (`mimic.sections`, Y1): keep chief complaint, history of present illness,
     brief hospital course, procedures, discharge diagnoses, discharge medications and pertinent results, in that order;
     then truncate from the end with a marker.
   - Count truncated items per template.

## 4. Tasks

### Y0 — Orientation, Nemotron serving, the MIMIC guard (~3 h, small GPU)

1. Read this ADVISORY, PROGRAM D41–D50, and osler_v3's FINDINGS and NEXT. Run ruff and the suite, deselecting
   `test_padding_trained.py::test_trained_checkpoint_logits_are_padding_invariant` and `test_overfit_64_items` as osler_v3 did.
   Record the counts.
2. **Serve Nemotron** from `envs/vllm/`, as `scripts/osler_v4/serve_nemotron.sh`:
   ```
   vllm serve nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4 --served-model-name nemotron \
     --host 127.0.0.1 --port 8001 --tensor-parallel-size 1 --max-model-len 16384 \
     --trust-remote-code --language-model-only --reasoning-parser nemotron_v3 \
     --speculative-config.method mtp --speculative-config.num_speculative_tokens 5 \
     --gpu-memory-utilization 0.95
   ```
   - The flags are from vLLM's recipe for this model; `--language-model-only` drops the vision encoder, which we do not
     use.
   - If it fails to start, adjust in this order and record each attempt: lower `--gpu-memory-utilization`; add
     `--kv-cache-dtype fp8`; drop MTP; lower `--max-model-len` to 12,288.
   - Run a test with three short prompts, then stop the server.
   - **Timebox: 3 hours.** If Nemotron cannot be served, mark Y3 `BLOCKED — Nemotron does not serve on this pod`, write the
     error into Questions, and continue. Do not fall back to Gemma without the operator.
3. **Throughput:** 200 requests of a 3,000-token note with a 600-token output, concurrency 32, reasoning off. Record output
   tokens/s, requests/h and peak memory.
4. **MIMIC guard:** in `src/meddecide/teacher/client.py`, a request flagged `contains_mimic=True` raises unless the base
   URL host is loopback. Add a unit test. Record that `OPENAI_*` is not used anywhere in this loop's MIMIC code paths
   (grep).
5. Record the files present under `data/physionet/` from `./dm status` (read only).

### Y1 — MIMIC foundation (CPU, ~4 h)

`src/meddecide/mr/mimic.py` (extend), `scripts/osler_v4/y1_mimic_base.py`:

1. Read discharge and radiology notes and the `hosp` tables with pandas or pyarrow. Write private parquet under
   `data/mimic_build/` (gitignored); check that `.gitignore` covers it.
2. **Section parser** `mimic.sections(text) -> dict[str, str]` for MIMIC-IV discharge notes: headers such as "Chief
   Complaint:", "History of Present Illness:", "Brief Hospital Course:", "Discharge Diagnosis:", "Discharge Medications:".
   Unit tests on synthetic strings only, never MIMIC text.
3. **Admission table:**
   - hadm_id → subject_id, split, late_anchor, ICD version, coded diagnoses (ordered by seq_num), procedures, MS-DRG and APR
     levels, discharge note id and its token count;
   - **ICD-10 admissions** (icd_version 10) are the coding substrate; ICD-9 admissions are recorded and unused.
4. Report counts in `loops/osler_v4/mimic.md`: patients and admissions per split, notes per split, token-length
   percentiles, ICD-10 share, DRG type counts. Aggregates only.

### Y2 — MIMIC structured-gold templates (CPU, ~6 h)

Every label comes from a structured field, through code. For each template, write the counts and every drop reason.

Volume per template, before mixing:
- up to 40,000 train items;
- 500 dev items (patients of the dev split);
- 500 test items (late-anchor patients of the test split).

Shared rules:
- Hard negatives everywhere: a negative must be clinically adjacent to the positive (same ICD category, same DRG family,
  same lab panel, same drug class). Record how each negative was drawn.
- Options are shuffled. Code titles come from `d_icd_diagnoses` and `d_icd_procedures`; DRG titles from `drgcodes`
  (HCFA type only).

| template id | type | state | question → gold (structured source) |
|---|---|---|---|
| `mimic_icd_support_v1` | choice (as `mr_icd_support_template_v1`, setting inpatient) | discharge note | the family's coded code, or "none" (`mimic.family_label`; the existing V5c design, 1:1 negatives, half hard: the condition word appears in the note) |
| `mimic_icd_specificity_v1` | choice, 3–6 options | discharge note | which billable code in this 3-character category was assigned (`diagnoses_icd`). Options: the category's billable siblings. Drop if more than one is coded in the category |
| `mimic_principal_dx_v1` | choice, 4–6 options | discharge note, with the Discharge Diagnosis section removed | which of the admission's coded diagnoses was sequenced first (seq_num 1). Options: the principal plus 3–5 of its secondary codes |
| `mimic_msdrg_v1` | choice, 4–6 options | discharge note | the admission's MS-DRG (HCFA). Options: its severity siblings (same base title with MCC / with CC / without) plus DRGs of the same MDC that occur in the data |
| `mimic_drg_cc_level_v1` | choice: MCC / CC / neither | discharge note | the severity split parsed by code from the MS-DRG title ("W MCC", "W CC", "W CC/MCC", "W/O CC/MCC", …). Only DRGs whose title carries a split marker |
| `mimic_apr_severity_v1` | score 1–4 | discharge note | APR-DRG severity level (number only) |
| `mimic_apr_mortality_v1` | score 1–4 | discharge note | APR-DRG risk-of-mortality level (number only) |
| `mimic_pcs_support_v1` | noul | discharge note | was this ICD-10-PCS code assigned? Positives: coded PCS codes; negatives: PCS codes of the same body system that are coded in other admissions with the same principal-diagnosis category |
| `mimic_lab_validation_v1` | noul | a lab page rendered by code from this admission's `labevents` (real values, `d_labitems` names, the reference range when present) | does the admission's lab evidence meet the code's criterion for the condition? Conditions and criteria are fixed in code (below). Both coded and uncoded conditions; balanced |
| `mimic_med_given_v1` | noul | discharge note | was this drug administered during the stay (an eMAR "Administered" event)? Negatives: drugs **ordered** in `prescriptions` for this admission but never administered, then drugs given in other admissions with the same principal category |
| `mimic_rad_exam_v1` | choice, 4–6 options | radiology report with its exam-name header masked by code | which exam was performed (`radiology_detail` exam_name). Options: exams of the same modality |

**Clinical-validation criteria** (`mimic_lab_validation_v1`; any value in the admission window):
- hyponatremia: sodium < 135 mmol/L;
- hypokalemia: potassium < 3.5;
- hyperkalemia: potassium > 5.0;
- anemia: hemoglobin < 13 g/dL (male) or < 12 (female);
- thrombocytopenia: platelets < 150 K/µL;
- leukocytosis: WBC > 11 K/µL;
- hypoglycemia: glucose < 70 mg/dL;
- hypomagnesemia: magnesium < 1.7 mg/dL;
- lactic acidosis: lactate > 2.0 mmol/L.

Unit checks drop rows whose `valueuom` does not match the criterion's unit (count them). AKI is excluded (§3.4).

Then:
- **Screens** (D35) on each template's dev items:
  - the O3 Naive Bayes screen (≥ 0.90 is a shortcut and keeps the template out of the MIMIC audit suite verdict);
  - majority rate.

  Run zero-shot Qwen3.5-4B for the ceiling screen in Y4.
- **Leakage:** check by note id and text hash against every test set, and between the splits.
- Write `loops/osler_v4/mimic.md` §2. It holds aggregates only.

**Y2b — the late MIMIC sources** run when each file set exists (re-checked at Y4 and before B3's decision). Never wait for them.
- **MIMIC-IV-ED:** `mimic_ed_acuity_v1` (score 1–5, triage `acuity`), from the triage chief complaint, vitals and pain
  score rendered by code. `mimic_ed_disposition_v1` (choice: admitted / home / other, from `edstays.disposition`), from the
  same state plus the ED diagnoses list.
- **MedNLI:** `mednli_v1` (choice: entailment / contradiction / neutral), from its train split only; its dev split is dev.
  Its test split is a test row in the MIMIC suite (additional, not in the verdict).
- **MIMIC-III:** `mimic_page_category_v1` (V5c step 1, `iter_mimic3_pages`). Exclude MDACE admissions (§3.3).

### Y3 — Nemotron-crafted minimal pairs over MIMIC notes (GPU, ≤ 10 h build)

**`mimic_doc_support_v1`** (noul: "Is ICD-10-CM <code> <title> supported by the documentation in this note?").

1. **Code chooses** an ICD-10 admission of the **train or dev split** and one of its coded **secondary** diagnoses.
   - The diagnosis is not principal and not held out.
   - It comes from a fixed list of chapters where documentation drives the code: E, D, F, G, J, K, N, R and I (minus I50).
2. **The original note** (sections as in §3.8) with this code is a **positive** item, provided verification (i) below
   confirms it is documented.
3. **Nemotron edits** the note: remove every statement and every piece of evidence specific to that condition, including
   treatments given only for it; change nothing else; leave no gap marker. Reasoning is off by default (pilot below).
4. **The edited note** with the same code is a **negative** item. The pair shares a split.
5. **Verification** (all must pass):
   - (i) an independent call (different system prompt, temperature 0, reasoning off) answers "Does this note document
     <condition>?". It must say **yes** for the original and **no** for the edit. If the original reads no, drop the pair:
     the code was not documented in this text.
   - (ii) string rules: no keyword from the code's title, nor from its code-built synonym list (title words minus stop
     words, plus the curated ICD table's condition words for that family), remains in the edit;
   - (iii) the edit keeps ≥ 70 % of the original's tokens, and ≤ 5 % of its tokens are new: no rewrite, no invented
     facts;
   - (iv) the edit keeps every other coded diagnosis that the original documents: recheck 2 random other codes with call
     (i) and require yes.
6. **Cue check:**
   - A Naive Bayes classifier on note text alone (no question), trained to tell edited notes from originals on train, must
     score AUC < 0.75 on dev.
   - If it scores higher, find the tell (word lists or formatting) and fix the edit prompt once.
   - If it is still ≥ 0.75, the template is BLOCKED.
7. **Pilot:** 200 pairs with reasoning off and 200 with reasoning on (medium effort). Record keep rate, agreement per
   check, pairs per GPU-hour, and the cue AUC.
   - **Pre-registered:** use the mode with the higher kept pairs per GPU-hour among modes with keep rate ≥ 40 %.
   - Spot-read 20 kept and 20 rejected pairs. Never paste them into committed files.
8. **Build** up to 30,000 pairs (60,000 items) within 10 GPU-hours, plus 250 dev pairs (dev patients).
   - Test pairs come from late-anchor test patients: 250 pairs, built the same way. They are an **additional** test
     template, not in the verdict, because the teacher touched them.
9. Write `loops/osler_v4/crafted.md`.

### Y4 — Fast dev v4, CG v2, the MIMIC audit suite, baselines (GPU, ~5 h)

1. **Fast dev v4** = osler_v3's fast dev v3, plus up to 400 dev items per new template (Y2, Y2b if built, Y3).
   - Fix it before any Stage A run. Record its partitions in the manifest. The MIMIC templates form a **MIMIC partition**.
2. **CG v2** = osler_v3's CG partition plus general-clinical dev items from **non-panel** sources with verified licences.
   - Candidates: DDXPlus validation (CC BY), HeadQA dev (if its licence checks out), MedNLI dev (if present), PubMedQA
     labelled dev.
   - Run the near-duplicate check (13-gram) against all seven panel datasets and drop any hit.
   - It must hold ≥ 6,000 items. Fix it now. The external panel never enters.
3. **The MIMIC audit suite v1** (D49) = the late-anchor test items of the Y2 templates (and Y2b if built by then).
   - It **excludes** templates that fail D35's screens on their dev items: ceiling ≥ 0.97 for zero-shot Qwen3.5-4B, or a
     Naive Bayes screen ≥ 0.90.
   - Excluded templates are reported, never in the verdict.
   - **Freeze it** (manifest with ids, hashes and counts per template under `data/bench/osler_mimic_v1/`; only the
     manifest's aggregate is committed) before any Stage A run.
4. **Baselines on the MIMIC suite and fast dev v4's MIMIC partition:** zero-shot pplx-flash (authors' code),
   MedDecider-4B, osler_v2's Osler-4B. Also MedDecider-9B as an additional row. All local.

### Y5 — pplx 27B soft labels (GPU, ≤ 8 h; only if D48 is confirmed in STATE §2)

1. Identify the newest **Apache-2.0** pplx decision model of about 27B on the Hub (`perplexity-ai/pplx-decider-v1.1-27b`
   or later). Pin the revision and record its licence.
2. Score it through its authors' code or our `PplxDecisionModel` port. Show argmax parity on 40 items, as D39 did for the
   flash model.
   - **Timebox: 3 hours.** If it fails, Y5 is BLOCKED and A5 is NOT MEASURED.
3. Score the **anchor pool**:
   - all general (non-domain-prefix) train items of the C1 mix plus osler_v3's broad replay;
   - excluding the knowledge dev v2 items, every CG item and every item whose text hash matches any test or panel item.
4. Save the full option distributions (private, `outputs/osler_v4/Y5/`).
5. Also score CG v2 and the knowledge dev v2. These are dev diagnostics only, recorded as the teacher's own accuracy.

### Y6 — Stage A: recipe (GPU, 2 new runs, ~5 h each)

Each run: pplx base, osler_v2's C1 mix, 13,992 steps, the D40 recipe, the same seed as osler_v3, and evaluations on fast
dev v4.

| run | recipe | source |
|---|---|---|
| A0 | control | **osler_v3's A0 checkpoint, re-scored** on fast dev v4 (no retraining) |
| A1, A3 | KL 0.1; broad replay | osler_v3's checkpoints, re-scored |
| A4 | `kl_to_base: 0.1` **and** osler_v3's broad replay (A3's mix) | new run |
| A5 | broad replay + **distillation** `distill_weight: 0.5` × KL(27B ‖ model) on anchor items that have Y5 soft labels, with `kl_to_base: 0` | new run; only if Y5 is DONE |

**Choice rule** (pre-registered; paired bootstrap within templates, 1,000 resamples, seed 0):
- A run **qualifies** if both lower bounds against A0 hold:
  - its MR partition (osler_v3's, without the MIMIC partition) > −0.01;
  - its knowledge dev > −0.01.
- **Choose the qualifying run with the highest CG v2 point estimate** (its CG v2 accuracy minus zero-shot pplx's).
- A difference under 0.003 from A0 is a tie, and A0 wins it.

A2 (KL 0.5) is not re-scored: it adds nothing A1 does not test. Write `loops/osler_v4/stage_a.md` before Stage B. Implementing `distill_weight` (loss, config, test) is part of this task
if D48 is confirmed. Reuse `losses.kl_to_reference` with the stored distributions as the reference.

### Y7 — Stage B: data blocks under keep rule v2 (GPU, ~5 h each)

The chosen recipe, 13,992 steps. Each block adds one source to the **last kept** block (D34).
- Caps: no template above 8 % of a block; replay ≥ 20 %.
- **MIMIC templates as a whole ≤ 35 %** of a block.

**Keep rule v2 (D47)**, against the last kept block (paired bootstrap within templates; the new templates are removed
from the MR, general and knowledge partitions as in D-X8-1):

| check | lower bound must be |
|---|---|
| **own:** the new templates' fast-dev items | > +0.02 |
| MR partition (osler_v3's) | > −0.015 |
| general | > −0.01 |
| knowledge | > −0.02 |
| CG v2 | > −0.01 |

| block | adds |
|---|---|
| B0 | Stage A's chosen run (no new run) |
| B1 | Y2 MIMIC structured-gold templates |
| B2 | Y3 `mimic_doc_support_v1` pairs |
| B3 | Y2b late sources (ED, MedNLI, MIMIC-III) built before B2's decision; otherwise NOT MEASURED |
| B4 | osler_v3's X1 real sources (RAGTruth, DrugProt), re-judged under rule v2 |

Write `loops/osler_v4/blocks.md` with every check's point and interval, and the MIMIC partition (labelled) for every
block.

### Y8 — Final Osler-4B (GPU, ~20 h)

- The kept mix, on the chosen recipe.
- **Step budget 63,304** (the smallest multiple of 8 at or above 1.5 × 42,200), with **10 evaluations** on fast dev v4.
- Selection by the pooled template macro, as before. Record the CG v2 and MIMIC partitions at every evaluation
  (labelled; not in the selection).
- Temperatures fitted on dev and saved with the checkpoint.
- Refuse to run twice (the `arm_result.json` guard, as osler_v3 X9).

### Y9 — Evaluation and Gate O4 (GPU, ~6 h)

Score the final model, calibrated (`load_any`), on:
- the MR suite v1 (verdict and reported-only);
- the MIMIC audit suite v1;
- the held-out files (including the MIMIC held-out items);
- the external panel;
- the knowledge test;
- the additional test templates (Y3 pairs, Y2b if built).

**Apply Gate O4 (D49) exactly.**

**Size-class scoreboard** (additional rows, labelled): Osler-4B against every open model of 10B or fewer parameters that we
can run locally, on every scope:
- zero-shot pplx-flash;
- MedDecider-4B and -9B;
- JEV-9B (osler_v0's harness);
- osler_v2's Osler-4B.

Add pplx 27B as a ceiling reference. Report the panel per dataset.

Write `gate_o4.md` and `results.md`.

### Y10 — Memorisation and PHI audit (small GPU, ~2 h; required because MIMIC is in the final mix)

1. **Membership inference:** the final model's per-item loss on MIMIC train items against test items of the same
   templates (2,000 each). Report the AUC; 0.5 means no measurable memorisation.
2. **Verbatim probe:** export the model to a plain `lm_head` (D23 export path). Take 300 train-split discharge-note prefixes
   of 128 tokens, generate 64 tokens greedily, and count continuations matching the true note on ≥ 50 consecutive tokens.
   Do the same with the zero-shot base as the control.
3. Record counts only, in `loops/osler_v4/audit.md`. This is a report for the release decision, not a gate.

### Y11 — Findings and closure — HARD STOP

FINDINGS:
- Summary first;
- which §2 outcomes apply;
- the Stage A choice, the blocks, and the final's CG and MIMIC trends;
- the size-class scoreboard;
- the five-claim spot-check.

Then NEXT.md, `Loop status: STOPPED`, commit, push, stop.

## 5. Task table (seed STATE.md with this)

| id | task | GPU | deps |
|---|---|---|---|
| Y0 | Orientation, Nemotron serving, MIMIC guard | small | — |
| Y1 | MIMIC foundation (tables, sections, splits) | no | Y0 |
| Y2 | MIMIC structured-gold templates (+ Y2b late sources when present) | no | Y1 |
| Y3 | Nemotron-crafted minimal pairs (`mimic_doc_support_v1`) | yes | Y0, Y1 |
| Y4 | Fast dev v4, CG v2, MIMIC audit suite v1, baselines | yes | Y2, Y3 |
| Y5 | pplx 27B soft labels (only if D48 confirmed) | yes | Y0 |
| Y6 | Stage A: A4, A5 + re-scored A0/A1/A3 + choice | yes | Y4, Y5 |
| Y7 | Stage B: blocks B1–B4 under keep rule v2 | yes | Y6 |
| Y8 | Final Osler-4B (63,304 steps) | yes | Y7 |
| Y9 | Evaluation, Gate O4, size-class scoreboard | yes | Y8 |
| Y10 | Memorisation and PHI audit | small | Y8 |
| Y11 | Findings and closure — HARD STOP | no | all |

Order:
- **GPU sequence:** Y0 → Y3 (teacher) → Y5 (27B) → Y4 → Y6 → Y7 → Y8 → Y9 → Y10.
- Y1 and Y2 are CPU tasks and run while the teacher works (Y3 needs only Y1).
- If D48 is not confirmed, Y5 is `BLOCKED — D48 not confirmed`, A5 is NOT MEASURED, and Y6 runs A4 only.
- If Y3 is blocked, Y4 proceeds without its template and B2 is NOT MEASURED.
- labevents may still be downloading at Y2: build the other templates and `mimic_lab_validation_v1` last. If labevents is
  absent at Y4, that template is NOT MEASURED in the suite.

GPU-hour estimate (planning guesses, not measurements):
- Y0 2, Y3 10, Y5 6, Y4 5, Y6 10, Y7 20, Y8 20, Y9 6, Y10 2;
- **about 80 GPU-hours, 3.5–4 days.**

## 6. Out of scope

- Osler-9B; architecture, attention or base changes; adapter variants beyond D33.
- Hosted teachers for anything; any hosted call with MIMIC content.
- Unofficial copies of credentialed data (D37).
- Training on any test split, the external panel (or any copy of a panel dataset), MDACE, held-out items, the knowledge
  dev v2 or CG items.
- The claims template (`clinclaim_grounding_v1`); restyle pages; synthetic charts. All are closed for this loop.
- MIMIC ICU tables (chartevents) and microbiology. They are a later loop (sepsis and respiratory-failure validation).
- Outcome prediction (readmission, length of stay, mortality).
- The HF encrypted backup (`./dm push`); releasing or uploading anything.

## 7. Failure modes to avoid

1. **"Coded" is not "documented".** `mimic_icd_*` labels are the coder's assignments, which include coding errors and
   omissions. Say so in `mimic.md`. Y3's verification (i) is what links a code to its documentation.
2. **The Discharge Diagnosis section gives the answer away.** The ICD and DRG templates keep it, because coders read it; the
   Naive Bayes screen decides whether a template is a shortcut. `mimic_principal_dx_v1` removes it by design. Report
   per-template screen scores.
3. **Long notes change the step time.** Measure s/step on B1's first 150 steps. If the 13,992-step block would exceed 7 h,
   record it; do not change the cap mid-loop.
4. **Edited notes carry tell-tale traces.** Y3's cue check is mandatory, and the cue AUC is reported for the final pairs as
   well as the pilot.
5. **Distillation targets leaking evaluation items.** Y5's exclusion list is checked by hash before training. Record the
   overlap count, which must be 0.
6. **MIMIC dominating the mix.** The 35 % cap applies; report the realised share per block and in the final.
7. **A teacher that silently reasons.** With reasoning off, check that responses carry no reasoning content (the
   `nemotron_v3` parser separates it). Record output-token counts.
