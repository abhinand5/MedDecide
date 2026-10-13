# ADVISORY — osler_v1: medical-record audit data, MIMIC, teacher prose

Planned 2026-10-13 by the advisor (Claude Opus) with the operator, after the osler_v0 review. Read in full at the start
of every iteration. Decisions referenced here are in `docs/plans/PROGRAM.md` §3 (D26–D31 are new).

## 1. Where things stand

osler_v0 (`loops/osler_v0/FINDINGS.md`) ended with **Gate O1 FAIL** for Osler-4B and Osler-9B:

- Both beat every baseline on the fresh seen-template set by 9–13 points and on the robustness pack.
- Both lost the external clinical panel to MedDecider of their size (4B −5.3, 9B −1.7 points).
- Osler-4B failed the knowledge guard (−1.6 [−2.9, −0.2]); the 9B passed.
- The three heads (L, P, N) were within about half a point; the dev rule kept L, the option-code head.
- The arms took about 35 GPU-hours, mostly because of small token batches and gradient checkpointing.

The operator's direction (2026-10-13): **data is the moat; stop architecture work** (D26). The target audience is
payment-integrity and medical-record audit work (D29), alongside general clinical decisions. This loop builds the first
record-audit data from three sources, fixes the knowledge loss with more replay, makes training faster, compares data
blocks with short matched runs, trains one Osler-4B and one Osler-9B on the kept mix, and applies Gate O2 (D31).

Code written at planning time (committed with this ADVISORY; tests in `tests/test_mr_v1.py`):

| module | what it does |
|---|---|
| `src/meddecide/mr/criteria.py` | 15 clinical indicators with published thresholds; the evidence rule (current encounter, not a baseline, room air for SpO2, MAR "given" only, unit conversion, blood-gas pairing, KDIGO creatinine rise) |
| `src/meddecide/mr/pages.py` | the 12 page classes (operator's taxonomy, vitals → flowsheet) and renderers for lab, flowsheet, MAR, bill and ED-triage pages in several layouts, including title-less continuation pages |
| `src/meddecide/mr/noise.py` | fax banners, page footers, OCR-style character noise that never alters a number |
| `src/meddecide/mr/icd.py` | 12 curated ICD-10-CM families with a resolver to the one supported code by the Official Guidelines (inpatient vs outpatient uncertain diagnoses, ruled out, history, family, specificity); the CMS order-file parser and `verify_codes` |
| `src/meddecide/mr/generators.py` | `mr_page_category_tabular_v1`, `mr_indicator_evidence_v1` (minimal-pair twins built from traps), `mr_icd_support_template_v1`; the held-out lists |
| `src/meddecide/mr/teacher_prose.py` | teacher prose from code-built facts plus re-extraction verification; refuses credentialed sources on a hosted teacher |
| `src/meddecide/mr/mimic.py` | MIMIC-III note → page class mapping, MIMIC-IV-Note readers, page chunking, ICD-10 family codes per admission, patient splits, the late-anchor rule |
| `src/meddecide/teacher/client.py` | adds `chat`, `model`, `local` and extra headers (from environment variables only) |
| `src/meddecide/train/arm_run.py` | **fix:** temperatures are fitted on the selected checkpoint and saved with it (osler_v0 deviation 59) |
| `scripts/osler_v1/v1_build_mr_generators.py` | V1's builder: splits by patient, the O3 shortcut screen, held-out items moved to `heldout_test.jsonl` |

A probe build (1,500 train patients per generator) ran at planning time; it is not a result. It showed the screen
pattern to expect: the indicator and ICD generators pass the O3 screen; the tabular page-category generator fails it
(Naive Bayes 1.0). That failure is expected and accepted (§3.6).

## 2. The question this loop answers

Can Osler-4B learn medical-record audit decisions (page category, ICD-10-CM code support, clinical-indicator evidence)
well enough to beat MedDecider-4B and its own base on them, without losing the external clinical panel or the base
model's knowledge? Which data sources make the difference: replay, code-generated records, teacher prose, real MIMIC
text?

Outcomes and what each implies (write the one that applies into FINDINGS):

1. **Gate O2 passes for both sizes.** The recipe is the base for the payer loop (claim edits, DRG, coverage).
2. **MR suite passes, panel or knowledge fails.** The audit data works but costs general skill. The next loop is
   replay and broad clinical data, not more audit data.
3. **MR suite fails against MedDecider.** If baselines are strong on MR tasks, our data adds too little. Look at
   which block helped on dev and on which template the loss sits.
4. **A block is not kept.** Report it with its dev numbers. A source that does not help on dev is dropped, not tuned.
5. **The held-out indicator or ICD family transfers** (reported separately, never in a verdict). Evidence that the
   model learned the rule, not the items.

## 3. Ground rules specific to this loop

1. **Labels by code, text from three sources (D27).** Every label is computed by code (`criteria`, `icd`) or taken
   from a structured MIMIC field (note category, coded ICD-10 diagnoses). Teacher prose is kept only when
   re-extraction returns the same facts (`teacher_prose.verify`). Never use a teacher's answer, option distribution or
   correctness judgement.
2. **MIMIC is local (D28).** MIMIC text and values never leave this machine. Never send them to the hosted teacher:
   `teacher_prose.generate` raises for credentialed sources unless `TEACHER_LOCAL=1`, and you must never set
   `TEACHER_LOCAL=1` for a hosted endpoint. Never commit, print into STATE/CLAIMS or paste any MIMIC text, value
   or identifier. Aggregates only.
3. **Held-out lists (fixed now, never changed):**
   - indicator `aki_creatinine_rise`;
   - ICD family `heart_failure`.

   Excluded from every training source (generated, teacher and MIMIC rows alike: drop MIMIC ICD rows of that family).
   Reported separately, never in a verdict or a keep decision.
4. **Splits by patient.** Generated items: `split_of(patient id)`. MIMIC: `mimic.subject_split` (dataset-qualified).
   The MIMIC-IV part of every **test** set uses only patients whose `anchor_year_group` passes `late_anchor`; record how
   many test rows that removes.
5. **Selection on dev only.** Block keep decisions, checkpoint selection and temperatures use dev splits. The external
   panel, MR test sets, held-out items and MDACE never influence a choice.
6. **The O3 screen.** Run it on every generated and teacher source. A failure is a defect to fix, except
   `mr_page_category_tabular_v1`: page category is lexical by nature (a lab report says "lab"), so its failure is
   recorded as expected (deviation), and its share of the mix is capped (§V7).
7. **No CPT descriptors, no InterQual or MCG content, nothing from any payment-integrity vendor.** HCPCS Level II,
   ICD-10-CM, CMS public files and published clinical criteria only.
8. **Teacher budget.** At most $60 of hosted teacher spend in this loop, counted from the API's reported token usage at
   list price ($0.15/M input, $0.60/M output for V4.1 Flash unless the endpoint reports cost). Stop at the cap and
   record what was built.
9. **One GPU job at a time**; detached jobs with logs; poll at 20 minutes or more (AGENTS.md).
10. **Recipe constants.**
    - Base: Qwen3.5-4B and -9B at the osler_v0 revisions.
    - Readout: option-code head (D23, D26).
    - LoRA r=32, alpha=32. lr 1e-3 head, 1e-4 LoRA. Cosine with a floor of 0.1. lambda_brier 1.0.
    - Selection metric: template macro.
    - Batching: from V0's measurement.

## 4. Tasks

### V0 — Orientation, environment, speed recipe (GPU small, ~3 h)

1. Read this ADVISORY, PROGRAM D26–D31 and `loops/osler_v0/FINDINGS.md` and `NEXT.md`. Run the full test suite and ruff.
   Record commit, GPU and disk (du per AGENTS.md).
   Known at planning time:
   - `tests/test_padding_trained.py::test_trained_checkpoint_logits_are_padding_invariant` is a strict xfail from
     student_v1. It runs only when CUDA is free, and on 2026-10-13 it XPASSED (a pass that must be reviewed).
   - Record it as a question for the operator. Never edit or deselect it in committed config.
   - The rest of the suite passed with that test deselected.
2. **Teacher reachability.** If `TEACHER_BASE_URL`, `TEACHER_MODEL` and `TEACHER_API_KEY` are set (via
   `scripts/pod_env.sh`), call `TeacherClient.from_env().list_models()` and one `chat` with a harmless prompt. Record
   the status code, model id and token usage, never the key. If they are not set, record `teacher: not configured`. V4
   then waits as BLOCKED until the operator sets them; do not stop the loop.
3. **MIMIC availability.** Check for `data/physionet/` (files: MIMIC-III `NOTEEVENTS.csv.gz`; MIMIC-IV
   `hosp/diagnoses_icd.csv.gz`, `hosp/patients.csv.gz`, `hosp/admissions.csv.gz`; MIMIC-IV-Note
   `note/discharge.csv.gz`, `note/radiology.csv.gz`; MIMIC-IV-ED `ed/*.csv.gz`). The operator downloads them with
   wget under that directory, so the paths may carry the physionet.org/files/<dataset>/<version>/ prefix. Find them
   with `find`. Record present or absent and sizes. Absent means V5 waits as BLOCKED.
4. **Speed recipe** (pre-registered choice). On Qwen3.5-4B, mix v2 train rows, option-code head, 150 steps each, time
   these configurations (same seed):
   - (a) the osler_v0 recipe: batch_size 8, max_batch_tokens 8,192, gradient checkpointing on;
   - (b) batch_size 32, max_batch_tokens 32,768, gradient checkpointing off;
   - (c) batch_size 64, max_batch_tokens 65,536, gradient checkpointing off;
   - (d) as (b) with gradient checkpointing on (only if (b) runs out of memory).

   Record for each: peak GPU memory, real tokens per second, items per second, and the loss over the last 50 steps.
   **Choose** the configuration with the highest items per second whose peak memory is ≤ 85 GB. Record that choice
   as the loop's batching constant (STATE §2). Learning rates stay the same. Note it in Deviations as a recipe
   change (D26).
5. **Calibration fix check.** Run the arm path end-to-end on the tiny model (`tests/test_arm_run_o6.py`). Confirm the
   saved `model.json` carries the fitted temperatures.

Acceptance: `outputs/osler_v1/V0/{env.json,speed.json}` exist; the choice is in STATE §2.

### V1 — MR generators (CPU, ~1 h)

1. `uv run --frozen python scripts/osler_v1/v1_build_mr_generators.py` with the defaults: 6,000 train, 300 dev and 300
   test patients per generator.
2. Self-audit:
   - read 10 rendered items per generator in a fresh process (do not paste them anywhere committed);
   - check label balance;
   - confirm twins flip (indicator, ICD);
   - confirm the held-out rows are only in `heldout_test.jsonl`;
   - check the screen results against §3.6.
3. Write `loops/osler_v1/mr_data.md` (aggregates only): generators, counts per split, label balance, screen results,
   the held-out lists.

### V2 — ICD-10-CM verification (CPU + network, ~30 min)

Download the CMS ICD-10-CM code descriptions in tabular order for FY2026 (cms.gov, "ICD-10 code files"). Unzip it to
`data/cms/icd10cm/2026/`. Run `icd.verify_codes(icd.load_order_file(<order file>))`.

- Every curated code must exist and be billable. If any is missing or not billable, mark V2 `BLOCKED` with the codes
  and ask the operator. Never edit the curated table to pass.
- Record the file's sha256.

### V3 — Replay to 20 % (CPU + network, ~3 h)

Add general human-labelled decision data until general replay is at least 20 % of the training rows of block B0 (§V7).
Candidates, each with its licence verified from the source and recorded:
- ARC (Easy and Challenge; CC BY-SA 4.0);
- OpenBookQA (Apache-2.0);
- BoolQ (CC BY-SA 3.0);
- WinoGrande (CC-BY);
- HellaSwag (MIT);
- SNLI (CC BY-SA 4.0).

Use the existing converters' pattern (`meddecide.mix.converters`), train splits only, deterministic ids. Every filter
returns counts and reasons. Run the leakage check (`meddecide.mix.leakage`) against v0.2 test, the panel and the MR
test sets. If a candidate's licence is unclear, drop it (record it).

### V4 — Teacher prose (network, CPU; BLOCKED until a teacher is configured)

Only code-generated facts go to the teacher (§3.1, §3.2). Two prose generators, written as thin scripts over
`meddecide.mr.teacher_prose`:

- **`mr_icd_support_prose_v1`**: an `icd.Diagnosis` drawn as in `mr_icd_support_template_v1` (excluding the held-out
  family). A page class is drawn from {discharge_summary, progress_note, consult_note, admission}. The setting is
  inpatient or outpatient (outpatient only with progress_note or consult_note). The question and options are the
  template generator's; the gold is `icd.supported_code`. Twins: the same one-fact change as the template generator,
  written by a separate teacher call; both twins must verify or the pair is dropped.
- **`mr_indicator_prose_v1`**: a `ChunkFacts` drawn as in `mr_indicator_evidence_v1` (excluding the held-out
  indicator). The page class is progress_note, written as a physician or a nursing progress note. The gold is
  `criteria.supports`. Twins as in the generator.

Steps:
1. **Pilot:** 100 requests per generator. Record the verification pass rate, the failure reasons by count, tokens and
   cost.
   - If the pass rate is below 60 %, fix the prompt (`write_messages`), never the verifier, and re-pilot once.
   - If it is still below 60 %, mark V4 `BLOCKED` with the numbers.
2. **Build**, within the budget (§3.8): aim for 15,000 verified pairs per generator, split by patient (pid). Run the
   O3 screen on dev, and check that no teacher page contains an ICD code string (regex `[A-TV-Z][0-9][0-9AB]\.?[0-9A-TV-Z]{0,4}`).
   Drop such pages and count them.
3. Spot-read 20 verified pages per generator in a fresh process. Record the defects you see by type (not the text).

The OpenCode Go plan is designed for coding agents and monitors for abuse. Send at most 2 requests per second and
respect HTTP 429 with exponential backoff. If the endpoint starts refusing, stop and record it; do not rotate keys.

### V5 — MIMIC sources (CPU; BLOCKED until the files exist)

1. **Page category (real notes).** `mimic.iter_mimic3_pages` plus `iter_mimic4_notes` for discharge and radiology.
   - Record every kept and dropped reason count.
   - Sample per class to at most 6,000 train pages per class (all classes that MIMIC provides). Page 0 and later pages
     are both eligible. Take 500 dev and 500 test pages per class.
   - Options and question as `mr_page_category_tabular_v1`. The gold is the mapped class.
   - Template id `mimic_page_category_v1`.
2. **ICD support (real discharge summaries).** Join MIMIC-IV-Note discharge to `mimic.admission_family_codes` (ICD-10
   admissions only; `mimic.icd10_admissions`).
   - For each admission and each non-held-out family: label = `mimic.family_label`. Ambiguous rows are dropped (count
     them).
   - Positives: every coded family code. Negatives (gold "none"): admissions with no code of that family, sampled at
     most 1:1 with the positives per family. Half of the negatives are hard: the family's condition word appears in the
     summary.
   - State = the whole discharge summary, truncated at the prompt cap from the end with a marker, counted.
   - Options and question as `mr_icd_support_template_v1`, setting inpatient.
   - Template id `mimic_icd_support_v1`.
   - This label is the coder's, not a construction (structured-source gold, D11/D13); say so in `mr_data.md`.
3. **Splits:** `mimic.subject_split`, then the late-anchor rule for MIMIC-IV test rows (§3.4).
4. **MDACE** (if obtainable from its public annotations release; it needs the MIMIC-III notes it points into):
   evidence-span rows for evaluation only (`mdace_icd_evidence_v1`). If the release cannot be obtained or matched
   within 2 hours, write `NOT MEASURED — <reason>`.
5. Append the MIMIC part to `loops/osler_v1/mr_data.md` (aggregates only).

### V6 — MR evaluation suite and baselines (GPU, ~6 h)

1. **Assemble the MR test suite.** Test splits of V1 and V4 (if built), plus the late-anchor test split of V5 (if
   built). Keep the template ids distinct. Write the manifest (ids, hashes, counts per template) to
   `data/bench/osler_mr_v0/`.
   - Held-out items go to a separate held-out file.
   - Also assemble a **fast dev set**: up to 400 dev items per MR template; 3,000 items of the mix v2 dev split,
     stratified by template; and the tier-1 knowledge dev (all MedQA and MedMCQA dev items in mix v2 dev). This is the
     fixed dev for V8.
2. **Score the baselines on the MR test suite and its held-out file.** Same runners and protocol as O2 (authors' code,
   both orders for choice items):
   - zero-shot Qwen3.5-4B and -9B, MedDecider-4B and -9B, JEV-9B;
   - additional: LiquidAI/d1-3B through its own code in `envs/d1/` (its card asks for transformers ≥ 5.14).
     Timebox 2 h, else `NOT MEASURED`. It is an additional row only (LFM licence; D31).

   Apply D12 to the zero-shot cells.
3. Write `loops/osler_v1/mr_baselines.md`.

### V7 — Mix v3 blocks (CPU, ~2 h)

Cumulative blocks, each a superset of the one before:

| block | adds | notes |
|---|---|---|
| B0 | mix v2 (osler_v0) + V3 replay so replay ≥ 20 % | the knowledge fix and new control |
| B1 | + V1 generators | page category tabular capped at 2 % of rows |
| B2 | + V4 teacher prose | skipped (NOT MEASURED) if V4 is blocked |
| B3 | + V5 MIMIC | skipped if V5 is blocked |

- No template above 8 % of a block (D17).
- Held-out rows are excluded everywhere.
- Run the leakage check of each block against every test set (v0.2 test, the panel, MR test, held-out, MDACE).
- Write the manifest per block.

### V8 — Block runs (GPU, ~2–3 h each)

**One matched short run per block:**
- Qwen3.5-4B, the recipe constants, V0's batching.
- The **same number of optimiser steps** for every block: the step count that gives 80,000 realised items on B0 by the
  O6 batch replay (`scripts/osler/o6_replay_batches.py` pattern).
- Eval at 25 %, 50 %, 75 % and 100 % of steps on the fixed fast dev set.
- The selected checkpoint is the best on the fast dev's pooled template macro.

Record realised items and tokens per block.

**Keep rule (pre-registered; dev only; paired bootstrap within templates, 1,000 resamples, seed 0).** Let C be the
last kept block (C = B0 to start). Block X is kept iff all three hold:
- (a) MR dev template macro: X − C lower bound > 0;
- (b) general dev (mix v2 dev part) template macro: X − C lower bound > −0.01;
- (c) knowledge dev (MedQA + MedMCQA dev, micro accuracy): X − C lower bound > −0.02.

**Knowledge check on B0.** B0's knowledge dev minus zero-shot Qwen3.5-4B on the same items must have a lower bound
above −0.02.
- If it fails, run one alternative **B0b**: LoRA target modules restricted to attention projections, same data.
  Use B0b as C if it passes and B0 does not.
- Otherwise record the failure (ADVISORY §2 outcome 2 risk) and continue with B0.

Write `loops/osler_v1/blocks.md`: each block's dev numbers, the decisions, the realised items and tokens.

**If V4 or V5 is still blocked when V8's last available block finishes:**
- set V9 `BLOCKED — waiting for <teacher|MIMIC>`;
- write the question to the operator: "train the final models without it, or wait";
- set the loop STOPPED per GOAL.

Do not train the finals without MIMIC unless the operator answers.

### V9 — Final Osler-4B and Osler-9B (GPU)

- **The kept mix**, one pass capped at 250,000 realised items.
- **Eval cadence:** 8 evals per run on the fast dev.
- **Temperatures** are fitted on the dev splits of the kept mix and saved with the checkpoint (V0 step 5).
- **4B first, then 9B.** Same recipe; 9B memory may force a smaller batch. If so, record the change.

### V10 — Evaluation and Gate O2 (GPU, ~6 h)

**Score both models on the evaluation sets:**
- the MR test suite and its held-out file;
- the external panel and the knowledge items of the O2 common set (v0.2 MedQA + MedMCQA test).

Use the calibrated scorer (deviation 59 path: the checkpoint now carries its temperatures; verify they match the dev
fits).

**Apply Gate O2 (D31)** exactly; baselines from V6 and osler_v0 O2. Report separately:
- the held-out indicator and family;
- MDACE;
- the generated vs MIMIC parts of the MR suite;
- d1-3B and the osler_v0 models (additional).

Write `loops/osler_v1/gate_o2.md` and `results.md`.

### V11 — Memorisation audit (GPU small; only if MIMIC is in the final mix)

For 200 training discharge summaries and 200 late-anchor test discharge summaries:
1. Load the base LM with the trained LoRA (the generation path, not the readout), give it the first 256 tokens and
   greedy-decode 64 tokens.
2. Count verbatim 8-gram overlaps of the continuation with the true continuation.
3. Report both rates, and the same for the untrained base.

A training rate clearly above the test rate is recorded as a release blocker for the operator. Aggregates only.

### V12 — Findings and closure (no GPU) — HARD STOP

FINDINGS.md:
- Summary first;
- which §2 outcome applies;
- every number with a CLAIMS row;
- the five-claim spot-check re-run.

NEXT.md: proposals only.

Then set `Loop status: STOPPED`, commit, push, stop.

## 5. Definition of done

Every task `DONE` or `BLOCKED — <reason>`, with these files present and every number in CLAIMS:
- `mr_data.md`, `mr_baselines.md`, `blocks.md`, `gate_o2.md`, `results.md`, `FINDINGS.md` and `NEXT.md`;
- or the BLOCKED stop of V8 with FINDINGS for what ran.

## 6. Task table (seed STATE.md with this)

| id | task | GPU | deps |
|---|---|---|---|
| V0 | Orientation, environment, speed recipe | small | — |
| V1 | MR generators | no | V0 |
| V2 | ICD-10-CM verification | no | V0 |
| V3 | Replay to 20 % | no | V0 |
| V4 | Teacher prose | no | V1, V2 |
| V5 | MIMIC sources | no | V0 |
| V6 | MR evaluation suite + baselines | yes | V1 (V4, V5 if built) |
| V7 | Mix v3 blocks | no | V3, V6 |
| V8 | Block runs + keep rule | yes | V7 |
| V9 | Final Osler-4B and -9B | yes | V8 |
| V10 | Evaluation + Gate O2 | yes | V9 |
| V11 | Memorisation audit | small | V9 |
| V12 | Findings and closure — HARD STOP | no | all |

Order exceptions:
- V1–V5 may run in any order on CPU.
- V4 and V5 may run while a GPU task runs.
- If V4 or V5 becomes available after V6 ran, re-run only V6's assembly and baseline scoring for the new templates
  (record it). Never re-run blocks already decided.

## 7. Explicitly out of scope

- Head, attention or other architecture changes (D26). Thinking/adaptive modes.
- Claim edits (NCCI/MUE), DRG validation, coverage criteria (next loop, D29).
- Any training on test splits, the external panel, MDACE, held-out items.
- Sending MIMIC content to a hosted service. Self-hosting the teacher (operator's call).
- Releasing or uploading anything. 27B-class training.

## 8. Failure modes to avoid

1. **A teacher fact leak.** The teacher writes "consistent with sepsis criteria", or puts an ICD code in the page.
   The verifier does not catch judgement words. Spot-read pages and run the code regex (V4).
2. **Lexical shortcuts in rendered pages.** For example, "(prior admission)" alone deciding the label. The variety in
   `pages.py` reduces this, but read the Naive Bayes screen numbers; above 0.9 on an indicator source is a defect.
3. **MIMIC label noise read as model error.** Coded diagnoses miss conditions. Report the MIMIC ICD part separately and
   do not over-interpret small differences on it.
4. **Mixing patients across splits** through MIMIC-III/IV overlap. Use the late-anchor rule.
5. **A block "helps" through more steps or more tokens.** Blocks are matched by steps. Report realised items and tokens.
6. **Calibration regressions.** Every scored model must carry its temperatures. Check before V10.

## 9. If you finish early

Write NEXT.md's design notes for the payer loop:
- claim edits from CMS NCCI PTP and MUE quarterly files;
- MS-DRG validation from the CMS definitions manual;
- coverage criteria from NCDs/LCDs, CPT-free.

Read and summarise the file formats. Do not download large files and do not start building.
