# ADVISORY — osler_v2: real text with the attention-only adapter, replay 20 %, Osler-4B and -9B

Planned 2026-10-17 by the advisor (Claude Opus) with the operator, after the osler_v1 review. Read in full at the start of
every iteration. Decisions referenced here are in `docs/plans/PROGRAM.md` §3 (D33–D37 are new).

## 1. Where things stand

osler_v1 (`loops/osler_v1/FINDINGS.md`) ended with **Gate O2 FAIL** for Osler-4B (9B deferred):

| check | vs MedDecider-4B | vs zero-shot Qwen3.5-4B |
|---|---|---|
| MR suite | accuracy +0.081 [+0.070, +0.092] but Brier +0.035 (fails) | passes |
| External panel (non-inferiority, > −0.02) | −0.022 [−0.037, −0.009] (fails) | passes |
| Knowledge guard | — | **+0.044 [+0.032, +0.055] (passes)** |

What the loop learned:
- **The attention-only adapter fixed the knowledge loss.** osler_v0's 4B was −1.6 points; osler_v1's 4B is +4.4.
- **Audit skill on generated pages does not transfer to real text.** On generated items, +0.235 vs MedDecider-4B; on
  real text, −0.074. The loss sits on real-text ICD coding (CodiEsp, the clinical-case set), where Osler is wrong and
  confident. The kept block B1 held **no real clinical text**.
- **Real text helped on dev.** B3 raised the MR dev macro by +0.054, but was rejected for knowledge (−0.040) and general
  (−0.0035) margins. That was with the old all-layer adapter, which is the one that loses knowledge.
- **Synthetic charts** (operator, 2026-10-17): whole multi-page charts written by a local Gemma 4 31B for chart-level
  decisions (W3b). The claim that they beat real data is tested on real-text dev templates, not assumed.

Defects of the osler_v1 design, fixed here:
1. **Cumulative blocks.** B3 carried B2's rejected prose. Fix (D34): each block adds one source to the last kept block.
2. **Dev blind spots.** The fast dev had no teacher-prose templates, so B2 could not show a benefit. Fix (D34): the dev
   set holds every candidate source's dev items.
3. **An underpowered knowledge check.** At 500 items, a model 0.8 points from its base failed the −0.02 margin. Fix
   (D34): at least 4,000 knowledge dev items.
4. **A distorted MR suite.**
   - `pmc_similar_patient_v1` was 42 % of the MR test items and is solvable by word overlap (AUC 0.97).
   - Three templates sat at 1.000.

   Fix (D35): MR suite v1 drops ceiling and vocabulary-shortcut templates from the verdict, by a fixed rule.
5. **Replay below target** (12.6–15.3 %). Knowledge fell as the replay share fell. Fix (D36): ≥ 20 %, with share-alike
   sources allowed.
6. **Unrounded teacher values** ("lactate 2.873"), and a repeated phrase ("documented baseline") that went with gold "no".
   Fixed in code at planning time (`teacher_prose.chunk_facts` rounds as a record prints, and the prompt asks for varied
   wording).

**MIMIC:** the operator's PhysioNet approval is still pending. Official files only (D28, D37). V5c's code is reused as W4.
It never blocks the loop.

## 2. The question this loop answers

With the attention-only adapter, does real clinical text (CodiEsp, MTSamples, EHR-Rel) close the real-text ICD gap without
costing knowledge or general skill? And does Osler-4B or Osler-9B then pass Gate O3 against MedDecider of its size?

Outcomes (write the one that applies into FINDINGS):

1. **Gate O3 passes for a size.** That recipe is the base for the payer loop and the release track.
2. **The real-text block is kept but the gate fails on the panel.** The audit skill transfers; broad clinical data is the
   next lever.
3. **The real-text block is not kept on the attention-only adapter either.** Real text costs knowledge or general skill
   whatever the adapter. Look at which template causes it.
4. **The MR suite v1 fails on Brier only.** Calibration is the lever, not data: per-template temperatures or a
   confidence penalty.
5. **The 9B passes and the 4B does not.** Report the size effect.

## 3. Ground rules specific to this loop

1. **Everything from osler_v1 §3 still holds:**
   - labels by code or structured fields;
   - teacher prose only if verified;
   - MIMIC local only;
   - splits by patient;
   - selection on dev only;
   - no CPT, InterQual or MCG content;
   - one GPU job at a time.
2. **The local teacher (W3b)** is Gemma 4 31B QAT (Apache-2.0) served by vLLM on this GPU. It counts as a GPU job.
3. **No unofficial copies of credentialed data, ever (D37).** Never download or read `lucky9-cyou/MIMIC-IV`,
   `ricardosantoss/ehrcomplete_icdfiltered`, or any other Hub re-upload of MIMIC or other PhysioNet data, even if the
   operator's approval arrives during the loop. Official files under `data/physionet/` only.
4. **Adapter (D33):** LoRA on the attention projections only (osler_v1 B0b's `target_modules`), r=32, alpha=32. Every
   other recipe constant is as in osler_v1 (V0's batching: batch 8, 8,192 tokens, gradient checkpointing).
5. **Held-out lists are unchanged:**
   - indicator `aki_creatinine_rise`;
   - ICD family `heart_failure`;
   - osler_v0's held-out templates and generators.
6. **Teacher:**
   - **OpenCode Go:** the same settings and guards as osler_v1 V4 (thinking off; $14 per 5 hours; $36 per rolling week).
   - **DeepSeek API** (`deepseek-flash`, key from `DEEPSEEK_API_KEY` in `~/.bashrc`, never printed): allowed in this loop
     for code-generated synthetic facts only, with its own ledger capped at $9.00.
   - **Loop total:** at most $50 across both endpoints.
7. **Knowledge dev items never come from the test sets.** MedMCQA's official dev split *is* the v0.2 MedMCQA test set, and
   MedQA's test split is the v0.2 MedQA test set. Knowledge dev items are drawn from the **official train splits, from
   rows that no block trains on** (§W5).

## 4. Tasks

### W0 — Orientation (no GPU, ~1 h)

1. Read this ADVISORY, PROGRAM D33–D37, and `loops/osler_v1/FINDINGS.md` and `NEXT.md`. Run ruff and the suite, with the
   padding test deselected on the command line as in osler_v1.
2. **Teacher reachability:** OpenCode Go (`list_models`) and DeepSeek (`GET /user/balance`). Record the remaining
   allowance and balance, never a key.
3. **MIMIC files:** official files present? (expected: absent).
4. Record the commit, GPU and disk.

### W1 — Audits carried from osler_v1 NEXT (CPU, ~3 h)

1. **The too-good ICD template** (`mr_icd_support_template_v2`, 1.000 on 554 test items):
   - Build a rule-level probe that answers each test item from its rendered text with no model (parse the certainty,
     subject, status and attribute words). Report its accuracy.
   - Run an exact and near-duplicate (normalised text) overlap check of its test items against B1's training rows.
   - Report what explains 1.000, if anything. Never change the template.
2. **The O3 Naive Bayes screen on every MR test template**, on its dev items, including `pmc_similar_patient_v1`, which
   osler_v1 never screened. Record the screen value per template.
3. **The ceiling statistic:** zero-shot Qwen3.5-4B accuracy on each MR template's dev items. Score the dev items with the
   O2 zero-shot runner if needed; the GPU is idle at this point.
4. **Freeze the MR suite v1 (D35)** from steps 2 and 3:
   - `data/bench/osler_mr_v1/manifest.json` lists the verdict templates and the reported-only templates, each with its
     reason.
   - Write `loops/osler_v2/mr_suite_v1.md`.
   - This is frozen before any osler_v2 model result exists.

### W2 — Replay ≥ 20 % and EHR-Rel (CPU + network, ~3 h)

1. **Add share-alike and permissive replay** (D36), train splits only, licences recorded:
   - ARC Easy and Challenge (CC BY-SA 4.0);
   - BoolQ (CC BY-SA 3.0);
   - SNLI (CC BY-SA 4.0);
   - OpenBookQA, only if its licence can be verified from the source.

   Use the existing converter pattern. Run the leakage check against every test set. Target: replay ≥ 20 % of the
   **largest** block's training rows, so every block is at or above 20 %. Use one fixed replay set for all blocks.
2. **EHR-Rel** (`bigbio/ehr_rel`, Apache-2.0; source github.com/babylonhealth/EHR-Rel): concept pairs with human
   relatedness ratings.
   - Write it as a `score` item: "How related are these two clinical concepts?", with the levels taken from the dataset's
     rating scale (read it from the source; never invent it).
   - Gold = the rounded mean rating. If the source gives only a continuous mean, use the bins it defines; if none, record
     `BLOCKED — rating scale not defined` and skip.
   - Split by concept pair (stable hash), 80/10/10. Template `ehr_rel_relatedness_v1`.

### W3 — ICD prose v2 (network; optional; ~$10)

osler_v1's ICD prose failed its pilot at 59.8 % because the teacher wrote past and family conditions ambiguously.

Design v2:
- **One diagnosis per page, about the patient and current only.** Vary certainty (confirmed, probable, suspected, ruled
  out), setting and the family's attributes.
- The history and family twists stay in the template generator only.
- Twins change certainty or one attribute.
- Gold = `icd.supported_code`.
- Template `mr_icd_support_prose_v2`.

Pilot: 100 ids, threshold 60 %, one permitted prompt fix. Then build up to 12,000 verified pairs within the budget
(§3.5). If the pilot fails, record `BLOCKED` and continue.

### W3b — Whole synthetic charts, local Gemma 4 teacher (GPU, ~1 h set-up + pilot, build ≤ 12 h)

Operator decision (2026-10-17): synthetic charts written locally by Gemma 4 31B (Apache-2.0), crafted for chart-level audit
decisions. Code written at planning time: `src/meddecide/mr/charts.py`, `scripts/osler_v2/w3b_charts.py`,
`tests/test_charts_v2.py`.

**What a chart is.** One Synthea encounter rendered as 7–14 pages, all with one patient header:
- code-rendered pages (laboratory, flowsheet, MAR, bill);
- one **planted evidence page** for an indicator (the twin chart changes only this page);
- narrative pages written by Gemma: an admission note, 1–3 progress notes, an optional consult or nutrition note, and a
  discharge summary carrying a final diagnosis.

Narrative pages must state **no** vital signs or lab values (checked by re-extraction), so labels come from the code pages
and the diagnosis facts only.

Three templates:
- `chart_indicator_evidence_v1` (noul; twins);
- `chart_evidence_page_v1` (choice: which page documents it);
- `chart_icd_support_v1` (choice over the family codes + none, inpatient rules).

**Steps:**
1. **Environment** (`envs/vllm/`, never the main project environment):
   - `uv venv --python 3.12 envs/vllm/.venv`, then install `vllm` (0.31.0, the newest on PyPI on 2026-10-17) with a torch
     build that runs on this Blackwell GPU.
   - Download the chat template `examples/tool_chat_template_gemma4.jinja` from the vLLM GitHub tag matching the
     installed version into `envs/vllm/examples/`.
   - Caches stay on /workspace (`scripts/pod_env.sh`).
2. **Serve** with the operator's exact setup (detached, from `envs/vllm/`, log under `outputs/osler_v2/W3b/logs/`):
   ```
   export VLLM_USE_RUST_FRONTEND=1
   vllm serve google/gemma-4-31B-it-qat-w4a16-ct \
     --tensor-parallel-size 1 \
     --enable-auto-tool-choice \
     --tool-call-parser gemma4 \
     --chat-template examples/tool_chat_template_gemma4.jinja \
     --reasoning-parser gemma4 \
     --speculative-config '{"model":"google/gemma-4-31B-it-assistant","num_speculative_tokens":4}' \
     --language-model-only
   ```
   Add a flag only if the server will not start without it (for example `--max-model-len` for memory). Record any such
   flag as a deviation. The QAT weights are about 23 GB and the draft model about 1 GB.
3. **Client variables** (local; cost is counted as $0):
   ```
   TEACHER_BASE_URL=http://127.0.0.1:8000/v1  TEACHER_MODEL=google/gemma-4-31B-it-qat-w4a16-ct  TEACHER_LOCAL=1
   TEACHER_THINKING=none  TEACHER_EXTRA_BODY='{"chat_template_kwargs": {"enable_thinking": false}}'
   ```
   These are set for the W3b process only. Never set `TEACHER_LOCAL=1` with a hosted URL.
4. **Pilot:**
   ```
   scripts/osler_v2/w3b_charts.py --count 200 --out data/gen/osler_v2_charts_pilot
   ```
   Record:
   - charts kept and items by template;
   - drop reasons by count (`summary_*.json`);
   - charts per hour;
   - tokens.

   Read 10 charts in a fresh process; never paste them anywhere committed. Check for:
   - stated numbers in narrative pages;
   - repeated phrases that track a gold label;
   - name and identifier consistency;
   - whether the discharge-summary diagnosis wording matches its certainty.

   **Acceptance:** at least 50 % of the attempted encounters yield a full chart, and no label-tracking phrase is found.
   You may make one prompt fix if needed (the `NO_NUMBERS` or `extra` wording in `charts.py` only; never the verifier).
   If the pilot still fails, mark W3b `BLOCKED` and continue.
5. **Build:** `--count 20000` (resumable). Stop at 20,000 encounters or 12 hours of GPU time, whichever comes first. Then
   stop the server.
6. **Screens:**
   - Run the O3 Naive Bayes screen on each chart template's dev split.
   - Run the leakage check against every test set.
   - Write `loops/osler_v2/charts.md` (aggregates only).

W3b uses the GPU. Run it before W5 (no other GPU job may run at the same time), so that its dev items enter fast dev v2.

### W4 — MIMIC (CPU; only when the official files exist)

osler_v1 V5c, unchanged: page category (MIMIC-III notes, MIMIC-IV-Note), ICD support (MIMIC-IV discharge summaries with
coded ICD-10 families), patient splits, the late-anchor rule for test rows, and MDACE for evaluation only.

Re-check at W6 and again before W7. It never blocks the loop.

### W5 — Dev and evaluation assembly; new baselines (GPU, ~4 h)

1. **Fast dev v2** (D34), fixed before any block runs. It contains:
   - the osler_v1 fast dev (5,723 items);
   - up to 400 dev items per template of every candidate source not already in it: teacher prose
     (`mr_indicator_prose_v1`, `synthea_narrative_page_v1`, and `mr_icd_support_prose_v2` if W3 built it), the W3b chart
     templates if W3b built them, MTSamples,
     **CodiEsp dev** (carve 15 % of CodiEsp *train* cases by case id; they leave training), EHR-Rel, and MIMIC if W4 ran;
   - the **knowledge dev v2**: 1,000 MedQA rows and 3,000 MedMCQA rows from the official train splits that are not in
     any training block (remove them from every block if present), 4,000 items in all.

   The MR, general and knowledge partitions are recorded in its manifest.
2. **Zero-shot Qwen3.5-4B on the knowledge dev v2** (the C0 knowledge check needs it).
3. **Test sets of the new templates:** EHR-Rel test; prose v2 test if built; MIMIC late-anchor test if built.
   - Score them with the five baselines using the O2 runners and protocol.
   - d1-3B is an additional row.
   - JEV-9B and MedDecider-9B are needed for the 9B gate.
4. Write `loops/osler_v2/baselines_v2.md`.

### W6 — Blocks v2 and the keep rule (GPU, ~5.5 h per block)

All runs are attention-only (D33), on the same 13,992 steps as osler_v1's blocks, and scored on fast dev v2.

| block | contents | compared with |
|---|---|---|
| C0 | osler_v1 B1 (generators + Synthea) + the W2 replay (≥ 20 %) | knowledge check against zero-shot on knowledge dev v2 |
| C1 | C0 + real text: MTSamples, CodiEsp train (minus the dev carve-out), EHR-Rel train. `pmc_similar_patient_v1` joins only if W1's screen put it below 0.90 | C0 |
| C2 | last kept + teacher prose (osler_v1 indicator and narrative prose; ICD prose v2 if built) | last kept |
| C3 | last kept + W3b charts (all three chart templates), if built | last kept |
| C4 | last kept + MIMIC (W4), only if the official files exist before C3's keep decision | last kept |

Rules:
- **Keep rule unchanged (D34):** paired bootstrap within templates, 1,000 resamples, seed 0. Lower bounds: MR > 0,
  general > −0.01, knowledge > −0.02. The MR partition includes every MR template of fast dev v2, so a source's own
  templates count.
- **The real-text question for the charts (labelled; never part of the rule).** Beside the keep rule, report C3's delta
  against its comparator on the **real-text** dev templates only (MTSamples, CodiEsp dev, EHR-Rel, MIMIC if present).
  This is the evidence for or against "synthetic charts make real text better".
- **If C0 fails its knowledge check against zero-shot,** record it and continue: the guard on the test set decides
  later. There is no adapter fallback in this loop.
- **Caps:** no template above 8 % of a block; the tabular page-category templates at 2 % together.
- Write `loops/osler_v2/blocks.md`.

### W7 — Final Osler-4B and Osler-9B (GPU)

- The kept mix, attention-only, one pass capped at 250,000 realised items.
- 8 evals on fast dev v2; temperatures fitted on dev and saved with the checkpoint.
- **4B first, then 9B.** If the 9B needs a smaller batch for memory, record it.
- If MIMIC arrives after C3's decision, it does **not** enter the finals of this loop.

### W8 — Evaluation and Gate O3 (GPU, ~8 h)

1. **Score both models, calibrated:**
   - MR suite v1, including the reported-only templates;
   - the held-out files;
   - the external panel;
   - the knowledge items.
2. **Apply Gate O3 (D35)** exactly, for both sizes.
3. **Additional rows**, labelled:
   - generated vs real-text parts;
   - the reported-only templates;
   - held-out items;
   - d1-3B;
   - osler_v1's Osler-4B;
   - panel superiority.
4. Write `gate_o3.md` and `results.md`.

### W9 — Memorisation audit (only if MIMIC is in a final mix) — as osler_v1 V11.

### W10 — Findings and closure — HARD STOP

FINDINGS.md:
- Summary first;
- which §2 outcome applies;
- the five-claim spot-check re-run.

NEXT.md: proposals only. Set `Loop status: STOPPED`, commit, push, stop.

## 5. Task table (seed STATE.md with this)

| id | task | GPU | deps |
|---|---|---|---|
| W0 | Orientation | no | — |
| W1 | Audits; MR suite v1 frozen | small | W0 |
| W2 | Replay ≥ 20 %; EHR-Rel | no | W0 |
| W3 | ICD prose v2 (optional) | no | W0 |
| W3b | Whole synthetic charts, local Gemma 4 teacher | yes | W0 |
| W4 | MIMIC (only when the official files exist) | no | W0 |
| W5 | Fast dev v2, knowledge dev v2, new test templates and baselines | yes | W1, W2, W3b (W3, W4 if built) |
| W6 | Blocks v2 + keep rule | yes | W5 |
| W7 | Final Osler-4B and Osler-9B | yes | W6 |
| W8 | Evaluation + Gate O3 | yes | W7 |
| W9 | Memorisation audit (only with MIMIC) | small | W7 |
| W10 | Findings and closure — HARD STOP | no | all |

Order exceptions:
- W1–W4 may run in any order; W3 and W4 may run while a GPU task runs. W3b is a GPU task: it runs alone, before W5.
- W4 never blocks a task.
- If W3 finishes after W5 assembled the dev set, its templates are **not** added (the dev set is fixed); W3's prose is
  then dropped from C2 and recorded.

## 6. Explicitly out of scope

- Head, attention-mask or base-model changes; adapter variants other than D33's.
- Unofficial copies of credentialed data (D37).
- Claim edits, DRG validation, coverage criteria.
- Training on any test split, the external panel, MDACE, held-out items or the knowledge dev v2.
- Releasing or uploading anything.

## 7. Failure modes to avoid

1. **Knowledge dev leaking into training.** Remove the 4,000 knowledge dev rows from every block and check it.
2. **MR suite v1 drift.** The suite is frozen at W1 before any osler_v2 model result. Never move a template in or out
   afterwards, even when a result looks unfair.
3. **The 9B outgrowing memory at batch 8 / 8,192 tokens.** Record the change; never shorten the prompt cap silently.
4. **Teacher wording cues.** Spot-read 20 prose v2 pages, and list repeated phrases that correlate with a gold label.
5. **Calibration.** Every scored model must carry its temperatures; V9/V10's check from osler_v1 is reused.
