# ADVISORY — osler_v3: keep the base's general skill (KL-to-base), real sources + Gemma-crafted real-text items

Planned 2026-10-18 by the advisor (Claude Opus) with the operator, after the osler_v2 review. Read in full at the start of
every iteration. New decisions: D41–D44 in `docs/plans/PROGRAM.md` §3 (D26–D40 still hold).

## 1. Where things stand

osler_v2 (`loops/osler_v2/FINDINGS.md`): Osler-4B on the pplx-decider-flash-v1.2 base (D39), kept mix C1 (osler_v1 generators +
Synthea + the 20 % replay + real text: MTSamples, CodiEsp, EHR-Rel). Gate O3:

| check | result |
|---|---|
| MR suite v1 vs MedDecider-4B | **+0.263** [+0.251, +0.276], Brier −0.308 (passes) |
| MR suite v1 vs zero-shot pplx-flash | **+0.158** [+0.142, +0.173], Brier −0.243 (passes) |
| External panel vs MedDecider-4B (non-inferiority > −0.02) | −0.004 [−0.020, +0.011] (passes) |
| **External panel vs zero-shot pplx-flash** | **−0.020 [−0.034, −0.006] (fails)** |
| Knowledge vs zero-shot pplx-flash | +0.002 [−0.006, +0.011] on 5,435 paired items. Recorded NOT MEASURED over 21 unpaired items; D44(a) now handles this |

What osler_v2 learned:
- **Real text is the lever for the record-audit domain.** C1 gave +0.075 on MR dev with no general or knowledge cost.
- **Teacher prose from synthetic facts does not help.** C2 was rejected, as osler_v1 B2 was.
- **The one remaining failure is forgetting.** Fine-tuning drifts 2 points from pplx-flash's general clinical skill on the
  external panel.
- **The Gemma chart pilot worked** (85–90 % of charts kept), but W3b was blocked by the certainty word of one template.
  D44(b) fixes the rule.

The code written at planning time for this loop is the **KL-to-base regulariser** (D41):
- `kl_to_base` and `kl_exclude_template_prefixes` in the config;
- `MedDecideModel.reference_option_probs`;
- `losses.kl_to_reference`;
- trainer logging of `kl` and `n_anchor` per step;
- tests in `tests/test_trainer_o6.py`.

On pplx with every item an anchor, it measured +27 % step time (1.03 s against 0.81 s) at an unchanged 53 GB peak.

**MIMIC:** still not approved (one week). Official files only (D28, D37). It never blocks the loop.

## 2. The questions this loop answers (by experiment, not by argument)

1. **Does KL-to-base close the panel gap without losing the MR gain?** At what strength, and does broader replay do the same
   job? (Stage A, four matched runs.)
2. **Do the new real sources add anything on top?** RAGTruth, DrugProt and MultiCaRe-derived items. (Stage B.)
3. **Do Gemma-crafted items over real text help, where teacher prose over synthetic facts did not?** (Stage B.)
4. **Do whole synthetic charts help chart-level audit?** (Stage B.)
5. **Does the final Osler-4B pass Gate O3?**

Outcomes (write the ones that apply into FINDINGS):
- **(1) Gate O3 passes:** the recipe goes to the 9B and the payer loop.
- **(2) KL closes the panel gap on test but costs MR:** report the trade-off curve over the Stage A strengths.
- **(3) No Stage A variant beats the control on the dev proxy:** the panel gap is not forgetting that dev can see. Look at
  the panel per dataset.
- **(4) A Stage B source is not kept:** report it with its dev numbers; it is dropped, not tuned.

## 3. Ground rules specific to this loop

1. **Everything from osler_v2 §3 still holds:** base pplx-flash (D39), attention-only LoRA (D33), the D40 speed recipe, no
   unofficial credentialed data (D37), labels by code or structured fields, selection on dev only, held-out lists
   unchanged.
2. **KL-to-base (D41).** Anchors are items whose template does **not** start with a domain prefix. The domain prefixes are
   fixed now and are the same in every run:
   ```
   mr_, synthea_, chart_, codiesp_, mtsamples_, mimic_, ehr_rel_, pmc_, ragtruth_, drugprot_, multicare_, clinclaim_, restyle_
   ```
   A new template family added during the loop must get a prefix from this list, or be recorded as an anchor in
   Deviations.
3. **Gemma-crafted items (D43):**
   - only over **real** text;
   - the label is the code's choice;
   - an independent verification call must agree;
   - contradictions are checked by string rules.

   Never keep an item whose verification disagrees, and never relabel one.
4. **Never train on** `MaziyarPanahi/MedExQA-clean` or any other copy of an external-panel dataset (D42). Never use JEV's
   predictions from the RAGTruth re-packaging as labels.
5. **The MR suite v1 and its verdict templates stay frozen** (osler_v2 W1). New test templates from this loop are
   reported as additional rows, never in the Gate O3 verdict, so the verdict stays comparable with osler_v2.
6. **Gemma** runs from osler_v2's `envs/vllm/` with the operator's serving command (osler_v2 ADVISORY W3b). It is a GPU job.
7. **Teacher budget (hosted):** none planned. Everything teacher-side is local Gemma.

## 4. Tasks

### X0 — Orientation (~1 h)

1. Read this ADVISORY, PROGRAM D41–D44, and `loops/osler_v2/FINDINGS.md` and `NEXT.md`. Run ruff and the suite, with the
   padding test deselected on the command line.
2. Check that the official MIMIC files are absent (expected) and that `envs/vllm/` serves Gemma (start and stop it).
3. **KL timing:** 150 training steps on the C1 rows with `kl_to_base: 0.5` and the domain prefixes of §3.2, on pplx.
   Record s/step, the anchor share and peak memory.

### X1 — Real sources (CPU + network, ~4 h)

Every converter returns counts and reasons, records licences per source, and splits by document or context (stable
hash). Run the leakage check against every test set (v0.2 test, the panel, the MR suite v1, held-out, the knowledge test).

1. **RAGTruth** from its original release (github.com/ParticleMedia/RAGTruth, MIT; the `wandb/RAGTruth-processed` mirror is
   acceptable if its revision is pinned). Labels come only from the human span annotations. Templates:
   - `ragtruth_response_grounded_v1` (noul): does the response contain no annotated hallucination?
   - `ragtruth_sentence_supported_v1` (noul): does this sentence of the response overlap no annotated hallucination
     span?
   - `ragtruth_halluc_type_v1` (choice over the annotated types plus none), when the source labels the type.

   Split by source context id, so one context never crosses splits. Cap at 20 sentence items per response.
2. **DrugProt** (`OpenMed/drugprot-parquet`, CC BY 4.0; train and validation). Template `drugprot_relation_v1`: for a
   chemical–gene pair marked in the abstract, which relation holds? Options are the annotated relation types plus "no
   relation".
   - Negatives are co-occurring pairs in the same sentence with no annotated relation, sampled at most 1:1 with the
     positives.
   - Train on the train split. Use the validation split as dev and test, split by PMID.
3. **MultiCaRe** (`OpenMed/multicare-cases`, CC BY 4.0). Inspect the fields first, and build direct decision items **only**
   from structured fields that are non-trivial to infer: **not** sex, and **not** age when the text states it. If none
   qualifies, record it and build none. MultiCaRe's main use is as the real-text substrate for X3. Split by case id.
4. Write `loops/osler_v3/sources.md`: counts, licences, leakage results, and the screen per template (the O3 Naive Bayes
   screen on dev).

### X2 — Broad replay for ablation A3 (CPU + network, ~3 h)

Assemble **human-labelled** general classification and NLI datasets with verified permissive licences (CC BY, CC BY-SA,
MIT, Apache; recorded per source):
- at least 8 datasets beyond the osler_v2 replay;
- train splits only;
- no overlap with any test set;
- never an LLM-labelled set.

Target: replay at 35 % of the A3 mix (osler_v2 C1 + this broad replay). Timebox 3 h, and record what was found if the
target is not met.

### X3 — Gemma-crafted items over real text (GPU, ≤ 6 h build)

The substrate is real text: MultiCaRe cases, MTSamples, CodiEsp **train**, and PMC-Patients. Use train splits only. A
document's split carries over to every item made from it.

1. **`clinclaim_grounding_v1`** (choice: supported / contradicted / not mentioned), with options in random order:
   - Code draws the label, and for "contradicted" also the **edit kind**: a number, laterality, negation, a drug or a
     temporal order.
   - Gemma writes one short claim about the passage, following the label and edit kind.
   - **Verification:**
     - a second Gemma call, with a different system prompt and temperature 0, classifies the claim against the passage;
       keep the item only if the classes agree;
     - for "contradicted", a string rule checks that the edited element exists in the passage and differs in the claim
       (for example, the number in the claim does not occur in the passage);
     - for "supported" and "not mentioned", reject claims that copy more than 80 % of a passage sentence verbatim.
   - Balance the three labels. Passages are 1–4 paragraphs, cut at paragraph boundaries.
2. **`restyle_page_category_v1`** (choice over the 12 page classes):
   - Code draws a page class from the narrative classes (discharge summary, admission, progress note, consult, ED,
     nutrition, radiology).
   - Gemma rewrites a real case into that page type, keeping its clinical facts.
   - Verification: teacher_prose kind `page`, the page type extracted independently.
   - Items are capped so no class exceeds 25 %.
3. **Pilot:** 300 items per template. Record the keep rate, the agreement rate and spot-read defects (never pasted into
   committed files). Acceptance: keep rate ≥ 40 %, and no label-tracking phrase other than the labelled fact (D44(b)).
   One prompt fix is allowed per template.
4. **Build:** up to 40,000 claim items and 12,000 restyled pages, within 6 GPU-hours.

### X4 — Charts (GPU, ≤ 6 h build)

Rebuild osler_v2 W3b's charts (`scripts/osler_v2/w3b_charts.py`) with D44(b):
- the certainty word of the target diagnosis is the labelled fact, not a cue;
- any other label-tracking cue blocks **its own template only**.

The osler_v2 pilots count as the pilot; re-run the label-tracking check on the first 500 built charts. Build up to 8,000
charts within 6 GPU-hours, then the screens and `loops/osler_v3/charts.md`.

### X5 — MIMIC (only when the official files exist) — as osler_v1 V5c. Never blocks anything.

### X6 — Fast dev v3, the clinical-general proxy, and new baselines (GPU, ~4 h)

1. **Fast dev v3** = osler_v2's fast dev v2, plus up to 400 dev items per new template (X1, X3, X4, and X5 if built). It is
   fixed before any Stage A run, and its partitions are recorded in its manifest.
2. **The clinical-general (CG) partition** is the dev proxy for the external panel. It is fixed now, before any run, and
   comprises:
   - the knowledge dev v2 (4,000 MedQA and MedMCQA train hold-outs);
   - the fast dev v2 items of the clinical, non-domain templates of mix v2: `medqa`, `medmcqa`, `pubmedqa`,
     `mmlu_medical`, `medquad`, `pubhealth`, `nfcorpus` and `trec_covid` families. List the exact template ids in the
     manifest.

   It must hold ≥ 4,000 items. The external panel itself never enters.
3. **Zero-shot pplx-flash on the CG partition** (its authors' code, as in osler_v2 W5) gives each variant's drift from its
   base.
4. **The five baselines and zero-shot pplx-flash on the new test templates** (X1, X3, X4). These are additional rows only.

### X7 — Stage A: recipe ablations (GPU, 4 runs, ~4 h each)

Each run: pplx base, osler_v2's C1 mix (or C1 + X2's replay for A3), 13,992 steps, the D40 recipe, the same seed, and
evaluations on fast dev v3.

| run | recipe |
|---|---|
| A0 | control: osler_v2's recipe (`kl_to_base: 0`) |
| A1 | `kl_to_base: 0.1`, the §3.2 anchors |
| A2 | `kl_to_base: 0.5`, the §3.2 anchors |
| A3 | `kl_to_base: 0`, plus X2's broad replay (35 %) |

**Choice rule (pre-registered; paired bootstrap within templates, 1,000 resamples, seed 0):**
- A variant **qualifies** if its MR partition minus A0's MR has a lower bound > −0.01 (A0 always qualifies).
- **Among qualifying runs, choose the one with the highest point estimate of (its CG accuracy − zero-shot pplx-flash's CG
  accuracy)**: the closest to its base on the clinical-general proxy.
- A difference under 0.002 from A0 is a tie, and A0 wins it.

Write the choice, every run's MR, CG, knowledge and general partitions, and their mean KL to the base on CG items to
`loops/osler_v3/stage_a.md` before Stage B starts.

If X2 found no broad replay, A3 is NOT MEASURED, and the rule runs over A0–A2.

### X8 — Stage B: data blocks on the chosen recipe (GPU, ~4 h each)

Same steps, recipe and keep rule as osler_v2 (D34): lower bounds MR > 0, general > −0.01, knowledge > −0.02 against the
last kept block. Also report the CG partition for every block (labelled, outside the rule). Every block adds one source to
the last kept block.

| block | adds |
|---|---|
| B0 | Stage A's chosen run (no new run) |
| B1 | X1 real sources (RAGTruth, DrugProt, MultiCaRe items if any) |
| B2 | X3 Gemma-crafted items (claims and restyled pages) |
| B3 | X4 charts |
| B4 | X5 MIMIC, only if its files exist before B3's decision |

Caps as before: no template above 8 % of a block; replay at least 20 % of every block. Write `loops/osler_v3/blocks.md`.

### X9 — Final Osler-4B (GPU, ~12 h)

- The kept mix, on the chosen recipe (with its KL setting).
- One pass, capped at 250,000 realised items, with 8 evaluations on fast dev v3.
- Temperatures fitted on dev and saved with the checkpoint.
- Selection by the pooled template macro (as before). Also record the CG partition at every evaluation (labelled).

### X10 — Evaluation and Gate O3 (GPU, ~5 h)

Score the final model, calibrated (via `load_any`), on:
- the MR suite v1, verdict and reported-only;
- the held-out files;
- the external panel;
- the knowledge test;
- the new test templates (additional).

**Apply Gate O3 (D35, D39, D44) exactly.**

Additional rows (labelled):
- osler_v2's Osler-4B on every scope;
- the panel per dataset, against both baselines;
- panel superiority;
- every Stage A run on the panel. Score all four on the panel test **after** the final verdict is written, for the
  trade-off curve; never used for a choice.

Write `gate_o3.md` and `results.md`.

### X11 — Memorisation audit (only with MIMIC in the final mix). X12 — Findings and closure — HARD STOP

FINDINGS:
- Summary first;
- which §2 outcomes apply;
- the Stage A trade-off;
- the five-claim spot-check.

NEXT.md. Then `Loop status: STOPPED`, commit, push, stop.

## 5. Task table (seed STATE.md with this)

| id | task | GPU | deps |
|---|---|---|---|
| X0 | Orientation, KL timing | small | — |
| X1 | Real sources: RAGTruth, DrugProt, MultiCaRe | no | X0 |
| X2 | Broad replay for A3 | no | X0 |
| X3 | Gemma-crafted items over real text | yes | X0, X1 |
| X4 | Charts with the D44 rule | yes | X0 |
| X5 | MIMIC (only when the official files exist) | no | X0 |
| X6 | Fast dev v3, CG proxy, new baselines | yes | X1, X2, X3, X4 (X5 if built) |
| X7 | Stage A: recipe ablations A0–A3 + choice | yes | X6 |
| X8 | Stage B: data blocks | yes | X7 |
| X9 | Final Osler-4B | yes | X8 |
| X10 | Evaluation + Gate O3 | yes | X9 |
| X11 | Memorisation audit (only with MIMIC) | small | X9 |
| X12 | Findings and closure — HARD STOP | no | all |

Order:
- X1, X2 and X5 are CPU tasks and may overlap GPU tasks.
- X3 and X4 are GPU tasks (Gemma) and run one at a time before X6.
- If X3 or X4 is blocked, X6 proceeds without its templates, and the block is NOT MEASURED.

## 6. Out of scope

- Osler-9B; architecture, attention or base changes; adapter variants beyond D33.
- Hosted teachers; unofficial copies of credentialed data.
- Training on any test split, the external panel (or any copy of a panel dataset), MDACE, held-out items or the knowledge
  dev v2.
- Releasing or uploading anything.

## 7. Failure modes to avoid

1. **Stage A at block length may not predict drift at final length.** The final run is 3× longer. Record the CG partition
   at every final evaluation and report the trend. Never switch recipes mid-run.
2. **A KL anchor set that includes domain templates holds back the domain gain.** Check the anchor share per block against
   the §3.2 prefixes.
3. **Instruction-labelled claims whose verifier shares the writer's blind spots.** Report the agreement rate per edit kind,
   and spot-read 30 kept and 30 rejected items.
4. **Restyled pages that keep the source's tell-tale formatting** (journal style, citations). Screen with Naive Bayes on
   dev; above 0.9 is a defect for that template.
5. **RAGTruth's three task types share contexts.** Split by context id, never by row.
