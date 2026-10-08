# ADVISORY — student_v1: correct G1, fix the padding bug, diversify the data, and compare three decision heads on held-out templates

**Read `AGENTS.md` first** (repo root). It carries the evidence rules (R1–R10), the
readout-health gate (D12), guardrails, and loop mechanics. This document is the work plan.
Read it in full at the start of every loop iteration.

**Status:** authored 2026-10-08 from the operator + advisor review of `student_v0`.
Program context: `docs/plans/PROGRAM.md` (decisions D17–D19 were added for this loop).

---

## 1. Where things stand

From `loops/student_v0/` (claims `S###`; read the **advisor note at the top of its
FINDINGS.md** first — it corrects that loop's headline):

- **MedDecide-0.8B** (frozen `Qwen/Qwen3.5-0.8B` + LoRA r=8 + pointer head, CE + Brier, warmup
  + cosine) was trained for one full pass of 212,481 gold items (S084). The selected checkpoint
  is **step 1,000**; the converged step-44,000 checkpoint is worse on test (S082).
- **Against zero-shot `Qwen/Qwen3.5-0.8B`** (letter readout, D12): on **seen** fresh templates
  macro accuracy **0.8894 vs 0.7262** (+0.1633 [+0.1530, +0.1741]) and Brier **0.1286 vs
  0.3584**; on **held-out** fresh templates macro **0.5544 vs 0.5720** (−0.0176 [−0.0301,
  −0.0048]) (S093/S094). Per held-out template: `pubmed_humans` +0.091, `fda_boxed_warning`
  +0.007, `ct_arm_role` −0.079, `ct_phase` −0.090.
- **Tier-1 knowledge MCQ did not improve** despite MedQA/MedMCQA train splits in the mix: MedQA
  0.393 vs zero-shot 0.405, MedMCQA 0.362 vs 0.378 (S093).
- **G1 per D16 was never computed**: `g1.md` used the Base ablation as a baseline; JEV-9B on
  v0.2 is `NOT MEASURED` (S095).
- **Known defects:** trained-head logits depend on batch composition (fp32 2.51e-2, 17.9× the
  control; S091); the gradient tail is length-driven (S090); 0 `score` training items (S053);
  no PubMed pre-window data (S058); the 2,019-item dev sample inverted a model ranking (S079).
- **Assets on disk:** 88 checkpoints of run 3 (`outputs/student_v0/S9_run3/checkpoints/`) and
  of the Base ablation (`outputs/student_v0/S10/checkpoints/`), the v0.2 benchmark
  (`data/bench/v0.2/`), the training mix (`data/train/student_v0/`).

**New since student_v0:** Unsloth ships a decision-model trainer (`FastDecisionModel`,
`DecisionTrainer`) with a **Clef-style joint head** — the prompt holds the input followed by
every question and its options, the LLM reads it once, and a small head scores every option of
every question jointly. Its documented defaults: LoRA r=16, LR 2e-4, **head LR 1e-4**, cosine,
long inputs truncated from the end of the input. Docs:
`https://unsloth.ai/docs/basics/train-your-own-decision-model-with-unsloth.md`. Its published
example dataset `LocalLLaMA/typed-decisions` (Apache-2.0; 1,200 train / 400 test; business
workflows; gold = teacher distributions) is **not medical and has LLM-derived gold**, so it is
used here **only** to validate the Unsloth pipeline, never as MedDecide training data.

## 2. The question this loop answers

**Why does the trained model not transfer to held-out decision types, and does the choice of
decision head matter?** Three heads, trained on the same diversified gold data with the same
budget, are compared on the same held-out templates:

- **A — pointer head** (ours, padding bug fixed),
- **B — letter-readout LoRA** (plain fine-tuning of the option-letter logits; the "no special
  head" control),
- **C — Clef-style joint head via Unsloth**.

Outcomes and what they decide:
- **Any arm beats zero-shot on held-out templates (D16 rule)** → that arm's head + the diversified
  data is the recipe to scale (loop 3 ladder; teacher data in loop 2).
- **All arms fail on held-out, all pass on seen** → the limit is data diversity, not the head;
  the next loop brings the teacher (diverse real documents, soft labels) forward.
- **B ≈ A ≈ C everywhere** → the head is not the lever; keep the simplest (B) and say so in the
  paper.
- **The step-trajectory (V3) shows held-out accuracy falling with training** → template
  over-fitting; the next loop uses early stopping on a held-out-like dev signal and more
  templates per step.

## 3. Ground rules specific to this loop

- **Branch:** `loop/student_v1`. Never commit to `main` or any other loop branch.
- **Write paths:** `src/`, `scripts/` (add files; do not change `run_loop.sh` or
  `pod_env.sh`), `tests/`, `configs/`, `envs/`, `tools/`, `docs/benchmark/`,
  `loops/student_v1/`, `outputs/student_v1/`, `data/` (new subdirectories only), root
  `pyproject.toml` / `uv.lock` / `.gitignore` (append only). **Read-only:** `AGENTS.md` (except
  V0's "Started" field), `docs/plans/`, every earlier `loops/*/`, `data/bench/v0.1/`,
  `data/bench/v0.2/`, `data/train/student_v0/`, `outputs/student_v0/`.
- **Benchmark:** **v0.2 unchanged** — no new test templates in this loop, so every number is
  comparable with student_v0. New *training* templates are train-only (pre-window records).
- **Held-out templates (D14) stay the same:** `ct_phase_choice_v1`,
  `fda_boxed_warning_noul_v1`, `pubmed_humans_noul_v1`, `ct_arm_role_noul_v1`. No item of these
  templates, from any window, in any training mix, dev selection set, or temperature fit.
- **Model selection and temperature fitting use seen-template dev items only.** Held-out dev
  items may be scored for the V3 trajectory analysis and reported, but never used to select a
  checkpoint, a configuration or a temperature.
- **Training sources (D17):** tier-1 official train splits; pre-window structured-gold items
  (records dated before 2026-03-01); and `train-candidate` datasets from
  `docs/benchmark/dataset_catalog.md` whose label provenance is **human** or **structured** and
  whose licence permits training. Never LLM-labelled gold (this excludes
  `LocalLLaMA/typed-decisions` from training), never UMLS/MIMIC, never teacher outputs.
- **Arms are matched (D18):** same training mix, same item order seed, same step budget, same
  base checkpoint (`Qwen/Qwen3.5-0.8B`), same max prompt length, same evaluation. Head-specific
  hyperparameters follow each head's documented defaults and are recorded.
- **Unsloth lives in its own environment** (`envs/unsloth/`, its own `uv` project, versions
  pinned) so it cannot change the main project's torch / transformers versions. Record both
  environments' versions in every arm's provenance.
- **GPU:** one RTX PRO 6000 (96 GB). One GPU job at a time; CPU work may overlap.
  **Do not spawn parallel sub-agents for GPU work.** Poll long jobs at ≥ 20-minute intervals.
- **Timeboxes:** ~2 h per integration problem (Unsloth included); each training arm has a fixed
  step budget (V6–V8), not a wall-clock box.
- **Hard stop at V10.** Do not start the next loop.
- **Wording:** describe templates as "record–claim consistency" / "document-grounded
  verification"; never name application domains, industries, companies or products in the
  repository (Unsloth and the baseline models are named as tools, which is fine).

## 4. Tasks

Each task: mark IN_PROGRESS (`date -u +%FT%TZ`), fill the STATE checklist, do the work, run the
acceptance check, self-audit (R6) into `outputs/student_v1/<task>/SELF_AUDIT.md`, append CLAIMS
rows (`V###`), mark DONE/BLOCKED, log, commit
(`loop(student_v1): V<n> <DONE|BLOCKED> — <headline>`), push `loop/student_v1`.

### V0 — Orientation, snapshot, Unsloth environment (~1 h)

**Do:** fill "Run started" in STATE and "Started" in AGENTS.md. Snapshot git commit, v0.2
manifest and training-mix hashes, test count. Create `envs/unsloth/` as a separate `uv` project
with `unsloth` (latest release that documents `FastDecisionModel`), pin it in its lockfile, and
verify on the GPU that `FastDecisionModel.from_pretrained("unsloth/Qwen3.5-0.8B")` (or
`Qwen/Qwen3.5-0.8B`) loads in **bf16** (`load_in_4bit=False`). Record versions of torch,
transformers and unsloth in both environments.

**Acceptance:** `outputs/student_v1/V0/{snapshot,envs}.json`; Unsloth load PASS or
`BLOCKED — <reason>` (if blocked, V5 and V8 are BLOCKED and the loop continues with arms A and
B).

### V1 — JEV-9B on v0.2, and G1 recomputed per D16 (GPU ~1 h)

**Why:** student_v0's verdict used the wrong baseline; the correct G1 has never been computed.

**Do:** run JEV-9B through its own protocol (as in `bench_v0_fix0` F7, fused kernels now bound)
on the full v0.2 test set; report coverage per template and the reason for every skipped item.
Make the baseline set in `scripts/bench/g1.py` explicit and default to D16 (`zeroshot`,
`jev9b`); an ablation may be passed only under an `--additional` flag and is never part of the
verdict. Recompute G1 for the student_v0 model (run 3, step 1,000) against zero-shot and JEV-9B,
seen and held-out separately.

**Acceptance:** `loops/student_v1/g1_student_v0_rerun.md` with the D16 verdict, every paired
difference with its CI and item count; a unit test proving an ablation passed without
`--additional` is rejected.

### V2 — Padding fix + its measured effect (GPU small, ~2 h)

**Do:** make the pointer head padding-invariant (e.g. gather option and answer states only from
real positions and make every hidden-state read independent of the pad layout; if the base's
linear-attention layers leak pad state, score without padding — length-uniform batches or batch
size 1 at evaluation). Turn the S9-diag padding probe into a unit test on a **trained**
checkpoint: fp32, 8 items, alone vs padded batch, tolerance 1e-3. Then quantify the bug's effect:
score student_v0's step-1,000 checkpoint on the v0.2 test set both ways (old batched path vs
fixed path) and report the per-template accuracy and Brier differences.

**Acceptance:** the trained-checkpoint padding test passes; `outputs/student_v1/V2/effect.json`
with both scorings; the fixed path is the default for every later evaluation.

### V3 — Checkpoint trajectory: does held-out skill rise or fall with training? (GPU ~1.5 h, no training)

**Do:** for every 4th of student_v0 run 3's 88 checkpoints (22 points, always including step
1,000 and the last), score with the V2 fixed path: (a) the seen-template dev sample, (b) **all**
v0.2 dev items of the four held-out templates. Plot and tabulate accuracy, macro accuracy and
Brier vs step for seen and held-out separately. Report the Spearman correlation between step and
held-out accuracy, with a bootstrap CI. This is analysis only — nothing is selected from it.

**Acceptance:** `loops/student_v1/trajectory.md` (table + description); the figure under
`outputs/student_v1/V3/`.

### V4 — Training data v1: diversity (CPU + network, ~5 h)

**Why:** student_v0 learned its seen templates and did not transfer; ADVISORY §2 branch 2 says
data diversity comes first.

**Do:**
1. **PubMed pre-window** (missing in student_v0, S058): build from a **deterministic, recorded**
   subset of PubMed baseline files (choose the file count to fit ~2 h; record the file names and
   hashes). Recorded subsampling is allowed; silent subsampling is not.
2. **More decision types, train-only.** Add **≥ 8 new pre-window templates** across
   ClinicalTrials.gov, openFDA and PubMed using structured fields (examples: masking, allocation
   ratio class, enrollment-size band as `score`, primary purpose, intervention type, product
   type, route, MeSH check tags such as age group and sex — **not** the humans/animals tag —
   publication type), plus **≥ 2 new record–claim consistency designs** that follow the S2
   role-binding rules (not arm role, which is held out). None may ask about a field a held-out
   template asks about. Each passes the screen on its pre-window dev split (gold-in-
   state check, string-presence baseline for claim templates, BoW macro < 0.90).
3. **`score` items:** from a source with graded human or structured labels in the catalog, or
   structured ordinal fields (e.g. enrollment-size bands). **Never** a field a held-out template
   asks about (trial phase, boxed warning, humans check tag, arm role) — not even re-phrased as
   a different question type. Target ≥ 5,000 train and ≥ 200 seen-template dev items;
   otherwise `NOT MEASURED — <reason>`.
4. **Catalog sources:** add the catalog's `train-candidate` datasets that meet D17 (human or
   structured labels, permissive licence), converted to typed decisions, each recorded with
   licence and provenance.
5. **Mix:** tier-1 train splits + all of the above. **No template above 8 % of the mix**;
   record per-template counts. Dev for selection = seen-template v0.2 dev + new templates'
   pre-window dev, **template-stratified, ≥ 4,000 items**.
6. **Leakage check** (as S6, mechanical): no v0.2 test/dev record id or normalised state hash in
   training; no record dated ≥ 2026-03-01; no held-out template id or held-out question in
   training or the selection dev set.

**Acceptance:** `data/train/student_v1/{train,dev}.jsonl` + manifest (per source × template ×
qtype counts, licences, provenance); leakage check 0 on every kind; the number of distinct
training templates is reported and is ≥ 1.5× student_v0's.

### V5 — Unsloth pipeline validation and data conversion (GPU small, ~2 h)

**Do:** (1) In `envs/unsloth/`, run Unsloth's documented recipe on `LocalLLaMA/typed-decisions`
(`all` subset) with `Qwen3.5-0.8B` at **bf16**, `max_steps=300`, and report held-out accuracy
before/after on its test split. This only validates that the integration works (accuracy rises
well above its pre-training value); it is not a MedDecide result, and the model is discarded.
(2) Write a converter from MedDecide items to Unsloth's decision dataset format (state, questions
with type / instructions / criteria, gold), with a round-trip test (convert → Unsloth
`build_dataset` → count skipped items, 0 skipped expected; any skip counted with its reason).
(3) Write an evaluator that runs an Unsloth-trained model on v0.2 items and emits the same
prediction JSONL format as `run_student.py`, so `g1.py` scores it unchanged.

**Acceptance:** validation numbers in CLAIMS; converter and evaluator tests pass; an untrained
Unsloth head scores near chance on 200 v0.2 dev items with the same metrics code.

### V6 — Arm A: pointer head on data v1 (GPU, ~1.5 h)

Recipe: student_v0 run 3's (LoRA r=8, alpha 16, CE + 1.0·Brier, warmup 3 % + cosine, head LR
1e-3, LoRA LR 2e-4, batch 8, prompt cap 8,192) with the V2 padding fix. **Step budget 15,000**,
dev eval every 500 steps on the V4 dev set, checkpoint at every eval, selection by dev macro
accuracy (Brier tie-break), temperatures per qtype on seen dev. Log per-step grad norm with batch
max length.

### V7 — Arm B: letter-readout LoRA (GPU, ~1.5 h)

The same prompt as the zero-shot letter readout (D12 protocol), the same LoRA (r=8, alpha 16) on
the same modules, **no new head**: loss = cross-entropy over the option-letter logits (softmax
restricted to the offered letters) + 1.0·Brier; LoRA LR 2e-4, warmup + cosine; identical data,
order seed, step budget (15,000), eval cadence, selection rule and temperature fitting as arm A.
D12 applies to its evaluation (label mass, greedy agreement, below-chance, constant answer).

### V8 — Arm C: Clef-style joint head via Unsloth (GPU, ~1.5 h)

Unsloth `FastDecisionModel` + `DecisionTrainer` on `Qwen/Qwen3.5-0.8B` in **bf16**, its
documented defaults (LoRA r=16, alpha 16, LR 2e-4, head LR 1e-4, cosine), `max_seq_length`
8,192, identical training items (V5 converter), **step budget 15,000** (or one pass if smaller —
record which), eval every 500 steps on the V4 dev set, checkpoint at every eval, the same
selection rule, and Unsloth's own `calibrate` on seen dev only. Record any default that had to
change and why.

**Acceptance for V6–V8:** each arm has a selected checkpoint, its dev trajectory, temperature(s),
throughput, wall-clock, and the grad-norm tail (p50 / p95 / max); a run that diverges (dev macro
< 0.50 on two consecutive evals or pre-clip grad norm > 5,000) is stopped, recorded as diverged,
and evaluated at its best saved checkpoint.

### V9 — Evaluation and G1 per arm (GPU ~2 h)

**Do:** evaluate every arm's selected checkpoint on the v0.2 test set (fixed padding path for A;
D12 for B; the V5 evaluator for C). Then:
- **G1 per arm (D16)** vs zero-shot `Qwen/Qwen3.5-0.8B` and JEV-9B, on seen and held-out
  templates separately, plus the strict slice.
- **Arm-vs-arm paired comparisons** (A−B, A−C, B−C) on held-out and on seen templates, same
  paired bootstrap, item counts stated.
- Per-template tables, tier 1, long-record slice, HLE (supplementary), selective accuracy at 50 /
  80 / 90 % coverage, option-shuffle flip rate, latency per question and per record.
- Also report student_v0's model (step 1,000, fixed path) as a reference row, labelled as such.

**Acceptance:** `loops/student_v1/g1.md` (per-arm verdicts + arm-vs-arm) and
`loops/student_v1/arms.md` (all tables), every number with a CLAIMS row.

### V10 — Findings and closure (no GPU, ~2 h) — HARD STOP

Write `loops/student_v1/FINDINGS.md` (Summary first; state which §2 outcome applies, per arm),
`NEXT.md` (proposals), fill STATE's closure feed. **Closure requirements — check each before
stopping:** (1) every STATE row is `DONE` or `BLOCKED — <reason>` (student_v0 left two rows
`IN_PROGRESS`); (2) the five-claim spot-check is **appended with re-run outputs** (student_v0
never appended it); (3) every number has a CLAIMS row. Set `Loop status: STOPPED`, commit, push,
stop.

## 5. Definition of done

- STATE.md — every task `DONE` or `BLOCKED` with a reason.
- CLAIMS.md — every reported number, with artifact and recompute command.
- FINDINGS.md — Summary first, full-coverage body, five-claim spot-check appended with outputs.
- Committed: code, tests, configs, `envs/unsloth/` project files (no weights), the loop's
  markdown reports. Not committed: items, training data, predictions, checkpoints.

## 6. Task table (seed STATE.md with this)

| id | task | GPU | deps | status |
|---|---|---|---|---|
| V0 | Orientation, snapshot, Unsloth environment | smoke | — | PENDING |
| V1 | JEV-9B on v0.2 + G1 recomputed per D16 | yes | V0 | PENDING |
| V2 | Padding fix + measured effect | small | V0 | PENDING |
| V3 | Checkpoint trajectory (seen vs held-out) | yes | V2 | PENDING |
| V4 | Training data v1 (diversity) | no | V0 | PENDING |
| V5 | Unsloth validation + converter + evaluator | small | V0, V4 | PENDING |
| V6 | Arm A: pointer head | yes | V2, V4 | PENDING |
| V7 | Arm B: letter-readout LoRA | yes | V4 | PENDING |
| V8 | Arm C: Clef-style head via Unsloth | yes | V5 | PENDING |
| V9 | Evaluation + G1 per arm + arm-vs-arm | yes | V1, V6, V7, V8 | PENDING |
| V10 | Findings and closure — HARD STOP | no | all | PENDING |

Order of the table is the execution order, except: V4 (CPU/network) may run while V1–V3 use the
GPU. A BLOCKED V8 (Unsloth) never blocks V9 for arms A and B, or V10.

## 7. Explicitly out of scope

| item | why |
|---|---|
| Teacher (DeepSeek) labels, UMLS / ontology synthetic data | loop 2 (teacher endpoint not provisioned) |
| MIMIC / PhysioNet data | later loop |
| New **test** templates or any change to v0.2 | keeps every number comparable with student_v0 |
| Models above 0.8B; the Base checkpoint | this loop isolates head and data at one size |
| Training on `LocalLLaMA/typed-decisions` or any LLM-labelled dataset | D17; it validates Unsloth only |
| 4-bit training for the matched arms | all arms bf16, so precision is not a confound |
| Changing D12, D14, D16 or the selection rule after results | settled; disagreements go to questions-for-operator |
| Publishing anything, pushing to the HF Hub | operator decision after review |

## 8. Failure modes to avoid

1. **The wrong G1 baseline (student_v0's error).** G1 baselines are zero-shot and JEV-9B only;
   ablations and sibling arms are reported as *additional*, never in the verdict.
2. **Unmatched arms.** A difference between heads is meaningful only if data, order, steps,
   prompt cap and evaluation match. Record every difference in a per-arm provenance table.
3. **Held-out signal leaking into selection.** Held-out dev items are for the V3 analysis only;
   any use in selection, temperature fitting or hyperparameter choice voids the held-out result.
4. **Unsloth's truncation.** It cuts the *end of the input* to fit `max_seq_length`; check that
   the question and options survive for every item (count items where the state was cut) and
   report it beside arm C's numbers.
5. **Divergence again.** Keep the grad-norm/length log; apply the V6–V8 tripwire.
6. **Diversity without screening.** Every new training template passes the screen on its dev
   split; a template a string rule solves teaches a shortcut.
7. **Closure gaps (student_v0's error).** Rows left `IN_PROGRESS`, a spot-check promised and
   not appended.
8. **Parallel GPU agents** (student_v0: OOMs, contention, cost). One GPU job; no GPU sub-agents.
9. **Relabelling an unwelcome result (R4); citing artifacts that do not exist (R2); impossible
   arithmetic (R5).**

## 9. If you finish early

1. A second seed of the best arm (same data, different seed) for seed variance on the held-out
   differences.
2. The best arm at 30,000 steps, to see whether held-out accuracy keeps moving.
3. The GPU byte-identity check on the fused-kernel path for arm A (student_v0 NEXT §7).
