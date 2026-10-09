# ADVISORY — osler_v0: competitor scoreboard, clinical generators, and the first Osler-4B / -9B

**Read `AGENTS.md` first** (repo root). It carries the evidence rules (R1–R10), the
readout-health gate (D12, amended for trained readouts by D21), guardrails, and loop mechanics.
This document is the work plan. Read it in full at the start of every loop iteration.

**Status:** authored 2026-10-09 from the operator + advisor review of `student_v1`.
Program context: `docs/plans/PROGRAM.md` (decisions D20–D24 were added for this loop).

---

## 1. Where things stand

From `loops/student_v1/` (claims `V###`):

- Three matched heads on `Qwen/Qwen3.5-0.8B` (LoRA + pointer / letter readout / Unsloth joint
  head, 15,000 steps, 482,889 rows over 24 templates) finished within 0.6 pts of each other on
  overall test accuracy: A 0.7591, C 0.7545, B 0.7528 (V113). **The head was not the lever at
  0.8B.**
- All pass G1 on seen fresh templates; **none transfers to held-out templates** (A and C FAIL;
  B NOT MEASURED because D12 failed on label mass). Held-out skill falls as training goes on
  (`loops/student_v1/trajectory.md`).
- **Decision-path knowledge collapsed**: MedQA / MedMCQA fell to about the majority-class rate on
  every arm, below zero-shot 0.8B (`loops/student_v1/arms.md`).
- Arm B's D12 failure was the readout normalising over the full vocabulary: the top token was an
  offered letter in 152/152 probe items, but 20–33 % of the mass sat on non-offered letters
  (V100, V101). D21 fixes the gate for trained readouts; D23's separate head removes the cause.
- The training data's option order carried a strong position prior (V085). Every arm here
  permutes option order for every question type, `noul` included.
- Open items carried from `loops/student_v1/NEXT.md`: HLE not scored; option-shuffle flip rate
  not measured; long-record slice membership not stored; prompt cap truncates 5.9 % of test
  items (three openFDA templates); the record–claim designs and catalog sources of V4 were never
  built; JEV-9B has 2,192 items over its cap.

**New since student_v1 (external; vendor claims, unverified until re-measured — R1):**

- **lion-ai MedDecider** 4B / 9B / 27B / 31B (CC BY-NC 4.0): rank-32 LoRA on Qwen3.5-4B / -9B,
  Qwen3.8-27B, gemma-4-31b-it; next-token option-letter readout; both option orders averaged;
  one temperature per question type; one epoch over ~119k decisions including **code-generated
  clinical notes with minimal-pair twins**, robustness items (none-of-the-above, planted
  instructions), NLI sets, MedQA/MedMCQA, DDXPlus and general replay (MNLI, BoolQ, ARC,
  CommonsenseQA). Its card reports an external panel of unseen exams and unseen clinical problem
  types; we re-measure on our own items.
- **perplexity-ai pplx-decider-v1.1-27b** (Apache-2.0): Qwen3.8-27B with **non-causal attention
  in the full-attention layers** (linear-attention layers unchanged) and a **separate `[255, d]`
  readout** whose row i maps to vocabulary token `token_ids[i]`; temperature applied once,
  softmax over the valid candidates of the current decision; extra human-labelled training data
  from tasksource.
- **edgeevals.ai System One board:** a private 669-case clinical decision suite (triage, notes,
  criteria, coding; severe errors zero a group; a 6-perturbation robustness pack). We cannot run
  it; we build our own equivalents of its suite shapes and perturbations and say so.

## 2. The question this loop answers

**Can a 4B decision model trained on gold-by-construction clinical decisions plus diverse
human-labelled data beat the best open medical decision model of its size, without losing the
base model's knowledge, and which readout should the Osler series use?**

Outcomes and what they decide:
- **Osler-4B and Osler-9B pass Gate O1 (D24)** → the recipe is the Osler recipe; the next loop
  adds teacher data (loop 2) and the release work.
- **Pass on our fresh tier, fail on the external panel** → we learned our benchmark, not
  medicine; the next loop is data breadth (teacher, more generators), not scale.
- **Knowledge guard fails** → replay ratio / LoRA scope / early stopping is the next loop's
  first job, before any scaling.
- **Arms within noise at 4B** → keep the option-code readout (simplest to serve) and say so.
- **Non-causal arm wins clearly** → record its serving cost (custom attention mask; no stock
  GGUF path) beside its gain; the operator decides.

## 3. Ground rules specific to this loop

- **Branch:** `loop/osler_v0` (cut from `dev`). Never commit to `main`, `dev` or another loop
  branch.
- **Write paths:** `src/`, `scripts/` (add files; do not change `run_loop.sh` or `pod_env.sh`),
  `tests/`, `configs/`, `envs/`, `tools/`, `docs/benchmark/`, `loops/osler_v0/`,
  `outputs/osler_v0/`, `data/` (new subdirectories only), root `pyproject.toml` / `uv.lock` /
  `.gitignore` (append only). **Read-only:** `AGENTS.md` (except O0's "Started" field),
  `docs/plans/`, every earlier `loops/*/`, `data/bench/v0.1/`, `data/bench/v0.2/`,
  `data/train/student_v0/`, `data/train/student_v1/`, earlier `outputs/*/`.
- **Benchmark v0.2 unchanged.** New evaluation sets (external panel, robustness pack, held-out
  generators) are **added** under `data/bench/v0.3_ext/` with their own manifest; v0.2 numbers
  stay comparable with every earlier loop.
- **Held-out (never trained on, never used for selection or temperatures):** the four D14
  templates (`ct_phase_choice_v1`, `fda_boxed_warning_noul_v1`, `pubmed_humans_noul_v1`,
  `ct_arm_role_noul_v1`) and the fields they ask about; the **held-out generators** fixed in O3
  before any model is trained; every dataset in the external panel (O1) — none of its splits,
  train included, enters training.
- **Model selection and temperature fitting** use seen-template dev items and seen-generator dev
  items only.
- **Training sources (D13, D17, D20):** tier-1 official train splits; pre-window structured-gold
  items; catalog datasets with human or structured labels and a training-compatible licence;
  gold-by-construction generated decisions (no LLM writes text or labels); general
  human-labelled decision data as replay. Never LLM-labelled gold, teacher outputs, UMLS / MIMIC,
  or any dataset in the external panel.
- **Competitors run through their authors' inference code** (each in its own `envs/<name>/` if
  its pins conflict with ours), with their own prompts, readouts, option-order averaging and
  temperatures. Where an author protocol cannot be reproduced within the timebox, the row is
  `NOT MEASURED — <reason>`; never substitute our harness silently. Our harness's zero-shot
  reading of a competitor may be reported only as an **additional**, labelled row.
- **Licences:** MedDecider is CC BY-NC — evaluation only, never a training signal or an
  initialisation. pplx-decider is Apache-2.0 — evaluation only in this loop as well.
- **Matched 4B arms (D23):** same mix, same item order seed, same option permutation, same
  example budget, same base revision, same prompt cap, same evaluation. Head-specific
  hyperparameters are recorded per arm.
- **GPU:** one RTX PRO 6000 (96 GB). One GPU job at a time; CPU work may overlap. No
  sub-agents for GPU work. Poll long jobs at ≥ 20-minute intervals.
- **Timeboxes:** ~2 h per integration problem (each competitor counts as one). Training arms
  have fixed example budgets, not wall-clock boxes.
- **Hard stop at O13.** Do not start the next loop.
- **Wording:** never name application domains outside medicine, industries, companies or
  products in the repository, except the baseline models and tools by their model ids.

## 4. Tasks

Each task: mark IN_PROGRESS (`date -u +%FT%TZ`), fill the STATE checklist, do the work, run the
acceptance check, self-audit (R6) into `outputs/osler_v0/<task>/SELF_AUDIT.md`, append CLAIMS
rows (`O###`), mark DONE/BLOCKED, log, commit
(`loop(osler_v0): O<n> <DONE|BLOCKED> — <headline>`), push `loop/osler_v0`.

### O0 — Orientation, snapshot, environments, throughput (smoke GPU, ~2 h)

**Do:** fill "Run started" in STATE and "Started" in AGENTS.md. Snapshot git commit, v0.2
manifest hash, training-mix hashes. Download (to `HF_HOME`) and record revisions of
`Qwen/Qwen3.5-0.8B`, `-4B`, `-9B` and every competitor in O2. For each competitor, install its
inference code in its own env if needed and run its card's example on the GPU (smoke). Record
from `Qwen/Qwen3.5-4B`'s config which layers are full attention and which are linear attention
(needed by arm N). Measure training throughput at 4B and 9B (LoRA r=32, bf16, batch of 8 at the
mix's median length, 50 steps) and write the projected wall-clock of one O6 arm.

**Acceptance:** `outputs/osler_v0/O0/{snapshot,envs,throughput}.json`; one smoke row per
competitor (PASS or `BLOCKED — <reason>`).

### O1 — External clinical panel + robustness pack (CPU + network, ~5 h)

**Why:** our fresh tier is home ground for us; the claim against competitors needs neutral
ground and the perturbations a deployed decision model meets.

**Do:**
1. **External panel** (test-only, never trained on): public medical sets with human or
   structured labels and a licence allowing evaluation, converted to typed decisions — unseen
   exams (e.g. MedXpertQA text, MMLU-Pro health, Medbullets, MedExQA, MedConceptsQA) and unseen
   clinical problem types (e.g. NLI4CT, PubMed 200k RCT sentence roles, adverse-drug-effect
   detection, symptom → diagnosis, medical question pairs, MedQuAD question types). Per
   dataset: licence, revision, split used, item count, converter test. Subsample deterministic
   and recorded, ≤ 1,000 items per dataset. Check none overlaps the training catalog or any
   training mix (record-id and normalised-text hash).
2. **Robustness pack** over a fixed, recorded sample of v0.2 fresh test items and the external
   panel (≤ 4,000 base items): (a) option order reversed; (b) question re-worded by a fixed
   template paraphrase table (no LLM); (c) irrelevant text padded into the state (sentences
   from unrelated records of the same source); (d) a planted instruction in the state telling
   the model to pick a wrong option; (e) the gold option replaced by a distractor plus a
   "none of these" option that becomes gold; (f) the same item repeated with a different item
   id (determinism). Each perturbation records the base item id.
3. **Fixes carried from student_v1:** store the long-record slice membership for v0.2 (ids
   only); flag the three openFDA templates whose items exceed 8,192 tokens and record the cap
   each model is evaluated at.

**Acceptance:** `data/bench/v0.3_ext/` + manifest (per dataset × qtype counts, licences, hashes);
overlap check 0; unit tests for every converter and perturbation (gold preserved or moved as
specified).

### O2 — Competitor scoreboard (GPU, ~15–25 h) — the first GPU job of this loop

**Do:** run on v0.2 test, the external panel and the robustness pack, each through its authors'
code: `lion-ai/MedDecider-4B`, `-9B`, `-27B`, `-31B`; `perplexity-ai/pplx-decider-v1.1-27b`
(its non-causal mode as saved); JEV-27B; Clef and Clef-Flash (if open weights are published).
Also zero-shot `Qwen/Qwen3.5-4B` and `-9B` (our harness, D12) on the external panel and the pack,
and JEV-9B on the external panel and the pack. Report coverage per set and the reason for every
skipped item. Hosted APIs are out of scope (no keys provisioned): `NOT MEASURED`.

**Acceptance:** `loops/osler_v0/scoreboard.md`: per model × set: n, accuracy, macro, Brier,
ECE, coverage, wall-clock per 1k items; robustness flip rates; every number in CLAIMS.

### O3 — Clinical decision generators, gold by construction (CPU, ~6 h)

**Why (D20):** the decision types users ask about — what a note says about a patient, whether a
patient meets a criterion, what a written policy implies, which code applies — have no
structured-field source at scale; code can build them with exact labels.

**Do:** a generator library in `src/meddecide/gen/` that builds a state from structured parts
(problems, medications, allergies, labs with units and reference ranges, vitals, timing,
family history, negations, social history) and computes the label from the same parts.
Families (≥ 8 generators across them):
- **Note facts with minimal-pair twins:** the twin changes one fact (negated vs affirmed,
  patient vs family member, past vs current, allergy vs current medication, value inside vs
  outside a range) and the gold flips. Both twins are in the same split.
- **Criterion eligibility:** a criterion list + a note → meets / does not meet / **insufficient
  information** (the note lacks the deciding fact).
- **Policy triage:** a written policy (thresholds, red flags) stated in the state + a
  presentation → the level the policy assigns (`score`).
- **Code assignment:** a short code list with definitions + a note → the code, or **none of
  these**.
Surface variation by template tables (section order, abbreviations, units, lengths, filler);
no LLM text. **Hold out ≥ 2 generators entirely** (one note-fact family, one of eligibility /
triage / coding) — fixed in `loops/osler_v0/heldout.md` before O6 starts. Each generator passes
the screen on its dev split (gold-in-state check; a string-presence / BoW baseline macro <
0.90; minimal-pair twins defeat any single-token rule).

**Acceptance:** generators with unit tests (label recomputed from parts; twin flips gold);
screen report; train / dev / test counts per generator; held-out list committed.

### O4 — Training mix v2 (CPU, ~4 h)

**Do:** the student_v1 mix (`data/train/student_v1/`, read-only) + O3 seen generators + the V4
items student_v1 did not build (≥ 2 record–claim consistency designs; catalog
`train-candidate` sources meeting D17) + **general human-labelled replay** (permissive licence,
not in the external panel; e.g. NLI, boolean QA, science and commonsense multiple choice) at a
recorded share (default 20 %) + robustness augmentation on training items (option order, padding,
planted instruction, none-of-these; ≤ 15 % of the mix). No template or generator above 8 % of
the mix. Dev for selection = seen-template v0.2 dev + seen-generator dev + replay dev,
template-stratified, ≥ 6,000 items. Leakage check as student_v1 V4, extended to the external
panel and held-out generators.

**Acceptance:** `data/train/osler_v0/{train,dev}.jsonl` + manifest (source × template × qtype
counts, licences, provenance); leakage check 0 on every kind.

### O5 — Readouts: option-code head, non-causal mode, export (GPU small, ~4 h)

**Do:** in `src/meddecide/model/`:
1. **Option-code head (D23):** a `[K, d]` readout (K = 255; codes A–Z then two-letter codes,
   each one token in the base tokenizer — verify), rows initialised from `lm_head` rows of the
   code tokens; logits gathered for the offered codes only; one temperature per qtype applied
   once. Unit test: at initialisation, its probabilities equal the zero-shot letter readout
   renormalised over the offered letters (tolerance 1e-4, fp32).
2. **Export:** write the trained head into a copy of `lm_head` (row i → token id i) with the LoRA
   kept separate (and, as a second artefact, merged); test that the exported causal LM gives the
   same offered-code probabilities as the native path (tolerance 1e-3, fp32, 50 items).
3. **Non-causal option (arm N):** a flag that makes the full-attention layers bidirectional on
   the decision path, leaving linear-attention layers unchanged; unit test that the causal path
   is untouched when the flag is off.
4. **Pointer head and head-only** at 4B: the student_v1 pointer head (padding fix kept) and a
   mode that trains the option-code head with no LoRA.
5. Padding invariance test on each path (batch-1 vs padded batch, tolerance 1e-3), and
   generation byte-identity with the adapter off.

**Acceptance:** all tests pass; `outputs/osler_v0/O5/readout_checks.json`.

### O6–O9 — Four matched Osler-4B arms (GPU)

Common recipe (all four): base `Qwen/Qwen3.5-4B` (revision recorded), bf16, base frozen, mix v2,
same order seed and option permutation (every qtype), prompt cap 16,384, CE + 1.0·Brier over
the offered options, LoRA r=32 alpha 32 on all linear layers (not for H), LR 1e-4 LoRA / 1e-3
head, 3 % warmup, cosine to 10 %, **example budget = one pass over mix v2 or 200,000 examples,
whichever is smaller** (record which), dev eval every 2,000 examples × 8, checkpoint at every
eval, selection by dev macro (Brier tie-break), temperatures per qtype on dev. Grad-norm and
batch-max-length logged per step; divergence tripwire as student_v1 (dev macro < 0.50 on two
consecutive evals, or pre-clip grad norm > 5,000 — then stop, record, use best saved checkpoint).
Also log tier-1 dev accuracy (MedQA / MedMCQA dev) at every eval, as a diagnostic only.

- **O6 — Arm L: option-code head, causal** (the default).
- **O7 — Arm P: pointer head.**
- **O8 — Arm N: option-code head, non-causal full-attention layers.**
- **O9 — Arm H: option-code head only, no LoRA** (head LR 1e-3; all else equal).

**Acceptance for each:** selected checkpoint, dev trajectory, temperatures, throughput,
wall-clock, grad-norm tail (p50 / p95 / max), tier-1 dev trajectory.

### O10 — Head choice at 4B (no GPU, ~1 h)

Pre-registered rule, dev only (never test, never held-out): start with L. Switch to P or N only
if its dev macro exceeds L's by ≥ 1.0 pt with the paired bootstrap CI lower bound > 0 on dev;
if both qualify, take the larger gain. Switch to H if its dev macro is within 0.5 pt of L's (it
leaves the decision path's knowledge untouched). Write `loops/osler_v0/head_choice.md` with the
numbers and the rule's verdict.

### O11 — Osler-9B and Osler-0.8B with the chosen head (GPU)

The O6–O9 recipe on `Qwen/Qwen3.5-9B` and `Qwen/Qwen3.5-0.8B` (r=32 at 9B; r=16 at 0.8B, as a
recorded deviation if memory or speed requires), the chosen head, same mix and budget rule.

### O12 — Evaluation and Gate O1 (GPU, ~6 h)

Evaluate every arm (O6–O9) and O11 model on v0.2 test, the external panel, held-out generators
and the robustness pack, in **both option orders averaged** and in single order (both reported;
the gate uses both-order averaged for every model that supports it, as the competitors do).
Then:
- **Gate O1 (D24)** for Osler-4B (the chosen arm) and Osler-9B — verdicts, every paired
  difference with CI and item count, the knowledge guard.
- **Additional (never in a verdict):** every O2 model; arm-vs-arm at 4B; Osler-0.8B vs zero-shot
  0.8B, JEV-9B and the student_v1 arms; tier 1, HLE (supplementary), long-record slice,
  per-template tables, selective accuracy at 50 / 80 / 90 % coverage, auto-accept rate at 95 % and
  99 % precision (thresholds fitted on dev, applied to test), latency per question and per
  record, and Osler-4B quantised to ≤ 4 GB (GGUF Q4_K_M / Q5_K_M via the O5 export) agreement
  and accuracy — or `NOT MEASURED — <reason>`.

**Acceptance:** `loops/osler_v0/gate_o1.md` and `loops/osler_v0/results.md`; every number in
CLAIMS.

### O13 — Findings and closure (no GPU, ~2 h) — HARD STOP

Write `loops/osler_v0/FINDINGS.md` (Summary first; state which §2 outcome applies),
`NEXT.md` (proposals), fill STATE's closure feed. Closure requirements: every STATE row `DONE`
or `BLOCKED — <reason>`; the five-claim spot-check appended with re-run outputs; every number
has a CLAIMS row. Set `Loop status: STOPPED`, commit, push, stop.

## 5. Definition of done

- STATE.md — every task `DONE` or `BLOCKED` with a reason.
- CLAIMS.md — every reported number, with artifact and recompute command.
- FINDINGS.md — Summary first, full-coverage body, five-claim spot-check appended with outputs.
- Committed: code, tests, configs, env project files (no weights), the loop's markdown reports.
  Not committed: items, generated data, training data, predictions, checkpoints.

## 6. Task table (seed STATE.md with this)

| id | task | GPU | deps | status |
|---|---|---|---|---|
| O0 | Orientation, snapshot, envs, competitor smoke, throughput | smoke | — | PENDING |
| O1 | External clinical panel + robustness pack | no | O0 | PENDING |
| O2 | Competitor scoreboard | yes | O0, O1 | PENDING |
| O3 | Clinical generators (gold by construction) + held-out list | no | O0 | PENDING |
| O4 | Training mix v2 | no | O1, O3 | PENDING |
| O5 | Readouts: option-code head, non-causal mode, export | small | O0 | PENDING |
| O6 | Arm L: option-code head (4B) | yes | O4, O5 | PENDING |
| O7 | Arm P: pointer head (4B) | yes | O4, O5 | PENDING |
| O8 | Arm N: non-causal option-code head (4B) | yes | O4, O5 | PENDING |
| O9 | Arm H: head only, no LoRA (4B) | yes | O4, O5 | PENDING |
| O10 | Head choice at 4B (dev rule) | no | O6–O9 | PENDING |
| O11 | Osler-9B + Osler-0.8B reference | yes | O10 | PENDING |
| O12 | Evaluation + Gate O1 | yes | O2, O11 | PENDING |
| O13 | Findings and closure — HARD STOP | no | all | PENDING |

Execution order is the table order, except: O3 and O4 (CPU) may run while O2 uses the GPU; O5
may use the GPU between O2 jobs. A BLOCKED arm in O7–O9 never blocks O10 (the rule runs over the
arms that finished); a BLOCKED O6 blocks O10 and O11 (L is the reference arm). A BLOCKED
competitor never blocks anything.

## 7. Explicitly out of scope

| item | why |
|---|---|
| Teacher (DeepSeek) labels, UMLS / ontology synthetic data | loop 2 |
| LLM-written training text or labels of any kind | D17, D20 |
| MIMIC / PhysioNet data | later loop |
| Hosted competitor APIs (Jev, Decisions APIs, Workers AI) | no keys provisioned; `NOT MEASURED` |
| Training an Osler model above 9B | after this loop's review |
| MedDecider or pplx outputs / weights as training signal or initialisation | licence (MedDecider) and independence of the comparison |
| Any change to v0.2, D12 for zero-shot cells, D14, D16, D24 or the O10 rule after results | settled; disagreements go to questions-for-operator |
| RLCD / RL objectives, adaptive thinking | later ablations |
| Publishing anything, pushing to the HF Hub, submitting to external leaderboards | operator decision after review |

## 8. Failure modes to avoid

1. **Home-ground win only.** A gain on our fresh tier that does not appear on the external panel
   is reported as exactly that (§2 outcome 2), never as "best in class".
2. **Unfair competitor rows.** A competitor run outside its authors' protocol understates it.
   Use their code, prompts, both-order averaging and temperatures, or mark `NOT MEASURED`.
3. **Knowledge collapse again.** Watch the tier-1 dev trajectory every eval; it is a diagnostic,
   never a selection signal, but a fall below zero-shot is written into STATE as it happens.
4. **Generator shortcuts.** A generator a string rule or a single token solves teaches a
   shortcut; minimal-pair twins and the screen are mandatory.
5. **Held-out leakage.** Held-out generators, D14 templates and the external panel never touch
   training, selection or temperatures; any touch voids those results.
6. **Unmatched arms.** Record every difference between O6–O9 in a provenance table.
7. **Position prior.** Option permutation for every qtype in every arm (V085).
8. **Closure gaps; parallel GPU jobs; overwriting a running chain script** (see the memory note
   in `loops/student_v1/NEXT.md` §5).
9. **Relabelling an unwelcome result (R4); citing artifacts that do not exist (R2); impossible
   arithmetic (R5).**

## 9. If you finish early

1. A second seed of Osler-4B for seed variance on the gate differences.
2. Osler-4B at two example budgets (half, double) to see whether held-out and tier-1 move.
3. Arm L with a replay share of 35 % if the knowledge guard failed.
