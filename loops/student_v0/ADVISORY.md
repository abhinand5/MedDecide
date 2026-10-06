# ADVISORY — student_v0: benchmark v0.2, a structured-gold training set, and the first trained MedDecide (0.8B)

**Read `AGENTS.md` first** (repo root). It carries the evidence rules (R1–R10), the
readout-health gate (D12), guardrails, and loop mechanics. This document is the work plan.
Read it in full at the start of every loop iteration.

**Status:** authored 2026-10-06 from the operator + advisor review of `bench_v0_fix0`.
Program context: `docs/plans/PROGRAM.md` (decisions D13–D16 were added for this loop).

---

## 1. Where things stand

Measured and trustworthy (all from `loops/bench_v0_fix0/`, claims X001–X039):

- **Benchmark v0.1** — `data/bench/v0.1/{tier1,fresh}/`: 22,594 tier-1 items (gold verified
  against raw sources, 0 mismatches of 15,915 checkable — X004) and 23,582 fresh items on the
  window 2026-03-01 → build date, class-balanced, with a strict slice of 2,896 items dated
  ≥ 2026-09-10 (X012–X014). 16 of 22 templates kept by the balanced screen (X017).
- **Harness** — letter readout for all three question types, validated against
  lm-evaluation-harness (`choice` X041 in bench_v0, `noul` X009), plus the D12
  readout-health gate (X011).
- **Zero-shot ladder** (`baselines_v0_1.md`, X039) and **decision-model baselines**
  (`decision_models_v0_1.md`, X030, X036–X038). Reference points for this loop, fresh tier,
  accuracy:

  | template | Qwen3.5-0.8B (zero-shot) | Qwen3.5-9B | JEV-9B |
  |---|---|---|---|
  | `ct_randomised_noul_v1` | 0.7955 | 0.8840 | 0.8765 |
  | `ct_healthy_volunteers_noul_v1` | 0.6035 | 0.7355 | 0.7200 |
  | `fda_boxed_warning_noul_v1` | 0.6157 | 0.7618 | 0.8254 (1,249/1,910 items) |
  | `pubmed_humans_noul_v1` | 0.6345 | 0.8678 | 0.8571 |
  | `ct_phase_choice_v1` | 0.3455 | 0.4327 | 0.4745 |
  | `fda_class_choice_v1` | 0.8400 | 0.9005 | 0.9451 (3,154/4,000 items) |
  | `pubmed_mesh_major_choice_v1` | 0.9872 | 0.9950 | 0.9960 |

**Known defects carried into this loop** (fix them first, S1):

1. `nfcorpus_graded_score_v1` offers a level no item in its pool has (X029) — every cell on
   it is invalid.
2. `pubmed_mesh_major_choice_v1` and `fda_class_choice_v1` do not discriminate between
   models (0.84–0.996 for everything from 0.8B to 9B): their distractors are unrelated,
   so the task is topic matching. They need near-miss distractors.
3. The D12 gate does not catch a **constant-answer** model: LFM2.5-350M sits at exactly
   0.50 on balanced `noul` templates and passes.
4. Prediction files are append-only across re-runs (P7 in `bench_v0_fix0/NEXT.md`).
5. Qwen3.5's fused kernels (`causal_conv1d`, `flash-linear-attention`) are not installed;
   that caused 30 GB allocations on long prompts, an 8,192-token cap on JEV-9B's openFDA
   cells, and ~3× slower inference (P4).

**Advisor note on `bench_v0_fix0/FINDINGS.md` Summary item 7:** its sentence "on the fresh
tier the same models are at ~0.50 on every `noul` template and near chance on `choice`" holds
only for LFM2.5-350M. From 0.8B up, models score 0.60–0.88 on fresh `noul` and 0.35–0.99
on fresh `choice` (table above). Use the table, not that sentence.

## 2. The question this loop answers

**Does the MedDecide recipe — a frozen base, a decision-path LoRA and a pointer head,
trained with cross-entropy + Brier on gold labels — add real skill over the zero-shot base
model and over an existing 9B decision model, including on decision types it never saw in
training?**

- **G1 PASS** (definition in S11) → scale up and add teacher + ontology data (loops 2–3).
- **G1 FAIL on held-out templates but PASS on seen ones** → the model learned templates,
  not decisions; the next loop's priority is data diversity, not scale.
- **G1 FAIL everywhere** → diagnose the recipe (readout, capacity, loss, data) at 0.8B before
  spending anything larger.

## 3. Ground rules specific to this loop

- **Branch:** `loop/student_v0`. Never commit to `main` or to `loop/bench_v0`.
- **Write paths:** `src/`, `scripts/` (add files; do not change `run_loop.sh` or
  `pod_env.sh`), `tests/`, `configs/`, `tools/`, `docs/benchmark/`, `loops/student_v0/`,
  `outputs/student_v0/`, `data/`, root `pyproject.toml` / `uv.lock` / `.gitignore`
  (append only). **Read-only:** `AGENTS.md` (except S0's "Started" field),
  `docs/plans/`, `loops/bench_v0/`, `loops/bench_v0_fix0/`, `data/bench/v0.1/`,
  `data/bench/{tier1,fresh}/`.
- **Benchmark v0.2** is written to `data/bench/v0.2/`. v0.1 stays untouched; unchanged
  templates are copied with identical `item_id`s so earlier predictions remain comparable.
- **Training data** goes to `data/train/student_v0/` (gitignored). **Test splits never
  enter training** — enforced by a check (S6), not by intention.
- **Allowed training sources:** official *train* splits of tier-1 datasets and
  **pre-window structured-gold** items (records dated **before 2026-03-01**, built by the
  same builders). Not allowed: UMLS/SNOMED/MIMIC data, teacher (DeepSeek) labels, any LLM
  label, any HF dataset found by S13 (catalog only), HLE (test only).
- **Held-out templates (D14):** `ct_phase_choice_v1`, `fda_boxed_warning_noul_v1`,
  `pubmed_humans_noul_v1`, and one record–claim consistency template chosen in S2 (record
  which). No item of a held-out template — from any time window — is in training data.
- **Readout gate:** D12 applies to every zero-shot letter-readout cell. The trained model
  reads options through its pointer head, so D12's label-mass check does not apply to it,
  but its constant-answer check (S1) and the below-chance check do.
- **GPU:** one RTX PRO 6000 (96 GB). **One GPU job at a time** — the previous loop ran two,
  hit repeated OOMs and lost ~1 h. CPU-only tasks (S13, report writing) may run while a GPU
  job runs.
- **Timeboxes:** ~2 h per integration problem, ~4 h per training run (S9, S10). If a run
  cannot finish in its box, stop it at the last checkpoint and evaluate that checkpoint —
  record the deviation.
- **Hard stop at S14.** Do not start loop 2.
- **Wording.** Describe the new template family as "record–claim consistency" or
  "document-grounded verification". Do not name application domains, industries,
  companies or products anywhere in the repository.

## 4. Tasks

Each task: mark IN_PROGRESS (`date -u +%FT%TZ`), fill the STATE checklist, do the work, run
the acceptance check, self-audit (R6) into `outputs/student_v0/<task>/SELF_AUDIT.md`, append
CLAIMS rows (`S###`), mark DONE/BLOCKED, log, commit
(`loop(student_v0): S<n> <DONE|BLOCKED> — <headline>`), push `loop/student_v0`.

### S0 — Orientation, snapshot, fused kernels (GPU smoke, ~1 h)

**Why:** a known 3× slowdown and the long-prompt OOMs come from missing kernels; this loop
trains on long records.

**Do:** fill "Run started" in STATE and "Started" in AGENTS.md. Snapshot: git commit,
v0.1 manifest hashes, test count. Install `causal_conv1d` and `flash-linear-attention`
(versions compatible with torch 2.14 / CUDA 13 / sm_120) into the project env. Measure
before/after on Qwen3.5-0.8B: tokens/s and peak memory for one 8k-token and one 16k-token
prompt. If the kernels cannot be built for this GPU within the timebox, record
`NOT MEASURED — <reason>` and continue with a 16k prompt cap.

**Acceptance:** `outputs/student_v0/S0/snapshot.json` + `kernels.json` (before/after numbers
or the failure reason); `uv run pytest` passes.

### S1 — Benchmark v0.2 fixes (CPU + small GPU, ~3 h)

**Why:** three templates are invalid or non-discriminative, and the gate has a blind spot.

**Do:**
1. **Graded-score option set:** rebuild `nfcorpus_graded_score` so every offered level
   occurs in the template's pool (include judged-non-relevant passages as the lowest level
   where qrels provide them, otherwise offer only the levels present). New template id
   `_v2`. Unit test: for every built item, each offered level has ≥ 1 item with that gold in
   the template's test split.
2. **Near-miss distractors** for `pubmed_mesh_major_choice` and `fda_class_choice` (`_v2`):
   MeSH distractors are **siblings in the MeSH tree** (same parent tree number) using NLM's
   public MeSH descriptor file (record its year and hash); FDA class distractors are
   classes that share the gold class's **mechanism of action or physiologic effect** in the
   openFDA `pharm_class_moa` / `pharm_class_pe` fields where available, else same route.
   Record the distractor rule per item in `meta`.
3. **Constant-answer check** added to the D12 gate (`src/meddecide/eval/health.py`): a cell
   fails `READOUT_FAIL — degenerate` if its most-predicted option is the argmax for
   ≥ 0.90 of items on a template whose majority share is ≤ 0.60. Unit test with the
   LFM2.5-350M pattern.
4. **Prediction files** carry a `run_id`; readers dedupe by `(run_id, item_id)` and report
   raw vs unique row counts.

**Acceptance:** `data/bench/v0.2/` built with a manifest; the three `_v2` templates pass the
screen (gold-in-state check, BoW macro < 0.90, `n_test ≥ 200`); the old `_v1` versions are
excluded from v0.2 headline tables (kept in the manifest as `superseded`); tests pass.
Zero-shot Qwen3.5-0.8B and 9B on the three `_v2` templates: report them, and the new
templates must **not** be saturated (9B accuracy ≤ 0.90) — if one is, record it and propose
a fix rather than loosening the rule.

### S2 — Record–claim consistency templates (CPU, ~4 h)

**Why:** real document work is mostly verification — "does this record support this claim?".
The current fresh tier asks the model to infer a hidden field; it never asks it to check a
stated one. This family is the closest public analogue, with gold known by construction.

**Design rules (all must hold; record how each template satisfies them):**
1. Gold comes from **structured fields only** (D6). The claim is built from the record's own
   field (supported) or from a near-miss value (not supported).
2. **Role-binding, not string-matching.** A string-presence rule must not solve it: the
   swapped value must *also* appear in the state, in a different role. Examples of the
   pattern:
   - a trial's **experimental** intervention vs its **comparator / placebo** arm (both listed
     in the arms, the arm-type labels removed from the state);
   - a trial's **primary** vs **secondary** outcome (both listed, section labels removed);
   - a field value of this record vs the same field of a **closely related** record
     (same condition / same class) when the role design is impossible for that field.
3. Balanced 50/50 supported vs not supported per template and split.
4. A **multi-field variant** (`choice`): the state plus 3–5 stated fields, exactly one of
   which was swapped; question "which stated field is not supported by the record?".
5. Build for both windows: **fresh** (≥ 2026-03-01, test/dev as in v0.1) and **pre-window**
   (< 2026-03-01, training only — used in S6).

At least **3 templates across ≥ 2 sources** (ClinicalTrials.gov, openFDA; PubMed optional),
including ≥ 1 multi-field variant. Choose one to hold out (D14) and record which and why
(prefer the one whose role structure differs most from the others).

**Acceptance:** templates built into `data/bench/v0.2/fresh/`; each passes the screen and
additionally a **string-presence baseline** (pick the option whose value appears verbatim in
the state; for `noul`, say "supported" iff the claimed value appears) with macro accuracy
≤ 0.60. Zero-shot 0.8B and 9B numbers recorded; documented in
`docs/benchmark/consistency_templates.md`.

### S3 — Long-record slice and shared-prefix measurement (GPU, ~2 h)

**Why:** real records are long and carry many checks each; per-record cost matters, not
just per-question cost.

**Do:**
1. **Long-record slice:** tag every v0.2 fresh test item whose rendered prompt exceeds
   **8,192** tokens (Qwen3.5 tokenizer) as `long_record`; if the slice has < 300 items,
   rebuild the affected openFDA / ClinicalTrials.gov templates with a larger state budget
   (up to 32,768 tokens, recorded) so the slice reaches ≥ 300 items where the window allows.
   Report every model on the slice separately.
2. **Shared-prefix measurement:** group fresh test items by source record (same state,
   several templates). For zero-shot Qwen3.5-0.8B, measure per-record latency for (a) one
   prompt per question vs (b) the state as a shared prefix with each question appended,
   reusing the KV cache of the prefix. Report records/s and questions/s for both, and check
   that (b) gives the same argmax as (a) on ≥ 99 % of questions.

**Acceptance:** slice counts per template in the manifest; `outputs/student_v0/S3/prefix.json`
with both throughputs and the agreement rate.

### S4 — HLE medical subset, supplementary test (CPU, ~1 h)

**Do:** build from the official `cais/hle` (gated — needs the operator's HF account to have
accepted its terms): category Biology/Medicine, answer type multiple-choice, no image, as
`choice` items in a separate `data/bench/v0.2/supplementary/hle_med.jsonl`. Record the
count, revision and licence. Run zero-shot 0.8B and 9B. Never train on it. Report it as
supplementary only, with its sample size and CI. If access is denied:
`BLOCKED — cais/hle terms not accepted` and continue.

**Acceptance:** item count + manifest, or BLOCKED with the reason.

### S5 — Training-data builders: tier-1 train splits (CPU, ~2 h)

**Do:** loaders for official **train** splits: MedQA (train), MedMCQA (train; cap 60,000,
seeded), SciFact (train claims), NFCorpus (train qrels, same pool rules as the fixed
`_v2` template), MedQuAD rows not used by any tier-1 split. **Excluded:** PubMedQA
`pqa_artificial` (heuristic labels — not gold), MMLU (no medical train split), any test or
validation split. Apply the F1 gold verifier (`scripts/bench/verify_gold.py` pattern) to
every train source.

**Acceptance:** `data/train/student_v0/tier1_train.jsonl` + manifest (counts per source,
licence, revision); gold verification 0 mismatches or each mismatch explained; 0 items
whose normalised (state + question) text hash appears in any v0.2 test or dev split (count
the removals).

### S6 — Pre-window structured-gold training data (CPU + network, ~4 h)

**Do:** run the v0.2 fresh builders (including S2's consistency templates) on records
dated **before 2026-03-01**: ClinicalTrials.gov first-posted 2023-01-01 → 2026-02-28;
openFDA labels first effective before 2026-03-01; PubMed records from update/baseline
files dated before 2026-03-01. Balanced per template; cap **20,000** items per template;
**exclude the held-out templates entirely**. Then assemble the training mix:

- `tier1_train` (S5) + `prewindow_structured` (this task), with the mixture recorded;
- a **dev** set for model selection and temperature fitting = v0.2 tier-1 dev + v0.2 fresh
  dev (non-held-out templates only);
- the **leakage check** (mechanical, part of acceptance): no `source_record_id` and no
  normalised state hash from any v0.2 test or dev item appears in training; no record dated
  ≥ 2026-03-01 appears in training; no held-out template id appears in training.

**Acceptance:** `data/train/student_v0/{train,dev}.jsonl` + `manifest.json` with counts per
source × template × qtype; the leakage check passes with its counts in CLAIMS.

### S7 — Training code: LoRA + pointer head (GPU smoke, ~4 h)

**Why:** this is the MedDecide architecture (D3).

**Do:** in `src/meddecide/model/` and `src/meddecide/train/`:
- **Base frozen.** A LoRA adapter on the base (rank and target modules in config; start
  r = 16 on attention + MLP projections) that is active only on the decision path.
- **Pointer head.** Render state, question and options with a marker token per option.
  Score each option from the hidden states at its marker and at the answer position (e.g. a
  small MLP or bilinear form). Softmax over **exactly the offered options**; supports 2–64
  options; `noul` = 2 options, `score` = ordered levels.
- **Loss:** cross-entropy + λ·Brier (λ in config, start 1.0). **Option-order augmentation:**
  shuffle options per example every epoch.
- **Calibration:** one temperature per `qtype` fitted on the dev set after training.
- **Inference:** returns the full distribution and the expected level for `score`; batched;
  records latency.
- **Tests:** (1) output is a proper distribution over the offered options only;
  (2) permuting options permutes the output within tolerance; (3) with the adapter
  disabled, greedy generation of the base on 20 fixed prompts is **byte-identical** to the
  untouched base model; (4) overfit test: 64 training items reach ≥ 0.95 training accuracy.

**Acceptance:** all four tests pass; a 200-step smoke run's loss decreases; throughput
(tokens/s, items/s) recorded and used to size S9's step count to its timebox.

### S8 — Evaluation path for the trained model (GPU, ~1 h)

**Do:** extend the harness so a trained checkpoint is scored on the same items, metrics and
probes as the zero-shot baselines: accuracy (± bootstrap CI), macro accuracy, Brier, ECE,
option-shuffle flip rate, constant-answer check, latency, and **selective accuracy** —
accuracy on the most-confident 50 %, 80 % and 90 % of items (coverage curve). Add the
selective-accuracy metric to the zero-shot path too.

**Acceptance:** a randomly initialised head scores near chance with no `READOUT_FAIL` from
the shape checks; a unit test covers the selective-accuracy computation.

### S9 — Train MedDecide-0.8B (instruct base) (GPU, timebox ~4 h)

**Do:** train on `Qwen/Qwen3.5-0.8B` with the S6 mix, one pass (or the number of steps that
fits the box), evaluating on dev every N steps; keep the best dev checkpoint (dev macro
accuracy, Brier as tiebreak); fit temperatures on dev. Log loss curves and the full config.

**Acceptance:** checkpoint (adapter + head) saved under `outputs/student_v0/S9/`; dev metrics
per template; training wall-clock and tokens seen recorded.

### S10 — Ablation: MedDecide-0.8B from the Base checkpoint (GPU, timebox ~3 h)

**Do:** identical recipe on `Qwen/Qwen3.5-0.8B-Base`, same data and step budget. If the
remaining session time cannot hold both S10 and S11, run S11 first on the S9 model and mark
S10 `BLOCKED — time` with the reason.

**Acceptance:** as S9.

### S11 — Gate G1 evaluation (GPU, ~3 h)

**Do:** evaluate the S9 model (and S10 if available) on v0.2 **test**: tier 1, fresh (full
window and strict slice), held-out templates, consistency templates, long-record slice, HLE
(supplementary). Baselines **on identical item sets**: zero-shot Qwen3.5-0.8B (letter
readout, D12 gate) and JEV-9B (own protocol, re-run on v0.2 and on any items it previously
skipped, now that kernels are installed — report its coverage).

**G1 definition (D16).** Over the **headline fresh set** = fresh templates kept by the v0.2
screen, excluding `pubmed_mesh_major_choice_v1` and `fda_class_choice_v1` (superseded by
`_v2`), reported separately for **seen** and **held-out** templates:

- per baseline B ∈ {zero-shot Qwen3.5-0.8B, JEV-9B}, compute the item-paired difference
  MedDecide − B in (a) macro accuracy across templates and (b) mean Brier, with a paired
  bootstrap 95 % CI (resample items within templates, 1,000 resamples);
- **G1 PASS** iff, against **both** baselines, the accuracy-difference CI lower bound is
  > 0 **and** the Brier-difference CI upper bound is < 0, on **seen** templates;
- the **held-out** result is reported with the same statistics and the same rule, labelled
  separately; it decides which branch of §2 applies.

Also report: tier-1 per template, consistency templates, long-record slice, strict slice,
selective-accuracy curves, flip rates, latency per question and per record.

**Acceptance:** `outputs/student_v0/S11/g1.json` + committed `loops/student_v0/g1.md` with the
verdict, every difference with its CI, and the item counts behind each.

### S12 — Generation byte-identity and the audit read-back (no/low GPU, ~1 h)

**Do:** (1) re-run S7's byte-identity test on the trained S9 adapter (adapter off ⇒ base
generation unchanged on 20 fixed prompts). (2) If the operator's audit file exists at
`outputs/bench_v0_fix0/F10/audit_v0.jsonl` (the operator plans to complete it during this
loop), report accept / reject / note counts per template and list templates with a reject
rate ≥ 10 %; otherwise record `NOT MEASURED — audit pending`.

**Acceptance:** identity result in CLAIMS; audit summary or the pending note.

### S13 — Candidate dataset catalog (CPU, may run during GPU jobs, ~2 h)

**Why:** many medical datasets exist on HF; quality, licences and labels vary. The next
review should choose from a vetted list, not a search result.

**Do:** catalog ≥ 30 candidate HF datasets relevant to medical decisions. For each: id,
size, label provenance (human / structured / heuristic / LLM-generated / unknown), licence,
convertibility to typed decisions (`noul` / `choice` / `score`), overlap risk with
MedDecide-Bench test sources, and a verdict — `train-candidate`, `eval-candidate`, or
`reject` — with a one-line reason. Prefer datasets whose labels are human or structured.
**Documentation only:** download metadata and at most a sample of rows; train on nothing.

**Acceptance:** committed `docs/benchmark/dataset_catalog.md`.

### S14 — Findings and closure (no GPU, ~2 h) — HARD STOP

**Why:** the loop's output is only as good as its record. This task writes the artifact the
next loop and the operator actually read.

**Do:** write `loops/student_v0/FINDINGS.md` following the FINDINGS structure (Summary first:
the question, what we did, what we found, what it means, what did not work — plain
language, every number with its denominator and CLAIMS id). State the G1 verdict and which
branch of §2 applies. Every summary claim must exist, with more detail, in the body. Pick
five claims spanning the loop, re-run their recompute commands in fresh processes, and
append the spot-check. Fill STATE's closure feed. Write `loops/student_v0/NEXT.md`
(proposals, not decisions). Set `Loop status: STOPPED`, commit, push, stop.

**Acceptance:** FINDINGS.md with (1) a Summary a non-specialist can follow, (2) a body
covering every summary claim, (3) the five-claim spot-check with re-run outputs; every STATE
task DONE or BLOCKED; pushed.

## 5. Definition of done

- STATE.md — every task `DONE` or `BLOCKED` with a reason.
- CLAIMS.md — every reported number, with artifact and recompute command.
- FINDINGS.md — Summary first, full-coverage body, five-claim spot-check appended.
- Committed: code, tests, configs, `docs/benchmark/` additions, the loop's markdown reports.
  Not committed: items, training data, predictions, checkpoints.

## 6. Task table (seed STATE.md with this)

| id | task | GPU | deps | status |
|---|---|---|---|---|
| S0 | Orientation, snapshot, fused kernels | smoke | — | PENDING |
| S1 | Benchmark v0.2 fixes | small | S0 | PENDING |
| S2 | Record–claim consistency templates | small | S1 | PENDING |
| S3 | Long-record slice + shared-prefix measurement | yes | S2 | PENDING |
| S4 | HLE medical subset (supplementary test) | small | S1 | PENDING |
| S5 | Tier-1 train-split builders | no | S1 | PENDING |
| S6 | Pre-window structured-gold data + training mix + leakage check | no | S2, S5 | PENDING |
| S7 | Training code: LoRA + pointer head | smoke | S0 | PENDING |
| S8 | Evaluation path for trained models | yes | S7 | PENDING |
| S9 | Train MedDecide-0.8B (instruct) | yes | S6, S8 | PENDING |
| S10 | Ablation: MedDecide-0.8B from Base | yes | S9 | PENDING |
| S11 | Gate G1 evaluation | yes | S9, S3, S4 | PENDING |
| S12 | Byte-identity + audit read-back | small | S9 | PENDING |
| S13 | Candidate dataset catalog | no | S0 | PENDING |
| S14 | Findings and closure — HARD STOP | no | all | PENDING |

Order of the table is the execution order, except: S13 is CPU-only and may run whenever a
GPU job is in flight; S11 may run before S10 when time is short (S10 rules). A blocked S4
or S10 never blocks S11 or S14.

## 7. Explicitly out of scope

| item | why |
|---|---|
| Teacher (DeepSeek) labels, UMLS / ontology synthetic data | loop 2 |
| MIMIC / PhysioNet data, MedDecide-clinical | later loop |
| Models above 0.8B, the 4B bake-off, 27B baselines | loop 3, after G1 |
| Joint schema head, RLCD, adaptive thinking | gated on loop-4 evidence |
| Training on any dataset found by S13, or on HLE | S13 is documentation only; HLE is test-only |
| Changing the D12 thresholds, the G1 definition, or the held-out template list | settled (D12, D14, D16); disagreements go to questions-for-operator |
| Tuning on any test split, per-model prompt tuning for baselines | biases every comparison |
| Retrieval track | separate repo |
| Publishing anything (weights, data, posts) | operator decision after review |

## 8. Failure modes to avoid

1. **Leakage into training.** The structured builders make it easy to train on the very
   records being tested. The S6 check (record ids, normalised state hashes, dates, held-out
   template ids) is the guard; a G1 PASS with a failing leakage check is void.
2. **String-matching templates.** A consistency template where the claimed value appears in
   the state only when it is true is solved by a string rule. The S2 string-presence baseline
   (≤ 0.60) is the detector.
3. **Template learning mistaken for decision skill.** Seen-template gains can come from
   template quirks. The held-out templates are the measurement; report them with the same
   weight as seen ones.
4. **Comparing on different item sets.** JEV-9B skipped long prompts before. Every
   comparison in G1 uses the identical item set; report coverage beside every number.
5. **Pipeline bugs reported as findings** (bench_v0's failure). A trained model at or below
   chance, a constant answer, or a calibration that is "perfect" on a tiny cell is a defect
   to investigate first.
6. **Two GPU jobs at once** (bench_v0_fix0's deviation, ~1 h of OOM re-runs).
7. **Dependency shortcuts** (bench_v0_fix0 closed F9 before its dependency F6). Mark a task
   DONE only after its dependencies are DONE.
8. **Overruled objection — teacher chosen without comparison.** Not exercised in this loop
   (no teacher), recorded so it is not forgotten.
9. **Relabelling an unwelcome result (R4).** If G1 fails, report it plainly and name the
   §2 branch. Do not change the headline set, the baselines or the statistic afterwards.
10. **Citing artifacts that do not exist (R2); impossible arithmetic (R5).**

## 9. If you finish early

In priority order, without starting anything in §7:
1. Run S10 if it was skipped for time.
2. A second seed of S9 (same data, different seed) to report seed variance on the G1
   differences.
3. A rank-8 vs rank-32 LoRA comparison at fixed steps, scored on dev only.
