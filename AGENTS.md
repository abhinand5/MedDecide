# AGENTS.md — MedDecide operating guide and loop pointer

> **What this file is:** the first file any agent reads in this repo. It carries
> (1) what MedDecide is, (2) the project rules, (3) the universal loop rules,
> (4) which loop is currently active and where its artifacts live.
>
> **Deliberate deviation from the autoresearch skill defaults:** this file **is
> committed**, and loop artifacts live in `loops/<loop>/` and **are committed**.
> The operator reviews progress by pulling the repo on another machine, so the
> loop's memory must travel through git. Raw outputs stay in the gitignored
> `outputs/`. Do not "fix" this layout.
>
> **How to update:** when a new loop starts, replace only the "Current loop" section.
> Universal rules change rarely — when they do, the change traces to a specific failure.

---

## Current loop

- **Loop:** `osler_v3` (loop 1f — keep the base's general skill with KL-to-base; real sources + Gemma-crafted real-text items)
- **Mission:** Stage A recipe ablations on osler_v2's kept mix (control, KL-to-base 0.1 and 0.5, broad replay; D41) with a
  pre-registered choice on a clinical-general dev proxy; Stage B data blocks (RAGTruth, DrugProt, MultiCaRe; Gemma-crafted
  items over real text, D43; synthetic charts under the D44 cue rule; official MIMIC if it arrives); final Osler-4B on
  pplx-decider-flash; Gate O3 (D35, D39, D44) — then hard-stop for review.
- **Branch:** `loop/osler_v3` (never commit to `main`, `dev` or any other loop branch)
- **Artifacts:** `loops/osler_v3/` (GOAL.md, ADVISORY.md, STATE.template.md, KICKOFF.md)
- **Run state:** `loops/osler_v3/STATE.md`
- **Outputs:** `outputs/osler_v3/<task-id>/` (gitignored)
- **Claims:** `loops/osler_v3/CLAIMS.md`
- **Style:** advisory
- **Started:** `<fill in at X0>`

---

## What MedDecide is

MedDecide is an open, calibrated **medical decision model** in the "System One" style:
given a state (a note, a trial record, a question, a query + passage) and a list of
typed questions, it returns a probability for every allowed option in **one prefill
pass** — no text generation. Question types:

| type | shape | answer |
|---|---|---|
| `noul` | yes/no | p(yes) |
| `choice` | 2–255 named options | distribution over options |
| `score` | 2–10 ordered levels | distribution + expected level |

The program goal is a community-defining open model plus **MedDecide-Bench** (public,
regenerable, contamination-resistant), released with code. The full program — stages,
gates, settled decisions — is in [docs/plans/PROGRAM.md](docs/plans/PROGRAM.md). Read it
once at the start of a loop.

Background reading on decision models (vendor numbers in it are **unverified** claims):
the operator's survey is summarised in `docs/plans/PROGRAM.md` §2; do not cite numbers
from it as measured facts.

---

## Project rules

### Settled decisions (do not reopen inside a loop)

These were decided by the operator. If evidence contradicts one, record it in STATE.md
"Questions for the operator" — do not act on it.

- **Model ladder (D22, D39):** Osler-4B on `Qwen/Qwen3.5-4B` (or on `perplexity-ai/pplx-decider-flash-v1.2`, itself
  Qwen3.5-4B, if D39's dev rule chooses it) and Osler-9B on `Qwen/Qwen3.5-9B` are the targets; Osler-0.8B on `Qwen/Qwen3.5-0.8B` is a reference trained with the same
  recipe. The MedGemma bake-off and LFM2.5-350M are dropped from training.
- **Architecture (D23, D26):** frozen base + decision-path LoRA + a separate option-code readout
  (rows initialised from `lm_head`, softmax over the offered codes, exportable to a plain
  `lm_head`). osler_v0's dev rule kept it over the pointer and non-causal arms; from osler_v1 the
  architecture is frozen and loops compare data blocks. Adaptive thinking remains a later
  ablation.
- **Teacher (D30):** `deepseek-ai/DeepSeek-V4.1-Flash`. A hosted endpoint (zero data retention) may
  write prose from code-generated facts; anything that reads MIMIC uses a self-hosted teacher.
- **Benchmark:** two tiers. Tier 1 = established public test sets (+ contamination
  probe). Tier 2 = **fresh** items from records dated after the ladder models' dates
  (window start 2026-03-01, D11; strict slice ≥ 2026-09-10 reported separately), with
  gold derived **only from structured source fields**. The SoTA claim rests on tier 2.
- **Readout-health gate (D12):** no zero-shot letter-readout accuracy is reported unless
  its (model, template) cell passes the gate; failing cells read `READOUT_FAIL — <check>`.
  Trained readouts that normalise over the offered options use the D21 variant (label mass
  is a diagnostic, not a gate).
- **Training data (D13, D17, D20, D27, D28):** official train splits, **pre-window** structured-gold items
  (records dated before 2026-03-01), vetted catalog datasets with human or structured
  labels and a permissive licence, gold-by-construction generated decisions (code computes the
  label), MIMIC notes with structured labels (patient splits), teacher prose written from code-built
  facts and kept only when re-extraction returns the same facts, and general human-labelled replay.
  Never test splits, never teacher-written labels or judgements, never an external-panel dataset.
- **Held-out templates (D14):** some fresh templates are excluded from training entirely
  and reported separately; the list is fixed per loop in its ADVISORY.
- **Gates G1 (D16), O1 (D24) and O2 (D31):** defined in `docs/plans/PROGRAM.md`; never changed after
  results. Each names its baselines exactly — an ablation, a sibling arm or any other model is
  never part of a verdict.
- **Release policy:** model weights, benchmark (public-source items only), and code are
  open. **Training data stays private.**

### Data rules (hard)

- **Public repo.** This GitHub repo is public. Never commit raw benchmark items,
  predictions, teacher labels, UMLS-derived text, or MIMIC-derived text. Commit code,
  configs, manifests (IDs, hashes, counts), and aggregate results only.
- **UMLS / SNOMED / RxNorm-restricted strings** taken from licensed UMLS or SNOMED release files are private-only. They
  never go to a hosted third-party API — only to self-hosted models. Display names that a public, permissively licensed
  dataset already publishes (e.g. Synthea's condition and medication names) are not covered by this rule (operator
  answer, osler_v1 Question 11).
- **MIMIC and other PhysioNet credentialed data** (D28) may only be processed on
  operator-controlled machines by local models; never sent to a hosted third-party
  API (including the hosted teacher); never printed into anything committed. It is a training and
  evaluation source from osler_v1; a memorisation audit precedes any release.
- **Test splits are evaluation-only.** Never fit temperatures, thresholds, prompts, or
  templates on a test split. Dev/train splits only.
- **No silent drops.** Every filter returns counts and reasons.

### Environment rules

- Work only inside the repo's `uv` environment (`uv run ...`). Python 3.12.
- All caches on the large volume: `HF_HOME=/workspace/.hf_home`,
  `UV_CACHE_DIR=/workspace/.uv_cache`, `TMPDIR=/workspace/tmp`. The root disk is 20 GB —
  never download models or datasets to `/root` or `/tmp`.
- `/workspace` is a network filesystem: prefer fewer, larger files (parquet/JSONL) over
  many small ones.
- One GPU job at a time. Long jobs run under `nohup`/`setsid` with logs in
  `outputs/<loop>/<task>/logs/`; check completeness of outputs, not just exit codes.
- Secrets come from environment variables (`HF_TOKEN`, `TEACHER_BASE_URL`,
  `TEACHER_API_KEY`). Never write a secret into any file.

### Code style

- Reusable logic in `src/meddecide/`; scripts in `scripts/` are thin CLI wrappers.
- Typed dataclasses or pydantic models for anything crossing module boundaries.
- Deterministic IDs (stable hashes) for items and templates.
- Small pure functions for parsing, templating, metrics — with unit tests.
- `pathlib.Path`; no hardcoded paths outside config defaults.
- `uv run ruff check .` and `uv run pytest` pass before any commit.

### Git

- Each loop works on the branch named in "Current loop" above (usually
  `loop/<loop-name>`; a repair loop may continue its parent's branch). Check
  `git branch --show-current` at the start of every session and switch to that branch if
  needed. **Never commit to or push `main`** — the operator merges after review.
- Commit loop progress at least after every task: STATE.md, CLAIMS.md, code, committed
  reports. Push that branch. Message format:
  `loop(<loop-name>): <task-id> <DONE|BLOCKED> — <headline>`.
- Never force-push, never rewrite history, never delete branches.

---

## Universal rules

### Evidence discipline

- **R1 — Every number has a source.** Every reported number carries the artifact it
  came from and a command that recomputes it, recorded as a row in the loop's
  CLAIMS.md (`id | claim | value | artifact | recompute command | task`). A number
  without a CLAIMS row is not a result.
- **R2 — Cite only what exists.** Before citing an artifact, check that it exists
  (`ls`/`test -f`). Never cite a path you expect to create later.
- **R3 — Unmeasured is a valid result.** `NOT MEASURED — <reason>` and
  `BLOCKED — <reason>` are correct, expected, valuable outcomes. A fabricated or
  estimated-but-unlabelled number is a critical failure.
- **R4 — Never relabel an unwelcome result.** If a metric returns a bad answer, report
  the bad answer. Do not swap in a different metric, subset, or split and present it as
  the result. Additional analyses are allowed only when labelled as additional.
- **R5 — Arithmetic must close.** Denominators are stated; percentages match their
  counts; parts sum to totals; splits are disjoint. Check before reporting.
- **R6 — Self-audit every task.** Before marking a task DONE, write
  `outputs/<loop>/<task>/SELF_AUDIT.md`: re-derive each headline number from the raw
  artifact in a fresh process, check denominators, compare with expectations
  (chance level, majority baseline, published values), and list anything suspicious.
- **R7 — Provenance.** Record for every run: git commit, command, config, seed, model
  ids + revisions, dataset ids + revisions, hardware, wall-clock.
- **R8 — Never edit a check to pass it.** Changing a test, acceptance criterion,
  threshold, or gold label to make something pass is forbidden. That is `BLOCKED`.
- **R9 — Raw outputs are immutable.** Do not edit raw predictions or downloaded data.
  Derived artifacts are regenerated by scripts, never hand-edited.
- **R10 — Measured vs believed.** Hypotheses are written as prose without numbers and
  are labelled as hypotheses. Only measured values appear in tables of results.

### Loop mechanics

- **Every wake:** read this file → the loop's GOAL.md → ADVISORY.md → STATE.md. Where
  STATE.md disagrees with your memory, **the file wins** — your context may have been
  compacted.
- Take the **first** `PENDING` task whose dependencies are `DONE`. Mark it
  `IN_PROGRESS` with a UTC timestamp (`date -u +%FT%TZ`) **before** working. Fill the
  STATE.md checklist for it.
- On finishing: run the acceptance check, self-audit (R6), append CLAIMS rows, mark
  `DONE` or `BLOCKED — <reason>`, append the iteration-log entry, commit.
- A `BLOCKED` task never blocks unrelated work. Record it, write the question into
  STATE.md "Questions for the operator", move on.
- **Hard stops.** When the plan says STOP, stop: write what is asked, commit, push,
  and end the session. Do not start the next loop, ever.

### Guardrails

- Do not start anything listed in the loop's out-of-scope section.
- Do not delete files outside `outputs/<current loop>/` and `/workspace/tmp`.
- Do not stop, delete, or reconfigure pods, volumes, HF repos, or remote services.
- Do not publish anything (no public HF uploads, no making repos public, no posts).
- Timebox integrations: if wiring up one baseline model or one data source fails for
  more than ~2 hours of effort, mark it `NOT MEASURED — <reason>` and move on.

### When you are unsure

Prefer the conservative reading of the plan. Write the ambiguity, the options, and the
one you would pick into STATE.md "Questions for the operator". If the ambiguity blocks
the task, mark it `BLOCKED`; otherwise proceed with the conservative option and record
it under "Deviations from the plan".

---

## Prior loops

| loop | outcome | findings |
|---|---|---|
| `osler_v2` | pplx-decider-flash-v1.2 chosen as the base on dev (MR +3.8, knowledge +9.2 over Qwen3.5-4B at C0); training 1.8× faster (attention-only LoRA, checkpointing only above 8,192 tokens). Real text (CodiEsp, MTSamples, EHR-Rel) kept (+7.5 MR dev); teacher prose rejected again; Gemma charts blocked by a certainty-word cue check. **Gate O3 FAIL** on one check: MR suite +26.3 vs MedDecider-4B and +15.8 vs its base; external panel −0.4 vs MedDecider-4B but −2.0 [−3.4, −0.6] vs zero-shot pplx-flash (limit −2.0); knowledge +0.2 on paired items. | `loops/osler_v2/FINDINGS.md` |
| `osler_v1` | Synthetic-first medical-record audit data (generators, Synthea, MTSamples, CodiEsp, PMC-Patients, 31,000 verified teacher pages), replay 15 %, four data blocks (B1 kept: generators + Synthea; teacher prose and real text not kept), attention-only final Osler-4B (9B deferred). **Gate O2 FAIL**: MR suite +8.1 accuracy vs MedDecider-4B but Brier worse (real-text ICD, overconfident); external panel −2.2 (limit −2.0). Knowledge guard passes at +4.4 — the attention-only adapter fixed osler_v0's loss. Generated MR items +23.5 vs MedDecider-4B; real text −7.4. The MR suite was dominated by a vocabulary-solvable template and three ceiling templates. | `loops/osler_v1/FINDINGS.md` |
| `osler_v0` | Scoreboard of 11 models on a 59,538-item common set (v0.2, a 4,353-item external clinical panel, a 20,109-item robustness pack); nine clinical generators (two held out); three matched 4B arms within half a point (dev rule kept the option-code head); Osler-9B and -0.8B. **Gate O1 FAIL** for both sizes: they beat every baseline on fresh seen templates (+9 to +13 pts) and robustness, but lose the external panel to MedDecider (4B −5.3, 9B −1.7) and the 4B fails the knowledge guard (−1.6). Calibration temperatures were not saved with checkpoints (fixed in osler_v1). | `loops/osler_v0/FINDINGS.md` |
| `bench_v0` | Built MedDecide-Bench v0 (21,202 tier-1 + 41,502 fresh items), a harness whose `choice` readout matches lm-evaluation-harness within 0.5 pts, and 264,478 ladder predictions. **Advisor review found defects** — MedMCQA key off by one, `noul`/`score` readout broken, imbalanced templates, narrow window — that invalidate its MedMCQA and relevance findings; repaired in `bench_v0_fix0`. T10 (teacher) and T11 (audit) blocked on the operator. | `loops/bench_v0/FINDINGS.md` (read with `loops/bench_v0_fix0/CORRECTIONS.md`) |
| `bench_v0_fix0` | Repaired loop 0: MedMCQA key fixed (0 gold mismatches of 15,915 verified), `noul`/`score` readout fixed (label mass 0.996+), D12 readout-health gate added, benchmark rebuilt as v0.1 (22,594 tier-1 + 23,582 balanced fresh items, window 2026-03-01), all 6 ladder models + JEV-9B + 2 Laya models measured, all 53 bench_v0 claims dispositioned (4 withdrawn). Teacher gate still blocked; audit staged. Advisor note: its Summary item 7 ("~0.50 on every fresh `noul`") holds only for the 350M model. | `loops/bench_v0_fix0/FINDINGS.md` |
| `student_v1` | Three matched heads on Qwen3.5-0.8B (pointer, letter-readout LoRA, Unsloth joint head; 15k steps, 482,889 rows, 24 templates) finished within 0.6 pts overall (A 0.7591, C 0.7545, B 0.7528). All pass G1 on seen fresh templates; held-out: A and C FAIL, B NOT MEASURED (D12 label-mass failure on 8/11 fresh templates — mass on non-offered letters). Held-out skill falls with training; decision-path MedQA/MedMCQA fell to about the majority rate. | `loops/student_v1/FINDINGS.md` |
| `student_v0` | Benchmark v0.2 (45,009 items; record–claim consistency, long-record slice, HLE supplementary), a 212,481-item gold training mix with a clean leakage check, and the first MedDecide-0.8B (LoRA r=8 + pointer head; selected checkpoint step 1,000). vs zero-shot 0.8B: seen fresh templates macro +0.163, Brier −0.230; held-out templates macro −0.018 (no transfer). Its own `g1.md` used the Base ablation as a baseline; per D16, G1 is incomplete on seen (JEV-9B not run) and FAIL on held-out — see the advisor note. Two runs failed first (length-curriculum bug; divergence); padding makes trained scores batch-dependent. | `loops/student_v0/FINDINGS.md` (advisor note at top) |

---

## Repo layout

```
AGENTS.md            this file (committed)
README.md
docs/plans/          program plan and stage specs (committed)
docs/benchmark/      benchmark schema + datasheet (committed)
loops/<loop>/        GOAL, ADVISORY, STATE, CLAIMS, FINDINGS, KICKOFF (committed; aggregates only)
src/meddecide/       library code
scripts/             thin CLI wrappers
tests/               unit tests
configs/             configs
outputs/<loop>/      raw outputs, logs, predictions (gitignored)
data/                downloaded + built data (gitignored)
```
