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

- **Loop:** `osler_v0` (loop 1c — competitor scoreboard, clinical generators, Osler-4B / -9B)
- **Mission:** measure every open competitor through its authors' code on v0.2, a new external clinical panel and a robustness pack; build gold-by-construction clinical generators (D20); train three matched Osler-4B arms (option-code head, pointer, non-causal; D23), choose a head by a pre-registered dev rule, train Osler-9B and an Osler-0.8B reference; apply Gate O1 (D24) — then hard-stop for review.
- **Branch:** `loop/osler_v0` (never commit to `main`, `dev` or any other loop branch)
- **Artifacts:** `loops/osler_v0/` (GOAL.md, ADVISORY.md, STATE.template.md, KICKOFF.md)
- **Run state:** `loops/osler_v0/STATE.md`
- **Outputs:** `outputs/osler_v0/<task-id>/` (gitignored)
- **Claims:** `loops/osler_v0/CLAIMS.md`
- **Style:** advisory
- **Started:** `<fill in at O0>`

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

- **Model ladder (D22):** Osler-4B on `Qwen/Qwen3.5-4B` and Osler-9B on `Qwen/Qwen3.5-9B`
  are the targets; Osler-0.8B on `Qwen/Qwen3.5-0.8B` is a reference trained with the same
  recipe. The MedGemma bake-off and LFM2.5-350M are dropped from training.
- **Architecture (D23):** frozen base + decision-path LoRA + a separate option-code readout
  (rows initialised from `lm_head`, softmax over the offered codes, exportable to a plain
  `lm_head`). Pointer head and non-causal full attention are matched ablations;
  the head is chosen by the loop's pre-registered dev rule. Adaptive thinking remains a later
  ablation.
- **Teacher:** `deepseek-ai/DeepSeek-V4.1-Flash`, self-hosted by the operator on
  4×RTX PRO 6000 when a task needs it.
- **Benchmark:** two tiers. Tier 1 = established public test sets (+ contamination
  probe). Tier 2 = **fresh** items from records dated after the ladder models' dates
  (window start 2026-03-01, D11; strict slice ≥ 2026-09-10 reported separately), with
  gold derived **only from structured source fields**. The SoTA claim rests on tier 2.
- **Readout-health gate (D12):** no zero-shot letter-readout accuracy is reported unless
  its (model, template) cell passes the gate; failing cells read `READOUT_FAIL — <check>`.
  Trained readouts that normalise over the offered options use the D21 variant (label mass
  is a diagnostic, not a gate).
- **Training data (D13, D17, D20):** official train splits, **pre-window** structured-gold items
  (records dated before 2026-03-01), vetted catalog datasets with human or structured
  labels and a permissive licence, gold-by-construction generated decisions (code writes the
  text and computes the label), and general human-labelled replay. Never test splits, never
  LLM-written text or LLM-labelled gold, never an external-panel dataset.
- **Held-out templates (D14):** some fresh templates are excluded from training entirely
  and reported separately; the list is fixed per loop in its ADVISORY.
- **Gates G1 (D16) and O1 (D24):** defined in `docs/plans/PROGRAM.md`; never changed after
  results. Each names its baselines exactly — an ablation, a sibling arm or any other model is
  never part of a verdict.
- **Release policy:** model weights, benchmark (public-source items only), and code are
  open. **Training data stays private.**

### Data rules (hard)

- **Public repo.** This GitHub repo is public. Never commit raw benchmark items,
  predictions, teacher labels, UMLS-derived text, or MIMIC-derived text. Commit code,
  configs, manifests (IDs, hashes, counts), and aggregate results only.
- **UMLS / SNOMED / RxNorm-restricted strings** are private-only. They never go to a
  hosted third-party API — only to self-hosted models.
- **MIMIC and other PhysioNet credentialed data** may only be processed on
  operator-controlled machines by local models; never sent to a hosted third-party
  API; never printed into anything committed. (Not used before the clinical loop.)
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
