# GOAL — student_v0 (benchmark v0.2, structured-gold training data, and the first trained MedDecide-0.8B, judged by gate G1)

MedDecide is an open, calibrated medical decision model: a frozen base LLM plus a
decision-path LoRA and a pointer head that returns a probability for every offered option
in one forward pass. The two previous loops built and repaired the measuring stick —
MedDecide-Bench v0.1 (tier 1 = established test sets with verified gold; tier 2 = "fresh"
items from records dated after 2026-03-01 with gold from structured fields), a validated
harness, the D12 readout-health gate, and zero-shot baselines for the model ladder and for
JEV-9B and Laya. This loop trains for the first time. It first fixes three benchmark
defects and adds a **record–claim consistency** template family, a long-record slice and an
HLE supplementary test (v0.2); then builds a **gold-only** training set from official train
splits and **pre-window structured-gold** items (records dated before 2026-03-01, built by
the same builders, with some templates held out entirely); then implements and trains
**MedDecide-0.8B** on `Qwen/Qwen3.5-0.8B`, with a Base-checkpoint ablation; and finally
applies **gate G1**: does the trained model beat zero-shot Qwen3.5-0.8B and JEV-9B, on
accuracy and Brier with paired confidence intervals, on seen and on held-out templates? It
ends with a hard stop for operator review.

## Read first, in this order

1. `AGENTS.md` (repo root) — project rules, R1–R10, D12, loop mechanics, guardrails.
2. `docs/plans/PROGRAM.md` — the program and settled decisions (note D12–D16).
3. `loops/bench_v0_fix0/FINDINGS.md` and `loops/bench_v0_fix0/NEXT.md` — the benchmark this
   loop starts from (read the advisor note in ADVISORY §1 about its Summary item 7).
4. `loops/student_v0/ADVISORY.md` — this loop's work plan. Read in full at the start of
   every iteration.
5. `loops/student_v0/STATE.md` — what has already been done. If it still contains the
   `<fill in bootstrap>` placeholders, this is the first iteration: start with S0.

Do not improvise the plan. Where STATE.md disagrees with your recollection, the file
wins — your context may have been compacted since the last iteration.

## Each iteration

Read STATE.md. Take the **first** `PENDING` task whose dependencies are `DONE` (exceptions
in ADVISORY §6: S13 may run while a GPU job runs; S11 may precede S10 when time is short).
Mark it `IN_PROGRESS` with a timestamp from `date -u +%FT%TZ`, copy its steps into the STATE
checklist, execute it, run its acceptance check, self-audit (R6), append to
`loops/student_v0/CLAIMS.md`, mark it `DONE` or `BLOCKED — <reason>` (timestamp from
`date -u`), log it, and commit (`loop(student_v0): S<n> <DONE|BLOCKED> — <headline>`) and push
`loop/student_v0`. Mark a task DONE only after all its dependencies are DONE.

A `BLOCKED` task does not block unrelated work. Record it, write the question into
STATE.md's questions-for-operator section, and move on.

Long GPU jobs: launch detached (`setsid nohup ... &`) with logs under
`outputs/student_v0/<task>/logs/`, record the PID in STATE, then poll the log and the output
files at intervals of 10 minutes or more. Never run two GPU jobs at once. Never assume a job
succeeded from its exit code — check that outputs exist and are complete.

## The contract

Every number you report carries the artifact it came from and a command that recomputes
it. `NOT MEASURED — <reason>`, `BLOCKED — <reason>` and `READOUT_FAIL — <check>` are
correct, expected, valuable outcomes; a fabricated number is a critical failure. If G1
fails, report the failure and the ADVISORY §2 branch it implies — never change the headline
set, the baselines or the statistic after seeing results. A model scoring below chance, a
constant answer, or a training run that looks too good is a defect to investigate before it
is a finding. Test splits never enter training; the S6 leakage check proves it.

## Do not

- Train on any test split, any LLM-produced label, teacher (DeepSeek) outputs, UMLS /
  SNOMED / MIMIC data, HLE, or any dataset from the S13 catalog.
- Train on any item of a held-out template (`ct_phase_choice_v1`,
  `fda_boxed_warning_noul_v1`, `pubmed_humans_noul_v1`, and the consistency template chosen
  in S2), from any time window.
- Fit temperatures, thresholds, prompts, templates or checkpoints on a test split.
- Let a language model produce or correct any benchmark gold label.
- Change the D12 thresholds, the G1 definition (D16), or the held-out list (D14).
- Loosen an acceptance rule or edit a check to make it pass (that is `BLOCKED`).
- Run two GPU jobs at once.
- Commit benchmark items, training data, predictions, checkpoints, tokens or secrets.
- Edit `loops/bench_v0/`, `loops/bench_v0_fix0/`, `data/bench/v0.1/`, `AGENTS.md` (except
  S0's "Started" field) or `docs/plans/`.
- Delete, stop or reconfigure pods, volumes, HF repos or remote services; upload anything
  public, including model weights.
- Name application domains, industries, companies or products anywhere in the repository.
- Start loop 2 or anything listed in ADVISORY.md's out-of-scope section.

## Done when

- Every task in STATE.md is `DONE` or `BLOCKED — <reason>`.
- `loops/student_v0/g1.md` states the G1 verdict with every paired difference, its CI and
  its item count, for seen and held-out templates separately.
- `loops/student_v0/FINDINGS.md` exists with a Summary section a non-specialist can follow,
  a body covering every summary claim, and the five-claim spot-check appended.
- Every reported number has a row in `loops/student_v0/CLAIMS.md`.
- `loops/student_v0/NEXT.md` lists proposals (not decisions) for the next loop.
- Everything committable is committed and pushed to `loop/student_v0`.

Then set `Loop status: STOPPED` at the top of STATE.md, commit, push, **stop**, and report:
which tasks completed, which blocked and why, the G1 verdict, and what you would do next.

Also set `Loop status: STOPPED` (and commit + push) if, before S14, every remaining
`PENDING` task is blocked on the operator — in that case still write FINDINGS.md for what
was done. Never set it for any other reason, and never leave it unset at the end of S14.
