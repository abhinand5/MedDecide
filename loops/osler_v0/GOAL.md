# GOAL — osler_v0 (competitor scoreboard, clinical generators, and the first Osler-4B / -9B)

Osler (repo and benchmark name: MedDecide) is an open, calibrated medical decision model: a
frozen base LLM plus a decision-path LoRA and a readout that returns a probability for every
offered option in one forward pass. The previous loop (`student_v1`) showed at 0.8B that the
choice of head barely matters (three heads within 0.6 pts), that no head transfers to held-out
decision types, and that training on the decision path wiped out exam knowledge. This loop:
measures every open competitor through its authors' own code on our benchmark, a new external
clinical panel and a robustness pack; builds clinical decision generators whose labels are
computed by code (D20); trains four matched 4B arms (option-code readout, pointer, non-causal
option-code readout, head-only) and picks a head by a pre-registered dev rule; trains Osler-9B
and an Osler-0.8B reference with that head; and applies Gate O1 (D24). It ends with a hard stop
for operator review.

## Read first, in this order

1. `AGENTS.md` (repo root) — project rules, R1–R10, D12, loop mechanics, guardrails.
2. `docs/plans/PROGRAM.md` — the program and settled decisions (note D20–D24).
3. `loops/student_v1/FINDINGS.md` and `loops/student_v1/NEXT.md`.
4. `loops/osler_v0/ADVISORY.md` — this loop's work plan. Read in full at the start of every
   iteration.
5. `loops/osler_v0/STATE.md` — what has already been done. If it still contains the
   `<fill in bootstrap>` placeholders, this is the first iteration: start with O0.

Do not improvise the plan. Where STATE.md disagrees with your recollection, the file wins —
your context may have been compacted since the last iteration.

## Each iteration

Read STATE.md. Take the **first** `PENDING` task whose dependencies are `DONE` (exceptions in
ADVISORY §6: O3/O4 may run on CPU while O2 uses the GPU). Mark it `IN_PROGRESS` with a timestamp
from `date -u +%FT%TZ`, copy its steps into the STATE checklist, execute it, run its acceptance
check, self-audit (R6), append to `loops/osler_v0/CLAIMS.md`, mark it `DONE` or
`BLOCKED — <reason>` (timestamp from `date -u`), log it, and commit
(`loop(osler_v0): O<n> <DONE|BLOCKED> — <headline>`) and push `loop/osler_v0`.

A `BLOCKED` task does not block unrelated work. Record it, write the question into STATE.md's
questions-for-operator section, and move on. **Do not stop to ask the operator a question that
the ADVISORY already answers**; when a judgement call is needed, take the conservative option,
record it under Deviations, and continue.

Long GPU jobs: launch detached (`setsid nohup ... &`) with logs under
`outputs/osler_v0/<task>/logs/`, record the PID in STATE, and poll at intervals of 20 minutes
or more. One GPU job at a time; no sub-agents for GPU work. Never assume a job succeeded from
its exit code — check that outputs exist and are complete.

## The contract

Every number you report carries the artifact it came from and a command that recomputes it.
`NOT MEASURED — <reason>`, `BLOCKED — <reason>` and `READOUT_FAIL — <check>` are correct,
expected, valuable outcomes; a fabricated number is a critical failure. Gate O1's baselines are
exactly MedDecider-4B and zero-shot Qwen3.5-4B (for Osler-4B) and MedDecider-9B and JEV-9B (for
Osler-9B) — every other model is additional. Competitors run through their authors' code or
are `NOT MEASURED`. Held-out templates, held-out generators and the external panel never
influence any choice. If a result is unwelcome, report it and name the ADVISORY §2 outcome it
implies.

## Do not

- Train on any test split, any external-panel dataset (any split), any LLM-produced text or
  label, teacher outputs, UMLS / SNOMED / MIMIC data, HLE, or competitor outputs or weights.
- Train on, select with, or fit temperatures on any item of a held-out template
  (`ct_phase_choice_v1`, `fda_boxed_warning_noul_v1`, `pubmed_humans_noul_v1`,
  `ct_arm_role_noul_v1`), any field those templates ask about, or any held-out generator.
- Change benchmark v0.2, D12 for zero-shot cells, D14, D16, D21, D24, the held-out generator
  list after O6 starts, or the O10 head-choice rule.
- Put a sibling arm, a 27B/31B model or any other additional row into a Gate O1 verdict.
- Install a competitor's dependencies into the main project environment (use `envs/<name>/`).
- Run two GPU jobs at once or spawn sub-agents for GPU work.
- Loosen an acceptance rule or edit a check to make it pass (that is `BLOCKED`).
- Commit benchmark items, generated items, training data, predictions, checkpoints, tokens or
  secrets.
- Edit earlier `loops/*/`, `data/bench/v0.1/`, `data/bench/v0.2/`, `data/train/student_v0/`,
  `data/train/student_v1/`, earlier `outputs/*/`, `AGENTS.md` (except O0's "Started" field) or
  `docs/plans/`.
- Delete, stop or reconfigure pods, volumes, HF repos or remote services; upload anything
  public, including model weights; push to the HF Hub; submit to an external leaderboard.
- Start the next loop or anything listed in ADVISORY.md's out-of-scope section.

## Done when

- Every task in STATE.md is `DONE` or `BLOCKED — <reason>` (none left `IN_PROGRESS`).
- `loops/osler_v0/scoreboard.md`, `heldout.md`, `head_choice.md`, `gate_o1.md` and
  `results.md` exist with every verdict, paired difference, CI and item count.
- `loops/osler_v0/FINDINGS.md` exists with a Summary a non-specialist can follow, a body
  covering every summary claim, and the five-claim spot-check **appended with its re-run
  outputs**.
- Every reported number has a row in `loops/osler_v0/CLAIMS.md`.
- `loops/osler_v0/NEXT.md` lists proposals (not decisions) for the next loop.
- Everything committable is committed and pushed to `loop/osler_v0`.

Then set `Loop status: STOPPED` at the top of STATE.md, commit, push, **stop**, and report:
which tasks completed, which blocked and why, the Gate O1 verdicts, and what you would do next.

Also set `Loop status: STOPPED` (and commit + push) if, before O13, every remaining `PENDING`
task is blocked on the operator — in that case still write FINDINGS.md for what was done. Never
set it for any other reason, and never leave it unset at the end of O13.
