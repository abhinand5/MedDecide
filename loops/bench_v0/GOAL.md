# GOAL — bench_v0 (build MedDecide-Bench v0, validate the harness, measure baselines, gate the teacher)

MedDecide is an open, calibrated medical decision model: it reads a state and typed
questions (`noul` / `choice` / `score`) and returns a probability for every allowed
option from one forward pass, without generating text. Before any model is trained,
this loop builds the measurement foundation: **MedDecide-Bench v0** (tier 1 =
established public test sets with a contamination probe; tier 2 = fresh items from
ClinicalTrials.gov, openFDA drug labels and PubMed records dated after every model's
training cutoff, with gold taken only from structured source fields), an **eval
harness** validated against an independent reference implementation, a **zero-shot
baseline table** over the model ladder and the open decision models that fit one GPU,
and a **teacher pipeline gate** for the self-hosted DeepSeek-V4.1-Flash teacher. It
exists because the predecessor project trusted an eval its own pipeline produced and
lost to BM25 in public; this program measures first. Its output decides how loop 1
(first trained student) starts. It ends with a hard stop for operator review.

## Read first, in this order

1. `AGENTS.md` (repo root) — project rules, the universal rules (R1–R10), loop
   mechanics, guardrails. Read fully.
2. `docs/plans/PROGRAM.md` — the program, the settled decisions, why this loop exists.
   Read once per session.
3. `loops/bench_v0/ADVISORY.md` — the work plan. Read in full at the start of every
   iteration.
4. `loops/bench_v0/STATE.md` — what has already been done. If it still contains the
   `<fill in bootstrap>` placeholders, this is the first iteration: start with T0.

Do not improvise the plan. Where STATE.md disagrees with your recollection, the file
wins — your context may have been compacted since the last iteration.

## Each iteration

Read STATE.md. Take the **first** `PENDING` task whose dependencies are `DONE` (one
documented exception for T10 in ADVISORY §6). Mark it `IN_PROGRESS` with a UTC
timestamp, copy its steps into the STATE checklist, execute it, run its acceptance
check, self-audit its numbers (R6), append to `loops/bench_v0/CLAIMS.md`, then mark it
`DONE` or `BLOCKED — <reason>`, log it in STATE.md's iteration log, and commit
(`loop(bench_v0): T<n> <DONE|BLOCKED> — <headline>`), pushing if the remote accepts.

A `BLOCKED` task does not block unrelated work. Record it, write the question into
STATE.md's questions-for-operator section, and move to the next task whose
dependencies are satisfied.

Long GPU jobs: launch detached (`setsid nohup ... &`) with logs under
`outputs/bench_v0/<task>/logs/`, then poll the log and the output files. Never assume a
job succeeded from its exit code — check that the outputs exist and are complete.

## The contract

Every number you report carries the artifact it came from and a command that recomputes
it. `NOT MEASURED — <reason>` and `BLOCKED — <reason>` are correct, expected, valuable
outcomes; a fabricated number is a critical failure. If a metric returns an unwelcome
answer, report the unwelcome answer — never construct a substitute metric and present its
output as the result. Benchmark gold never comes from a language model. Test splits are
never used to fit anything. No item text, prediction, or teacher output is ever
committed — the repository is public.

## Do not

- Train or fine-tune anything (no LoRA, no pointer head, no gradient steps).
- Use MIMIC, PhysioNet, UMLS, SNOMED, or any credentialed / licensed-vocabulary data.
- Commit benchmark items, model predictions, teacher outputs, tokens, or secrets.
- Fit temperatures, thresholds, prompts, or templates on any test split.
- Let a language model produce or correct any benchmark gold label.
- Tune prompts per model to improve baseline scores.
- Loosen an acceptance tolerance or edit a check to make it pass (that is `BLOCKED`).
- Delete, stop, or reconfigure pods, volumes, HF repos, the teacher machine, or any
  remote service; upload anything public.
- Edit `AGENTS.md` (except the "Started" field in T0) or `docs/plans/`.
- Start loop 1 or anything listed in ADVISORY.md's out-of-scope section.

## Done when

- Every task in STATE.md is `DONE` or `BLOCKED — <reason>`.
- `loops/bench_v0/FINDINGS.md` exists with a Summary section a non-specialist can
  follow, a body covering every summary claim, and the five-claim spot-check appended.
- Every reported number has a row in `loops/bench_v0/CLAIMS.md`.
- `loops/bench_v0/NEXT.md` lists proposals (not decisions) for loop 1.
- Everything committable is committed and pushed.

Then set `Loop status: STOPPED` at the top of STATE.md, commit, push, **stop**, and
report: which tasks completed, which blocked and why, the headline findings, and what
you would do next. Do not continue past this point.

Also set `Loop status: STOPPED` (and commit + push) if, before T12, every remaining
`PENDING` task is blocked on the operator — in that case still write FINDINGS.md for
what was done (T12 covers a partial loop too). An outer script re-invokes you until
it sees `STOPPED`, so never set it for any other reason, and never leave it unset at
the end of T12.
