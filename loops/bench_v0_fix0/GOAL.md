# GOAL — bench_v0_fix0 (repair the benchmark and harness, then re-measure every question type)

Loop `bench_v0` built MedDecide-Bench v0, an eval harness, and zero-shot baselines, but
an advisor review found that part of its record is wrong: the MedMCQA answer key is off by
one (every gold shifted one letter, all "A" answers dropped), the yes/no (`noul`) and
graded (`score`) items are rendered and read incorrectly (label mass ~0.002 instead of
~0.99), several fresh templates are dominated by one class, and the fresh window was
narrower than necessary. Two bench_v0 headline findings ("MedMCQA collapse is position
bias", "relevance judging is at chance") are artefacts of those defects. This loop fixes
the defects, adds a mechanical **readout-health gate** so a pipeline bug can never again
be reported as a finding, rebuilds the benchmark as **v0.1** (fresh window from
2026-03-01, class-balanced, with a strict post-teacher slice), re-runs the ladder
baselines, adds the decision-model baselines and the contamination probe, and records
corrections to bench_v0 without editing it. Its output decides whether loop 1 (the first
trained student) can start on this benchmark. It ends with a hard stop for operator review.

## Read first, in this order

1. `AGENTS.md` (repo root) — project rules, R1–R10, loop mechanics, guardrails.
2. `docs/plans/PROGRAM.md` — the program and settled decisions (note D11, D12).
3. `loops/bench_v0/FINDINGS.md` — what the previous loop did and claimed.
4. `loops/bench_v0_fix0/ADVISORY.md` — this loop's work plan. §1 lists the defects with
   evidence. Read in full at the start of every iteration.
5. `loops/bench_v0_fix0/STATE.md` — what has already been done. If it still contains the
   `<fill in bootstrap>` placeholders, this is the first iteration: start with F0.

Do not improvise the plan. Where STATE.md disagrees with your recollection, the file
wins — your context may have been compacted since the last iteration.

## Each iteration

Read STATE.md. Take the **first** `PENDING` task whose dependencies are `DONE` (the F8
exception in ADVISORY §6). Mark it `IN_PROGRESS` with a timestamp from `date -u +%FT%TZ`,
copy its steps into the STATE checklist, execute it, run its acceptance check, self-audit
(R6), append to `loops/bench_v0_fix0/CLAIMS.md`, mark it `DONE` or `BLOCKED — <reason>`
(timestamp from `date -u`), log it, and commit
(`loop(bench_v0_fix0): F<n> <DONE|BLOCKED> — <headline>`) and push `loop/bench_v0`.

A `BLOCKED` task does not block unrelated work. Record it, write the question into
STATE.md's questions-for-operator section, and move on.

Long GPU jobs: launch detached (`setsid nohup ... &`) with logs under
`outputs/bench_v0_fix0/<task>/logs/`, then poll the log and the outputs. Check that
predictions count equals items count; never trust an exit code alone.

## The contract

Every number you report carries the artifact it came from and a command that recomputes
it. `NOT MEASURED — <reason>`, `BLOCKED — <reason>` and `READOUT_FAIL — <check>` are
correct, expected, valuable outcomes; a fabricated number is a critical failure. A model
scoring below chance, or a label mass far below 1, is a pipeline defect until proven
otherwise — investigate it, do not explain it. If a metric returns an unwelcome answer,
report it; never construct a substitute metric and present it as the result. Benchmark
gold never comes from a language model. Test splits are never used to fit anything. No
item text, prediction, or teacher output is ever committed — the repository is public.

## Do not

- Train or fine-tune anything.
- Use MIMIC, PhysioNet, UMLS, SNOMED, or any credentialed / licensed-vocabulary data.
- Commit benchmark items, model predictions, teacher outputs, tokens, or secrets.
- Edit any file in `loops/bench_v0/` (corrections go in `loops/bench_v0_fix0/CORRECTIONS.md`).
- Overwrite the v0 benchmark data in `data/bench/tier1/` or `data/bench/fresh/`
  (v0.1 goes to `data/bench/v0.1/`), or edit `configs/bench_v0.yaml`.
- Report any (model, template) accuracy that did not pass the readout-health gate.
- Fit temperatures, thresholds, prompts, or templates on any test split.
- Let a language model produce or correct any benchmark gold label.
- Tune prompts per model.
- Loosen an acceptance tolerance or edit a check to make it pass (that is `BLOCKED`).
- Write a timestamp that did not come from `date -u`.
- Commit to or push `main`; delete, stop, or reconfigure pods, volumes, HF repos, the
  teacher machine, or any remote service; upload anything public.
- Edit `AGENTS.md` (except this loop's "Started" field in F0) or `docs/plans/`.
- Start loop 1 or anything listed in ADVISORY.md's out-of-scope section.

## Done when

- Every task in STATE.md is `DONE` or `BLOCKED — <reason>`; none is `PENDING`.
- `loops/bench_v0_fix0/FINDINGS.md` exists with a Summary section a non-specialist can
  follow (stating which bench_v0 findings were wrong and what replaced them), a body
  covering every summary claim, and the five-claim spot-check appended.
- `loops/bench_v0_fix0/CORRECTIONS.md` dispositions every affected bench_v0 claim.
- Every reported number has a row in `loops/bench_v0_fix0/CLAIMS.md`.
- `loops/bench_v0_fix0/NEXT.md` lists proposals (not decisions) for loop 1.
- Everything committable is committed and pushed to `loop/bench_v0`.

Then set `Loop status: STOPPED` at the top of STATE.md, commit, push, **stop**, and
report: which tasks completed, which blocked and why, the headline findings, and what you
would do next. Do not continue past this point.

Also set `Loop status: STOPPED` (and commit + push) if, before F11, every remaining
`PENDING` task is blocked on the operator — in that case still write FINDINGS.md for what
was done. Never set it for any other reason, and never leave it unset at the end of F11.
