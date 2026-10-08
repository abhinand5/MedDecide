# GOAL — student_v1 (correct G1, fix the padding bug, diversify the training data, and compare three decision heads on held-out templates)

MedDecide is an open, calibrated medical decision model: a frozen base LLM plus a
decision-path LoRA and a decision head that returns a probability for every offered option in
one forward pass. The previous loop (`student_v0`) trained the first MedDecide-0.8B. Read
correctly (see the advisor note at the top of `loops/student_v0/FINDINGS.md`), it beats
zero-shot Qwen3.5-0.8B by a wide margin on decision types it trained on, but **not** on four
held-out decision types it never saw; G1 against its second baseline (JEV-9B) was never
computed; and a padding bug makes its scores depend on batch composition. This loop: computes G1
properly; fixes and measures the padding bug; checks across the saved checkpoints whether
held-out skill rises or falls with training; builds a **more diverse** gold training set
(PubMed, ≥ 8 new train-only decision types, more record–claim designs, `score` items, vetted
catalog sources); and trains **three heads under matched conditions** — A our pointer head, B a
plain letter-readout LoRA, C a Clef-style joint head through Unsloth — then compares them on
the same held-out templates with the pre-registered G1 rule. It ends with a hard stop for
operator review.

## Read first, in this order

1. `AGENTS.md` (repo root) — project rules, R1–R10, D12, loop mechanics, guardrails.
2. `docs/plans/PROGRAM.md` — the program and settled decisions (note D16–D19).
3. `loops/student_v0/FINDINGS.md` (start with the advisor note at its top) and
   `loops/student_v0/NEXT.md`.
4. `loops/student_v1/ADVISORY.md` — this loop's work plan. Read in full at the start of every
   iteration.
5. `loops/student_v1/STATE.md` — what has already been done. If it still contains the
   `<fill in bootstrap>` placeholders, this is the first iteration: start with V0.

Do not improvise the plan. Where STATE.md disagrees with your recollection, the file wins —
your context may have been compacted since the last iteration.

## Each iteration

Read STATE.md. Take the **first** `PENDING` task whose dependencies are `DONE` (exception in
ADVISORY §6: V4 may run on CPU while V1–V3 use the GPU). Mark it `IN_PROGRESS` with a timestamp
from `date -u +%FT%TZ`, copy its steps into the STATE checklist, execute it, run its acceptance
check, self-audit (R6), append to `loops/student_v1/CLAIMS.md`, mark it `DONE` or
`BLOCKED — <reason>` (timestamp from `date -u`), log it, and commit
(`loop(student_v1): V<n> <DONE|BLOCKED> — <headline>`) and push `loop/student_v1`. Mark a task
DONE only after all its dependencies are DONE.

A `BLOCKED` task does not block unrelated work. Record it, write the question into STATE.md's
questions-for-operator section, and move on. **Do not stop to ask the operator a question that
the ADVISORY already answers**; when a judgement call is needed, take the conservative option,
record it under Deviations, and continue.

Long GPU jobs: launch detached (`setsid nohup ... &`) with logs under
`outputs/student_v1/<task>/logs/`, record the PID in STATE, and poll at intervals of 20 minutes
or more. One GPU job at a time; no sub-agents for GPU work. Never assume a job succeeded from its
exit code — check that outputs exist and are complete.

## The contract

Every number you report carries the artifact it came from and a command that recomputes it.
`NOT MEASURED — <reason>`, `BLOCKED — <reason>` and `READOUT_FAIL — <check>` are correct,
expected, valuable outcomes; a fabricated number is a critical failure. The G1 baselines are
zero-shot `Qwen/Qwen3.5-0.8B` and JEV-9B — nothing else enters a G1 verdict. Arms are compared
only on identical item sets under matched training conditions. Held-out templates and held-out
dev items never influence any choice. If a result is unwelcome, report it and name the
ADVISORY §2 outcome it implies.

## Do not

- Train on any test split, any LLM-produced label (including `LocalLLaMA/typed-decisions`),
  teacher outputs, UMLS / SNOMED / MIMIC data, HLE, or catalog datasets that fail D17.
- Train on, select with, or fit temperatures on any item of a held-out template
  (`ct_phase_choice_v1`, `fda_boxed_warning_noul_v1`, `pubmed_humans_noul_v1`,
  `ct_arm_role_noul_v1`), or on any field those templates ask about.
- Change benchmark v0.2, D12, D14, D16 or the selection rule.
- Put an ablation or a sibling arm into a G1 verdict.
- Install Unsloth into the main project environment (it lives in `envs/unsloth/`).
- Run two GPU jobs at once or spawn sub-agents for GPU work.
- Loosen an acceptance rule or edit a check to make it pass (that is `BLOCKED`).
- Commit benchmark items, training data, predictions, checkpoints, tokens or secrets.
- Edit earlier `loops/*/`, `data/bench/v0.1/`, `data/bench/v0.2/`, `data/train/student_v0/`,
  `outputs/student_v0/`, `AGENTS.md` (except V0's "Started" field) or `docs/plans/`.
- Delete, stop or reconfigure pods, volumes, HF repos or remote services; upload anything
  public, including model weights; push to the HF Hub.
- Name application domains, industries, companies or products anywhere in the repository.
- Start the next loop or anything listed in ADVISORY.md's out-of-scope section.

## Done when

- Every task in STATE.md is `DONE` or `BLOCKED — <reason>` (none left `IN_PROGRESS`).
- `loops/student_v1/g1_student_v0_rerun.md`, `g1.md` and `arms.md` exist with every verdict,
  paired difference, CI and item count.
- `loops/student_v1/FINDINGS.md` exists with a Summary a non-specialist can follow, a body
  covering every summary claim, and the five-claim spot-check **appended with its re-run
  outputs**.
- Every reported number has a row in `loops/student_v1/CLAIMS.md`.
- `loops/student_v1/NEXT.md` lists proposals (not decisions) for the next loop.
- Everything committable is committed and pushed to `loop/student_v1`.

Then set `Loop status: STOPPED` at the top of STATE.md, commit, push, **stop**, and report:
which tasks completed, which blocked and why, the per-arm G1 verdicts, and what you would do
next.

Also set `Loop status: STOPPED` (and commit + push) if, before V10, every remaining `PENDING`
task is blocked on the operator — in that case still write FINDINGS.md for what was done. Never
set it for any other reason, and never leave it unset at the end of V10.
