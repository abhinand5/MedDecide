# student_v0 — RUN STATE

> **This file is the loop's memory.** The live copy is `loops/student_v0/STATE.md`
> (committed, so the operator can review it remotely). Read it at the start of every
> iteration, before the plan. If it disagrees with your recollection, **the file
> wins** — your context may have been compacted since the last iteration.
>
> **How to update:** set the task `IN_PROGRESS` with a UTC start time *before* working;
> on completion set `DONE` or `BLOCKED — <reason>`, fill the finished time, and append a
> ≤5-line entry to the iteration log. Never delete a log entry; append only. Timestamps
> are `date -u +%FT%TZ`. Never paste item text, predictions, or secrets into this file.

Loop status: `RUNNING`  <!-- set to STOPPED at the hard stop (S14), or when no PENDING task can proceed without the operator -->
Run started (UTC): `2026-10-06T18:06:22Z`
Last updated (UTC): `2026-10-06T18:27:30Z`
Iterations so far: `1`

---

## 1. Task board

Statuses: `PENDING` → `IN_PROGRESS` → `DONE` | `BLOCKED — reason`

| id | task | GPU | deps | status | started (UTC) | finished (UTC) |
|---|---|---|---|---|---|---|
| S0 | Orientation, snapshot, fused kernels | smoke | — | DONE | 2026-10-06T18:06:22Z | 2026-10-06T18:27:30Z |
| S1 | Benchmark v0.2 fixes | small | S0 | PENDING | | |
| S2 | Record–claim consistency templates | small | S1 | PENDING | | |
| S3 | Long-record slice + shared-prefix measurement | yes | S2 | PENDING | | |
| S4 | HLE medical subset (supplementary test) | small | S1 | PENDING | | |
| S5 | Tier-1 train-split builders | no | S1 | PENDING | | |
| S6 | Pre-window structured-gold data + training mix + leakage check | no | S2, S5 | PENDING | | |
| S7 | Training code: LoRA + pointer head | smoke | S0 | PENDING | | |
| S8 | Evaluation path for trained models | yes | S7 | PENDING | | |
| S9 | Train MedDecide-0.8B (instruct) | yes | S6, S8 | PENDING | | |
| S10 | Ablation: MedDecide-0.8B from Base | yes | S9 | PENDING | | |
| S11 | Gate G1 evaluation | yes | S9, S3, S4 | PENDING | | |
| S12 | Byte-identity + audit read-back | small | S9 | PENDING | | |
| S13 | Candidate dataset catalog | no | S0 | PENDING | | |
| S14 | Findings and closure — HARD STOP | no | all | PENDING | | |

Rules: take the **first** `PENDING` task whose deps are all `DONE` (exceptions in
ADVISORY §6: S13 may run while a GPU job runs; S11 may precede S10 when time is short).
Never run two GPU jobs at once. A `BLOCKED` task does not block
unrelated work — record it and move on.

---

## 2. Key values discovered during the run

Fill these in as they are measured; later tasks read them from here rather than
recomputing or guessing.

| key | value | source | task |
|---|---|---|---|
| fused kernels installed (yes/no, versions) | **yes** — `causal_conv1d` 1.7.0, `flash-linear-attention` 0.5.2 (triton 3.8.0); both bound by transformers 5.18 (`fallback_warnings` 2 → 0) | `outputs/student_v0/S0/kernels.json` | S0 |
| Qwen3.5-0.8B tokens/s at 8k / 16k prompt, before → after kernels | **40,809 → 124,526** (8k) and **36,399 → 106,697** (16k), best of 5 after warmup at length | `outputs/student_v0/S0/kernels.json` | S0 |
| v0.2 manifest sha256 | | `data/bench/v0.2/manifest.json` | S1 |
| consistency templates built (ids) + the held-out one | | `docs/benchmark/consistency_templates.md` | S2 |
| long-record slice size (items, templates) | | | S3 |
| HLE-med item count | | | S4 |
| training items per source (tier1_train / prewindow) | | `data/train/student_v0/manifest.json` | S5/S6 |
| leakage check result (counts removed / found) | | | S6 |
| training throughput (tokens/s, items/s) | | | S7 |
| S9 steps, tokens seen, wall-clock, best dev step | | | S9 |
| G1 verdict (seen / held-out) | | `loops/student_v0/g1.md` | S11 |

---

## 3. Current task checklist

**Fill this in before starting any task**, by copying that task's concrete steps out of
the plan as unticked boxes. Tick each one the moment it is finished, not at the end.
This is the only record of progress *within* a task — without it, a compaction mid-task
leaves you unable to tell what you already did.

Clear this section and write the new task's checklist when you start the next task; the
completed checklist goes into the iteration-log entry.

```
Task in flight: none — S0 finished 2026-10-06T18:27:30Z
Working dir:    outputs/student_v0/S0/

- [x] fill "Run started" in STATE + "Started" in AGENTS.md
- [x] snapshot.json: git commit, v0.1 manifest hashes, test count
- [x] install causal_conv1d + flash-linear-attention for torch 2.14 / cu130 / sm_120
- [x] measure Qwen3.5-0.8B tokens/s + peak memory, 8k and 16k prompt, before -> after kernels
- [x] kernels.json written (numbers, or NOT MEASURED with reason)
- [x] acceptance check run and passed (snapshot.json + kernels.json exist; uv run pytest passes)
- [x] self-audit (R6) written to SELF_AUDIT.md
- [x] CLAIMS.md rows appended (S001–S009)
- [x] STATE updated, committed, pushed
```

---

## 4. Iteration log (append only, newest last)

### S0 — DONE — 2026-10-06T18:27:30Z
- What ran: `uv pip install causal-conv1d flash-linear-attention` (TORCH_CUDA_ARCH_LIST=12.0, built from source ~13 min); `scripts/student/s0_kernels.py --kernels off|auto --lengths 8192 16384 --reps 5 --warmup-at-length`; `scripts/student/s0_snapshot.py`
- Output: `outputs/student_v0/S0/{snapshot.json,kernels.json,kernels_before.json,kernels_after.json,SELF_AUDIT.md,pytest.xml}`
- Headline: fused kernels are installed **and bound** (transformers fallback warnings 2 → 0), and Qwen3.5-0.8B prefill goes **40,809 → 124,526 tok/s at 8k (3.05×)** and **36,399 → 106,697 tok/s at 16k (2.93×)**, peak allocated 2.33 → 1.97 GiB at 8k (S001–S006).
- Surprises: a single-shot measurement said the kernels made it **36× slower**; the cause was Triton JIT-compiling on the first call at each new sequence length (~7.5 s) landing inside the timed region — re-measured with a warmup pass at length (S007). Also, the first binding check read `func.__module__`, which `functools.wraps` copies from the *torch* function, so it reported a fallback that was not happening.
- Next: S1 (benchmark v0.2 fixes) is unblocked.

## 5. Blocked items

| id | what is blocked | exact reason | what would unblock it |
|---|---|---|---|

---

## 6. Questions for the operator

Anything you could not resolve without a human. Be specific enough to answer without
re-reading the run: state the ambiguity, the options, and which you would pick.

---

## 7. Deviations from the plan

Any place you departed from GOAL/ADVISORY, with the reason. An empty section is the
expected outcome. Editing code or a check to make it pass is never an acceptable
deviation — that is a `BLOCKED`.

---

## 8. Closure summary feed

<!-- Filled at closure, feeding FINDINGS.md's Summary section. One row per major
     outcome, in the order a human should hear them. Each claim cites its CLAIMS id;
     failures and blocked tasks get rows too. -->

| # | outcome (one sentence, plain language) | claims | artifact |
|---|---|---|---|
| | | | |
