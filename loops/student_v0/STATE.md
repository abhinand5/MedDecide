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
Run started (UTC): `<fill in bootstrap>`
Last updated (UTC): `<fill in every iteration>`
Iterations so far: `<increment each wake>`

---

## 1. Task board

Statuses: `PENDING` → `IN_PROGRESS` → `DONE` | `BLOCKED — reason`

| id | task | GPU | deps | status | started (UTC) | finished (UTC) |
|---|---|---|---|---|---|---|
| S0 | Orientation, snapshot, fused kernels | smoke | — | PENDING | | |
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
| fused kernels installed (yes/no, versions) | | `outputs/student_v0/S0/kernels.json` | S0 |
| Qwen3.5-0.8B tokens/s at 8k / 16k prompt, before → after kernels | | | S0 |
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
Task in flight: none
Working dir:    outputs/student_v0/<id>/

- [ ] <step 1>
- [ ] <step 2>
- [ ] acceptance check run and passed
- [ ] self-audit (R6) written to SELF_AUDIT.md
- [ ] CLAIMS.md rows appended
- [ ] STATE updated, committed, pushed
```

---

## 4. Iteration log (append only, newest last)

<!-- Template for each entry:
### <task-id> — <DONE|BLOCKED> — <UTC timestamp>
- What ran: <command or script>
- Output: <path>
- Headline: <one number or one sentence, with its CLAIMS id>
- Surprises: <anything unexpected, or "none">
- Next: <what this unblocks>
-->

<!-- The Headline line matters beyond this file: these headlines are the raw material
     FINDINGS.md's Summary section is written from at closure. A headline that only
     makes sense with full context ("done, see report") starves the summary. Write
     each one so a reader who has not seen the task can repeat it: what was measured,
     what came out, with what denominator. -->

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
