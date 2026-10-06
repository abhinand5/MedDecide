# bench_v0_fix0 — RUN STATE

> **This file is the loop's memory.** The live copy is `loops/bench_v0_fix0/STATE.md`
> (committed, so the operator can review it remotely). Read it at the start of every
> iteration, before the plan. If it disagrees with your recollection, **the file
> wins** — your context may have been compacted since the last iteration.
>
> **How to update:** set the task `IN_PROGRESS` with a UTC start time *before* working;
> on completion set `DONE` or `BLOCKED — <reason>`, fill the finished time, and append a
> ≤5-line entry to the iteration log. Never delete a log entry; append only. Timestamps
> are `date -u +%FT%TZ`. Never paste item text, predictions, or secrets into this file.

Loop status: `RUNNING`  <!-- set to STOPPED at the hard stop (F11), or when no PENDING task can proceed without the operator -->
Run started (UTC): `<fill in bootstrap>`
Last updated (UTC): `<fill in every iteration>`
Iterations so far: `<increment each wake>`

---

## 1. Task board

Statuses: `PENDING` → `IN_PROGRESS` → `DONE` | `BLOCKED — reason`

| id | task | GPU | deps | status | started (UTC) | finished (UTC) |
|---|---|---|---|---|---|---|
| F0 | Orientation and baseline snapshot | no | — | PENDING | | |
| F1 | MedMCQA key fix + raw-gold verification for all tier-1 sources | no | F0 | PENDING | | |
| F2 | `noul`/`score` readout fix + reference validation + health gate | yes | F0 | PENDING | | |
| F4 | Fresh tier v0.1: window 2026-03-01, balanced, strict slice | no | F0 | PENDING | | |
| F3 | Template screen v2 (on v0.1) | no | F1, F4 | PENDING | | |
| F5 | Tier-1 contamination probe | yes | F1, F2 | PENDING | | |
| F6 | Ladder baselines on v0.1 with health gate | yes | F2, F3 | PENDING | | |
| F7 | Decision-model baselines | yes | F3 | PENDING | | |
| F8 | Teacher pipeline gate | teacher | F1, F2, F4 | PENDING | | |
| F9 | Corrections record | no | F1, F2, F6 | PENDING | | |
| F10 | Regenerate operator audit sample | no | F3 | PENDING | | |
| F11 | Findings and closure — HARD STOP | no | all | PENDING | | |

Rules: take the **first** `PENDING` task whose deps are all `DONE` (F8 exception in
ADVISORY §6). Never run two GPU tasks at once. A `BLOCKED` task does not block
unrelated work — record it and move on.

---

## 2. Key values discovered during the run

Fill these in as they are measured; later tasks read them from here rather than
recomputing or guessing.

| key | value | source | task |
|---|---|---|---|
| v0.1 fresh window start / end | | `docs/benchmark/fresh_window.md` | F4 |
| strict-slice start | 2026-09-10 | ADVISORY §1 item 4 | F4 |
| MedMCQA gold mismatches after fix (of N) | | `outputs/bench_v0_fix0/F1/gold_verification.json` | F1 |
| `noul` reference agreement (pts) | | | F2 |
| `choice` MedQA reproduction (vs bench_v0) | | | F2 |
| balancing K per template | | | F4 |
| kept / dropped templates (v2 screen) | | | F3 |
| cells passing / failing the health gate | | | F6 |
| teacher logprob API shape | | | F8 |

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
Working dir:    outputs/bench_v0_fix0/<id>/

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
