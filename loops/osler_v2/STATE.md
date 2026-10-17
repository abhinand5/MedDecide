# osler_v2 — RUN STATE

> **This file is the loop's memory.** The live copy is `loops/osler_v2/STATE.md`
> (committed, so the operator can review it remotely). Read it at the start of every
> iteration, before the plan. If it disagrees with your recollection, **the file
> wins** — your context may have been compacted since the last iteration.
>
> **How to update:** set the task `IN_PROGRESS` with a UTC start time *before* working;
> on completion set `DONE` or `BLOCKED — <reason>`, fill the finished time, and append a
> ≤5-line entry to the iteration log. Never delete a log entry; append only. Timestamps
> are `date -u +%FT%TZ`. Never paste item text, predictions, or secrets into this file.

Loop status: `RUNNING`  <!-- set to STOPPED at the hard stop (W10), or when no PENDING task can proceed without the operator -->
Run started (UTC): `<fill in bootstrap>`
Last updated (UTC): `<fill in every iteration>`
Iterations so far: `<increment each wake>`

---

## 1. Task board

Statuses: `PENDING` → `IN_PROGRESS` → `DONE` | `BLOCKED — reason`

| id | task | GPU | deps | status | started (UTC) | finished (UTC) |
|---|---|---|---|---|---|---|
| W0 | Orientation | no | — | PENDING | | |
| W1 | Audits; MR suite v1 frozen | small | W0 | PENDING | | |
| W2 | Replay ≥ 20 %; EHR-Rel | no | W0 | PENDING | | |
| W3 | ICD prose v2 (optional) | no | W0 | PENDING | | |
| W3b | Whole synthetic charts, local Gemma 4 teacher | yes | W0 | PENDING | | |
| W4 | MIMIC (official files only; only when they exist) | no | W0 | PENDING | | |
| W5 | Fast dev v2, knowledge dev v2, new test templates and baselines | yes | W1, W2, W3b (W3, W4 if built) | PENDING | | |
| W6 | Blocks v2 + keep rule | yes | W5 | PENDING | | |
| W7 | Final Osler-4B (9B deferred, D39) | yes | W6 | PENDING | | |
| W8 | Evaluation + Gate O3 (Osler-4B; 9B NOT MEASURED) | yes | W7 | PENDING | | |
| W9 | Memorisation audit (only with MIMIC) | small | W7 | PENDING | | |
| W10 | Findings and closure — HARD STOP | no | all | PENDING | | |

Rules: take the **first** `PENDING` task whose deps are all `DONE` (exceptions in ADVISORY §5). Never run two GPU jobs at
once. A `BLOCKED` task does not block unrelated work. No row may be left `IN_PROGRESS` at closure.

---

## 2. Key values discovered during the run

| key | value | source | task |
|---|---|---|---|
| teacher allowance / DeepSeek balance; MIMIC files | | | W0 |
| too-good ICD probe result; NB screen per template; ceiling statistic; MR suite v1 verdict templates | | `loops/osler_v2/mr_suite_v1.md` | W1 |
| replay share of the largest block; sources and licences; EHR-Rel items | | | W2 |
| ICD prose v2 pilot pass rate; pairs built; spend | | | W3 |
| charts: pilot keep rate, charts/h, items per template, NB screens | | `loops/osler_v2/charts.md` | W3b |
| fast dev v2 partitions; knowledge dev v2 size; zero-shot knowledge dev | | | W5 |
| speed check: real tokens/s and peak per backbone | | | W0 |
| base choice (C0p − C0q, three partitions) and the chosen base | | `loops/osler_v2/blocks.md` | W6 |
| kept blocks and dev deltas | | `loops/osler_v2/blocks.md` | W6 |
| Osler-4B: base, steps, selected step, dev template macro | | | W7 |
| Gate O3 verdict (4B; 9B NOT MEASURED) | | `loops/osler_v2/gate_o3.md` | W8 |

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
Working dir:    outputs/osler_v2/<id>/

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
