# osler_v1 — RUN STATE

> **This file is the loop's memory.** The live copy is `loops/osler_v1/STATE.md`
> (committed, so the operator can review it remotely). Read it at the start of every
> iteration, before the plan. If it disagrees with your recollection, **the file
> wins** — your context may have been compacted since the last iteration.
>
> **How to update:** set the task `IN_PROGRESS` with a UTC start time *before* working;
> on completion set `DONE` or `BLOCKED — <reason>`, fill the finished time, and append a
> ≤5-line entry to the iteration log. Never delete a log entry; append only. Timestamps
> are `date -u +%FT%TZ`. Never paste item text, predictions, or secrets into this file.

Loop status: `RUNNING`  <!-- set to STOPPED at the hard stop (V12), or when no PENDING task can proceed without the operator -->
Run started (UTC): `<fill in bootstrap>`
Last updated (UTC): `<fill in every iteration>`
Iterations so far: `<increment each wake>`

---

## 1. Task board

Statuses: `PENDING` → `IN_PROGRESS` → `DONE` | `BLOCKED — reason`

| id | task | GPU | deps | status | started (UTC) | finished (UTC) |
|---|---|---|---|---|---|---|
| V0 | Orientation, environment, speed recipe | small | — | PENDING | | |
| V1 | MR generators | no | V0 | PENDING | | |
| V2 | ICD-10-CM verification | no | V0 | PENDING | | |
| V3 | Replay to 20 % | no | V0 | PENDING | | |
| V4 | Teacher prose | no | V1, V2, V5a | PENDING | | |
| V5a | Synthea | no | V0 | PENDING | | |
| V5b | Real clinical text | no | V2 | PENDING | | |
| V5c | MIMIC sources (only when the files exist) | no | V0 | PENDING | | |
| V6 | MR evaluation suite + baselines | yes | V1, V5a, V5b (V4, V5c if built) | PENDING | | |
| V7 | Mix v3 blocks | no | V3, V6 | PENDING | | |
| V8 | Block runs + keep rule | yes | V7 | PENDING | | |
| V9 | Final Osler-4B and -9B | yes | V8 | PENDING | | |
| V10 | Evaluation + Gate O2 | yes | V9 | PENDING | | |
| V11 | Memorisation audit | small | V9 | PENDING | | |
| V12 | Findings and closure — HARD STOP | no | all | PENDING | | |

Rules: take the **first** `PENDING` task whose deps are all `DONE` (exceptions in ADVISORY §6: V1–V5b in any order on CPU;
V4 and V5a–V5c may run while a GPU task runs; V5c runs only when its files appear and never blocks). Never run two GPU jobs at once. A `BLOCKED` task does not block unrelated work.
No row may be left `IN_PROGRESS` at closure.

---

## 2. Key values discovered during the run

| key | value | source | task |
|---|---|---|---|
| teacher: configured? model id, status | | `outputs/osler_v1/V0/env.json` | V0 |
| MIMIC files present (which, sizes) | | `outputs/osler_v1/V0/env.json` | V0 |
| batching constant (batch_size, max_batch_tokens, gradient checkpointing); items/s; peak GB | | `outputs/osler_v1/V0/speed.json` | V0 |
| MR generators: items per split, screen results | | `loops/osler_v1/mr_data.md` | V1 |
| ICD-10-CM FY2026 order file sha256; verify result | | `data/cms/icd10cm/2026/` | V2 |
| replay share of B0; sources and licences | | | V3 |
| teacher: pilot pass rate; pairs built; spend | | | V4 |
| Synthea: encounters, coverage, items per template | | `loops/osler_v1/mr_data.md` | V5a |
| real text: MTSamples pages per class; CodiEsp items; PMC pairs | | `loops/osler_v1/mr_data.md` | V5b |
| MIMIC (if it arrives): pages per class; ICD rows per family; late-anchor removals | | `loops/osler_v1/mr_data.md` | V5c |
| MR baselines (template macro per model) | | `loops/osler_v1/mr_baselines.md` | V6 |
| blocks: rows, replay share; steps per block run | | | V7 |
| kept blocks and their dev deltas | | `loops/osler_v1/blocks.md` | V8 |
| Osler-4B / -9B: selected step, dev template macro | | | V9 |
| Gate O2 verdicts | | `loops/osler_v1/gate_o2.md` | V10 |

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
Working dir:    outputs/osler_v1/<id>/

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
