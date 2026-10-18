# osler_v3 — RUN STATE

> **This file is the loop's memory.** The live copy is `loops/osler_v3/STATE.md`
> (committed, so the operator can review it remotely). Read it at the start of every
> iteration, before the plan. If it disagrees with your recollection, **the file
> wins** — your context may have been compacted since the last iteration.
>
> **How to update:** set the task `IN_PROGRESS` with a UTC start time *before* working;
> on completion set `DONE` or `BLOCKED — <reason>`, fill the finished time, and append a
> ≤5-line entry to the iteration log. Never delete a log entry; append only. Timestamps
> are `date -u +%FT%TZ`. Never paste item text, predictions, or secrets into this file.

Loop status: `RUNNING`  <!-- set to STOPPED at the hard stop (X12), or when no PENDING task can proceed without the operator -->
Run started (UTC): `<fill in bootstrap>`
Last updated (UTC): `<fill in every iteration>`
Iterations so far: `<increment each wake>`

---

## 1. Task board

Statuses: `PENDING` → `IN_PROGRESS` → `DONE` | `BLOCKED — reason`

| id | task | GPU | deps | status | started (UTC) | finished (UTC) |
|---|---|---|---|---|---|---|
| X0 | Orientation, KL timing | small | — | PENDING | | |
| X1 | Real sources: RAGTruth, DrugProt, MultiCaRe | no | X0 | PENDING | | |
| X2 | Broad replay for A3 | no | X0 | PENDING | | |
| X3 | Gemma-crafted items over real text | yes | X0, X1 | PENDING | | |
| X4 | Charts with the D44 rule | yes | X0 | PENDING | | |
| X5 | MIMIC (official files only; only when they exist) | no | X0 | PENDING | | |
| X6 | Fast dev v3, CG proxy, new baselines | yes | X1, X2, X3, X4 (X5 if built) | PENDING | | |
| X7 | Stage A: recipe ablations A0–A3 + choice | yes | X6 | PENDING | | |
| X8 | Stage B: data blocks | yes | X7 | PENDING | | |
| X9 | Final Osler-4B | yes | X8 | PENDING | | |
| X10 | Evaluation + Gate O3 (Osler-4B; 9B deferred) | yes | X9 | PENDING | | |
| X11 | Memorisation audit (only with MIMIC) | small | X9 | PENDING | | |
| X12 | Findings and closure — HARD STOP | no | all | PENDING | | |

Rules: take the **first** `PENDING` task whose deps are all `DONE` (order exceptions in ADVISORY §5). Never run two GPU jobs
at once. A `BLOCKED` task does not block unrelated work. No row may be left `IN_PROGRESS` at closure.

---

## 2. Key values discovered during the run

| key | value | source | task |
|---|---|---|---|
| KL timing: s/step, anchor share, peak (pplx, kl 0.5) | | | X0 |
| real sources: items per template, licences, screens, leakage | | `loops/osler_v3/sources.md` | X1 |
| broad replay: datasets, licences, A3 replay share | | | X2 |
| Gemma-crafted: pilot keep / agreement rates; items built | | | X3 |
| charts: items built; cue check | | `loops/osler_v3/charts.md` | X4 |
| fast dev v3 partitions; CG partition size and templates; zero-shot pplx on CG | | | X6 |
| Stage A: MR, CG, knowledge, general per run; mean KL; the chosen recipe | | `loops/osler_v3/stage_a.md` | X7 |
| Stage B: kept blocks and dev deltas | | `loops/osler_v3/blocks.md` | X8 |
| final Osler-4B: steps, selected step, CG trend | | | X9 |
| Gate O3 verdict (4B) | | `loops/osler_v3/gate_o3.md` | X10 |

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
Working dir:    outputs/osler_v3/<id>/

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
