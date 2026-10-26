# osler_v5 — RUN STATE

> **This file is the loop's memory.** The live copy is `loops/osler_v5/STATE.md`
> (committed, so the operator can review it remotely). Read it at the start of every
> iteration, before the plan. If it disagrees with your recollection, **the file
> wins** — your context may have been compacted since the last iteration.
>
> **How to update:** set the task `IN_PROGRESS` with a UTC start time *before* working;
> on completion set `DONE` or `BLOCKED — <reason>`, fill the finished time, and append a
> ≤5-line entry to the iteration log. Never delete a log entry; append only. Timestamps
> are `date -u +%FT%TZ`. Never paste item text, predictions, or secrets into this file.

Loop status: `RUNNING`  <!-- set to STOPPED at the hard stop (Z7), or when no PENDING task can proceed without the operator -->
Run started (UTC): `<fill in bootstrap>`
Last updated (UTC): `<fill in every iteration>`
Iterations so far: `<increment each wake>`

---

## 1. Task board

Statuses: `PENDING` → `IN_PROGRESS` → `DONE` | `BLOCKED — reason`

| id | task | GPU | deps | status | started (UTC) | finished (UTC) |
|---|---|---|---|---|---|---|
| Z0 | Orientation and recipe code (D51–D54) | no | — | PENDING | | |
| Z1 | Probes on the osler_v4 final (blend curve, readout reset) | yes | Z0 | PENDING | | |
| Z2 | Recipe arms R0–R3 and the choice | yes | Z0 | PENDING | | |
| Z3 | Final Osler-4B (63,304 steps) | yes | Z2 | PENDING | | |
| Z4 | Blend α on dev and export | yes | Z3 | PENDING | | |
| Z5 | Evaluation, Gate O5, scoreboard | yes | Z4 | PENDING | | |
| Z6 | Membership audit | small | Z4 | PENDING | | |
| Z7 | Findings and closure — HARD STOP | no | all | PENDING | | |

Rules: take the **first** `PENDING` task whose deps are all `DONE` (order exceptions and the GPU sequence in ADVISORY §5).
Never run two GPU jobs at once. A `BLOCKED` task does not block unrelated work. No row may be
left `IN_PROGRESS` at closure.

---

## 2. Key values discovered during the run

**Operator confirmations** (filled by the operator before kickoff; the loop never edits this line):
- D51–D55 (per-item weighting, option-count augmentation and positional health, blend, dev composite, Gate O5): `yes` — recorded 2026-10-26 by the advisor on the operator's standing instruction in chat ("I agree with whatever you go with"), after the independent review (commit a9ca386).

| key | value | source | task |
|---|---|---|---|
| suite counts (passed / skipped / deselected); disk headroom test | | | Z0 |
| mix composition SHA-256 (must equal 9d25e30f…) | | | Z0 |
| closed-set list (template, label-set size, reason); expected share of trained items with > 26 options | | | Z0 |
| S, its parts and positional health for zero-shot pplx, A0, osler_v4 final (additional) | | | Z0 |
| probes: blend curve on the osler_v4 final; readout-reset DDXPlus split | | `loops/osler_v5/probes.md` | Z1 |
| arms R0–R3: S, parts, knowledge, positional health, MIMIC gradient share; the chosen arm | | `loops/osler_v5/arms.md` | Z2 |
| final: selected step; S trend; positional health per evaluation | | | Z3 |
| blend: S, parts, CG floor per α; the chosen α | | `loops/osler_v5/blend.md` | Z4 |
| Gate O5 verdict; release targets | | `loops/osler_v5/gate_o5.md` | Z5 |
| membership AUC | | `loops/osler_v5/audit.md` | Z6 |

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
Working dir:    outputs/osler_v5/<id>/

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
