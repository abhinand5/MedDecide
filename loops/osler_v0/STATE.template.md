# osler_v0 — RUN STATE

> **This file is the loop's memory.** The live copy is `loops/osler_v0/STATE.md`
> (committed, so the operator can review it remotely). Read it at the start of every
> iteration, before the plan. If it disagrees with your recollection, **the file
> wins** — your context may have been compacted since the last iteration.
>
> **How to update:** set the task `IN_PROGRESS` with a UTC start time *before* working;
> on completion set `DONE` or `BLOCKED — <reason>`, fill the finished time, and append a
> ≤5-line entry to the iteration log. Never delete a log entry; append only. Timestamps
> are `date -u +%FT%TZ`. Never paste item text, predictions, or secrets into this file.

Loop status: `RUNNING`  <!-- set to STOPPED at the hard stop (O12), or when no PENDING task can proceed without the operator -->
Run started (UTC): `<fill in bootstrap>`
Last updated (UTC): `<fill in every iteration>`
Iterations so far: `<increment each wake>`

---

## 1. Task board

Statuses: `PENDING` → `IN_PROGRESS` → `DONE` | `BLOCKED — reason`

| id | task | GPU | deps | status | started (UTC) | finished (UTC) |
|---|---|---|---|---|---|---|
| O0 | Orientation, snapshot, envs, competitor smoke, throughput | smoke | — | PENDING | | |
| O1 | External clinical panel + robustness pack | no | O0 | PENDING | | |
| O2 | Competitor scoreboard | yes | O0, O1 | PENDING | | |
| O3 | Clinical generators (gold by construction) + held-out list | no | O0 | PENDING | | |
| O4 | Training mix v2 | no | O1, O3 | PENDING | | |
| O5 | Readouts: option-code head, non-causal mode, export | small | O0 | PENDING | | |
| O6 | Arm L: option-code head (4B) | yes | O4, O5 | PENDING | | |
| O7 | Arm P: pointer head (4B) | yes | O4, O5 | PENDING | | |
| O8 | Arm N: non-causal option-code head (4B) | yes | O4, O5 | PENDING | | |
| O9 | Head choice at 4B (dev rule) | no | O6–O8 | PENDING | | |
| O10 | Osler-9B + Osler-0.8B reference | yes | O9 | PENDING | | |
| O11 | Evaluation + Gate O1 | yes | O2, O10 | PENDING | | |
| O12 | Findings and closure — HARD STOP | no | all | PENDING | | |

Rules: take the **first** `PENDING` task whose deps are all `DONE` (exceptions in ADVISORY §6:
O3/O4 may run on CPU while O2 uses the GPU; O5 may use the GPU between O2 jobs). Never run two
GPU jobs at once. A `BLOCKED` task does not block unrelated work — record it and move on. No row
may be left `IN_PROGRESS` at closure.

---

## 2. Key values discovered during the run

Fill these in as they are measured; later tasks read them from here rather than
recomputing or guessing.

| key | value | source | task |
|---|---|---|---|
| model revisions (Qwen3.5 0.8B/4B/9B; each competitor) | | `outputs/osler_v0/O0/snapshot.json` | O0 |
| Qwen3.5-4B full-attention layer indices | | `outputs/osler_v0/O0/envs.json` | O0 |
| projected wall-clock of one 4B arm / of the 9B run | | `outputs/osler_v0/O0/throughput.json` | O0 |
| competitor smoke results | | | O0 |
| external panel: datasets, items; robustness pack: base items × perturbations | | `data/bench/v0.3_ext/manifest.json` | O1 |
| scoreboard headline (MedDecider-4B / -9B / pplx v1.1 on v0.2 fresh and external panel) | | `loops/osler_v0/scoreboard.md` | O2 |
| generators: count, held-out list, train/dev/test items | | `loops/osler_v0/heldout.md` | O3 |
| mix v2: items, sources, replay share, max template share; leakage result | | `data/train/osler_v0/manifest.json` | O4 |
| readout checks (init equality, export round-trip, padding, byte-identity) | | `outputs/osler_v0/O5/readout_checks.json` | O5 |
| arms L / P / N: examples seen, selected step, dev macro, tier-1 dev at selection, wall-clock | | | O6–O8 |
| chosen head and the rule's verdict | | `loops/osler_v0/head_choice.md` | O9 |
| Osler-9B / Osler-0.8B: selected step, dev macro | | | O10 |
| Gate O1 verdicts (4B, 9B) incl. knowledge guard | | `loops/osler_v0/gate_o1.md` | O11 |

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
Working dir:    outputs/osler_v0/<id>/

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
