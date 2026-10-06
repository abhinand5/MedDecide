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
Run started (UTC): `2026-10-06T05:47:36Z`
Last updated (UTC): `2026-10-06T05:57:21Z`
Iterations so far: `1`

---

## 1. Task board

Statuses: `PENDING` → `IN_PROGRESS` → `DONE` | `BLOCKED — reason`

| id | task | GPU | deps | status | started (UTC) | finished (UTC) |
|---|---|---|---|---|---|---|
| F0 | Orientation and baseline snapshot | no | — | IN_PROGRESS | 2026-10-06T05:47:36Z | |
| F1 | MedMCQA key fix + raw-gold verification for all tier-1 sources | no | F0 | DONE | 2026-10-06T05:48:40Z | 2026-10-06T05:57:21Z |
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
| MedMCQA gold mismatches after fix (of N) | **0 of 4,183** (all 8 tier-1 sources: 0 of 15,915 checked) | `outputs/bench_v0_fix0/F1/gold_verification.json` | F1 |
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
Task in flight: F2 (`noul`/`score` readout fix + reference validation + health gate)
Working dir:    outputs/bench_v0_fix0/F2/

- [ ] render every item's options with letter labels; read letter tokens (same path as validated choice)
- [ ] noul: "A. Yes" / "B. No"; score: A = lowest level; record letter -> option-key map per item
- [ ] keep the choice path byte-identical; re-run T7's MedQA comparison to prove it
- [ ] validate noul against lm-evaluation-harness (boolq or PubMedQA yes/no) within +/-2 pts
- [ ] implement the readout-health gate (median label mass >= 0.5, greedy agreement >= 0.9,
      CI upper bound not below chance) with unit tests
- [ ] 200 noul + 200 score items for 0.8B and 4B: median label mass >= 0.9, greedy agreement >= 0.9
- [ ] acceptance check run and passed
- [ ] self-audit (R6) written to SELF_AUDIT.md
- [ ] CLAIMS.md rows appended
- [ ] STATE updated, committed, pushed
```

---

## 4. Iteration log (append only, newest last)

### F1 — DONE — 2026-10-06T05:57:21Z
- What ran: fixed `_medmcqa_item` (cop is 0-based), wrote `scripts/bench/verify_gold.py` and
  `src/meddecide/bench/verify.py`, rebuilt tier-1 as v0.1, ran the independent gold check twice.
- Output: `outputs/bench_v0_fix0/F1/` (gold_verification.json, SELF_AUDIT.md),
  `data/bench/v0.1/tier1/` (manifest, acceptance, audit).
- Headline: **0 gold mismatches of 15,915 checked items across all eight tier-1 sources** (X004);
  MedMCQA now has 4,183 test items with gold A 1,348 / B 1,085 / C 925 / D 825, identical to the raw
  `cop` counts (X002, X003).
- Surprises: verification also caught a *real* second loader bug - three nfcorpus score items
  carried a grade borrowed from another query's pool (X005) - plus two defects in my own checker
  (label-vs-key comparison, pooled qrels) that produced false alarms before the real bug surfaced.
- Next: F2 (`noul`/`score` readout), which gates every GPU number in this loop.


### F0 — DONE — 2026-10-06T05:48:40Z
- What ran: `git rev-parse HEAD`, `sha256sum` over every v0 benchmark JSONL, manifest totals via
  `jq`, `uv run pytest --junit-xml`, `uv run ruff check .`.
- Output: `outputs/bench_v0_fix0/F0/{start.json,pytest.txt,pytest.xml,SELF_AUDIT.md}`.
- Headline: the repair loop starts from a provable snapshot — commit `88dea18`, 11 v0 files
  hashed, tier-1 21,202 items and fresh 41,502 items (reproducing bench_v0 C013/C018), test suite
  74 passed / 0 failed (X001).
- Surprises: the repo's `-q` addopts suppresses pytest's console summary under `--tb=no`, so the
  count had to come from the JUnit XML; the first parse attempt raised IndexError and is recorded
  in the audit rather than hidden.
- Next: F1 (MedMCQA key fix and gold verification for all eight tier-1 sources).

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
