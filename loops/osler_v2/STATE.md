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
Run started (UTC): `2026-10-17T13:34:35Z`
Last updated (UTC): `2026-10-17T14:06:18Z`
Iterations so far: `1`

---

## 1. Task board

Statuses: `PENDING` → `IN_PROGRESS` → `DONE` | `BLOCKED — reason`

| id | task | GPU | deps | status | started (UTC) | finished (UTC) |
|---|---|---|---|---|---|---|
| W0 | Orientation | no | — | DONE | 2026-10-17T13:34:35Z | 2026-10-17T14:06:18Z |
| W1 | Audits; MR suite v1 frozen | small | W0 | DONE | 2026-10-17T14:07:22Z | 2026-10-17T14:13:48Z |
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
| teacher allowance / DeepSeek balance; MIMIC files | OpenCode Go reachable (HTTP 200, 37 ids, chat OK; 5-hour and weekly allowances NOT MEASURED, not exposed to the client); DeepSeek balance USD 9.88 (loop cap $9.00); MIMIC core 0 of 5 files | `loops/osler_v2/CLAIMS.md` W003–W005; `outputs/osler_v2/W0/env_probe.json` | W0 |
| too-good ICD probe result; NB screen per template; ceiling statistic; MR suite v1 verdict templates | Probe 0.612 (339/554); overlap with B1 0 exact / 0 near-dup; NB dev macro 1.000 (page tabular), 0.796 (indicator), 0.804 (ICD), 1.000 (synthea page), 0.576 (synthea indicator), 0.590 (MTSamples), 0.520 (PMC); ceiling 1.000 / 0.695 / 0.710 / 0.995 / 0.788 / 0.629 / 0.780; verdict 13,103 test items, reported only 6,191; codiesp and mkurman screens NOT MEASURED (no dev) | `loops/osler_v2/mr_suite_v1.md`; `outputs/osler_v2/W1/screens.json`; `icd_audit.json`; CLAIMS W009–W016 | W1 |
| replay share of the largest block; sources and licences; EHR-Rel items | | | W2 |
| ICD prose v2 pilot pass rate; pairs built; spend | | | W3 |
| charts: pilot keep rate, charts/h, items per template, NB screens | | `loops/osler_v2/charts.md` | W3b |
| fast dev v2 partitions; knowledge dev v2 size; zero-shot knowledge dev | | | W5 |
| speed check: real tokens/s and peak per backbone | Qwen 6,802 tok/s, peak 58.4 GB; pplx-flash 6,469 tok/s, peak 57.1 GB (both pass: ≥ 5,500, ≤ 85 GB) | `loops/osler_v2/CLAIMS.md` W007–W008; `outputs/osler_v2/W0/speed_summary` logs | W0 |
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
Task in flight: none (W1 DONE at 2026-10-17T14:13:48Z)
Working dir:    outputs/osler_v2/W1/

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

### W0 — DONE — 2026-10-17T14:06:18Z
- What ran: `scripts/osler_v2/w0_speed.py --steps 300 --dyn 8192` (Qwen and `--pplx`); `scripts/osler_v2/w0_env_probe.py`; ruff; full pytest gate with the padding strict-xfail deselected on the command line only.
- Output: `outputs/osler_v2/W0/` (speed logs, `env_probe.json`, `pytest_gate.log`, `ruff.log`, `SELF_AUDIT.md`).
- Headline: speed Qwen 6,802 real tok/s (peak 58.4 GB), pplx-flash 6,469 (peak 57.1 GB); both pass the 5,500 / 85 GB rule (CLAIMS W007–W008). Pytest gate: 730 passed, 1 skipped, 1 deselected, exit 0 (W002).
- Surprises: (1) first Qwen speed attempt hit CUDA OOM because my own pytest run (started with the wrong deselect) held about 89 GB; rerun after it exited. (2) that pytest run failed the padding test, which is deselected by osler_v1 practice and was not edited. (3) non-fatal allocator warnings in both speed logs. (4) MIMIC files absent (expected).
- Next: W1 (CPU audits) and W2 (replay and EHR-Rel) are unblocked.

### W1 — DONE — 2026-10-17T14:13:48Z
- What ran: `scripts/osler/o2_run_zeroshot.py` (Qwen3.5-4B, mr_fastdev, 2,397 rows, GPU, 4 min); `scripts/osler_v2/w1_screens.py`; `scripts/osler_v2/w1_icd_audit.py`; `scripts/osler_v2/w1_freeze_suite.py`.
- Output: `outputs/osler_v2/W1/` (screens.json, icd_audit.json, zeroshot/, SELF_AUDIT.md); `data/bench/osler_mr_v1/manifest.json`; `loops/osler_v2/mr_suite_v1.md` (committed).
- Headline: MR suite v1 frozen (D35): 13,103 of 19,294 test items in the verdict; the two page-category templates reported only (ceiling 1.000 and 0.995; NB 1.000). codiesp and mkurman screens NOT MEASURED (no dev items). ICD too-good probe 0.612 with zero training overlap (CLAIMS W015–W016), so it is still unexplained.
- Surprises: (1) the ceiling statistic is on the 400-item fast-dev sample (D-W1-1); (2) a one-item tie in the synthea ceiling (decision unchanged); (3) pmc passes D35 but its word-overlap AUC is high (Q3).
- Next: W2 (replay ≥ 20 %, EHR-Rel) and W3/W4 status; W3b needs the vLLM environment (GPU, ~1 h set-up).

---

## 5. Blocked items

| id | what is blocked | exact reason | what would unblock it |
|---|---|---|---|

---

## 6. Questions for the operator

Anything you could not resolve without a human. Be specific enough to answer without
re-reading the run: state the ambiguity, the options, and which you would pick.

---

**Q1 (W1, 2026-10-17T14:13:48Z) — mkurman_icd_support_v1 has no dev split and can never be screened on dev.** It is evaluation-only (osler_v1 V5b), so the D35 rules cannot trigger for it, and it stays in the verdict (13,103 verdict test items include its 1,490). Options: (a) keep it in the verdict with its screens NOT MEASURED (current; the literal D35 text); (b) report it only, which removes 1,490 test items where osler_v1's loss sat (V185), and would look like cherry-picking unless the operator decides it. I would keep (a). Please confirm before W6.

**Q2 (W1, 2026-10-17T14:13:48Z) — codiesp_icd_support_v1's screen is measured at W5, after the CodiEsp dev carve-out.** I will apply D35 to its dev items at W5 and record the result before any osler_v2 block (W6). If the operator wants a different rule for this, say so before W5.

**Q3 (W1, 2026-10-17T14:13:48Z) — pmc_similar_patient_v1 (8,100 test items, 42 % of the test items) passes D35 (NB 0.520), but osler_v1 measured word-overlap AUC 0.97 (V151).** The AUC is not a D35 criterion, so the template stays in the verdict; I will report the AUC as an additional row. Confirm that this is the right reading.

## 7. Deviations from the plan

Any place you departed from GOAL/ADVISORY, with the reason. An empty section is the
expected outcome. Editing code or a check to make it pass is never an acceptable
deviation — that is a `BLOCKED`.

---

- **D-W1-1 (W1, 2026-10-17T14:13:48Z):** the ceiling statistic uses each template's 400-item fast-dev sample (the `mr_fastdev` file), not its full dev split: the zero-shot runner reads the bench item schema and the full dev splits are in the generator schema. The NB screens use the full dev splits. The decisions do not depend on it (page-category templates are excluded by NB; the others are 0.69–0.79 on 400 items, far below 0.97). Recorded in `outputs/osler_v2/W1/SELF_AUDIT.md`.
- **D-W1-2 (W1, 2026-10-17T14:13:48Z):** codiesp_icd_support_v1 and mkurman_icd_support_v1 have no dev items, so D35's two rules cannot trigger: both stay in the verdict with their screens NOT MEASURED (the literal D35 text excludes only ceiling and vocabulary-shortcut templates). See question Q1.
- **Tie (W1):** one zero-shot item (synthea_page, 6956ca9e6ee8d55b) has two options with equal maximum probability; the repo's `np.argmax` convention (first index) is used. The decision is unchanged either way (see SELF_AUDIT).

## 8. Closure summary feed

<!-- Filled at closure, feeding FINDINGS.md's Summary section. One row per major
     outcome, in the order a human should hear them. Each claim cites its CLAIMS id;
     failures and blocked tasks get rows too. -->

| # | outcome (one sentence, plain language) | claims | artifact |
|---|---|---|---|
| | | | |

