# osler_v4 — RUN STATE

> **This file is the loop's memory.** The live copy is `loops/osler_v4/STATE.md`
> (committed, so the operator can review it remotely). Read it at the start of every
> iteration, before the plan. If it disagrees with your recollection, **the file
> wins** — your context may have been compacted since the last iteration.
>
> **How to update:** set the task `IN_PROGRESS` with a UTC start time *before* working;
> on completion set `DONE` or `BLOCKED — <reason>`, fill the finished time, and append a
> ≤5-line entry to the iteration log. Never delete a log entry; append only. Timestamps
> are `date -u +%FT%TZ`. Never paste item text, predictions, or secrets into this file.

Loop status: `RUNNING`  <!-- set to STOPPED at the hard stop (Y11), or when no PENDING task can proceed without the operator -->
Run started (UTC): `2026-10-21T18:26:40Z`
Last updated (UTC): `2026-10-21T18:53:35Z`
Iterations so far: `1`

---

## 1. Task board

Statuses: `PENDING` → `IN_PROGRESS` → `DONE` | `BLOCKED — reason`

| id | task | GPU | deps | status | started (UTC) | finished (UTC) |
|---|---|---|---|---|---|---|
| Y0 | Orientation, Nemotron serving, MIMIC guard | small | — | DONE | 2026-10-21T18:26:40Z | 2026-10-21T18:53:35Z |
| Y1 | MIMIC foundation (tables, sections, splits) | no | Y0 | PENDING | | |
| Y2 | MIMIC structured-gold templates (+ Y2b late sources when present) | no | Y1 | PENDING | | |
| Y3 | Nemotron-crafted minimal pairs (`mimic_doc_support_v1`) | yes | Y0, Y1 | PENDING | | |
| Y4 | Fast dev v4, CG v2, MIMIC audit suite v1, baselines | yes | Y2, Y3 | PENDING | | |
| Y5 | pplx 27B soft labels (only if D48 confirmed) | yes | Y0 | PENDING | | |
| Y6 | Stage A: A4, A5 + re-scored A0/A1/A3 + choice | yes | Y4, Y5 | PENDING | | |
| Y7 | Stage B: blocks B1–B4 under keep rule v2 | yes | Y6 | PENDING | | |
| Y8 | Final Osler-4B (63,304 steps) | yes | Y7 | PENDING | | |
| Y9 | Evaluation, Gate O4, size-class scoreboard | yes | Y8 | PENDING | | |
| Y10 | Memorisation and PHI audit | small | Y8 | PENDING | | |
| Y11 | Findings and closure — HARD STOP | no | all | PENDING | | |

Rules: take the **first** `PENDING` task whose deps are all `DONE` (order exceptions and the GPU sequence in ADVISORY §5).
Never run two GPU jobs at once (the Nemotron server counts). A `BLOCKED` task does not block unrelated work. No row may be
left `IN_PROGRESS` at closure.

---

## 2. Key values discovered during the run

**Operator confirmations** (filled by the operator before kickoff; the loop never edits these two lines):
- D48 (pplx 27B soft-label distillation on general items): `yes` (operator, 2026-10-21, session message during Y0)
- D46, D47, D49, D50 (MIMIC design, keep rule v2, Gate O4, final length): `yes` (operator, 2026-10-21, same message)

| key | value | source | task |
|---|---|---|---|
| suite counts (passed / skipped / deselected) | baseline 840 / 1 / 2 (exit 0); after Y0 860 / 1 / 2 (exit 0); ruff clean | `outputs/osler_v4/Y0/logs/pytest_final.log`; CLAIMS C001–C004 | Y0 |
| Nemotron serving: flags that worked, out tok/s, requests/h, peak memory | committed flags, no fallback; ready 6 min 43 s after launch; KV cache 382,713 tokens; A (600-token output, 64-way, 200 req): 1,509.4 out tok/s, 9,056.7 req/h; B (60-token): 20,751 req/h; GPU peak 96,971 MiB of 97,887 | `outputs/osler_v4/Y0/serve_check.json`; CLAIMS C006–C013 | Y0 |
| MIMIC: patients / admissions / notes per split; ICD-10 share; token p50/p95 | | `loops/osler_v4/mimic.md` | Y1 |
| MIMIC templates: train/dev/test items, NB screen, majority rate per template | | `loops/osler_v4/mimic.md` | Y2 |
| minimal pairs: pilot keep rate per mode, chosen mode, cue AUC, pairs built | | `loops/osler_v4/crafted.md` | Y3 |
| fast dev v4 partitions; CG v2 size and sources; MIMIC suite v1 templates and items | | | Y4 |
| pplx 27B: revision, licence, parity, anchor items scored, overlap check (must be 0) | | | Y5 |
| Stage A: MR, knowledge, CG v2 per run; the chosen recipe | | `loops/osler_v4/stage_a.md` | Y6 |
| Stage B: kept blocks; own / MR / general / knowledge / CG v2 per block; MIMIC share | | `loops/osler_v4/blocks.md` | Y7 |
| final: steps, selected step, CG v2 and MIMIC trend | | | Y8 |
| Gate O4 verdict (4B) | | `loops/osler_v4/gate_o4.md` | Y9 |
| audit: membership AUC; verbatim matches (model vs base) | | `loops/osler_v4/audit.md` | Y10 |

---

## 3. Current task checklist

**Fill this in before starting any task**, by copying that task's concrete steps out of
the plan as unticked boxes. Tick each one the moment it is finished, not at the end.
This is the only record of progress *within* a task — without it, a compaction mid-task
leaves you unable to tell what you already did.

Clear this section and write the new task's checklist when you start the next task; the
completed checklist goes into the iteration-log entry.

```
Task in flight: none (Y0 closed 2026-10-21T18:53:35Z; its checklist is in the §4 entry)
Working dir:    none
Server:         none running (Nemotron stopped 2026-10-21 18:48Z; GPU 0 MiB)
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

### Y0 — DONE — 2026-10-21T18:53:35Z
- What ran: `uv run ruff check .` (pass); `uv run pytest` with the two ADVISORY deselects, before (baseline) and after the Y0
  changes; `scripts/osler_v4/serve_nemotron.sh` launched detached 18:39:03Z (vLLM PID 420612, EngineCore 421021), then
  `scripts/osler_v4/y0_serve_check.py`; server stopped by SIGINT (GPU 0 MiB at 18:48:23Z); `./dm status` read only.
- Output: `outputs/osler_v4/Y0/` (serve_check.json, SELF_AUDIT.md, logs/; gitignored).
- Headline: Nemotron 3.5 Super serves on this pod with the committed flags, no fallback (ready 6 min 43 s after launch; KV cache
  382,713 tokens). 200 requests, 2,897 prompt tokens on average, forced 600-token output, concurrency 64, reasoning off:
  1,509.4 output tokens/s and 9,056.7 requests/h, 0 failed (C006–C013). The MIMIC loopback guard is in the teacher client
  (C004); pytest 860 passed, 1 skipped, 2 deselected after the changes (C002; baseline 840).
- Surprises: (1) the advisor's 1,234 tok/s and about 8,100 requests/h were exceeded; not investigated, and no decision depends
  on it. (2) GPU peak 96,971 MiB of 97,887: the server fills the card, so nothing else can run beside it. (3) Two pytest runs
  printed no summary: `-q` with the pyproject `addopts = "-q"` gives `-qq`; the reruns used the configured level. (4) The
  operator's data-update steer and the D48/D46/D47/D49/D50 confirmations came in mid-run (§7, §2).
- Checklist (completed): ruff and pytest counts; server up; three short prompts (reasoning off) and one reasoning-on prompt;
  throughput A and B; MIMIC guard, its tests and the OPENAI_ grep; `./dm status`; acceptance; SELF_AUDIT; CLAIMS C001–C014;
  STATE; commit.
- Next: Y1 (CPU, MIMIC foundation). Y2 (CPU, the templates, including the operator-steered ED and MedNLI templates) runs while
  Y3's teacher works. Y3 needs the server restarted. D48 is yes, so Y5 (27B) is no longer blocked; it follows Y3 in the GPU order.

---

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

- **operator steer 2026-10-21: data update** (session message during Y0; the operator's own words, in substance). The
  ADVISORY §1 and §5 statements on what is on disk are out of date. Checked with `ls data/physionet/` (and `./dm status`,
  read only):
  1. MIMIC-IV 3.1 `hosp` is complete, labevents included. `mimic_lab_validation_v1` is built in Y2 with the other
     templates. The ADVISORY §5 rule "labevents may be absent at Y2" no longer applies.
  2. MIMIC-IV-ED 2.2 (all 8 files) and MedNLI 1.0.0 (all 5 files) are complete. `mimic_ed_acuity_v1`,
     `mimic_ed_disposition_v1` and `mednli_v1` are built inside Y2 now, not deferred to Y2b. Their dev items enter fast dev
     v4 at Y4. The late-anchor test items of the two ED templates join MIMIC audit suite v1 under the same D35 screens.
     MedNLI test stays an additional row. These three form block B3.
  3. MIMIC-III 1.4 is still downloading. It is re-checked at Y4 and before the B2 keep decision (the operator's wording; the
     ADVISORY Y2b text says "B3's decision"). If present then, `mimic_page_category_v1` (MDACE admissions excluded) joins B3;
     otherwise B3 goes ahead without it. The loop never waits for it.
  4. `icu/` files (for example chartevents) are appearing on disk but stay out of scope (ADVISORY §6).
- Operator confirmations were given in the same message: D48 = yes; D46, D47, D49, D50 = yes. They are entered in §2 as the
  operator's values. Before this message they were placeholders, and D48 was read as not confirmed.

---

## 8. Closure summary feed

<!-- Filled at closure, feeding FINDINGS.md's Summary section. One row per major
     outcome, in the order a human should hear them. Each claim cites its CLAIMS id;
     failures and blocked tasks get rows too. -->

| # | outcome (one sentence, plain language) | claims | artifact |
|---|---|---|---|
| | | | |
