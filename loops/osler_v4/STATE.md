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
Last updated (UTC): `2026-10-21T19:44:00Z`
Iterations so far: `1`

---

## 1. Task board

Statuses: `PENDING` → `IN_PROGRESS` → `DONE` | `BLOCKED — reason`

| id | task | GPU | deps | status | started (UTC) | finished (UTC) |
|---|---|---|---|---|---|---|
| Y0 | Orientation, Nemotron serving, MIMIC guard | small | — | DONE | 2026-10-21T18:26:40Z | 2026-10-21T18:53:35Z |
| Y1 | MIMIC foundation (tables, sections, splits) | no | Y0 | DONE | 2026-10-21T18:54:18Z | 2026-10-21T19:44:00Z |
| Y2 | MIMIC structured-gold templates (+ Y2b late sources when present) | no | Y1 | PENDING | | |
| Y3 | Nemotron-crafted minimal pairs (`mimic_doc_support_v1`) | yes | Y0, Y1 | IN_PROGRESS (pilot done; build pending) | 2026-10-21T19:25:21Z | |
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
Task in flight: Y3 (started 2026-10-21T19:25:21Z). Y1 closed 19:41:36Z (its checklist is in the §4 entry).
Working dir:    outputs/osler_v4/Y3/ (pilot_off, pilot_on, build_off); private pairs under data/mimic_build/pairs/ (gitignored)
Server:         none running (Nemotron stopped 19:41Z so the full test suite could use the GPU). Restart before the build.

- [x] helper module (meddecide.mr.mimic_pairs) and 19 unit tests with a fake teacher
- [x] driver scripts/osler_v4/y3_pairs.py (pilot, build, cue, pilot-report, report); the six-attempt smoke test was deleted
- [x] pilot off (200 train attempts, 152 s) and pilot on (200 attempts, 687 s); pilot-report: no mode reaches a 40 % keep rate
- [x] mode decision: off (deviation, §7); loops/osler_v4/crafted.md pilot section written
- [ ] full-suite acceptance for the Y1 and Y3 code (ruff and pytest with the GPU free): running from 19:41Z
- [ ] restart Nemotron (serve_nemotron.sh, detached); `build --mode off --hours 8` (dev 250, test 250, train up to 40,000 kept)
- [ ] cue check on the build (AUC < 0.75 on dev, ADVISORY Y3 §6); crafted.md build and cue sections (report --mode off)
- [ ] SELF_AUDIT.md (outputs/osler_v4/Y3); CLAIMS rows C024 onward
- [ ] STATE updated, committed, pushed; Y3 DONE or BLOCKED
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

### Y1 — DONE — 2026-10-21T19:44:00Z
- What ran: `scripts/osler_v4/y1_mimic_base.py` on the official files (CPU; run 2 from 19:10:55Z to 19:20:33Z; the first run's log is kept as y1_mimic_base_run1.log); `tests/test_mimic_sections_v4.py` (12); full-suite acceptance with the GPU free (C024).
- Output: `data/mimic_build/` (private parquet, gitignored); `outputs/osler_v4/Y1/aggregates.json`; `loops/osler_v4/mimic.md` (generated; aggregates only).
- Headline: MIMIC-IV 3.1 hosp holds 364,627 patients and 546,028 admissions (train/dev/test admissions 436,063 / 55,580 / 54,385; 203,429 late-anchor). 331,793 discharge notes, 331,732 linked to an admission; token p50 3,046 and p95 5,631 with the pplx tokenizer; ICD-10 share 0.4663 of coded admissions (C015–C023).
- Surprises: the first run's split bookkeeping was wrong (fixed; see §7); 208,046 radiology notes are for subjects with no hospital admission; 1,749 discharge notes exceed the 8,192-token cap (cut by section priority in Y2 and Y3).
- Checklist (completed): schemas and tokenizer; `sections` with unit tests; admission table; private parquet; script run; `mimic.md` §1; acceptance; SELF_AUDIT; CLAIMS C015–C024; STATE; commit.
- Next: Y3 build (GPU) is running; Y2 templates (CPU) are next, including the ED and MedNLI templates from the operator steer.

---

## 5. Blocked items

| id | what is blocked | exact reason | what would unblock it |
|---|---|---|---|

---

## 6. Questions for the operator

Anything you could not resolve without a human. Be specific enough to answer without
re-reading the run: state the ambiguity, the options, and which you would pick.

1. **Y3 mode when no mode reaches the 40 % keep rate** (ADVISORY Y3 §7 is silent). Options: (a) reasoning off, the higher
   keep rate (0.2806) and 1,302.6 kept per GPU-hour, which is what the build uses; (b) reasoning on (0.1735; 178.1 per
   GPU-hour); (c) block Y3 and run B2 as NOT MEASURED. I picked (a) and recorded it under §7. Please confirm or redirect.
2. **Spot-reads of MIMIC pairs** (ADVISORY Y3 §7). The loop checks structure only (sentence counts, deleted numbers, the rule
   that dropped each attempt) and prints no note text, because this agent session runs on a hosted model. Options: (a) keep
   the structural checks; (b) you read 20 kept and 20 rejected pairs on the pod yourself and record the verdict in §6 or §7.
   I picked (a) until you say otherwise.
3. **Reasoning-on selections with no valid list** (76 of 200 attempts in the on pilot). Cause not investigated; likely the
   4,096-token cap is reached before the list (not verified). Not needed for the chosen mode; logged only.

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
- **Y1 (2026-10-21):** the split key for every MIMIC table is `subject_split(subject_id, "mimic4")` (ADVISORY §3.3). The osler_v1
  notes reader used `"mimiciv"`; it was changed to `"mimic4"` because no MIMIC output had ever been made with the old key.
- **Y1 first run defects, fixed in the reported run** (log kept as `outputs/osler_v4/Y1/logs/y1_mimic_base_run1.log`): the
  "split mismatch" count mixed orphan notes with real mismatches, and radiology splits went through the admissions map (208,046
  notes had none). The reported run keys every split by subject and separates each excluded group with its own count.
- **Y3 mode choice (pre-registered rule silent):** ADVISORY Y3 §7 chooses the mode "with the higher kept pairs per GPU-hour
  among modes with keep rate ≥ 40 %". Neither mode reaches 40 % (reasoning off 0.2806, on 0.1735; `pilot_report.json`).
  Taken as the conservative reading: reasoning off, because it has the higher keep rate and 7.3× the kept pairs per GPU-hour
  (1,302.6 vs 178.1). Recorded as a deviation; the operator may decide otherwise (§6).
- **Y3 spot-reads replaced by structural checks:** ADVISORY Y3 §7 asks for 20 kept and 20 rejected pairs to be read. Pair text
  is MIMIC text and this agent session is hosted, so the spot-check uses sentence counts, the deleted sentence numbers and the
  rule that dropped each attempt, and no note text is printed into the session (§6, question 2).
- **Y3 pilot cue AUC must be grouped by pair:** the first pilot cue check (random 5-fold) gave 0.056 and 0.014, which is a leak
  (each pair's twin is in the other fold). The reported value groups folds by `pair_id` (0.5719 off, 0.5658 on).
- **Y3 AKI and heart-failure targets:** codes in N17 (AKI) and I50 (heart failure, held out) are not pair targets at all;
  the AKI reading of ADVISORY §3.4 (AKI items to held-out only) is applied conservatively to pairs (no pair is built for them).
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
