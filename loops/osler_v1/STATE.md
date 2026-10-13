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
Run started (UTC): `2026-10-13T21:32:02Z`
Last updated (UTC): `2026-10-13T22:29:31Z`
Iterations so far: `2`

---

## 1. Task board

Statuses: `PENDING` → `IN_PROGRESS` → `DONE` | `BLOCKED — reason`

| id | task | GPU | deps | status | started (UTC) | finished (UTC) |
|---|---|---|---|---|---|---|
| V0 | Orientation, environment, speed recipe | small | — | DONE | 2026-10-13T21:32:02Z | 2026-10-13T22:29:31Z |
| V1 | MR generators | no | V0 | DONE | 2026-10-13T22:30:29Z | 2026-10-13T22:35:18Z |
| V2 | ICD-10-CM verification | no | V0 | DONE | 2026-10-13T22:43:25Z | 2026-10-13T22:44:06Z |
| V3 | Replay to 20 % | no | V0 | DONE | 2026-10-13T22:44:15Z | 2026-10-13T22:45:36Z |
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
| teacher: configured? model id, status | configured via the OPENAI_* fallback; model `deepseek-v4.1-flash`; models endpoint 200 (39 ids); one harmless chat call OK; list-price cost $0.00 | `outputs/osler_v1/V0/env.json` (CLAIMS V003) | V0 |
| MIMIC files present (which, sizes) | none: `data/physionet/` absent; 0 of 5 core files (NOTEEVENTS, diagnoses_icd, patients, admissions, note discharge); V5c stays PENDING | `outputs/osler_v1/V0/env.json` (CLAIMS V004) | V0 |
| batching constant (batch_size, max_batch_tokens, gradient checkpointing); items/s; peak GB | **batch 8, max_batch_tokens 8192, gradient checkpointing on** (the osler_v0 recipe, unchanged). 4.118 items/s steady state (4.114 first run); 3,661 real tokens/s; peak reserved 21.8 GB. b, c and e ran out of memory; d (batch 32, checkpointing) ran at 3.09 items/s | `outputs/osler_v1/V0/speed.json`, `speed/*.json` (CLAIMS V012–V019) | V0 |
| MR generators: items per split, screen results | train 28,032 / dev 1,394 / test 1,430 / heldout_test 176; indicator and ICD v2 pass the O3 screen (Naive Bayes 0.796 and 0.804); page category fails as expected (1.000); ICD v1 design failed (0.938) and was replaced (V1-D1) | `loops/osler_v1/mr_data.md`, `outputs/osler_v1/V1/SELF_AUDIT.md` (CLAIMS V022–V033) | V1 |
| ICD-10-CM FY2026 order file sha256; verify result | order file sha256 6dc95c9c7e96c734…; zip sha256 55a9124a27ca78a4…; 98,186 rows; the 48 curated codes: 0 missing, 0 not billable (PASS). FY2027 additional: 48 of 48 present and billable | `data/cms/icd10cm/2026/` (gitignored); `outputs/osler_v1/V2/verify_fy2026.json` (CLAIMS V035–V037) | V2 |
| replay share of B0; sources and licences | B0 replay share 0.1531 (100,132 of 653,914 train rows), **target 20 % NOT met** (38,314 rows short under D20). Sources: HellaSwag (MIT, 39,905), WinoGrande (CC-BY, 40,398); mix v2's CommonsenseQA and QASC (19,829) kept. Dropped: ARC, BoolQ, SNLI (CC BY-SA), OpenBookQA (licence unknown) | `outputs/osler_v1/V3/replay_manifest.json`, `outputs/osler_v1/V3/SELF_AUDIT.md` (CLAIMS V038–V041) | V3 |
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
Task in flight: none (V3 DONE 2026-10-13T22:45:36Z; its checklist is closed in the V3 iteration entry); V2 closed earlier (V2 DONE 2026-10-13T22:44:06Z; its checklist is closed in the V2 iteration entry); V1 closed earlier (V1 DONE 2026-10-13T22:35:18Z; its checklist is closed in the V1 iteration entry in section 4)
Working dir:    outputs/osler_v1/V1/

V1 checklist (closed): build (6,000/300/300 per generator); audit; screen checked against §3.6 (page category fails as expected; ICD v1 failed and was replaced by v2, V1-D1); ten items read per generator; mr_data.md; SELF_AUDIT.md; CLAIMS V022–V033; test_icd_template_v2.py and test_mr_v1.py pass.

V0 checklist (closed):
V0 checklist (closed):

V0 checklist (closed):
- [x] read ADVISORY, PROGRAM D26-D32, osler_v0 FINDINGS and NEXT (done at start)
- [x] environment: commit 0998e79, GPU RTX PRO 6000 97,887 MiB, disk (scripts/osler_v1/v0_env.py -> env.json)
- [x] ruff check . passed (2026-10-13T21:32Z)
- [x] tests: pytest verbose, test_padding_trained deselected on the command line only: 645 passed, 1 deselected, 2 warnings, exit 0 (outputs/osler_v1/V0/logs/pytest_full.log; CLAIMS V001). First run killed at 21:38 (no progress output); not counted.
- [x] record the padding XPASS as a question for the operator: observed XFAIL on 2026-10-13T21:45Z, the ADVISORY recorded XPASS at planning time (CLAIMS V006); not edited
- [x] teacher reachability: models endpoint 200, chat OK, model deepseek-v4.1-flash, cost 0.0 (env.json)
- [x] MIMIC availability: data/physionet absent, core set incomplete (env.json)
- [x] speed recipe: v0_speed.py + v0_speed_chain.sh; tests/test_speed_v0.py (9 pass); chain a-d, then e (V0-D1), a re-run with all records (V0-D7); choice a (CLAIMS V012–V018)
- [x] calibration fix check: tests/test_arm_run_o6.py: 3 of 3 passed; the saved model.json carries the fitted temperatures (CLAIMS V007)
- [x] acceptance check: outputs/osler_v1/V0/env.json and speed.json exist; the choice is in section 2
- [x] self-audit (R6) written to SELF_AUDIT.md: 12 PASS, 0 FAIL (CLAIMS V021)
- [x] CLAIMS.md rows appended (V001–V021)
- [x] STATE updated; committed and pushed (see the git log of loop/osler_v1)
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

### V0 — DONE — 2026-10-13T22:29:31Z
- What ran: `v0_env.py` (teacher probe, MIMIC presence, GPU, disk); ruff; pytest (final suite: 670 passed, 1 deselected, padding test
  deselected on the command line only); the padding test once, observed (not edited); `tests/test_arm_run_o6.py` (3 passed, calibration
  saved with the checkpoint); `v0_speed_chain.sh` (configs a, b, c, d) then config e (added, V0-D1) and a full-record re-run of a
  (V0-D7); `v0_speed.py --aggregate`; `v0_self_audit.py` (12 PASS).
- Output: `outputs/osler_v1/V0/` (`env.json`, `speed.json`, `speed/*.json`, `SELF_AUDIT.md`, `logs/`), all gitignored.
- Headline: batching constant = the osler_v0 recipe, unchanged (batch 8, max_batch_tokens 8,192, checkpointing on): 4.118 items per
  second steady state, peak reserved 21.8 GB (CLAIMS V017–V018). b, c and e ran out of memory; d ran at 3.09 items per second. The
  teacher answers (`deepseek-v4.1-flash`, models endpoint 200, one chat OK, list-price cost $0.00; CLAIMS V003). MIMIC: 0 of 5 core
  files (V004). Padding test: XFAIL on this run, not XPASS as the ADVISORY recorded (V006).
- Surprises: no batching recipe beat the osler_v0 one; the planned larger batches were slower or out of memory. The run is GPU-bound at
  about 3,660 real tokens per second, so a block of 80,000 realised items takes about 5.4 hours (ADVISORY assumed 2–3). LoRA and head
  initialisation were unseeded, now seeded (V0-D6). ADVISORY V3's candidate list conflicts with D20 (V0-D4).
- Next: V1 (MR generators), V2 (ICD order file, FY2026 acceptance), V3 (replay, permissive sources), V5a and V5b (CPU and network).
  V4 waits for V1, V2 and V5a. V5c is re-checked before V7.

### V1 — DONE — 2026-10-13T22:35:18Z
- What ran: `v1_build_mr_generators.py` (first build with the ICD v1 design, kept as evidence; second build with the ICD v2 design); `v1_audit.py` (writes `SELF_AUDIT.md`); `v1_generator_fingerprint.py` (refactor check, identical); ten items read per generator in a fresh process (terminal only).
- Output: `data/gen/osler_v1/` (gitignored); `outputs/osler_v1/V1/` (`screen.json`, `self_audit.json`, `SELF_AUDIT.md`, `screen_build1_icd_v1design.json`, `logs/`); `loops/osler_v1/mr_data.md` (aggregates).
- Headline: train 28,032 / dev 1,394 / test 1,430 / heldout_test 176 rows; patients 18,816 with none in two splits (CLAIMS V026–V027). Indicator and ICD v2 pass the O3 screen (Naive Bayes 0.796 and 0.804, dev); the page-category generator fails as ADVISORY §3.6 expects (1.000). The ICD v1 design failed (0.938) and was replaced (CLAIMS V022–V023, deviation V1-D1).
- Surprises: the planning-time probe passed the ICD v1 design at 1,500 patients, but it fails at 6,000 (more data, a stronger Naive Bayes). The ICD template's laterality wording repeats "knee". No held-out text reaches a training split (CLAIMS V028).
- Next: V2 (ICD order file, FY2026 acceptance), V3 (replay, permissive sources), V5a and V5b. V4's ICD prose must use the two-diagnosis design of v2 (a teacher page with the target and the distractors), or its screen would fail the same way.

### V2 — DONE — 2026-10-13T22:44:06Z
- What ran: `v2_icd_verify.py` (downloads the CMS FY2026 and FY2027 "code descriptions in tabular order" zips, unpacks under `data/cms/icd10cm/<year>/`, verifies the 48 curated codes with `icd.verify_codes`); spot-check of 12 codes against their titles and billable flags (CLAIMS V037); `outputs/osler_v1/V2/SELF_AUDIT.md`.
- Output: `outputs/osler_v1/V2/verify_fy2026.json`, `verify_fy2027.json`, `logs/verify.log` (gitignored); the zip and the order file stay in `data/cms/icd10cm/` (gitignored).
- Headline: FY2026 (ADVISORY acceptance year): 48 curated codes in 12 families; 0 missing; 0 not billable; 98,186 order rows (CLAIMS V035). The additional FY2027 check: 48 of 48 present and billable (V036). The curated table was not edited.
- Surprises: the zip carries two files with "order" in the name (`icd10cm_order_2026.txt` and `icd10cm_order_addenda_2026.txt`); the first run stopped on that and the script was fixed to take the exact order file (the addenda file is not used). FY2027 took effect on 2026-10-01 (Question 4).
- Next: V3 (replay, permissive sources), V5a (Synthea), V5b (real text). V4 waits for V1, V2 and V5a.

### V3 — DONE — 2026-10-13T22:45:36Z (target not met; see Question 2 and deviation V3-D1)
- What ran: `v3_replay.py` (downloads the train splits of HellaSwag and WinoGrande xl; converts with `meddecide.mix.replay_v1`; counts every kept and dropped row; leakage check against the osler_v0 protected sets); `v3_self_audit.py` (re-reads the rows; checks the MR dev, test and heldout_test text keys; 7 of 7 PASS).
- Output: `data/train/osler_v1/replay_v1_train.jsonl` (gitignored, 80,303 rows); `outputs/osler_v1/V3/replay_manifest.json`, `SELF_AUDIT.md`, `logs/replay.log` (gitignored).
- Headline: 80,303 permissive replay rows added (HellaSwag 39,905; WinoGrande 40,398), none dropped, none in any protected set. B0 replay share 0.1531 (100,132 of 653,914 rows); the 20 % target is not met: 138,446 replay rows would be needed, 38,314 more than the permissive pool gives (CLAIMS V038–V039).
- Surprises: the licence conflict between ADVISORY V3's candidate list and D20 (the three CC BY-SA sources would give the SNLI pool, 550k rows, and reach 20 %); OpenBookQA's own licence is not stated on the Hub.
- Next: V5a (Synthea) and V5b (real clinical text), both CPU and network; V4 once V1, V2 and V5a are done; V5c re-check before V7.

## 5. Blocked items

| id | what is blocked | exact reason | what would unblock it |
|---|---|---|---|

---

## 6. Questions for the operator

Anything you could not resolve without a human. Be specific enough to answer without
re-reading the run: state the ambiguity, the options, and which you would pick.

1. **Padding-invariance test (V0).** `tests/test_padding_trained.py` is a strict xfail. The ADVISORY recorded an XPASS on
   2026-10-13 (a pass that must be reviewed). On this run it XFAILs (CLAIMS V006, `outputs/osler_v1/V0/logs/padding_observed.log`).
   Which result is the reference? I left the test as committed, and it is deselected only on the command line. Pick: treat the
   XFAIL as current (the test's tolerance is still unmet), or find out what changed between the two runs.
2. **Replay share (V3).** With permissive sources only (HellaSwag and WinoGrande), the replay share is expected to reach about
   15 % of B0, not 20. This is arithmetic from the mix v2 manifest and the source sizes, not yet a measurement; V3 measures it
   (`outputs/osler_v1/V3/replay_manifest.json`, not yet written). The 20 % target needs either (a) CC BY-SA sources (SNLI alone, 550k rows,
   would reach it; BoolQ and ARC are small), which conflicts with D20's permissive rule, or (b) removing non-replay rows, which
   the ADVISORY does not allow. I would pick: keep permissive only and report the gap, unless the operator amends D20.
3. **Run time.** At the measured 4.11 items per second, one 80,000-item block (V8) takes about 5.4 hours, not the ADVISORY's 2–3,
   and the 250,000-item 4B final (V9) about 17 hours. The 9B is roughly twice that. The full V4 teacher target (80,000 pages at
   two requests per second) is about 22 hours. I would keep the ADVISORY sizes and let the operator cut scope.
4. **Code-set year (V2).** The ADVISORY names FY2026. FY2027 took effect on 2026-10-01 and is the current code set. V2 checks
   FY2026 as the acceptance year and FY2027 as an additional row. I would pick: keep FY2026 as written, and the operator decides
   whether the curated table should move to FY2027.

---

## 7. Deviations from the plan

Any place you departed from GOAL/ADVISORY, with the reason. An empty section is the
expected outcome. Editing code or a check to make it pass is never an acceptable
deviation — that is a `BLOCKED`.

- **V0-D1 (speed sweep, D26).** The pre-registered sweep (a, b, c, d) ran as written. Configuration e (the osler_v0 batch shape
  without gradient checkpointing) was added after a–d had been measured, because the planned set never tested that. It uses the
  same protocol and the same rule; checkpointing changes no number. e ran out of memory at step 22, so it is not eligible. The
  chosen constant is therefore a, which is the osler_v0 recipe unchanged. The planned larger batches were slower (d, 3.09 items/s)
  or out of memory on their first step (b, c). CLAIMS V012–V016.
- **V0-D2 (speed runs use a constant learning rate).** The sweep does not pass a schedule, so the learning rate is flat with no
  warmup. This isolates batching. The loss columns in the speed records are therefore not comparable with the arm runs, which use
  warmup and cosine decay. The choice rule uses throughput and peak memory only.
- **V0-D3 (how the speed numbers are measured).** Items per second are wall-clock over steps 11–150 (warm-up excluded), taken from
  the step timestamps. The trainer's `elapsed_s` is not used, because osler_v0 measured that it under-states wall-clock time (its
  deviations 47–48). The 85 GB cap is applied to peak reserved memory, which is the conservative reading.
- **V0-D4 (replay licences, D20 over the ADVISORY list).** ADVISORY V3 lists ARC, BoolQ and SNLI, which are CC BY-SA. D20 (settled)
  requires a permissive licence, and the osler_v0 review rejected BoolQ and SNLI on that ground, so the loop follows D20 and does
  not reopen it (AGENTS.md). OpenBookQA is dropped: its dataset licence is not stated on the Hub, and the ADVISORY says to drop an
  unclear licence. Kept: HellaSwag (MIT) and WinoGrande (CC-BY). Consequence: replay cannot reach 20 % (Question 2).
- **V0-D5 (padding test not reproduced).** The strict xfail was observed as XFAIL on 2026-10-13T21:45Z. The ADVISORY recorded XPASS at
  planning time. The test was not edited or deselected in committed config (Question 1).
- **V0-D6 (seeding the training path, R7).** `run_arm` now calls `torch.manual_seed(config.seed)` before the model is built. Before
  this, LoRA's random A matrices and the head started from different weights on every run, although the data order was seeded. The
  change touches no head, attention or base and no check. It is exercised by the V8 block runs. The V0 speed JSONs were produced before
  this change, so their loss columns vary between runs (config a: 1.2132 and 1.0880 for the same window, CLAIMS V019); their
  throughput does not (4.1137 and 4.1183, CLAIMS V017).
- **V1-D1 (ICD template design).** The ADVISORY V1 template (one diagnosis per page) failed the O3 screen at full size (Naive Bayes 0.938, dev 550; CLAIMS V022). The v2 template adds two distractor diagnoses and names the target condition in the question. The draw, the gold and the twin rule are unchanged. v1 stays in `generators.py` as the failing record. v2 is the registered generator, and its name is `mr_icd_support_template_v2`. The teacher ICD variant must use the same two-diagnosis design (V4).
- **V1-D2 (page category).** The page-category screen failure is accepted as ADVISORY §3.6 expects. Its share of any mix is capped in V7.
- **V1-D3 (twins).** The page-category generator emits one item per patient id, so the twin audit does not apply to it.
- **V3-D1 (replay target).** The 20 % replay share is not reached. D20 (settled) requires permissive licences; with the permissive pool (HellaSwag, WinoGrande, and the two sources already in mix v2) the share is 15.31 %. Nothing is removed from the non-replay rows to reach 20 %: the ADVISORY says 'add', so the gap is reported (Question 2). CLAIMS V038–V040.
- **V0-D7 (speed evidence).** The first run of config a stored only its last 50 step records, so its throughput could not be
  re-derived from the file. Config a was re-run with every record stored (`speed/a.json`; the first run is kept as `speed/a_run1.json`),
  and the choice is recomputed from these files (`outputs/osler_v1/V0/SELF_AUDIT.md`, 12 PASS). The driver now stores all records; the
  other configurations' files keep their last-50 records, so their summaries cannot be re-derived from disk. Their numbers are recorded as
  measured (CLAIMS V013–V016).

---

## 8. Closure summary feed

<!-- Filled at closure, feeding FINDINGS.md's Summary section. One row per major
     outcome, in the order a human should hear them. Each claim cites its CLAIMS id;
     failures and blocked tasks get rows too. -->

| # | outcome (one sentence, plain language) | claims | artifact |
|---|---|---|---|
| | | | |
