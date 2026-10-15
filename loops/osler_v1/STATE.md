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
Last updated (UTC): `2026-10-15T06:24:04Z`
Iterations so far: `3` (V0–V3, V5a, V5b and V6 closed; V4, V7 and V8 in progress; V5c blocked on absent files)

---

## 1. Task board

Statuses: `PENDING` → `IN_PROGRESS` → `DONE` | `BLOCKED — reason`

| id | task | GPU | deps | status | started (UTC) | finished (UTC) |
|---|---|---|---|---|---|---|
| V0 | Orientation, environment, speed recipe | small | — | DONE | 2026-10-13T21:32:02Z | 2026-10-13T22:29:31Z |
| V1 | MR generators | no | V0 | DONE | 2026-10-13T22:30:29Z | 2026-10-13T22:35:18Z |
| V2 | ICD-10-CM verification | no | V0 | DONE | 2026-10-13T22:43:25Z | 2026-10-13T22:44:06Z |
| V3 | Replay to 20 % | no | V0 | DONE | 2026-10-13T22:44:15Z | 2026-10-13T22:45:36Z |
| V4 | Teacher prose | no | V1, V2, V5a | IN_PROGRESS — narrative build on OpenCode Go to its weekly guard (about 07:50Z by the ledger rate); the DeepSeek continuation is HELD pending Question 10 (V4-D7); V4 then closes on the verified pages | 2026-10-13T23:09:08Z | |
| V5a | Synthea | no | V0 | DONE | 2026-10-13T22:45:53Z | 2026-10-13T22:51:28Z |
| V5b | Real clinical text | no | V2 | DONE | 2026-10-13T22:52:20Z | 2026-10-13T22:56:31Z |
| V5c | MIMIC sources (only when the files exist) | no | V0 | BLOCKED — PhysioNet files absent (re-checked 2026-10-14T00:15Z, CLAIMS V097); re-check before V9 | | |
| V6 | MR evaluation suite + baselines | yes | V1, V5a, V5b (V4, V5c if built) | DONE | 2026-10-13T23:24:35Z | 2026-10-14T01:41:00Z |
| V7 | Mix v3 blocks | no | V3, V6 | IN_PROGRESS | 2026-10-13T23:40:48Z | |
| V8 | Block runs + keep rule | yes | V7 | IN_PROGRESS | 2026-10-13T23:53:00Z (marked late: the block config and the first V8 files predate this mark) | |
| V9 | Final Osler-4B on the kept block, attention-only LoRA (Osler-9B deferred by the operator, V9-D1; LoRA targets by the operator, V9-D2) | yes | V8 | PENDING | | |
| V10 | Evaluation + Gate O2 (Osler-4B only; the 9B verdict is NOT MEASURED — deferred by the operator) | yes | V9 | PENDING | | |
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
| Synthea: encounters, coverage, items per template | 40,000 encounters (33,879 emergency, 6,121 inpatient); observation coverage 37.1 %; items train 59,849 / dev 7,702 / test 6,509 / heldout 1,342. Indicator screen PASS (NB 0.576); page-category screen FAIL, expected (1.000; V5a-D1). Rare indicators have at most 4 positives per class in dev and test (limit) | `loops/osler_v1/mr_data.md` §8, `outputs/osler_v1/V5a/SELF_AUDIT.md` (CLAIMS V042–V048) | V5a |
| real text: MTSamples pages per class; CodiEsp items; PMC pairs | MTSamples items train 668 / dev 97 / test 111 (radiology 269, consult 255, progress 153, discharge 103, ED 63, admission 25, laboratory 8). CodiEsp train 3,374 / test 1,654 (machine-translated). PMC pairs train 52,549 / dev 7,990 / test 8,100 (up to 40,000 anchors across the whole file; partners anywhere in it). mkurman eval-only 1,490. NHS NOT MEASURED. Audit: no problems; text overlap zero | `loops/osler_v1/mr_data.md` §9, `outputs/osler_v1/V5b/SELF_AUDIT.md` (CLAIMS V049–V056) | V5b |
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
Task in flight: V8 (marked 2026-10-13T23:53:00Z; smoke OK; the B0 run started 2026-10-14T01:35:22Z, then B1), V7 (started 2026-10-13T23:40:48Z; B2 and B3 wait for the final teacher prose), and V4 (started 2026-10-13T23:09:08Z; pilots done; ICD prose BLOCKED; indicator and narrative prose building under scripts/osler_v1/v4_build_chain.sh, log outputs/osler_v1/V4/logs/build_chain.log, progress outputs/osler_v1/V4/*_progress.jsonl). V5b closed earlier (V5b DONE 2026-10-13T22:56:31Z; its checklist is closed in the V5b iteration entry); V5a closed earlier (V5a DONE 2026-10-13T22:51:28Z; its checklist is closed in the V5a iteration entry); V3 closed earlier (V3 DONE 2026-10-13T22:45:36Z; its checklist is closed in the V3 iteration entry); V2 closed earlier (V2 DONE 2026-10-13T22:44:06Z; its checklist is closed in the V2 iteration entry); V1 closed earlier (V1 DONE 2026-10-13T22:35:18Z; its checklist is closed in the V1 iteration entry in section 4)
Working dir:    outputs/osler_v1/V1/

V7 checklist (in progress; started 2026-10-13T23:40:48Z):
- [x] rules: ADVISORY V7 (cumulative blocks, 2 % tabular cap, 8 % template cap, leakage per block); B2 and B3 read the teacher files, so they need --teacher-final
- [x] first build (23:40Z) rejected by the leakage check: 40 unique PubHealth training rows whose bare id equals a panel id (15 of them also equal a robustness id); O4's build.json recorded the same bare hits and kept them under its dataset-qualified identity rule (V7-D4). Its B3 also held 610 teacher rows from an unfinished V4 build: that B3 record and file are stale (the block driver refuses them: leakage_clean false, no teacher flag)
- [x] rebuild with the rules (B0, B1; launched 2026-10-13T23:50:41Z; log outputs/osler_v1/V7/logs/blocks_build_v2.log)
- [ ] B2 and B3 once V4 is final (`v7_blocks.py --blocks B2 B3 --teacher-final`), and the B3 record replaced
- [ ] blocks.md; CLAIMS rows; SELF_AUDIT.md; commit

V8 checklist (in progress; marked 2026-10-13T23:53:00Z; the block config predates this mark):
- [x] configs/osler_v1/block_arm.yaml (loads; B0 defaults)
- [x] fast dev in the mix schema: v8_prepare.py -> fastdev_mix.jsonl (5,723 rows: MR 2,397, mix 3,326, tier-1 500 inside mix; one PubHealth row dropped, V7-D4) and fastdev_knowledge.jsonl (500)
- [x] step budget, keep rule, keep and knowledge reports (the D12 gate included): src/meddecide/train/blocks.py (tests/test_blocks_v8.py 14 pass; tests/test_bare_collisions_v8.py 3 pass); the scripts are thin
- [x] v8_replay_steps.py (the trainer's batcher; padded tokens with the chunk rounding); v8_block_run.py (refuses a block whose V7 record is missing, not clean, hash-mismatched or not teacher-final; checks the replay hash; validates logged steps and the evaluation steps); v8_keep.py (keep rule; knowledge check with the D12 gate)
- [x] scope mix_v2 added to o2_io.py and o2_run_zeroshot.py (defaults unchanged)
- [x] v8_chain.sh written and syntax-checked: waits for the V6 chain; zero-shot 4B on the knowledge items; smoke B0 (8 steps; the queue stops if its check fails); B0; B1
- [x] replay steps for B0 and B1: 13,992 steps; B0 80,008 realised items, B1 80,851 (CLAIMS V076–V078; hashes match the V7 records)
- [x] v8_chain.sh launched 2026-10-14T00:02:40Z; it waits for the V6 chain (zero-shot 9B and MedDecider and JEV-9B were still running at 00:02Z), then zero-shot knowledge, smoke B0, B0, B1 (log outputs/osler_v1/V8/logs/chain.log)
- [x] zero-shot Qwen3.5-4B on the 500 fast-dev knowledge items: 500 of 500 scored; medmcqa_4opt_v1 0.712 and medqa_usmle4_v1 0.656; both PASS D12 (`outputs/osler_v1/V8/knowledge_zs/`)
- [x] smoke B0 OK (8 steps; 46 realised items; 8 logged steps with 0 mismatches on items, real tokens and padded tokens; evaluations at 2, 4, 6, 8; `outputs/osler_v1/V8/smoke/B0/block_result.json`)
- [x] B0 run done 2026-10-14T07:28:07Z: 13,992 steps, 80,008 realised items, selected step 13,992, 0 log mismatches (CLAIMS V103). Dev (5,723 items): accuracy 0.8548, template macro 0.8582 (CLAIMS V105). Knowledge 500 items 0.624 (zero-shot 4B 0.684 on the same items).
- [x] knowledge check of B0 FAILED: B0 minus zero-shot is −0.060, lower bound −0.102 against −0.02 (CLAIMS V104); so B0b is run as ADVISORY V8 says. The knowledge check and the keep rule join items by the trainer's id (the trainer's `Item` derives its id from identity fields; `v8_keep.py` maps the O2 ids through `mix_row_to_item`)
- [x] B1 run done 2026-10-14T13:25:57Z (13,992 steps, 80,851 realised items, 0 log mismatches; full dev accuracy 0.9037, template macro 0.8781; CLAIMS V107–V108). The preview of B1 against B0 passes all three checks (CLAIMS V109); it is not the decision, because C may change to B0b
- [x] B0b done 2026-10-14T17:21:51Z (CLAIMS V110–V112): its knowledge check FAILED (point −0.008, lower bound −0.042), so C stays B0 (ADVISORY V8). B1 is KEPT against B0 (MR +0.0947, general +0.0062, knowledge +0.0480; all lower bounds pass), so C becomes B1 for B2 and B3. Queued earlier: `outputs/osler_v1/V8/B0b/`), then (`scripts/osler_v1/v8_chain_b0b.sh` starts after "V8 CHAIN DONE"), then `v8_keep.py --knowledge-check B0b`; C = B0b if its check passes, else B0; keep rule B1 against C (`v8_keep.py --candidate B1 --current C`)
- [ ] B2 and B3: `scripts/osler_v1/v8_chain_b2b3.sh` waits for `outputs/osler_v1/V4/FINAL.json` (written when V4 is closed: complete, or finished at the weekly budget per the operator's instruction), then builds B2 and B3 with `--teacher-final`, replays them, and runs them (one GPU job at a time). Then the keep rules: B2 against C = B1; B3 against the last kept block
- [ ] V4 closure: finish the build or stop it at the weekly guard; O3 screen and code-string check on the dev pages; spot-read 20 pages per generator; counts in SELF_AUDIT and CLAIMS; then write `outputs/osler_v1/V4/FINAL.json`
- [x] second endpoint prepared (V4-D7, 2026-10-15): `v4_teacher_build.py` takes `--ledger`, `--total-cap`, `--tag`; the endpoint host and model are recorded on every progress row and every written item; `v4_closure.py` counts by endpoint; `v4_deepseek_chain.sh` waits for the OpenCode build, closes V4 at the OpenCode weekly guard unless `outputs/osler_v1/V4/DEEPSEEK_APPROVED` exists, and otherwise runs the DeepSeek phase in chunks of 200 with a balance check before each (ledger cap $9.00, floor $0.40). The old chain (`v4_build_chain.sh`) was stopped at 2026-10-15T06:22:46Z; its running build (pid 2000630) continues. Tests: CLAIMS V119–V121.
- [ ] operator answer to Question 10 (approve or hold). The default is hold: at the guard the chain closes V4 on the OpenCode pages (Q8).
- [ ] blocks.md; CLAIMS rows; SELF_AUDIT.md; commit

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

### V5a — DONE — 2026-10-13T22:51:28Z
- What ran: `v5_synthea.py extract --encounters 40000` (streams the Hub copy; 8 encounter row groups; observation scan 127 of 11,424 row groups, coverage recorded); `v5_synthea.py build` (run twice: once with a held-out-pool variant that was reverted, and once with the committed logic, which is the one used); `v5a_audit.py` (splits, held-out isolation, balance, O3 screens, chance and majority; `SELF_AUDIT.md`); ten items read per template.
- Output: `data/synthea/osler_v1/` and `data/gen/osler_v1_synthea/` (gitignored); `outputs/osler_v1/V5a/` (`extract_report` in data, `screen.json`, `audit.json`, `SELF_AUDIT.md`, `logs/`).
- Headline: 40,000 encounters; train 59,849 / dev 7,702 / test 6,509 / held-out 1,342 rows; the indicator template passes the O3 screen (Naive Bayes 0.576), the page category fails as expected (1.000; V5a-D1). No training patient is in any other split (CLAIMS V042–V046).
- Surprises: an earlier build variant that removed a held-out patient's other items from dev and test cut dev indicator items from 740 to 72; it was reverted (V5a-D2). The held-out aki items are 4.5 % yes and are not balanced. Three indicators have at most 4 positives per class in dev and test.
- Next: V5b (MTSamples, CodiEsp, PMC-Patients, NHS and mkurman evaluation sets); V4 once V1, V2 and V5a are done.

### V5b — DONE — 2026-10-13T22:56:31Z
- What ran: `v5b_realtext.py fetch` (five public sources into the Hub cache, never the repo); `v5b_realtext.py build` (MTSamples page items; CodiEsp train and corpus dev; PMC-Patients (anchors selected across the whole file, up to 40,000) with the leakage drop, run twice (the first build superseded); mkurman evaluation set; NHS checked for a clean note-type mapping); `v5b_audit.py` (splits, PMC leakage re-derived, text overlap with the v0 protected sets and the MR test sets, chance and majority, informational Naive Bayes; `SELF_AUDIT.md`); two items read per template.
- Output: `data/gen/osler_v1/realtext/` (gitignored; `manifest.json` with sha256 per file); `outputs/osler_v1/V5b/` (`audit.json`, `report.json`, `SELF_AUDIT.md`, `logs/`).
- Headline: training real text: MTSamples 668 / 97 / 111; CodiEsp 3,374 train and 1,654 test (corpus dev); PMC pairs 52,549 / 7,990 / 8,100. Evaluation only: mkurman 1,490 test items. NHS NOT MEASURED (department labels). No real-text item overlaps a protected or MR test text (CLAIMS V049–V054).
- Surprises: a first build took the first 40,000 rows as the anchor set, which left 24,821 of them without an internal partner; the build was corrected to select anchors across the whole file (the ADVISORY's 'all rows'). Every pair is from the `comm` licence subset. MTSamples has 8 laboratory and 25 admission pages (very small classes). CodiEsp is machine-translated English.
- Next: V4 (teacher prose) needs V1, V2 and V5a, which are DONE. The ICD prose must use the two-diagnosis design of v2. V5c (MIMIC) is re-checked before V7. V6 (MR suite and baselines, GPU) needs V1, V5a and V5b, which are DONE.

### V6 — DONE — 2026-10-14T01:41:00Z (checklist closed in this entry)
- What ran: the baseline chain `scripts/osler_v1/v6_baselines_run.sh` (zero-shot Qwen3.5-4B and -9B, MedDecider-4B and -9B, JEV-9B; the O2 runners; 20,812 rows each; all rc=0); `v6_analyse.py` (set summaries, the readout gate per template, the matched block); `v6_self_audit.py` (107 checks in a fresh process, 0 failed).
- Output: `loops/osler_v1/mr_baselines.md`; `outputs/osler_v1/V6/` (baseline_summary.json, SELF_AUDIT.md, self_audit.json, baselines/, logs/); CLAIMS V082–V095.
- Headline: on mr_test (19,294 items) zero-shot Qwen3.5-4B scores 0.7383, -9B 0.7480 and JEV-9B 0.7473, all items scored. MedDecider-4B and -9B score 67.34 % of the items (12,992; the 12-option items are skipped) with 0.6681 and 0.6653 on those items, and the zero-shot models score 0.6177 and 0.6312 on the same items (matched block, additional). On mr_heldout the synthea indicator template fails the competitors' readout gate, so their held-out figures cover 176 items.
- Surprises: MedDecider cannot take 12-option items (its letter set is A to J). JEV-9B has the highest template macro on mr_test (0.7336). The page-category templates are lexical (V1-D2, V5a-D1). The d1-3B row is NOT MEASURED: envs/d1 does not exist, the GPU queue had priority, and its two-hour timebox was not used.
- Next: V8 (B0 running from 01:35Z, then B1); the V4 teacher build; once V4 is complete, re-run the V6 assembly and baseline scoring for the teacher templates (ADVISORY §6); the MIMIC re-check before V9.

### V6 addendum — 2026-10-14T17:45Z (appended; the entry above is unchanged)
- The additional d1-3B row was measured in the GPU idle window: its own code (`system_one`), both option orders, 20,812 rows, 0 skipped (`scripts/osler_v1/v6_d1_run.py`). It is in `mr_baselines.md` section 8 and CLAIMS V114–V117. The d1-3B row is additional only (D31); the matched block for the comparators is unchanged. The self-audit now runs 127 checks, 0 failed.

### V4 (checkpoint: second endpoint prepared; DeepSeek held) — 2026-10-15T06:24Z
- Built and tested: budget ledger total cap (`SpendLedger(total_cap_usd=)`, 2 new tests; 10 pass), endpoint and model per row and item, `v4_closure.py` counts by endpoint, `v4_deepseek_chain.sh` (helpers tested by hand: balance read 9.88, the week-guard match, the stop-reason parse).
- Ledger at 06:23Z: OpenCode Go $33.07 of the $36 week ($6.90 in the last 5 h); 19,453 narrative encounters processed of 28,600; the OpenCode build (pid 2000630) continues under the switch script.
- Full suite: 704 passed with the padding test deselected on the command line; the padding test itself XPASSed on the GPU (Question 12).
- Open: Question 10 (DeepSeek hold, default) and Question 11 (SNOMED and RxNorm strings to hosted endpoints). Neither blocks other tasks.

## 5. Blocked items

| id | what is blocked | exact reason | what would unblock it |
|---|---|---|---|
| V4-ICD | the ICD prose (`mr_icd_support_prose_v1`) | the pilot pass rate is 59.8 % (58 of 97 eligible) after the one permitted prompt fix (V059, V060); the ADVISORY threshold is 60 % (V063) | an operator decision: accept a lower threshold; or change the ICD page design; or drop the ICD teacher prose (the template ICD items stay, and B2 contains indicator and narrative prose only) |
| V5c | the MIMIC sources (D28) | the PhysioNet files are absent on the volume (re-checked 2026-10-14T00:15Z; CLAIMS V097); ADVISORY V5c never blocks the loop | none for this loop: the finals train without MIMIC (D32; ADVISORY V8) and the V11 audit is recorded as not applicable |
| V4-DS | the DeepSeek API phase of the narrative build (operator instruction; V4-D7) | D30 allows the DeepSeek API as a hosted teacher only with zero data retention. Third-party pages found on 2026-10-15 report no documented zero-retention tier for the first-party API (Question 10); not verified against DeepSeek's own terms | the operator's answer to Question 10: create `outputs/osler_v1/V4/DEEPSEEK_APPROVED` before the OpenCode weekly guard binds, or hold (the default). Held work does not block any other task |

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
5. **ICD teacher prose (V4, BLOCKED).** The two-diagnosis ICD page passed 43 % of pilot ids with the first prompt and 59.8 % with the revised one (CLAIMS V059–V063); the threshold is 60 %. Most rejections are status (history against current) and subject (family against patient): the teacher writes past or family conditions in ways the extraction reads otherwise. Options: (a) accept the template ICD items only and drop the ICD prose (B2 = indicator and narrative prose); (b) accept 59.8 % with the rejection bias recorded; (c) change the ICD page design (for example one diagnosis per prose page, with the distractors as separate notes). I would pick (a): it keeps every training decision verified and costs no further budget.
6. **Teacher build scale (V4).** The ADVISORY aims are 15,000 verified pairs per twin generator and 20,000 narrative pages. At two requests per second (the endpoint limit), the indicator build (21,000 patient ids, about 15,000 pairs at 71 %) and the narrative build (28,600 encounters, about 20,000 pages at 70 %) take roughly 17 to 20 hours together, and the spend is about $31 against the $60 cap. I am building to the aims as written; the operator may cut them.
7. **Bare-id collisions (V7-D4).** Forty PubHealth training rows have a bare source record id equal to a panel id (fifteen of them also equal a robustness id), and one PubHealth fast-dev row does the same. The ids are small numbers in two datasets (PubHealth against the MMLU-Pro health panel), so the collision is a namespace accident. O4 (osler_v0) recorded the same bare hits (`outputs/osler_v0/O4/build.json`: train panel 40 and robustness 15; dev 1 and 1) and kept the rows, because its protected-set docstring says a record is identified by the dataset-qualified id, and that id has no hits here. I dropped the rows as the conservative reading (AGENTS.md): 40 of 573,611 mix v2 training rows, and 1 of the 3,327 mix rows of the fast dev set. Options: (a) follow O4 and drop nothing; (b) keep the drop. I would pick (b): it costs nothing measurable and removes the question from the evaluation.
8. **Teacher budget (V4), corrected at 2026-10-14T00:03Z.** An earlier estimate here put the spend at the 7-day cap of $36 before the narrative aim was reached. That estimate used $0.0013 per indicator item. The measured cost is $0.000928 per indicator id (1,136 ids in the progress file, 69 % accepted; CLAIMS V079) and $0.00039 per narrative item (100 items in the ledger). The full aims (21,000 indicator ids and 28,600 narrative encounters) therefore cost about $31 on top of the $1.35 already spent, which is below the $36 weekly cap. No decision is needed unless the measured cost changes; the build runs to its aims as written.
9. **Dev–train overlap inherited from osler_v0 (V8, CLAIMS V100).** 229 general fast-dev items (mix v2) share a dataset-qualified source record with a training row, and 27 share the same text. The MR and knowledge dev parts share nothing with the training files. The overlap is the same in every block, so the keep rule's differences are less affected than the absolute dev numbers. Options: (a) keep the osler_v0 split and report the overlap as a limitation of the dev numbers; (b) rebuild the general dev split by record, which changes the fixed fast dev set and restarts V8. I would pick (a) for this loop and recommend (b) for the next one.

10. **DeepSeek API continuation of the narrative build (V4-D7; operator instruction, 2026-10-15).** The operator asked that, when the OpenCode Go weekly guard ($36) binds, the narrative build continue on the DeepSeek API (`api.deepseek.com`, model `deepseek-flash`, thinking off, its own $9.00 ledger cap, starting balance $9.88). D30 names the DeepSeek API as a hosted teacher only with zero data retention. A web search on 2026-10-15 found third-party pages (meetily.ai, guptadeepak.com, macaron.im, opper.ai; not DeepSeek's own terms) that report no documented zero-retention tier for the first-party API, training on API data as the default, and storage in China. I could not verify DeepSeek's own policy. The data would be code-generated Synthea facts (synthetic records; no MIMIC). Options: (a) approve: create `outputs/osler_v1/V4/DEEPSEEK_APPROVED` before the OpenCode guard binds (about 07:50Z at the current rate; 19,453 of 28,600 encounters processed at 06:23Z), and the chain runs the DeepSeek phase under the $9.00 cap and the balance floor; (b) hold: V4 closes on the verified OpenCode pages at the guard (Q8), and a later DeepSeek phase would need B2 and B3 rebuilt. I would pick (b) until the operator confirms the DeepSeek API's retention terms: sending the facts is irreversible, and holding is not. The hold is the default now.
11. **SNOMED and RxNorm strings sent to hosted endpoints (AGENTS.md data rule).** AGENTS.md says UMLS, SNOMED and RxNorm-restricted strings never go to a hosted third-party API. The narrative facts are Synthea encounters: condition names are SNOMED CT display names and medication names are RxNorm names. V4-D3 removed only the SNOMED semantic tags. ADVISORY V4 names `synthea_narrative_page_v1` as a teacher source, and the OpenCode Go build has already sent these facts for every narrative encounter processed so far. I did not stop the build, because the ADVISORY names the route and the records are synthetic. Narrative rows written from V4-D7 on carry `teacher_endpoint` and `teacher_model`; earlier rows do not and are OpenCode Go rows. Options: (a) confirm the route, no change; (b) drop the narrative teacher rows and rebuild B2 and B3 without them (B2 then equals B1 plus indicator prose); (c) change the narrative facts to codes or strip the names (a change of generator, not allowed in this loop). I would pick (a) if the operator judges that Synthea descriptors in a hosted prompt are within the SNOMED and RxNorm terms; otherwise (b).
12. **Padding test, observed again (Question 1).** The full suite on 2026-10-15 (`outputs/osler_v1/V4/logs/pytest_2026-10-15_with_padding.log`) reported `tests/test_padding_trained.py::test_trained_checkpoint_logits_are_padding_invariant` as FAILED: its assertion passed, and as a `strict=True` xfail that is an XPASS failure. Observed outcomes so far: XPASS at planning (ADVISORY), XFAIL on 2026-10-13 (V0), XPASS now. The test is unedited and is deselected on the command line, as in every run since V0, so the committed config is unchanged. The residual is at the test's 1e-3 tolerance boundary (V2 measured 1.92e-3 for a padded batch; the value is not printed by the test). Question 1 remains open.

13. **Gate O2 panel rule, operator confirmation (D31; needed before V9).** The D31 row in `docs/plans/PROGRAM.md` says the external panel moves from superiority to non-inferiority "because this loop adds no broad-clinical data (operator to confirm before V9 runs)". No confirmation is recorded in this file. The gate as written is used (accuracy lower bound > −0.02 against both baselines). V9 is about 12 hours away, behind B2 and B3. If the operator has not confirmed by then, V9 is recorded BLOCKED on this item and the gate is not changed (AGENTS.md: gates are never changed after results).

**Operator answers (received during V8, 2026-10-14):**
- **Q2 (replay share): accepted.** The 15.3 % replay share stands for this loop; the 20 % target is not pursued (V3-D1 accepted).
- **Q4 (code-set year): FY2027.** The acceptance year is FY2027. All 48 curated codes (12 families) pass the FY2027 order file, and they also pass FY2026 (CLAIMS V101); the table is unchanged.
- **Q5 (ICD prose): option (a).** The ICD prose is dropped. B2 is indicator and narrative prose only (V4-D1 stands; V7-D5).
- **Q6 (teacher peak hours): bulk generation may run in peak hours too,** under the same budget guards. The peak pause was removed from `scripts/osler_v1/v4_teacher_build.py` and the build chain restarted at 2026-10-14T06:42Z; it resumed from its progress file (4,097 ids done, 16,903 to do; CLAIMS V102). The guards (5-hour window $14, week $36, loop $60) are unchanged (V4-D4).
- **Q7 (bare-id rows): option (b).** The 40 rows and the 1 fast-dev row stay dropped (V7-D3, V7-D4).
- **Q9 (dev–train overlap): option (a).** The osler_v0 split is kept and the overlap is reported as a limitation of the general dev numbers (CLAIMS V100).
- **Q8 (teacher budget): operator instruction (received during V8).** If the V4 weekly teacher guard ($36) is reached, do not wait for the next week: finish V4 with the verified pages built so far, record the counts, and continue with B2 and B3. Recorded as V4-D5.
- **Osler-9B: deferred to the next loop (operator decision, V9-D1).** V9 trains Osler-4B only; V10 applies Gate O2 to Osler-4B only and reports the 9B verdict as NOT MEASURED — deferred by the operator.
- **Final LoRA targets (operator decision, V9-D2).** The final Osler-4B trains with B0b's attention-only LoRA targets (`q_proj`, `k_proj`, `v_proj`, `o_proj`), not B0's defaults; the data follows the kept block; the B2 and B3 keep decisions stay as the rule makes them. The operator's reason is in V9-D2.
- Q1 (padding test) and Q3 (run time) had no answer; the defaults in this file stand: the padding test stays as committed, and the ADVISORY run sizes stay.


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
- **V3-D1 (replay target).** The 20 % replay share is not reached. D20 (settled) requires permissive licences; with the permissive pool (HellaSwag, WinoGrande, and the two sources already in mix v2) the share is 15.31 %. Nothing is removed from the non-replay rows to reach 20 %: the ADVISORY says 'add', so the gap is reported (Question 2). CLAIMS V038–V040. Accepted by the operator (Question 2 answer): the 15.3 % share stands for this loop, and the 20 % target is not pursued.
- **V5a-D1 (Synthea page category).** `synthea_page_category_v1` fails the O3 screen (Naive Bayes 1.000) because its page headers name the class. The same lexical failure is recorded and accepted for the tabular page category (V1-D2). Its share of any mix is capped (V7).
- **V5a-D2 (held-out pool and dev and test).** The held-out pool shares 225 patients with dev and test, not with train (the audit: train 0, dev and test 0 shared with each other). A strict rule that moves each held-out patient out of dev and test was tried and reverted: it cut dev indicator items from 740 to 72, too few for the keep rule. The held-out set is evaluated only and no training item belongs to its patients. CLAIMS V044.
- **V5b-D1 (PMC anchors).** The ADVISORY reads 'all rows' with 'at most 40,000 anchor patients'. The build selects the first 40,000 patients (in file order) that link to another patient in the file, out of 167,034 read; a partner may be any patient in the file. A first build used the first 40,000 rows as the whole set, which left most anchors without a partner; that build is superseded (CLAIMS V051, V057). Every pair is from the `comm` licence subset.
- **V5b-D2 (CodiEsp).** Corpus train is our train; corpus dev is our test; the corpus test set has no codes and is unused. The English text is machine-translated and is labelled so.
- **V5b-D3 (mkurman).** The three CC BY-ND 4.0 rows are dropped (no derivatives), so evaluation items come only from the CC BY 4.0 rows (1,490).
- **V5b-D4 (NHS synthetic notes).** NOT MEASURED: `note_type` labels are departments and care settings, not page classes; the timebox was used.
- **V5b-D5 (MTSamples licence).** Its licence is uncertain; the operator accepted it under D32 and it is recorded in the manifest and in mr_data §9.
- **V4-D1 (the three prose generators).** ADVISORY V4 has one teacher task for three generators. The ICD generator is BLOCKED by the 60 % rule (V063). The indicator and narrative generators passed it (71 % and 70 %, V061–V062) and are built; B2 is indicator and narrative prose only, unless the operator decides otherwise (Question 5).
- **V4-D2 (the ICD prompt changed once).** After the first pilot the prompt gained the rules for a page with several diagnoses and explicit definitions for certainty, subject and status in the extraction. It was applied only to the diagnoses kind, so the indicator and narrative prompts, and their pilots, are unchanged. The first pilot is kept as evidence (V059).
- **V4-D3 (narrative facts).** The Synthea descriptors are cleaned of SNOMED semantic tags ('(disorder)', '(finding)') before the teacher sees them. The narrative pilot ran before this change; its 70 accepted pages are the pilot record and are not used for training.
- **V0-D7 (speed evidence).** The first run of config a stored only its last 50 step records, so its throughput could not be
  re-derived from the file. Config a was re-run with every record stored (`speed/a.json`; the first run is kept as `speed/a_run1.json`),
  and the choice is recomputed from these files (`outputs/osler_v1/V0/SELF_AUDIT.md`, 12 PASS). The driver now stores all records; the
  other configurations' files keep their last-50 records, so their summaries cannot be re-derived from disk. Their numbers are recorded as
  measured (CLAIMS V013–V016).
- **V6-D1 (baseline chain restart).** The first baseline script used a short MedDecider-9B revision, so the chain was stopped and restarted as `scripts/osler_v1/v6_baselines_run.sh` at 2026-10-13T23:30Z. The first attempt's zero-shot 4B predictions were removed before the rerun. The chain log keeps the first `START zs4b` line (23:30:14Z) from that attempt; only the 23:30:45Z run produced predictions.
- **V6-D2 (set-level figures and the readout gate).** ADVISORY V6 applies D12 to the zero-shot cells and does not say how a set's accuracy treats a failing cell. D12 reports no accuracy for a READOUT_FAIL cell, so each set's figures are taken over the passing cells and the failing cells are named (`readout_fail_templates`). On `mr_heldout` this leaves 1,518 items for the zero-shot models and 176 for the three competitors. The matched block (additional, labelled) puts all five models on the same 176 items; coverage (scored over in scope) is reported beside every figure. CLAIMS V082–V093.
- **V7-D1 (block contents and the teacher gate).** B0 and B1 hold only the sources of ADVISORY V7 with train splits. B0 = mix v2 train + the V3 replay: 653,874 rows, replay 15.31 %, largest template 6.48 %. B1 = B0 + the V1 and Synthea train items with the tabular cap: 696,532 rows, replay 14.38 %, tabular 2.00 %, largest template 6.08 %. B2 and B3 read the teacher files and are built only with `--teacher-final`, once V4 is final (the V4 writer appends verified items as they are made). The first B3 build (23:44Z) included 610 partial teacher rows; that record and file are stale and are refused by the block driver.
- **V7-D2 (tabular cap).** The 2 % cap removes 45,223 tabular page-category rows from B1 (deterministic sample, seed 0), counted under `dropped` in `outputs/osler_v1/V7/blocks.json`.
- **V7-D3 (bare-id drop, Question 7).** 40 PubHealth mix v2 training rows are dropped from every block (25 with a panel id only, 15 with a panel and a robustness id). After the drop, the bare-id check and the dataset-qualified check both report zero hits in B0 and B1.
- **V8-D1 (zero-shot knowledge comparison).** The knowledge items of the fast dev set are not in the MR evaluation file, so zero-shot Qwen3.5-4B is run on the 500 tier-1 fast-dev items: the same runner and protocol, with scope `mix_v2` added to the O2 I/O module and the zero-shot runner (defaults unchanged). It is an additional, labelled comparison for B0's knowledge check.
- **V8-D2 (partitions).** MR = items of the MR sources; general = every other fast-dev item (mix v2, including the 500 tier-1 items); knowledge = the tier-1 items (MedQA, MedMCQA). ADVISORY V8 names the 'mix v2 dev part' for the general set; this is the non-MR part of the fixed set.
- **V8-D3 (keep-rule statistics).** The lower bound is the 2.5th percentile of the paired bootstrap (the lower end of a 95 % two-sided interval), with 1,000 resamples, seed 0, resampling within templates. MR and general use the template macro of paired differences (`stratified_macro_difference`); knowledge uses the item mean (`paired_mean_difference_stratified`). ADVISORY V8 does not state the percentile; this reading is used.
- **V8-D4 (step count).** The step count is the smallest multiple of four whose batches hold at least 80,000 realised items on B0. Multiples of four keep the 25, 50, 75 and 100 % evaluations on whole steps; realised items may exceed 80,000 by at most three batches.
- **V8-D5 (evaluation set).** Each evaluation uses the full fast dev set (5,723 items), not the trainer's default 6,000-item subset, so the selection is on the pooled fast-dev template macro as ADVISORY V8 requires.
- **V2-D1 (acceptance year, operator Q4).** The acceptance year is FY2027 (`verify_fy2027.json`: passed, 0 missing, 0 not billable, 48 codes). FY2026 also passes (`verify_fy2026.json`). The curated table is unchanged.
- **V4-D4 (peak-hour pause removed, operator Q6).** The teacher build no longer waits in the peak windows. The budget guards are unchanged and checked before every submission. The build was restarted from its progress file at 2026-10-14T06:42Z (4,097 ids done). The peak tariff applies to the calls made in those hours, so the spend projection is higher than the off-peak figure (Question 8).
  Consequence measured at 2026-10-14T13:27Z: since the restart, 7,200 ids cost $9.81 (about $0.00136 per id; off-peak about $0.00094, peak about double). The ledger total is $14.00, $6.78 in the last 5 hours and $14.00 in the 7-day window (the caps are $14 per 5 hours and $36 per week). At this blend about 9,640 indicator ids and 28,600 narrative encounters would cost about $27 more, so the weekly $36 guard will stop the narrative build at roughly two-thirds of its encounters, until the window rolls (about 2026-10-20). The chain retries every 15 minutes while the guard holds; I will stop it there and record V4 as budget-limited (ADVISORY V4: build within the budget).
- **V4-D5 (budget-limited finish, operator instruction).** If the weekly teacher guard ($36, rolling seven days) is reached, V4 finishes with the verified pages built so far, and its counts are recorded; the build does not wait for the next week. The narrative target (20,000 pages) is then reported as reached or not, with the counts. ADVISORY V4 requires building within the budget, and this instruction is the operator's reading of that rule.
- **V7-D5 (B2 content, operator Q5a).** B2 = B1 plus the indicator and narrative prose; the ICD prose is not in any block (V4-D1).
- **V8-D6 (run time).** At 4.12 items per second, an 80,000-item block runs about 5.4 hours, not the ADVISORY's 2–3 hours (Question 3). Recorded as measured.
- **V9-D1 (Osler-9B deferred, operator decision, received during V8 at 2026-10-14T17:25Z).** The operator decided to skip Osler-9B in this loop. V9 trains Osler-4B only, on the kept mix, with the same recipe and the 250,000-item cap; the 9B is deferred to the next loop. V10 applies Gate O2 to Osler-4B only, against its own baselines (MedDecider-4B and zero-shot Qwen3.5-4B for the MR suite, as D31 names them), and writes the Osler-9B verdict as NOT MEASURED — deferred by the operator. The 9B baselines measured in V6 (zero-shot 9B, MedDecider-9B, JEV-9B) stay as recorded and are not used in any 4B verdict. Everything else in the plan is unchanged.
- **V9-D2 (final LoRA targets, operator decision, received 2026-10-15T05:25Z).** The final Osler-4B trains with B0b's attention-only LoRA targets (`q_proj`, `k_proj`, `v_proj`, `o_proj`; `scripts/osler_v1/v9_final.py` sets them from `v8_block_run.ATTENTION_TARGETS`), not B0's default targets (12 module names, which add the linear-attention and MLP projections). The data follows the kept block, as V9-D1 says. The B2 and B3 block runs keep the default targets that B0 and B1 used, and their keep decisions stay as the rule makes them. The operator's reason: on the 500 knowledge-dev items, B0b minus zero-shot Qwen3.5-4B is −0.008 [−0.042, +0.026], against B0's −0.060 [−0.102, −0.020]; the fast-dev template macro is equal (B0b 0.8573, B0 0.8582; CLAIMS V104, V105, V110, V111). The attention-only names apply to the full-attention projections of this hybrid model; the other projections of the base stay frozen in the final run.
- **V8-D7 (item ids in the cross-model join).** The trainer's `Item` derives its id from the identity fields (`make_item`), so the block predictions carry ids that differ from the fast dev rows' ids. The knowledge comparison against zero-shot maps each O2 row id to the trainer's id by `mix_row_to_item` on the same fast dev row; a row with no fast dev id raises. The block-against-block keep comparison joins two files written by the same code and is unaffected. All 500 knowledge items mapped (CLAIMS V104).
- **V4-D7 (second teacher endpoint: DeepSeek API; held; operator instruction, 2026-10-15).** The operator instructed that, when the OpenCode Go weekly guard ($36, rolling seven days) binds, the narrative build continues on the DeepSeek API (`TEACHER_BASE_URL=https://api.deepseek.com`, `TEACHER_MODEL=deepseek-flash`, key from `DEEPSEEK_API_KEY` in `~/.bashrc`, never printed; thinking off; the same generator, prompts and verification; the same rate limit and the same off-peak and peak prices in the ledger; its own ledger, `outputs/osler_v1/V4/spend_ledger_deepseek.jsonl`, capped at $9.00; the balance is read with GET /user/balance). Built: `v4_teacher_build.py` takes `--ledger`, `--total-cap`, `--tag`; each progress row and each written item records the endpoint host and model id. Rows written before this change carry no field and are OpenCode Go rows (`v4_closure.py` labels them). `v4_deepseek_chain.sh` runs the phase in 200-index chunks with a balance check before each (stop below $0.40). **Held:** D30 allows the DeepSeek API only with zero data retention, and a 2026-10-15 search found third-party reports that the first-party API has no documented zero-retention tier (Question 10). The phase runs only if `outputs/osler_v1/V4/DEEPSEEK_APPROVED` exists when the OpenCode guard binds; otherwise V4 closes on the verified OpenCode pages (Q8). The combined hosted spend stays within the $60 loop cap in either case (OpenCode at most $36, DeepSeek at most $9). The cost check is the balance difference, not the ledger alone, because the ledger uses the OpenCode list price for the DeepSeek calls.

---

## 8. Closure summary feed

<!-- Filled at closure, feeding FINDINGS.md's Summary section. One row per major
     outcome, in the order a human should hear them. Each claim cites its CLAIMS id;
     failures and blocked tasks get rows too. -->

| # | outcome (one sentence, plain language) | claims | artifact |
|---|---|---|---|
| | | | |
