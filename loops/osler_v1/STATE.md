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
Last updated (UTC): `2026-10-15T14:31:06Z`
Iterations so far: `5` (V0–V3, V4, V5a, V5b and V6 closed; V7 built, blocks.md written for B0 to B1; V8 in progress: B0, B0b, B1 and B2 done; B1 KEPT against B0 (C = B1); B2 NOT KEPT against B1; the B3 run is in progress; Q13 confirmed, so V9 follows B3; V10 prepared on the CPU, not run; V5c blocked on absent files)

---

## 1. Task board

Statuses: `PENDING` → `IN_PROGRESS` → `DONE` | `BLOCKED — reason`

| id | task | GPU | deps | status | started (UTC) | finished (UTC) |
|---|---|---|---|---|---|---|
| V0 | Orientation, environment, speed recipe | small | — | DONE | 2026-10-13T21:32:02Z | 2026-10-13T22:29:31Z |
| V1 | MR generators | no | V0 | DONE | 2026-10-13T22:30:29Z | 2026-10-13T22:35:18Z |
| V2 | ICD-10-CM verification | no | V0 | DONE | 2026-10-13T22:43:25Z | 2026-10-13T22:44:06Z |
| V3 | Replay to 20 % | no | V0 | DONE | 2026-10-13T22:44:15Z | 2026-10-13T22:45:36Z |
| V4 | Teacher prose | no | V1, V2, V5a | DONE — closed 2026-10-15T07:56:54Z at the OpenCode Go weekly guard (V4-D5): indicator 14,903 verified pairs, narrative 16,450 verified pages; ICD prose dropped (Q5a; V4-ICD); the DeepSeek continuation was held and not run (V4-D7; Question 10) | 2026-10-13T23:09:08Z | 2026-10-15T07:56:54Z |
| V5a | Synthea | no | V0 | DONE | 2026-10-13T22:45:53Z | 2026-10-13T22:51:28Z |
| V5b | Real clinical text | no | V2 | DONE | 2026-10-13T22:52:20Z | 2026-10-13T22:56:31Z |
| V5c | MIMIC sources (only when the files exist) | no | V0 | BLOCKED — PhysioNet files absent (re-checked 2026-10-14T00:15Z, CLAIMS V097); re-check before V9 | | |
| V6 | MR evaluation suite + baselines | yes | V1, V5a, V5b (V4, V5c if built) | DONE | 2026-10-13T23:24:35Z | 2026-10-14T01:41:00Z |
| V7 | Mix v3 blocks | no | V3, V6 | IN_PROGRESS — B0, B1, B2 and B3 built (B2 734,514 rows and B3 792,260 rows, 2026-10-15T07:57–08:01Z); blocks.md pending | 2026-10-13T23:40:48Z | |
| V8 | Block runs + keep rule | yes | V7 | IN_PROGRESS | 2026-10-13T23:53:00Z (marked late: the block config and the first V8 files predate this mark) | |
| V9 | Final Osler-4B on the kept block, attention-only LoRA (Osler-9B deferred by the operator, V9-D1; LoRA targets by the operator, V9-D2) | yes | V8 | PENDING — unblocked: Q13 confirmed 2026-10-15; starts after the B3 keep decision | | |
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
| teacher: pilot pass rate; pairs built; spend | indicator pilot pass 71 %; indicator 14,903 verified pairs (29,806 rows; 21,000 patient ids; 70.97 % accepted); narrative 16,450 verified pages (23,351 encounter ids of 28,600; 70.45 % accepted); ICD prose not built (BLOCKED: pilot 59.8 %); OpenCode Go spend $36.0118 (week guard $36.00); the DeepSeek API was not used | `outputs/osler_v1/V4/closure.json`, `outputs/osler_v1/V4/SELF_AUDIT.md`, CLAIMS V126–V128 | V4 |
| Synthea: encounters, coverage, items per template | 40,000 encounters (33,879 emergency, 6,121 inpatient); observation coverage 37.1 %; items train 59,849 / dev 7,702 / test 6,509 / heldout 1,342. Indicator screen PASS (NB 0.576); page-category screen FAIL, expected (1.000; V5a-D1). Rare indicators have at most 4 positives per class in dev and test (limit) | `loops/osler_v1/mr_data.md` §8, `outputs/osler_v1/V5a/SELF_AUDIT.md` (CLAIMS V042–V048) | V5a |
| real text: MTSamples pages per class; CodiEsp items; PMC pairs | MTSamples items train 668 / dev 97 / test 111 (radiology 269, consult 255, progress 153, discharge 103, ED 63, admission 25, laboratory 8). CodiEsp train 3,374 / test 1,654 (machine-translated). PMC pairs train 52,549 / dev 7,990 / test 8,100 (up to 40,000 anchors across the whole file; partners anywhere in it). mkurman eval-only 1,490. NHS NOT MEASURED. Audit: no problems; text overlap zero | `loops/osler_v1/mr_data.md` §9, `outputs/osler_v1/V5b/SELF_AUDIT.md` (CLAIMS V049–V056) | V5b |
| MIMIC (if it arrives): pages per class; ICD rows per family; late-anchor removals | | `loops/osler_v1/mr_data.md` | V5c |
| MR baselines (template macro per model) | | `loops/osler_v1/mr_baselines.md` | V6 |
| blocks: rows, replay share; steps per block run | B0 653,874 rows (replay 15.31 %); B1 696,532 (14.38 %); B2 734,514 (13.63 %; largest template 5.77 %; tabular 2.00 %); B3 792,260 (12.64 %; largest template 6.63 %; tabular 2.00 %); steps per block run: the 13,992 of V8 (80,000 realised items on B0); the B2 and B3 counts are in the V7 logs, CLAIMS rows to follow | `outputs/osler_v1/V7/blocks.json`, `outputs/osler_v1/V7/logs/blocks_build_b2b3.log` | V7 |
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
Running jobs at 2026-10-15T14:28:08Z (GPU busy with the B3 run, 100 % utilisation):
- B3 run: pid 2275082 (`scripts/osler_v1/v8_block_run.py --block B3`, started by the chain at 2026-10-15T14:24:15Z; 792,260 train items; 13,992 steps), log `outputs/osler_v1/V8/logs/B3.log`, run record `outputs/osler_v1/V8/B3/logs/train.jsonl`. B2 finished at 2026-10-15T14:24:15Z (rc 0; NOT KEPT against B1, CLAIMS V140). The V4 build and its switch script exited (V4 closed at 2026-10-15T07:56:54Z).
- B2 and B3 chain: pid 1965958 (`scripts/osler_v1/v8_chain_b2b3.sh`), log `outputs/osler_v1/V8/logs/chain.log`. It built B2 and B3, replayed B0 to B3 (B2 82,826 realised items; B3 81,721), ran B2, and is running B3, one GPU job at a time.

Task in flight: V8 (B0, B0b, B1 and B2 done; B1 KEPT against B0, C = B1; B2 NOT KEPT against B1; B3 running, its keep decision is against B1), V7 (B0 to B3 built; blocks.md pending), V4 (DONE 2026-10-15T07:56:54Z). V0 to V3, V5a, V5b and V6 are closed (see the iteration log). V5c BLOCKED (files absent). V9 to V12 are PENDING.
Working dir:    outputs/osler_v1/V1/

V7 checklist (in progress; started 2026-10-13T23:40:48Z):
- [x] rules: ADVISORY V7 (cumulative blocks, 2 % tabular cap, 8 % template cap, leakage per block); B2 and B3 read the teacher files, so they need --teacher-final
- [x] first build (23:40Z) rejected by the leakage check: 40 unique PubHealth training rows whose bare id equals a panel id (15 of them also equal a robustness id); O4's build.json recorded the same bare hits and kept them under its dataset-qualified identity rule (V7-D4). Its B3 also held 610 teacher rows from an unfinished V4 build: that B3 record and file are stale (the block driver refuses them: leakage_clean false, no teacher flag)
- [x] rebuild with the rules (B0, B1; launched 2026-10-13T23:50:41Z; log outputs/osler_v1/V7/logs/blocks_build_v2.log)
- [x] B2 and B3 built after V4 closed (`v7_blocks.py --blocks B2 B3 --teacher-final`, chain at 2026-10-15T07:57:49Z to 08:01Z); the stale B3 record was replaced (CLAIMS V132–V134)
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
- [x] keep report follows the chain (2026-10-15T09:55Z): `starting_block` in `src/meddecide/train/blocks.py` (C = B0, or B0b when B0 fails and B0b passes; 2 new tests in tests/test_blocks_v8.py, 17 pass); `v8_blocks_report.py` compares B3 with the last kept block (B2 if kept, else B1), shows it as pending while an earlier decision is open, and refuses a keep record that names a different C. blocks.md regenerated: B1 KEPT against B0; B2 and B3 pending
- [x] B2 run done 2026-10-15T14:24:15Z (rc 0): 13,992 steps, 82,826 realised items, 0 log mismatches on 279 logged steps; full-dev accuracy 0.9015, template macro 0.8742 (CLAIMS V138–V139). Keep against C = B1: NOT KEPT (all three lower bounds fail; CLAIMS V140)
- [ ] B3 run (chain; started 2026-10-15T14:24:15Z; pid 2275082, `scripts/osler_v1/v8_chain_b2b3.sh`): its keep decision is against C = B1, the last kept block (`v8_keep.py --candidate B3 --current B1`). B3 contains B2's teacher prose, which failed on its own (Question 14)
- [x] V4 closure (2026-10-15T07:56:54Z): the O3 screen (indicator PASS; narrative FAIL, V4-D6); the code-string check (0 hits in both files); spot reads of 40 pages (terminal only); counts and spend in SELF_AUDIT (19 PASS) and CLAIMS V126–V131; FINAL.json written
- [x] second endpoint prepared (V4-D7, 2026-10-15): `v4_teacher_build.py` takes `--ledger`, `--total-cap`, `--tag`; the endpoint host and model are recorded on every progress row and every written item; `v4_closure.py` counts by endpoint; `v4_deepseek_chain.sh` waits for the OpenCode build, closes V4 at the OpenCode weekly guard unless `outputs/osler_v1/V4/DEEPSEEK_APPROVED` exists, and otherwise runs the DeepSeek phase in chunks of 200 with a balance check before each (ledger cap $9.00, floor $0.40). The old chain (`v4_build_chain.sh`) was stopped at 2026-10-15T06:22:46Z; its running build (pid 2000630) continues. Tests: CLAIMS V119–V121.
- [ ] operator answer to Question 10. The default (hold) was applied at the guard: the DeepSeek phase did not run, and V4 closed on the OpenCode pages. The answer is still open; a later DeepSeek phase would need B2 and B3 rebuilt (V4-D7)
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

V10 preparation (CPU; checked 2026-10-15, not yet run on the GPU; the verdict is NOT MEASURED until the Osler-4B scoring runs):
- [x] `scripts/osler_v1/v10_items.py`: 30,621 rows selected of 80,350 (MR test 19,294; MR held-out 1,518 of which 584 test; external panel 4,353; knowledge MedQA and MedMCQA test 5,456); 49,729 O2 rows not in V10, counted by benchmark; every row converts with `o2_row_to_item`; ids unique (manifest `outputs/osler_v1/V10/items_manifest.json`; CLAIMS V124)
- [x] `scripts/osler_v1/v10_score.py`: O2-format rows in both orders and single order; calibrated with the arm's dev fits; refuses a checkpoint whose saved calibration differs from the dev fits (ADVISORY V10). Smoke run on the B0b checkpoint: 16 of 16 rows scored, calibration match True (throwaway output, removed)
- [x] `src/meddecide/eval/gate_o2.py` (D31 rules as pure functions) and `tests/test_gate_o2.py` (9 pass); `scripts/osler_v1/v10_gate.py` writes `loops/osler_v1/gate_o2.md`. Wiring check on a scratch copy only: an oracle file (the gold labels) gave PASS, and MedDecider-4B's own rows gave NOT MEASURED for the 6,302 MR and 212 panel items it did not score. Neither is a result.
- [ ] V10 GPU scoring of Osler-4B (after V9); the 9B verdict stays NOT MEASURED (V9-D1); MDACE NOT MEASURED (data absent); the MIMIC part NOT MEASURED (V5c)

V9 checklist (opened 2026-10-15T14:31Z; unblocked by Q13):
- [ ] B3 run finished (the watcher `ban5c4mjh` waits for `END B3` in the chain log; B3 pid 2275082)
- [ ] B3 keep decision (`scripts/osler_v1/v8_keep.py --candidate B3 --current B1`), then `scripts/osler_v1/v8_blocks_report.py`; CLAIMS rows for the B3 run and its decision
- [ ] defect screen of B3 before V9 (below chance, a constant answer, divergence, knowledge below zero-shot, a result that looks too good); any defect is investigated and recorded, not tuned
- [ ] the kept block K recorded (B3 if kept, otherwise B1) in §2 and in the V9 provenance
- [ ] MIMIC re-check before V9 (`data/physionet` absent at 2026-10-15T14:31Z; V5c stays BLOCKED)
- [ ] replay for K at 250,000 items with eight evaluations (CPU): `scripts/osler_v1/v8_replay_steps.py --blocks K --target-items 250000 --multiple 8 --out outputs/osler_v1/V9`; the file hash must equal the block's V7 record
- [ ] `scripts/osler_v1/v9_final.py --block K`, launched detached (one GPU job; log `outputs/osler_v1/V9/logs/osler_4b.log`; PID recorded here), with the attention-only LoRA targets (V9-D2)
- [ ] completeness: `outputs/osler_v1/V9/osler_4b/block_result.json`, `checkpoints/best`, `arm_result.json` with the fitted temperatures; eight evaluations on the planned steps; 0 log mismatches
- [ ] V10 after V9: `scripts/osler_v1/v10_score.py` (detached GPU job, log `outputs/osler_v1/V10/logs/osler_4b.log`), then `scripts/osler_v1/v10_gate.py` → `loops/osler_v1/gate_o2.md`; `loops/osler_v1/results.md`; the Osler-9B verdict NOT MEASURED (V9-D1)
- [ ] V11: MIMIC is not in the final mix, so the memorisation audit is not applicable (record at closure)
- [ ] V12: `FINDINGS.md` (summary first; §2 outcome; every number with a CLAIMS row; the five-claim spot-check re-run), `NEXT.md` (proposals only), `Loop status: STOPPED`, commit, push, stop

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

### V10 (preparation; checkpoint) — 2026-10-15T06:45Z
- Built on the CPU and tested: the V10 items file and manifest, the calibrated scorer (smoke-tested on the GPU with 16 rows), the Gate O2 rules and their 9 tests, and the gate report. The verdict waits for the V9 checkpoint.
- The V4 closure tools are written and will run at closure: `v4_self_audit.py` (re-derives every headline count and reconciles the spend with the ledgers; the audit reads the files while the build runs, so its figures are final only after closure) and `v4_spot_read.py` (20 pages per generator, printed to the terminal only).
- Open: Q10 (DeepSeek hold, default), Q11 (SNOMED and RxNorm), Q12 (padding), Q13 (D31 confirmation before V9).

### V4 — DONE — 2026-10-15T07:56:54Z (closed at the OpenCode Go weekly guard; the DeepSeek continuation was held)
- What ran: `v4_teacher_build.py --mode build --gen narrative` on OpenCode Go (resumed from its progress file; the week reached $36.00 at 07:56Z); `v4_deepseek_chain.sh` held (no approval file; V4-D7); `v4_closure.py --finalize` (closure.json, FINAL.json); `v4_self_audit.py` (19 checks, 0 FAIL, fresh process); `v4_spot_read.py` (40 pages, terminal only; the defect types are below, no text); `v4_defect_counts.py` (aggregates).
- Output: `data/gen/osler_v1/teacher/{indicator,narrative}.jsonl` (gitignored); `outputs/osler_v1/V4/` (closure.json, FINAL.json, SELF_AUDIT.md, self_audit.json, defect_counts.json, progress files, ledgers, logs).
- Headline: indicator prose 14,903 verified pairs (29,806 rows) from 21,000 patient ids (70.97 % accepted; CLAIMS V126). Narrative prose 16,450 verified pages from 23,351 encounter ids (70.45 % accepted; the 20,000-page aim reached at 82.3 %; budget-limited, V4-D5; CLAIMS V127). OpenCode Go spend $36.0118 (CLAIMS V128). The DeepSeek API was not used.
- Surprises: the narrative O3 screen fails on the page class (V4-D6; CLAIMS V129); the indicator screen passes (CLAIMS V129). Defect types (CLAIMS V130–V131): unrounded decimal values in 33.4 % of indicator pages; the phrase 'documented baseline' in 70.0 % of indicator pages, with gold 'yes' at 47.05 % against 56.88 % without it; a residual SNOMED semantic tag '(morphologic abnormality)' in 624 narrative pages, so the V4-D3 cleaning was incomplete; bracketed placeholders; RxNorm tall-man lettering; a synthetic MRN (4471902) in 3,069 indicator and 1,837 narrative pages, with no label association.
- Next: the B2 and B3 runs (V8). Questions 10 and 11 stay open for the operator.

### V7 (checkpoint: B2 and B3 built) — 2026-10-15T08:01Z
- What ran: `v7_blocks.py --blocks B2 B3 --teacher-final` (chain, started 2026-10-15T07:57:49Z after FINAL.json).
- Headline: B2 734,514 rows (replay 13.63 %, largest template 5.77 %, tabular 2.00 %, 0 leakage hits; CLAIMS V132); B3 792,260 rows (replay 12.64 %, largest template 6.63 %, 0 leakage hits; CLAIMS V133). Narrative pages are 1.80 % of B2 rows (CLAIMS V134). The replay share falls with each teacher block (15.31 % on B0); the 20 % target stays unmet (Question 2).
- Next: the replays and the B2 and B3 runs (chain), then blocks.md and the keep decisions.

### V8 (checkpoint: B1 KEPT recorded; the keep report follows the chain) — 2026-10-15T10:09Z
- B1 is KEPT against B0: MR +0.0947 [+0.0807, +0.1098] (lower bound above 0), general +0.0062 [−0.0048, +0.0177] (above −0.01), knowledge +0.0480 [+0.0080, +0.0900] (above −0.02); CLAIMS V112. Both knowledge checks fail (B0 −0.0600, lower bound −0.1020; B0b −0.0080, lower bound −0.0420; CLAIMS V104, V111), so the starting C is B0 and the current C is B1.
- `starting_block` (`src/meddecide/train/blocks.py`) gives the starting C. `v8_blocks_report.py` compares B3 with the last kept block (B2 if kept, else B1), shows it as pending while an earlier decision is open, and raises on a keep record that names another C. blocks.md regenerated (B2 and B3 pending). Tests: 17 pass in `tests/test_blocks_v8.py`; full suite 715 passed, 1 deselected, exit 0 (CLAIMS V136).
- B2 run at step 4,000 of 13,992 at 10:09Z (pid 2180924; log `outputs/osler_v1/V8/logs/B2.log`; CLAIMS V137 for its step time). The chain (pid 1965958) starts B3 after it without waiting for the B2 decision.
- Next: `v8_keep.py --candidate B2 --current B1` when `END B2` appears in the chain log; then B3 against the last kept block; then the blocks report.

### V8 (checkpoint: B2 done and NOT KEPT against B1; B3 running) — 2026-10-15T14:28Z
- B2 finished 2026-10-15T14:24:15Z (rc 0): 13,992 steps, 82,826 realised items, the four evaluations on the full fast dev set, 0 log mismatches (CLAIMS V138–V139). Full-dev accuracy 0.9015, template macro 0.8742.
- Keep decision against C = B1 (CLAIMS V140): MR −0.0046 [−0.0181, +0.0072]; general −0.0038 [−0.0149, +0.0080]; knowledge −0.0220 [−0.0640, +0.0220]. Every lower bound fails, so B2 is NOT KEPT (ADVISORY §2 outcome 4); C stays B1.
- B3 (B2 plus V5b) is running in the chain and is decided against C = B1, the last kept block. Its teacher prose is Question 14.
- blocks.md regenerated (B3 pending). Next: `v8_keep.py --candidate B3 --current B1` when `END B3` appears in the chain log; then the blocks report; then the V9 check (Question 13).

### V8 (checkpoint: operator answers recorded; Gate O2 superiority rows) — 2026-10-15T14:31Z
- Operator answers: Q10 do not run the DeepSeek continuation (V4 stays closed); Q11 option (a), no change; Q12 no action; Q13 confirmed (D31 non-inferiority, superiority reported separately). V9 proceeds after B3. Q14 has no answer; its default is V8-D8.
- `v10_gate.py` now reports the external panel's superiority as labelled additional rows (accuracy lower bound above 0; never in the verdict), via `panel_superior` in `src/meddecide/eval/gate_o2.py`. tests/test_gate_o2.py: 11 pass (CLAIMS V141).
- B3 running (step 200 at 14:31Z). Next: the B3 keep decision, the defect screen, the kept block, the MIMIC re-check, the replay and V9 (detached).

### V8 (checkpoint: defect screen and the too-good investigation) — 2026-10-15T14:55Z
- Defect screen (`defect_screen` in `src/meddecide/train/blocks.py`, six tests; CLI `scripts/osler_v1/v8_defect_screen.py`) on the finished runs: B1 and B2 show no below-chance, constant-answer or divergence flag (CLAIMS V144). Both flag too good on the same four templates: mr_icd_support_template_v2, mr_page_category_tabular_v1, synthea_indicator_evidence_v1, synthea_page_category_v1.
- Investigation of the too-good flags: (1) no MR dev item shares text or a dataset-qualified record with any block's training file, B0 to B3 (CLAIMS V143); (2) the shortcut screens: the indicator (Naive Bayes 0.7964, V024), the ICD v2 template (0.804, V023) and the Synthea indicator (0.5757, V046) pass; the two page-category templates are lexical by design and fail as expected (1.000; V025, V045; ADVISORY §3.6, V5a-D1); (3) the splits are by patient (V1, V5a). No leakage path was found. The flags stay in the record. osler_v0 FINDINGS already measured a seen-template effect (+9 to +13 points on fresh seen templates), so the flags are expected for generated templates. An options-only check for the generated ICD and indicator templates is proposed for NEXT.md.
- The keep decisions and the V9 screen are unchanged by this entry. B3 is still running (step 550 at 14:44Z).

## 5. Blocked items

| id | what is blocked | exact reason | what would unblock it |
|---|---|---|---|
| V4-ICD | the ICD prose (`mr_icd_support_prose_v1`) | the pilot pass rate is 59.8 % (58 of 97 eligible) after the one permitted prompt fix (V059, V060); the ADVISORY threshold is 60 % (V063) | an operator decision: accept a lower threshold; or change the ICD page design; or drop the ICD teacher prose (the template ICD items stay, and B2 contains indicator and narrative prose only) |
| V5c | the MIMIC sources (D28) | the PhysioNet files are absent on the volume (re-checked 2026-10-14T00:15Z; CLAIMS V097); ADVISORY V5c never blocks the loop | none for this loop: the finals train without MIMIC (D32; ADVISORY V8) and the V11 audit is recorded as not applicable |
| V4-DS | the DeepSeek API phase (operator instruction; V4-D7) | not run: V4 closed at the OpenCode guard on 2026-10-15T07:56:54Z with no approval file; D30 requires zero data retention, and the third-party reports (Question 10) are unverified | the operator decided not to run the DeepSeek phase in this loop (Question 10, answered 2026-10-15); V4 stays closed as it is. It blocks nothing |

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
   **Status (2026-10-15T07:56:54Z):** the default (hold) applied at the guard. The DeepSeek phase did not run, and V4 closed on the OpenCode pages (16,450 narrative pages; CLAIMS V127). A later approval would need the narrative build resumed and B2 and B3 rebuilt before their runs.
11. **SNOMED and RxNorm strings sent to hosted endpoints (AGENTS.md data rule).** AGENTS.md says UMLS, SNOMED and RxNorm-restricted strings never go to a hosted third-party API. The narrative facts are Synthea encounters: condition names are SNOMED CT display names and medication names are RxNorm names. V4-D3 removed only the SNOMED semantic tags. ADVISORY V4 names `synthea_narrative_page_v1` as a teacher source, and the OpenCode Go build has already sent these facts for every narrative encounter processed so far. I did not stop the build, because the ADVISORY names the route and the records are synthetic. Narrative rows written from V4-D7 on carry `teacher_endpoint` and `teacher_model`; earlier rows do not and are OpenCode Go rows. Options: (a) confirm the route, no change; (b) drop the narrative teacher rows and rebuild B2 and B3 without them (B2 then equals B1 plus indicator prose); (c) change the narrative facts to codes or strip the names (a change of generator, not allowed in this loop). I would pick (a) if the operator judges that Synthea descriptors in a hosted prompt are within the SNOMED and RxNorm terms; otherwise (b).
   **Status (2026-10-15T07:56:54Z):** the OpenCode route sent the Synthea facts of every narrative encounter processed (23,351 ids). The 624 accepted pages that carry the SNOMED semantic tag '(morphologic abnormality)' show that the V4-D3 cleaning was incomplete (CLAIMS V131). No decision is recorded; the blocks are built as they stand, and the operator decides before any release.
12. **Padding test, observed again (Question 1).** The full suite on 2026-10-15 (`outputs/osler_v1/V4/logs/pytest_2026-10-15_with_padding.log`) reported `tests/test_padding_trained.py::test_trained_checkpoint_logits_are_padding_invariant` as FAILED: its assertion passed, and as a `strict=True` xfail that is an XPASS failure. Observed outcomes so far: XPASS at planning (ADVISORY), XFAIL on 2026-10-13 (V0), XPASS now. The test is unedited and is deselected on the command line, as in every run since V0, so the committed config is unchanged. The residual is at the test's 1e-3 tolerance boundary (V2 measured 1.92e-3 for a padded batch; the value is not printed by the test). Question 1 remains open.

13. **Gate O2 panel rule, operator confirmation (D31; needed before V9).** The D31 row in `docs/plans/PROGRAM.md` says the external panel moves from superiority to non-inferiority "because this loop adds no broad-clinical data (operator to confirm before V9 runs)". No confirmation is recorded in this file. The gate as written is used (accuracy lower bound > −0.02 against both baselines). V9 is about 12 hours away, behind B2 and B3. If the operator has not confirmed by then, V9 is recorded BLOCKED on this item and the gate is not changed (AGENTS.md: gates are never changed after results).

14. **Cumulative blocks and a non-kept source (V8; decided by the rule at the B3 keep).** B2 (B1 plus the V4 teacher prose) was NOT KEPT against B1 (CLAIMS V140). B3 is B2 plus the V5b real text, so B3 contains the prose, and its only comparison is against C = B1, the last kept block. If B3 is kept, the kept mix holds prose that failed on its own. ADVISORY §2 outcome 4 says a source that does not help on dev is dropped, not tuned. Options: (a) follow the rule: the kept mix is B3 if it passes against B1, and the B2 result is reported next to it; (b) drop the teacher rows from the final mix (B1 plus V5b), which no V8 run measured. I would pick (a): it is the pre-registered decision, and the operator's V9-D2 instruction says the B2 and B3 keep decisions stay as the rule makes them. Context (a lexical association, measured, not tested): indicator pages with the phrase 'documented baseline' have gold 'yes' in 47.05 % of pages with it against 56.88 % without (CLAIMS V130). If B3 is not kept, the final mix is B1 and this question closes.

**Operator answers (received 2026-10-15, after B2 and during B3):**
- **Q10 (DeepSeek continuation): do not run it in this loop.** V4 stays closed as it is (closed 2026-10-15T07:56:54Z on the OpenCode pages). No `DEEPSEEK_APPROVED` file exists and the DeepSeek ledger is unused.
- **Q11 (SNOMED and RxNorm strings to the hosted endpoint): option (a), confirm the route, no change.** The operator's reason: the Synthea display names are public synthetic strings, and the AGENTS.md rule targets licensed UMLS-file strings; the advisor will clarify that wording at the hard stop. The narrative pages stay as built (16,450 verified; the 624 pages with the residual SNOMED tag, CLAIMS V131, are unchanged).
- **Q12 (padding test): no action.** The test stays as committed and is deselected on the command line, as in every run.
- **Q13 (Gate O2 panel rule, D31): confirmed.** Gate O2 is applied as written: the external panel is non-inferior when the accuracy lower bound is above −0.02 against both baselines. Superiority is reported separately and never enters the verdict. V9 proceeds after B3.
- **Q14 (cumulative blocks): no answer given.** The default applies (V8-D8, Deviations): the kept mix follows the keep rule.

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

- **V4-D6 (narrative O3 failure, page category).** `synthea_narrative_page_v1` fails the O3 screen: the gold page class appears in the record text for 683 of 1,496 dev items (Naive Bayes macro 0.8923; CLAIMS V129). The narrative pages are page-category items (ADVISORY V4), and their headers name the class. This is the lexical failure ADVISORY §3.6 expects for page category, and V1-D2 and V5a-D1 accepted it for the other page-category generators. The narrative pages enter B2 and B3 as their own source (`mr_teacher`, skill `mr_narrative`): 1.80 % of B2 rows and 1.67 % of B3 rows (CLAIMS V134). The 2 % tabular cap of V7 does not count them, which is a judgement recorded here; the keep decisions show whether they help or hurt.
- **V4-D8 (narrative target and the defects found).** The 20,000-page narrative aim is not reached: 16,450 pages from 23,351 encounters at the OpenCode Go weekly guard, 82.3 % of the aim (V4-D5). The spot read (20 pages per generator, 40 in all) and the defect counts (CLAIMS V130–V131) found artefacts the verifier does not catch: unrounded decimal values; the rule-restating phrase 'documented baseline', which goes with gold 'no' in the indicator pages; bracketed placeholders; a residual SNOMED semantic tag in 624 narrative pages; and a synthetic MRN repeated across pages. They are recorded, not fixed: the build is closed, and rewriting verified text would change the verified set after its verification and cost budget. They are candidates for the next loop's generator revision. The keep decisions (V8) use the blocks as built.

---

- **V8-D8 (the kept mix when a superset of a non-kept block is decided, Question 14).** B2 (B1 plus the V4 teacher prose) was NOT KEPT against B1 (CLAIMS V140). B3 contains that prose, and its only comparison is against C = B1. The operator has not answered Question 14, so the default applies: the kept mix follows the keep rule (if B3 passes against B1, the final mix is B3 and the B2 result is reported beside it). Recorded as a deviation so the operator can redirect.

## 8. Closure summary feed

<!-- Filled at closure, feeding FINDINGS.md's Summary section. One row per major
     outcome, in the order a human should hear them. Each claim cites its CLAIMS id;
     failures and blocked tasks get rows too. -->

| # | outcome (one sentence, plain language) | claims | artifact |
|---|---|---|---|
| | | | |
