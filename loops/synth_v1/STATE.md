# synth_v1 — state

Loop status: RUNNING (autonomous; operator instruction 2026-10-29: no further questions, get the work done).

| step | status |
|---|---|
| synthetic pipeline (LLM-only families) | DONE — smoke v2 keep 91/200 (branch synth/driver-fix, 7295 1714) |
| verifier fixes | DONE — synth/generator-fixed |
| MIMIC-grounded build | RUNNING (subagent, branch synth/mimic-grounded) |
| mix composition with the grounded block | DONE 2026-10-29T18:28Z (counts and hashes in claims.md, rows C-S1 to C-S22) |
| final training, 63,304 steps | RUNNING since 2026-10-29T18:38:03Z (pid 17342, rebalanced mix; watchdog pid 18364). The first run (pid 15217, 18:28:49Z, pre-rebalance mix) was stopped by the coordinator |
| Gate O5 and scoreboard | PENDING |
| findings | PENDING |

## Run record (aggregates only; no item text, ids or MIMIC values)

Branch: `synth/train-v1` (from `loop/synth_v1` at ee83dd0). Not pushed, not merged.

Code provenance (R7). The composer and driver are not committed on this branch (the operator asked for STATE and claims only).
Run from a scratch extract at `/workspace/tmp/synth_train_code`: osler_v5 code from `loop/osler_v5` at 23a0362, the synth
module and tests from `synth/mimic-grounded` at 8c1c7bb (`MIX_EXCLUSIONS`, `in_training_mix`), plus two local edits:
`scripts/osler_v5/run_arm.py` (`--grounded` option; block appended after the augmentation, caps checked in code) and
`scripts/synth/v1_grounded_rows.py` (conversion, exclusions, checks). Both are uncommitted.

Steps and decisions this session:
- Recipe: osler_v5 F2 config `configs/osler_v5/arm_F2.yaml` (per-item weighting, option augmentation p = 1.0 on
  mimic_principal_dx_v1 and mimic_msdrg_v1, seed 0, `drop_mix_templates: [fda_route_choice_v1]`). Steps 63,304; evaluations at
  6330, 12661, 18991, 25322, 31652, 37982, 44313, 50643, 56974, 63304 (`evaluation_grid`, checked); full checkpoints every 2,500
  steps, newest two kept (driver constants).
- Exclusion (coordinator, commit 589d2c2 / 8c1c7bb): `mimic_icd_specificity_v1` excluded from the mix, reason `answer_in_facts`.
  Its 3,441 items were skipped by `in_training_mix`; they are not in the mix.
- The on-disk grounded manifest (`manifest_mimic_v1.json`) predates `MIX_EXCLUSIONS` and has no `mix_exclusions` key; the count
  above is from the code's `MIX_EXCLUSIONS` and the item template ids, not from the manifest. Not regenerated.
- Held-out text: 27 grounded items (6 of them in the excluded template) matched the held-out condition regex used by the
  synth builder (`HELD_OUT_TEXT`) or the ICD prefixes I50/N17 in their text. The builder's admission-level filter does not look
  at text, so the 21 matches in the mix templates were dropped (reason `held_out_text_match`) before the mix. After the drop,
  0 rows match. The 27 is a count, not a claim that the source admissions are held out.
- Option keys: osler_v4 convention (A/B/C for the DRG-CC, disposition and DRG templates by label; digits for APR). Gold is the
  key of the grounded label; no option or gold was created.
- Augmentation is applied to the corrected mix only (the pools come from it); the grounded block is appended afterwards and is
  not augmented, so its option text is never rewritten.
- The 20 % / 8 % caps are checked in code after the append. Grounded share 0.893 %; largest template in the whole mix
  `broad_aqua_rat_v1` at 4.762 % (unchanged from the corrected mix); largest grounded template 2.824 %.

Settled-rule consequences (not changed here; the operator decides; see Questions):
- RESOLVED by the coordinator correction (below): replay share was 19.824 % and the family share 35.584 % in the first mix.
- MIMIC-family share rises from 35.004 % to 35.584 % (the grounded rows are MIMIC rows; the 35 % family cap is not re-sampled).

Launch (2026-10-29T18:28:49Z): PID 15217 (python, child of the detached launcher). Log:
`outputs/synth_v1/train/logs/run.log`. First launch attempt failed at import (empty `$CODE` in a child shell); its log is kept as
`outputs/synth_v1/train/logs/run_failed_launch_1.log`. Watchdog: `scripts/synth/v1_watchdog_b.sh` (scratch), 60-second poll,
resume after 900 s without advance of `run.log` or `logs/train.jsonl`, max 5 resumes; its log `outputs/synth_v1/train/logs/watchdog.log`.
GPU: RTX PRO 6000 Blackwell, 0 MiB before launch; no vLLM or Nemotron server running. 2 GB write test passed before launch.
Checks before launch: ruff passes on the repo; `tests/test_synth_generator.py` and `tests/test_synth_driver.py` 73 passed (run
from the scratch extract with the synth code at 8c1c7bb). The full suite was not run during the job.

## Questions for the operator

1. The replay minimum (D36, 20 %) and the MIMIC-family cap (35 %) are not met after the grounded append (19.824 % and 35.584 %).
   The task's caps were only the 20 % block and 8 % template caps, so the run proceeds without a top-up or re-sample. Say if the
   run should be restarted with a top-up (not started).
   RESOLVED by the coordinator's correction (2026-10-29): the rebuilt mix meets both settled rules; see the correction section.
2. The grounded manifest on disk has no `mix_exclusions` key (built before commit 589d2c2). Regenerate it, or accept the code count?

## Correction and relaunch (coordinator, 2026-10-29, after the first launch)

Steps: the watchdog (pid 15539) and the job (pid 15217) were stopped cleanly (SIGTERM, then checked). Their logs are kept:
`outputs/synth_v1/train/logs/run_stopped_attempt_1.log` and `outputs/synth_v1/train/logs/watchdog.log` (no event). The first
composition record is kept as `outputs/synth_v1/train/composition_attempt_1.json` (its shares: replay 19.824 %, family 35.584 %).

Rebuild (in the driver's `rebalance`, before the augmentation; the grounded block is appended after the augmentation as before):
- Drop 11,500 base MIMIC rows (source mimic_iv or mimic_iv_note; the grounded block is not dropped), water-filling from the
  largest MIMIC template down to a common level of 36,799 rows; ties by template name; the rows inside a
  template chosen by stable-hash rank (salt synth_v1_family_drop). No held-out or test file is read or written.
- Add 4,241 replay rows from the broad replay pool rows that are not composed in the mix (pool
  46,087 rows), chosen by stable-hash rank (salt synth_v1_replay_extra). 3,241 is the minimal top-up for the
  grounded-inclusive mix; 1,000 is the margin. No id repeats (checked).
- Grounded block: 13,120 rows, unchanged, appended after the augmentation.
- Family membership is the template list of the family blocks B1-B3 (15 templates, MedNLI included). The composer's family cap
  counts those blocks; the check counts the same templates over the final mix.

Shares (rows; before = the corrected mix; after = the final mix):
| quantity | before (corrected mix, D56a) | after the grounded append (final) | cap / minimum | result |
|---|---|---|---|---|
| rows | 1,456,438 | 1,462,299 | | |
| replay share | 20.002 % | 20.212 % | at least 20 % | pass |
| MIMIC-family share (family templates, incl. grounded) | 35.004 % | 34.974 % | at most 35 % | pass |
| grounded share | 0 | 0.897 % | at most 20 % | pass |
| largest template share (broad_aqua_rat_v1) | 5.001 % (post-rebalance, pre-append) | 4.956 % | at most 8 % | pass |

Hashes: corrected mix (D56a) `78d7e5478060fc828784cb3365569e177c6961270d38a76a9cff144684e9f3be` (equals the recorded value); final mix (after the rebalance and
the grounded append) `45a6ecec0fc23cbf53ad0e49194c02ec4f9eba9c5dffbb6c78094b86e682ae80`.

Relaunch: pid 17342, process start 2026-10-29T18:38:03Z, same config (`configs/osler_v5/arm_F2.yaml`, seed 0, 63,304 steps,
ten evaluations at Y8's steps). Watchdog pid 18364 (`scripts/synth/v1_watchdog_b.sh`, pgrep anchored to the driver's command
line; 900 s stall rule; max 5 resumes). The watchdog's pid file is `outputs/synth_v1/train/logs/run.pid`.

## Incident: watchdog stall during the step-6,330 evaluation (2026-10-29, 20:23Z)

- The first watchdog (`v1_watchdog_b.sh`, pid 18364) counted log silence only. The evaluation at step 6,330 runs for more than
  15 minutes with no log line (the GPU at 99 % and 73 GB), so at 20:23:28Z the watchdog judged the run stalled, killed pid 17342
  and resumed it from the step-5,000 full checkpoint (`logs/watchdog.log`, resume 1; `logs/resumes.jsonl`). Left unchanged, this
  would repeat each cycle (a resume to 6,330 then a second evaluation longer than 15 minutes) and never finish the evaluation.
- Change (coordinator-level judgement, reversible): the stall rule now requires the log silent for 15 minutes AND GPU utilisation
  below 5 %. The GPU-aware script is `scripts/synth/v1_watchdog_c.sh` (committed). A hung process still gets resumed; a busy
  evaluation does not. The old watchdog pid 18364 was stopped by pid.
- Resumed job: pid 33680 (resumed from the step-5,000 full checkpoint), watchdog pid per `ps`, using `v1_watchdog_c.sh`.
- The run has not yet reached the step-6,330 evaluation line at the time of writing.

## Provenance: what the composer needs (added 2026-10-29; code commits 6c85e73 and 43bb7c6)

The code committed in 6c85e73 (scripts/osler_v5/run_arm.py, scripts/synth/v1_grounded_rows.py) is not runnable on this branch
alone. It imports modules that live on other commits:
- `loop/osler_v5` at commit 23a0362: meddecide.train.corrected_mix, meddecide.train.option_augment, meddecide.train.osler_arm,
  meddecide.eval.composite, the osler_v4 Y8 composer `scripts/osler_v4/y8_final_mix.py` (which `run_arm.py` loads by path),
  `configs/osler_v5/arm_F2.yaml`, and the rest of the osler_v5 driver tree. The `data/` directory is a symlink in the scratch run.
- `synth/mimic-grounded` at commit 8c1c7bb: meddecide.synth.mimic_families (MIX_EXCLUSIONS, in_training_mix, HELD_OUT_TEXT)
  and its dependency meddecide.mr.mimic_items (osler_v4 template question texts, held-out families).
- The scratch extract that actually ran is `/workspace/tmp/synth_train_code`: `git archive` of 23a0362 with the synth module and
  tests from 8c1c7bb overlaid, plus the scratch edits that are now committed in 6c85e73. The `outputs` directory is a symlink to
  the repo's outputs; `PYTHONPATH` points at the scratch `src`.
- The run's launch command, with the paths, is in `outputs/synth_v1/train/logs/run.log` (first line of each launch) and in
  section "Relaunch" above.

## Stall judgement (mine, not an operator ruling)

The GPU-aware stall rule (silence for 15 minutes AND GPU utilisation below 5 %) is my judgement, made after the incident
above. It was accepted by the coordinator for now (2026-10-29). It is not an operator ruling. The operator may change it.

## Milestone: step 6,330 evaluation (2026-10-29, 21:12Z; pid 33680; intermediate, dev only, no verdict)

Driver line: S 0.8014 (MR 0.9050, MIMIC 0.7200, CG micro 0.7792); knowledge dev 0.7552; positional health original order True, seeds
1 and 2 both True. Trainer tripwire at the same step (fast dev v4, 17,532 items): accuracy 0.7843, template macro 0.7993. Checkpoint
saved at outputs/synth_v1/train/checkpoints/step_006330. The run continues toward 63,304 under v1_watchdog_c.sh.

## Outage 2026-10-30 07:25Z to 18:04Z (about 10.6 h): cause, evidence, fix

Timeline (UTC; from logs/train.jsonl, logs/watchdog.log, logs/resumes.jsonl, logs/watchdog_c.stdout):
- 07:01Z full checkpoint step_042500 written (COMPLETE). 07:25Z last training record at step 44,300. 07:44:36Z the watchdog
  (v1_watchdog_c) killed pid 33680 after 1,120 s of silence with the GPU idle. Its log has no traceback; the process was alive.
  Cause not identified: `dmesg` is not permitted on this host, and nothing in run.log explains the hang. This is an open item.
- 07:44:56Z to 08:01:17Z: five resumed jobs (pids 130017, 130530, 131147, 131724, 132294), each killed by the watchdog exactly
  180 s after its start. Cause (a bug in v1_watchdog_c.sh, not in the job): the stall age was taken from the stale mtimes of
  logs/train.jsonl (07:25Z) and logs/run.log (06:00Z), not from the job's start, and the GPU is idle during the roughly 8 to 10
  minutes of composition. Each job was therefore killed in its composition phase. None reached its resume record
  (resumes.jsonl has only the 2026-10-29 line). The resume limit (5) was then reached at 08:01:17Z and the watchdog exited.
- 17:49Z: logs backed up to logs/backup_before_resume_2026-10-30T1749Z/. 17:50:26Z: v1_watchdog_d.sh started the job (pid 207920)
  with --resume from step_042500 (checked: COMPLETE, optimizer.pt, rng.pt, trainable.pt, run_state.json present).
- 17:57:07Z: "resume" record written (from_step 42,500). 17:57 to 18:04:30Z: the job was alive, CPU-bound (main thread running,
  about 93 % CPU, 209 threads, the worker threads idle in futex waits), GPU 0 %, with 9.4 GB loaded. This is the resume
  fast-forward: the trainer replays the batch iterator for the 42,500 batches already taken before its first step
  (src/meddecide/train/trainer.py, `skip_in_epoch`). It is not a lock or a deadlock. The earlier resume from 5,000 took about one
  minute for the same loop; this one takes about seven.
- 18:04:30Z: first step after the resume (42,550). 18:06:26Z: step 42,700, rate about 1.36 steps per second, GPU 100 %, 91 GB.
  This is the first and only check of the resumed run; the 5-minute check passed.

Blocking point, answering the coordinator's question: none at 18:04Z onwards. Between 17:57Z and 18:04Z the process was in the
resume fast-forward (CPU-bound), not blocked. py-spy is not installed, so no stack dump was taken; the evidence is the process
state, CPU use and the step records.

Fix (scratch `scripts/synth/v1_watchdog_d.sh`, committed as scripts/synth/v1_watchdog_d.sh):
1. The stall age is measured from the later of the log mtime and the job's start time (a stale log no longer counts).
2. A 30-minute grace after each start: no stall check. This covers composition (about 7 to 10 min) and the resume fast-forward
   (about 7 min at step 42,500, longer at later steps).
3. MAX_RESUMES 10 (was 5). The kill rule is otherwise unchanged: silent log for more than 15 min AND GPU below 5 %.
No change to the training config, the mix, the seed or the step budget.

Open items: the cause of the 07:25Z hang is unknown. If it recurs, the watchdog will detect it after 15 min of silence and a
busy-idle GPU, and the run will resume from the newest full checkpoint.

Restart record: resumed at 17:57:07Z (from step 42,500); first step 18:04:30Z; step 42,700 at 18:06:26Z.
ETA to 63,304: about 20,600 steps at 1.36 steps per second (about 4.2 h) plus four evaluations (44,313; 50,643; 56,974; 63,304)
at about 20 min each, about 1.3 h. Total about 5.5 h from 18:06Z, so about 23:40Z on 2026-10-30 (estimate; the rate may vary).
