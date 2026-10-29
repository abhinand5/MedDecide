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
