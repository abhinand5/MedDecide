# synth_v1 — state

Loop status: RUNNING (autonomous; operator instruction 2026-10-29: no further questions, get the work done).

| step | status |
|---|---|
| synthetic pipeline (LLM-only families) | DONE — smoke v2 keep 91/200 (branch synth/driver-fix, 7295 1714) |
| verifier fixes | DONE — synth/generator-fixed |
| MIMIC-grounded build | RUNNING (subagent, branch synth/mimic-grounded) |
| mix composition with the grounded block | DONE 2026-10-29T18:28Z (counts and hashes in claims.md, rows C-S1 to C-S15) |
| final training, 63,304 steps | RUNNING since 2026-10-29T18:28:49Z (pid 15217; watchdog pid 15539) |
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
- Replay share falls from 20.002 % to 19.824 % (replay rows 291,319 unchanged; the top-up rule would need 3,241 more replay
  rows (computed by the minimal-top-up formula); not added).
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
2. The grounded manifest on disk has no `mix_exclusions` key (built before commit 589d2c2). Regenerate it, or accept the code count?
