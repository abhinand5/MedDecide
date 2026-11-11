# STATE — osler_v6

Loop status: `RUNNING`
Branch: `loop/osler_v6`. Last updated (UTC): `2026-11-10T11:46Z` (Z0-Z5 DONE; Z6 IN_PROGRESS, resumed).
Task in flight: Z6 — resumed from full checkpoint step 20,000 at 11:37Z after a CUDA OOM; detached, uv PID `1046221`,
python PID `1046224`, log `outputs/osler_v6/Z6/final/logs/run_resume.log`; `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
The resumed run **passed the former crash point** (step 22,317) at ~13:48Z — the log is at step 22,406 with no OOM and the GPU
footprint is ~58 GB (against 97 GB before); the allocator setting fixed it. Four evaluations are now in (6,330 / 12,661 /
18,991 / 25,322; C036, C038, C039, C042), the last with positional PASS; under **A1** all of them are dev trajectory, and the
only candidate is the evaluation at step 63,304. Five evaluations are in (C036, C038, C039, C042, C043): S 0.796776,
0.791281, 0.818523 (positional FAIL), 0.822039, 0.827817 — the trajectory is rising. Last poll 2026-11-10T22:37Z: step
31,716 (~50%), ~3.1 s/step. Sixth evaluation (37,982, C044): S 0.837467 (new high) but **positional FAIL** on the original
order (the second miss after 18,991; under A1 it is trajectory only). Seventh evaluation (44,313, C045): S 0.840188 (new
high) but positional **FAIL** again (0.364730 vs 0.395210; the third original-order miss); a readout diagnostic (C046) shows
a strong key/position bias in the original order (BJ 128 of 499 two-letter-gold items) that the permutations remove. Under A1
the step-63,304 evaluation must pass or the final is BLOCKED. Eighth evaluation (50,643, C047): S 0.841495 (new high),
positional **PASS** again — the check has alternated (fails at 18,991 / 37,982 / 44,313, passes at the other five). Last poll
2026-11-11T16:15Z: step 50,759 (~80%). Ninth evaluation (56,974, C048): S 0.845382 (new high), positional **PASS** — the last
two evaluations pass. Last poll 2026-11-11T22:05Z: step 57,048 (~90%), ~3.2 s/step; the final evaluation at 63,304 is due
about 04:00Z on Nov 12, the A1 decision follows, then post-processing (~1 h).

## 1. Task board

| id | task | depends | status |
|---|---|---|---|
| Z0 | Setup: GO recorded, teacher revision checked, Z8 closure noted, AGENTS current loop | — | DONE 2026-11-07T17:40Z |
| Z1 | Anchor scoring to 400,000 items with the osler_v4 Y5 path (resume from 112,128; parity 40 of 40 re-run); outputs to `outputs/osler_v6/Z1/` | Z0 | DONE 2026-11-08T07:13Z — 385,536 scored (96.38 %), 14,464 cut by the 12-hour budget, audit PASS |
| Z2 | Dosing sampler (D57, 8 per batch, ≥3 labelled anchors, seeded) and the λ calibration probe (200 steps at λ = 1; λ then frozen; ADVISORY §2.4) | Z1 | DONE 2026-11-08T08:26Z — final probe s0 0.170491, λ = 3.243598 frozen; audit PASS (probe #1, random fill, superseded) |
| Z3 | Arm G1 (F2 + distillation, seed 0) at the frozen λ, 13,992 steps; check the 30–50 % band over its first 200 steps; G0 is the existing F2 seed-0 result C062 (checked, not re-scored) | Z2 | DONE 2026-11-08T21:31Z — 13,992 steps, band 0.355474 PASS, final S 0.825891 / knowledge 0.775 / positional PASS; audit PASS |
| Z4 | Arm G1 seed 1, only if G1 seed 0 passes positional health | Z3 | DONE 2026-11-09T10:52Z — 13,992 steps, band 0.365937 PASS, final S 0.826311 / knowledge 0.7708 / positional PASS; audit PASS |
| Z5 | Choice rule (ADVISORY §2.6), dev only. None qualifies → outcome (4): report and stop | Z3, Z4 | DONE 2026-11-09T10:52Z — chosen G1s0 (knowledge lower +0.006244, MR −0.0207, MIMIC −0.0104, CG 0.7945, positional PASS); seed-1 stability reported |
| Z6 | Final at 63,304 steps, positional health at each evaluation, checkpoint by S (only if Z5 chose a recipe) | Z5 | IN_PROGRESS 2026-11-09T10:52Z |
| Z7 | Gate O6 verdict, audit, FINDINGS.md, five-claim spot-check, closure | Z6 or Z5 | PENDING |

## 2. Operator decisions

- **GO for D57, D58 and the ADVISORY §2 design:** `yes` — 2026-11-07, operator instruction "start osler_v6 with everything intact".
- **Loop model:** `claude-sonnet-5-5` (operator instruction: no Haiku session; Sonnet 5.5 for mission-critical work). Start: tmux, `claude --model claude-sonnet-5-5`, then `/loop` with the KICKOFF prompt.
- **Autonomous run:** authorised by the operator on 2026-11-07 ("start osler_v6"); the GPU jobs follow GOAL.md's rules.
- **Osler_v5 Z8:** not run; closed by the operator on 2026-11-07. The record is on `loop/osler_v5` at commit `53ba2fe` (STATE board row, FINDINGS addendum). The copy of osler_v5 on this branch was replaced by the merge in §2c.
- **Base (ADVISORY §6, D59, Clef-Flash vs pplx-4B):** OPEN. Every run in this loop uses the pplx-decider-flash-v1.2 base (the same as
  osler_v5 F2), because §2 was written for it. §6 is not used until the operator decides.
- **Teacher:** `perplexity-ai/pplx-decider-v1.2-27b@cce0f3f80033e9d89d80831d5281fd72124aacce` (osler_v4 Y5; parity C072). The
  ADVISORY's earlier "v1.1" was a typo, corrected at Z0.

## 2b. Pre-registration clarifications (2026-11-07, before any run)

Made after an independent plan review (DeepSeek, 2026-11-07), before Z1. Each is a wording fix or a stated default; none changes a
number already measured.
- §2.6 knowledge bound: the text said "> 0" but its parenthesis said "must not reduce"; now a non-inferiority bound (> −0.01).
  Operator may override.
- §2.4 λ: the share formula and target are defined; the scaling is the closed form (0.4/0.6)·(1−s0)/s0. Calibration is a separate 200-step probe (Z2), so the D57 band applies to the arm's own first 200 steps. A miss blocks G1; λ is not re-tuned.
- §2.5/2.6 G0 is the reference, not a candidate.
- §2.6 seed rule: seed 0 is the candidate; seed 1 is a stability check.
- §2.7 checkpoint: "by S" now only among positional-health passes; the final is BLOCKED if none pass.
- §2.2 expected total about 380,000 (112,128 scored + about 268,000 in 12 GPU-hours).
- §2.8 Gate O6 is Gate O4's four checks, unchanged.
- Arm statistic: the same definition as the probe, pre-clip, mean of per-step ‖∇KL‖/(‖∇CE‖+‖∇KL‖) over the arm's first 200 steps. D57 says "applied gradient"; this run reads it as pre-clip, recorded here.
- Edge cases: s0 → 0 or 1 makes the closed form undefined; that is BLOCKED, not clamped.
- Z0 CLAIMS and SELF_AUDIT rows are deferred to the first Z1 batch (R1/R6 unmet for Z0 until then).
- Not changed: the G0′ control (sampler without KL) was not added; the package is the treatment. Operator may request it.

## 2d. Operator note (2026-11-10, for DeepSeek)

- **Checkpoint rule changed (ADVISORY §2.7 Amendment A1).** Z6 selects the final checkpoint at step **63,304** only. It is
  eligible only if it passes positional health in all three orders. The best-S rule is retired for Z6. Intermediate evaluations
  (6,330; 12,661; 18,991; 25,322; and the rest) are reported as a dev trajectory and are not candidates.
- **What you do:**
  1. Update `scripts/osler_v6/z6_select.py` so that it selects step 63,304 (not `select_by_s`), with the same positional-health
     check. If step 63,304 fails, it writes `selection.json` with outcome BLOCKED and exits non-zero. Add unit tests for both
     outcomes. Keep the calibration on dev fast dev v4 only.
  2. Report every evaluation in the Z6 table, including the ones that are not eligible. Do not drop any.
  3. Z7 scores `checkpoints/chosen_step_063304` if selected. If BLOCKED, Z7 reports the final as BLOCKED and does not substitute
     another checkpoint.
  4. Run ruff and pytest before any code commit. Commit this change on `loop/osler_v6` with a message stating A1.
- **What does not change:** the G1 and G0 definitions, the choice rule (§2.6), Gate O6's four checks and their limits, and the
  dev-only nature of Z5 and Z6.
- Operator: the decision is recorded here. No other pre-registered rule has been changed.

## 2c. osler_v5 merge (operator-approved 2026-11-07)

- Operator instruction: "merge required stuff" (2026-11-07), approving the merge of `loop/osler_v5` into this branch.
- Merge commit `bdbf32a` (parents `d731ab5` and `53ba2fe`). Changed paths: `loops/osler_v5/*`, `scripts/osler_v5/*`, `docs/paper/EVIDENCE.md`. Nothing under `loops/osler_v6/` changed.
- Code gate: `uv run --frozen ruff check .` passed. `uv run --frozen pytest -q` on the merge: one failure, `tests/test_padding_trained.py::test_trained_checkpoint_logits_are_padding_invariant` (strict XPASS, the student_v1 V2 BLOCKED item). The test file is identical to `dev` and the merge does not touch `tests/`, so the failure predates the merge. Baseline gate (the documented osler_v5 command, C001/C076: the same two deselects): ran to 100 %, exit 0. Pushed on this basis.
- Result: the osler_v5 copy on this branch reads `Loop status: STOPPED`; its Z8 row is `BLOCKED — NOT RUN`.
- Inherited cosmetic line: `loops/osler_v5/STATE.md` line 13 still says Z8 "is NOT started and waits for the operator's decision". The Z8 row is authoritative. Fix it on `loop/osler_v5` and re-merge, not here.

## 3. Facts found at Z0 (read-only)

- The teacher snapshot is present in `HF_HOME` at the pinned revision.
- osler_v4 Y5 has scored 112,128 anchor items (`outputs/osler_v4/Y5/anchor_probs.jsonl`); the pool is capped at 400,000
  (`scripts/osler_v4/y5_soft_labels.py`, POOL_CAP). The 112,128 are reusable by id and hash; they are not re-scored.
- The GPU was idle at Z0 (0 MiB).

## 4. Checklist — Z0 (done)

- [x] Read AGENTS.md, PROGRAM D48/D57/D58, ADVISORY, osler_v5 FINDINGS addendum.
- [x] Teacher revision checked against the cache and osler_v4 Y5 (cce0f3f8…).
- [x] ADVISORY teacher corrected (v1.1 → v1.2); status changed from DRAFT to GO recorded.
- [x] AGENTS.md Current loop section updated to osler_v6 (Started field set).
- [x] GOAL.md, STATE.md, KICKOFF.md committed.

## 4b. Checklist — Z1 (done)

- [x] Script `scripts/osler_v6/z1_anchor_scoring.py`: the osler_v4 Y5 path, Z1 output dir, reuse by id + text hash; `uv run --frozen ruff check .` and the pytest suite pass before its commit (C009).
- [x] Pool rebuilt with the current bench exclusions (all 73 `data/bench/**/*.jsonl`, including the osler_v4 y9 panel/knowledge/MIMIC sets created after Y5); its sha256 is osler_v4's `a9fab387…`, so no old scored row is excluded (C004).
- [x] Reuse: all 112,128 osler_v4 records verified against their pool row (item id, template, qtype, option count, probability sum); 0 dropped, positions preserved; written to `outputs/osler_v6/Z1/anchor_probs.jsonl` (C006).
- [x] Parity re-run: 40 of 40 argmax agreement on both batched paths; chosen path `reference_batched`; numbers bit-identical to osler_v4's (C007).
- [x] Anchor scoring detached under `--max-hours 12`, polled every 20+ minutes: stopped at the budget with 385,536 scored (112,128 reused + 273,408 new), 14,464 pool rows cut (rate 6.32 items/s < 10/s; the pre-registered cut, C012).
- [x] Join `data/train/osler_v6/z1_anchor_soft.jsonl`: 385,536 rows, sha `edd4c207…`, overlap with every bench row 0 (C013).
- [x] Summary numbers, `SELF_AUDIT.md` (PASS, `scripts/osler_v6/z1_audit.py`), CLAIMS rows, STATE exact, commit and push.

## 4c. Checklist — Z2 (done)

- [x] D57 dosing sampler and its tests (committed before Z1 closed): 3 labelled anchors in every formed batch, per-epoch permutation without replacement, seeded, statistics logged.
- [x] Pre-clip share probe in the trainer (`distill_probe`, `distill_probe_steps`), leaving the update unchanged.
- [x] Build `data/train/osler_v6/mix_f2_distill.jsonl`: 1,456,438 rows, 368,221 labelled by text (25.28 %; anchor 855,532), composition 78d7e547…, sha `91ff5362…`; the augmentation matches osler_v5's (37,844 + 37,843).
- [x] Fill order fixed before any arm result: length-bucketed inside blocks (D57's strata, quota and labelled draw unchanged); probe #1 (random fill) preserved and superseded (C022, C023).
- [x] 200-step probe at λ = 1 on the final sampler: 3 labelled anchors on every step, all finite; s0 = 0.170491; λ = 3.243598 frozen (no re-tune) (C019).
- [x] `SELF_AUDIT.md` (fresh-process re-derivation, `z2_audit.py`, PASS), CLAIMS rows, STATE exact, commit and push (C020).

## 4d. Checklist — Z3 (done)

- [x] Disk headroom 2 GB write before the job (1.9 s, 1.1 GB/s).
- [x] G1 seed 0 launched at the frozen λ with the D57 quota (min 3 labelled per batch), 13,992 steps, evaluations at 6,996 and 13,992; ran detached (PIDs 723838/723842, log `outputs/osler_v6/Z3/G1s0/logs/run.log`).
- [x] First-200-step band: mean pre-clip share 0.355474 in [0.30, 0.50] (`z3_audit.py`), 3 labelled on every step (C024).
- [x] Evaluations: positional health all three orders at 13,992 PASS; knowledge dev 0.775000; MR 0.928969, MIMIC 0.754192, CG 0.794511 for the choice rule (C027).
- [x] `SELF_AUDIT.md` (PASS), CLAIMS rows, STATE exact, commit and push (C028).

## 4e. Checklist — Z4 (done)

- [x] Disk headroom 2 GB write before the job (2.0 s, 1.1 GB/s).
- [x] G1 seed 1 launched at the frozen λ on `mix_f2_distill_s1.jsonl` (augmentation seed 1), quota 3, 13,992 steps, evaluations at 6,996 and 13,992; detached (PIDs 777164/777168, log `outputs/osler_v6/Z4/G1s1/logs/run.log`).
- [x] First-200-step band: mean 0.365937 in [0.30, 0.50] (C029).
- [x] Seed-1 evaluation at 13,992: S 0.826311, knowledge 0.770750, positional PASS, reported beside seed 0 (C032); audit PASS (C033).

## 4f. Checklist — Z6 (in progress)

- [ ] Disk headroom 2 GB write before the job.
- [ ] Final launched on G1s0's recipe (seed-0 config/mix, frozen λ 3.243597720934029, quota 3), 63,304 steps, 10 evaluations; detached with PID/log; resumes from full checkpoints if interrupted.
- [ ] **A1**: the final checkpoint is step 63,304 only (no other), eligible only if it passes positional health in all three orders; `z6_select.py` calibrates it on dev fast dev v4 and writes `chosen_step_063304`, or writes BLOCKED and exits non-zero; the intermediate evaluations are reported as a dev trajectory (never candidates).
- [ ] `SELF_AUDIT.md`, CLAIMS, STATE exact, commit/push.

## 5. Log

- 2026-11-07T17:40Z — Z0 DONE. Z1 is the next task. It needs its own output directory, so the Y5 script is run from a copy that
  writes to `outputs/osler_v6/Z1/` (no append to osler_v4 outputs). That copy is a code change: ruff and the suite run before its
  commit, not during the GPU run.
- 2026-11-07T19:08Z — Z1 progress. The script is `scripts/osler_v6/z1_anchor_scoring.py`: it imports the osler_v4 Y5 script by path
  and redirects every path to `outputs/osler_v6/Z1/` and `data/train/osler_v6/`; the pool rule, request mapping, teacher call and
  record format are osler_v4's code. Pool rebuilt against all 73 bench files (13 are newer than Y5: the y9 panel/knowledge/MR
  files, both MIMIC sets, membership, fastdev_v4_mimic, pairs/mednli test): identical sha `a9fab387…`, 0 new exclusions, so the
  reclaim is exact — 112,128 records reused in place (0 dropped, 0 mismatches, 0 overlap), 287,872 pool rows left to score.
  Parity re-run: 40/40 on both batched paths and the comparison numbers are bit-identical to osler_v4's `parity.json`; path
  `reference_batched`. Anchor job launched detached (uv 301442, python 301463) with a 12 GPU-hour budget; first flush 512 scored at
  7.07 items/s, 4,095 input tokens/s. Ruff clean; suite 1107 passed, 4 skipped, 2 deselected, exit 0 (168 s).

- 2026-11-07T19:32Z — Z1 anchor job healthy: 8,704 new items at 6.47 items/s (4,115 input tokens/s), GPU 100 %. While it
  runs (CPU work only), the Z2/Z3 code is written and committed: `meddecide.train.dosing` (D57 quota sampler: 3 labelled
  anchors per formed batch, per-epoch permutation without replacement, soft cap for a long first fill item, realised
  statistics), the trainer's pre-clip share probe (`distill_probe`; CE and λ·KL gradient norms recorded per step without
  changing the update), `scripts/osler_v6/build_mix.py` (osler_v5's corrected F2 mix + Z1 teacher labels attached, checked
  against the corrected composition hash), `scripts/osler_v6/z2_probe.py` (200 steps at λ = 1, closed form for λ) and
  `scripts/osler_v6/run_arm.py` (the osler_v5 driver reading the prepared mix, with `--distill-weight` and
  `--min-labelled`). Ruff clean; the suite (GPU hidden) exits 0.

- 2026-11-07T19:36Z — Z1 job still healthy (11,264 new at 6.34 items/s; records continuous, last pool index 123,391; no
  errors). Z5's choice rule is committed ahead of the arms: `meddecide.eval.osler_v6_choice` (the §2.6 checks with their
  strict limits, G0 never a candidate, seed 0 the only candidate) and `scripts/osler_v6/z5_choose.py` (osler_v5's paired
  dev statistics against the F2 seed-0 evaluation). The script fails early until the arm exists, as designed.

- 2026-11-07T19:34Z — Z1's join/summary chain launched detached (`scripts/osler_v6/z1_chain.sh`, chain PID 503528): it
  waits for the anchor job's python PID and then runs `join` (soft labels + the bench-overlap check) and `summary`, logging
  exit codes. So the Z1 data artifacts complete even if no session is awake when the 12-hour job ends. The chain does not
  start Z2 (the mix build waits for Z1 to be audited and marked DONE).

- 2026-11-07T19:56Z — Z1 job still healthy (20,480 new at 6.37 items/s). Z7's gate code is written ahead of the final and
  **validated against Gate O4**: `scripts/osler_v6/z7_gate.py` imports osler_v4's `y9_gate.py` unmodified and points only the
  final-model key at `outputs/osler_v6/Z7/baselines`; run on osler_v4's own predictions it reproduces
  `outputs/osler_v4/Y9/gate.json` exactly — pplx_flash FAIL with i PASS (+0.13871 [0.124266, 0.152998]), ii PASS (+0.165424
  [0.150232, 0.179508]), iii FAIL (−0.042145 [−0.05842, −0.026935]), iv PASS (−0.006691 [−0.01721, 0.004756]); meddecider_4b
  NOT MEASURED; all figures identical. So Gate O6 will be the same code path as Gate O4, with the same limits.

- 2026-11-07T20:16Z — Z1 job healthy at 27,648 new (6.36 items/s). The adapted arm driver's full data path was smoke-tested
  on a synthetic 600-row mix (200 labelled) in a temporary directory: it reads/verifies the mix manifest, converts every item,
  loads the dev sets (fast dev 351, CG 416, knowledge 200, positional 50 in `--smoke`) and writes `dry_run.json` with the
  13,992-step grid [6,996, 13,992]. No arm has run; the temporary data and output were removed. In-flight diagnostic on the
  new Z1 records (not a choice): 0 error statuses, probability sums 1.0, teacher argmax vs pool gold choice 0.8403 / noul
  0.802 / score 0.6557 — consistent with the teacher's osler_v4 dev diagnostic (0.8516 union).

- 2026-11-07T20:41Z — Z1 job healthy at 35,328 new (6.25 items/s; 147,456 records total). The Z6 checkpoint-selection rule
  is committed: `meddecide.eval.osler_v6_select` (highest S among the positional-health passes; no fallback, tie rejected)
  with 6 tests, and `scripts/osler_v6/z6_select.py` (selects, fits the chosen checkpoint's temperatures on dev fast dev v4
  only, saves `chosen_step_XXXXXX`). Suite on the full Z2-Z7 CPU code: 1136 passed, 4 skipped, 2 deselected, exit 0 (181 s);
  ruff clean (C010).

- 2026-11-07T22:16Z — Z1 job healthy at 72,704 new (6.28 items/s); the join/summary chain still waits on the python PID.
  The Z7 scoring path is committed too: `scripts/osler_v6/z7_score.py` verifies each gate set's SHA-256 against osler_v4's
  `sets_manifest.json`, refuses an uncalibrated checkpoint, and writes `outputs/osler_v6/Z7/baselines/final_<set>/`. All
  Z2-Z7 CPU code is now written; nothing else starts until Z1's audit.

- 2026-11-07T23:55Z — Z1 job healthy at 110,080 new (6.29 items/s, 4.86 h of the 12-hour budget). Z2's future audit is
  committed too: `scripts/osler_v6/z2_audit.py` re-derives s0, λ and the per-step D57 quota from `steps.jsonl` in a fresh
  process. Z5's G0 inputs were checked in place (17,532 fast-dev, 4,000 knowledge, 8,307 CG rows at step 13,992; S 0.8259,
  knowledge 0.7595, positional pass), so the choice rule's control is intact.

- 2026-11-08T00:02Z — Z1 job healthy at 111,104 new (6.29 items/s, 4.91 h). The D57 arm band check is now fully wired: the
  trainer's `distill_probe_steps` limits the pre-clip gradient-share measurement to the arm's first 200 steps (the probe run
  keeps 0 = all steps), `run_arm.py` turns it on with `log_every: 1` for G1, and `scripts/osler_v6/z3_audit.py` re-derives the
  band ([0.30, 0.50], mean of per-step shares at the frozen λ), the quota of 3 on every step, the step count, the evaluation
  grid and the loss health, and writes the Z3 SELF_AUDIT. Suite 1137 passed, 4 skipped, 2 deselected, exit 0; ruff clean (C011).

- 2026-11-08T07:13Z — **Z1 DONE.** The anchor job scored to the 12 GPU-hour budget and stopped: 385,536 of the 400,000 pool
  rows scored (96.38 %), 112,128 reused from osler_v4 + 273,408 new, 14,464 rows cut by the budget (rate 6.32 items/s, under
  the pre-registered 10 items/s threshold — the cut is the expected outcome, not a shortfall). The join wrote 385,536 soft
  labels (choice 269,282 / noul 97,974 / score 18,280), sha `edd4c207…`, benchmark overlap 0. The fresh-process audit
  (`scripts/osler_v6/z1_audit.py`) re-derived every number and returned PASS: pool sha equals osler_v4's, 112,128 reused with
  0 field/position mismatches, 0 argmax mismatches, probability sums off by 1.1e-16, parity identical to osler_v4, 0 problems.
  Z2 starts next: build the distilled mix, then the 200-step λ probe.

- 2026-11-08T07:22Z — Z2's mix built: `data/train/osler_v6/mix_f2_distill.jsonl` (8.5 GB, private), 1,456,438 rows, of which
  368,221 carry a Z1 teacher distribution (25.28 %); 855,532 anchor rows, 600,906 domain rows; composition `78d7e547…` (the
  D56a-corrected F2 mix), augmentation counts identical to osler_v5's. The 200-step λ probe was launched detached
  (uv PID 667812, python PID 667815, log `outputs/osler_v6/Z2/logs/probe.log`) on the now-idle GPU.

- 2026-11-08T08:00Z — **Z2 DONE.** The probe at λ = 1 ran 200 steps on the distilled mix with exactly 3 labelled anchors on
  every step: s0 = 0.162436, so λ = (0.4/0.6)·(1−s0)/s0 = **3.437525**, frozen (the audit re-derives both to 1e-12). The
  mix: 1,456,438 rows, 368,221 labelled (25.28 %). Measured efficiency fact: the dosed batches mix 3 short anchors with 5
  unlabelled items of mixed length, so mean padded/real tokens is 3.54 (mean 24,198 padded against 6,879 real per step) and
  the probe's step time is 7.76 s with its extra backward — F2's length-bucketed batches padded at 1.16x. The arms run with
  this same pre-registered sampler (no change after a run); the expected consequence is an arm much longer than F2's ~4.5 h,
  and the final longer than the ADVISORY's ~24 h planning estimate. This is recorded as a measured planning fact, not a
  design change; the ADVISORY §4 estimates are explicitly not measured.

- 2026-11-08T08:01Z — Z3 launched. The first launcher attempt passed an empty λ to the driver (a shell one-liner exited before
  printing it); the driver refused the empty float immediately and no arm started. The launcher is fixed (`assert` + non-empty
  check) and the arm relaunched with λ = 3.4375248931943583, uv PID 673265 / python PID 673269. The arm's first 200 steps are
  the D57 band check; the step log is per-step, so the band can be read at step 200 and the arm stopped early if it misses.

- 2026-11-08T08:27Z — **Z2 corrected and re-closed.** The first arm launch exposed a padding defect in the sampler as first
  implemented: with the fill drawn uniformly at random, three short anchors beside one long record padded whole batches to
  the record (probe #1: mean padded/real 3.54; the arm's 42 logged steps: 3.70 and 7.70 s/step with the probe's extra
  backward). The extrapolated arm (~17 h) and final (~75 h) were far outside the ADVISORY's planning estimates. The arm was
  stopped at step 42 — **before any evaluation, checkpoint or result**; its logs are preserved in
  `outputs/osler_v6/Z3/G1s0_aborted_padding/` (C022). D57 does not pin the fill order (only the quota of 3 labelled
  anchors, the without-replacement labelled draw, the seeding and the logged statistics), so the unlabelled fill stream is
  now length-bucketed inside blocks of 32 batches and the batches are shuffled inside the block — the same anti-padding,
  anti-curriculum mechanism as `iter_batches`; the strata, quota, draw and coverage are unchanged (C023). The calibration
  probe was re-run on the final sampler (probe #1 preserved in `Z2/probe_random_fill/`, its λ not used): s0 = 0.170491 →
  λ = **3.243598**, frozen; 200 steps, 3 labelled each, audit PASS, mean step 3.53 s and padding 2.39x (C019-C021). No
  pre-registered number (D57, D58, the frozen sets, the choice rule, Gate O6) was changed; the operator may review the
  clarification in §7.

- 2026-11-08T08:28Z — Z3 relaunched on the final sampler: λ = 3.243597720934029, uv PID 723838 / python PID 723842. The
  earlier arm (random fill, λ = 3.4375) is stopped and preserved under `outputs/osler_v6/Z3/G1s0_aborted_padding/`; it
  produced no evaluation, checkpoint or result.

- 2026-11-08T08:42Z — Z3 healthy at step 206 and the pre-registered D57 band check passes: the arm's own first 200 steps
  give a mean pre-clip share of **0.355474** (band [0.30, 0.50]), with 3 labelled anchors on every step (C024). The frozen
  λ is vindicated by the arm's own measurement; no re-tune. The probe's extra backward stops after step 200, so the rate
  should improve from 3.48 s/step to about 1.9 s/step; the arm should reach 13,992 steps in ~7.5 h (about 16:00Z), then
  the two evaluations and the Z3 audit.

- 2026-11-08T09:23Z — Z3 at step 1,064, post-probe step time ~2.8 s (1,064 steps in 56 min including load), GPU 79 GB,
  loss finite, 3 labelled on every step, no evaluations yet (first at 6,996). Revised finish ~19:20Z; then the two
  evaluations, the positional-health check and the Z3 audit.

- 2026-11-08T10:14Z — Z3 at step 2,098 (2.77 s/step); first evaluation still ahead at 6,996. While it runs, the **seed-1
  mix for Z4** was built ahead of time (CPU only): `data/train/osler_v6/mix_f2_distill_s1.jsonl`, 1,456,438 rows / 368,221
  labelled, augmentation seed 1, sha `fc8e6d04…` (C025). It is not used unless Z3's positional health passes and Z4 starts.

- 2026-11-08T14:42Z — Z3's first evaluation (step 6,996) completed and written: S 0.8163 (MR 0.9330, MIMIC 0.7342, CG v2
  0.7818), knowledge dev 0.7680, positional health in all three orders PASS; the same-step G0 record is S 0.8064 (0.9233 /
  0.7242 / 0.7718), knowledge 0.7500, so G1 leads on every dev component at the halfway point (C026). This is an interim dev
  measurement, not the choice: the pre-registered rule reads the 13,992 evaluation with paired bootstrap bounds. The eval
  took 177 s and its checkpoint is saved in `checkpoints/step_006996`; the arm continues at ~2.9 s/step toward 13,992
  (~20:10Z), then the second evaluation and the Z3 audit.

- 2026-11-08T21:34Z — **Z3 DONE.** G1 seed 0 trained to 13,992 steps at the frozen λ (12 h 58 min wall, ~2.9 s/step) and its
  fresh-process audit passed: the arm's own first-200-step mean share is 0.355474 (band [0.30, 0.50]), 3 labelled anchors
  on every step, 13,992 steps, evaluations [6,996, 13,992], 0 problems (C028). Final dev evaluation (13,992): S 0.825891,
  MR 0.928969, MIMIC 0.754192, CG v2 micro 0.794511, knowledge dev 0.775000, positional health in all three orders PASS
  (C027). Against G0 at the same step: knowledge +0.0155, CG +0.0101, MR -0.0086, MIMIC -0.0016, S -0.00003 — the paired
  bounds are Z5's job, but no component is near its pre-registered margin. Because the seed-0 result is a positional pass,
  ADVISORY §2.5 requires the seed-1 stability arm: **Z4 starts now** on the seed-1 mix (C025), same frozen λ and quota.

- 2026-11-08T21:50Z — Z4 (seed 1) is training: at step 224, its own first-200-step D57 band passes (mean share 0.365937;
  3 labelled on every step, C029), and the post-probe rate is ~2.1 s/step, so it should finish about 06:30Z (earlier than
  the 10:30Z planning estimate). The seed-1 result will be reported beside seed 0 and never replaces it.

- 2026-11-08T21:52Z — **Z5 preliminary: G1 seed 0 qualifies.** Running the pre-registered rule on the step-13,992 dev
  predictions gives no failed check: the knowledge paired lower bound is +0.006244 (limit −0.01), MR −0.020692 and MIMIC
  −0.010395 (limit −0.03), CG v2 micro 0.794511 against G0's 0.784399, positional health PASS; the highest-knowledge-dev
  qualifier is G1s0 (C030). The seed-1 stability arm is still training, so Z5 will be finalised (with the stability row
  reported beside seed 0) once Z4 completes. A bug in `z5_choose.py` (it looked for the seed-1 arm under Z3 instead of Z4)
  was found and fixed before this run.

- 2026-11-09T00:29Z — Z4 at step 3,580 (2.90 s/step), no evaluation yet (first at 6,996, ~03:00Z); expected finish ~10:00Z.
  The exact Z5/Z6/Z7 commands are recorded in §8 so a later session can continue without re-deriving them.

- 2026-11-09T03:42Z — Z4's first evaluation (step 6,996) is in: G1s1 S 0.8114 (MR 0.9216, MIMIC 0.7365, CG 0.7761),
  knowledge 0.7640, positional PASS. Seed 0 at the same step: S 0.8163, knowledge 0.7680; G0: S 0.8064, knowledge 0.7500.
  The seed difference is small (about half a point on S and knowledge, 1.1 points on MR) and does not change the seed-0
  candidate; the stability row is reported beside it, never instead of it (C031). Z4 continues to 13,992 (~10:40Z).

- 2026-11-09T10:52Z — **Z4 DONE and Z5 DONE.** The seed-1 stability arm finished 13,992 steps (band 0.365937 PASS, audit
  PASS) with final dev S 0.826311 / MR 0.930845 / MIMIC 0.757068 / CG 0.791020 / knowledge 0.770750 / positional PASS — within
  ~0.5 points of seed 0 on every component (C032, C033). With both arms in, the pre-registered choice rule was run for real:
  **G1s0 is the chosen recipe** (knowledge paired lower bound +0.006244, MR −0.020692, MIMIC −0.010395, CG 0.794511 ≥ 0.784399,
  positional PASS); G0 is not a candidate and the seed-1 arm is reported beside it (knowledge +0.01125 [+0.00274, +0.02025],
  MR −0.00670, MIMIC +0.00125) and never replaces it (C034). **Z6 starts now**: the 63,304-step final on G1s0's recipe, ten
  evaluations, checkpoint by S among positional passes.

- 2026-11-09T10:53Z — Z6 launched: the 63,304-step final at the frozen λ with the D57 quota, ten evaluations, on the
  seed-0 mix (uv PID 830482 / python PID 830492). At the arms' measured ~2.9 s/step the steps alone are ~51 h; with the ten
  evaluations (~20 min each) and post-processing the expected completion is about 2026-11-11T18:00Z. Full checkpoints every
  2,500 steps allow a resume if the pod interrupts.

- 2026-11-09T13:00Z — Z6 at step 2,572 (3.09 s/step), no evaluation yet (first at 6,330, ~16:15Z). Diagnostic: the final's
  own first-200-step mean share is 0.351422 with 3 labelled anchors on every step — the same dose the arms ran at (C035);
  the band rule is pre-registered for the arms, so this is a health check, not a gate.

- 2026-11-09T16:32Z — Z6's first evaluation (step 6,330) is in: S 0.796776, MR 0.899795, MIMIC 0.720461, CG 0.770073,
  knowledge 0.761500, positional PASS (C036). It is an earlier step than the arm's 6,996 evaluation, so no cross-arm claim
  is read. A reproducibility check shows the final's sampler stream is identical to G1s0's (same batch composition and
  n_teacher on all 6,467 common steps); losses differ from step 2 because bf16 GPU training is not bitwise deterministic
  (C037) — the final is a statistical replica of the chosen recipe, which is what it is designed to be.

- 2026-11-09T22:24Z — Z6's second evaluation (step 12,661): S 0.791281 (MR 0.894419, MIMIC 0.704173, CG 0.775250,
  knowledge 0.758500), positional PASS. S is below the first evaluation's 0.796776; this is not a defect: the driver lays the
  cosine schedule over the run's own length, so the final's schedule spans 63,304 steps while the arms' spanned 13,992, and
  the final's early evaluations sit at a much higher LR than the arms' same-step figures. Selection by S happens among the
  final's own ten evaluations, and the last ones (low LR) are the likely candidates.

- 2026-11-10T04:05Z — Z6's third evaluation (step 18,991) shows the positional-health instability the rule anticipates: it
  has the highest S so far (0.818523; MR 0.947022, MIMIC 0.740762) but fails the original-order check narrowly (two-letter
  0.388778 against 0.410180 = 0.5 × single-letter 0.820359), while both permuted orders pass. Per ADVISORY §2.7 it is **not
  eligible** for the final checkpoint; selection by S is among the passes only, and if no evaluation passes the final is
  BLOCKED and reported (C039). The remaining seven evaluations will show whether the model settles into a passing region.

- 2026-11-10T11:46Z — **Z6 was interrupted and resumed.** At 06:37Z, after step 22,317, the final died with a CUDA OOM
  raised inside a Triton autotune benchmark (`fla` l2norm backward) on the step following a 54,272-padded-token batch; the
  GPU was idle from then until the resume. Full checkpoints at 17,500 and 20,000 exist, so the run resumed from **step
  20,000** at 11:37Z with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (allocator setting only). The earlier three
  evaluations were preserved (`evaluations.json` backed up, and the driver fixed to merge pre-resume evaluations — commit
  53691b4). The pre-crash log is kept as `logs/train_until_crash.log`. The deterministic stream will re-reach the crash
  point in ~2.3 h; with the fresh allocator and expandable segments it should pass, and if it does not, the next mitigation
  is to pre-warm the autotuner for the long-sequence shape in a low-memory process before resuming.

- 2026-11-10T13:51Z — the resumed Z6 **passed the former crash point**: the log reached step 22,406 (past 22,317) with no OOM,
  100 % GPU and ~58 GB resident (the pre-crash run sat at 97 GB). The deterministic stream replayed through the 54k-token
  batch safely; `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is the change that made the difference. The run continues
  toward 63,304; the fourth evaluation (25,322) is due ~16:15Z.

- 2026-11-10T16:52Z — Z6's fourth evaluation (step 25,322): S 0.822039 (MR 0.941014, MIMIC 0.750818, CG 0.774287),
  knowledge 0.751750, positional PASS (original 0.5351 against 0.8244). The positional check recovered after the step-18,991
  miss. Under A1 none of these is a candidate: the evaluation at step 63,304 is the only one, and it is positional-gated.

- 2026-11-10T22:38Z — Z6's fifth evaluation (step 31,652): S 0.827817 (MR 0.956085, MIMIC 0.747782, CG 0.779583),
  knowledge 0.754750, positional PASS. It is the highest S in the trajectory so far and above G1s0's 13,992-step S 0.825891,
  but under A1 only step 63,304 is a candidate. The run is at step 31,716 (~50% of 63,304).

- 2026-11-11T04:29Z — Z6's sixth evaluation (step 37,982): S 0.837467 (MR 0.963514, MIMIC 0.769906, CG 0.778982),
  knowledge 0.764000, but positional health **fails on the original order** again (0.388778 against 0.431138; both permutations
  pass). The trajectory's S is rising while the positional check oscillates (fails at 18,991 and 37,982, passes at the other
  four). Under A1 this is trajectory evidence only: the final at 63,304 must itself pass all three orders to be selected.

- 2026-11-11T10:26Z — Z6's seventh evaluation (step 44,313): S 0.840188 (MR 0.966014, MIMIC 0.780263, CG 0.774287),
  knowledge 0.765250, but the original-order positional check fails a third time (two-letter 0.364730 against 0.395210; both
  permutations pass). Investigation (C046): on the 499 two-letter-gold DDX items the model predicts 407 two-letter keys but
  concentrates on BJ (128), BM (42), BC (38), BG (33), BD (31) — a key/position bias, not a constant answer; the permuted
  orders break it and pass. This is the D52b check doing its job. Under A1 the final checkpoint at 63,304 must itself pass all
  three orders; the intermediate misses are trajectory evidence only. The trend is a real risk that the final is BLOCKED, and
  if so Z7 will report the final as BLOCKED without substitution.

- 2026-11-11T16:16Z — Z6's eighth evaluation (step 50,643): S 0.841495 (MR 0.959522, MIMIC 0.781889, CG 0.783075),
  knowledge 0.763750, positional health **PASS** in all three orders. The check has alternated across the trajectory (fails at
  18,991, 37,982, 44,313; passes at the other five), so the step-63,304 outcome remains genuinely uncertain. Two evaluations
  remain (56,973 ~22:10Z; 63,304 ~04:00Z); under A1 the last one alone decides whether the final can be scored for Gate O6.

- 2026-11-11T22:06Z — Z6's ninth evaluation (step 56,974): S 0.845382 (MR 0.963445, MIMIC 0.787942, CG 0.784760),
  knowledge 0.767250, positional **PASS** in all three orders. Two evaluations pass in a row now; the trajectory's S is at its
  highest and comfortably above the arms' final 0.8259. The A1 decision rests on the last evaluation, at step 63,304
  (~04:00Z on Nov 12): a positional pass selects `chosen_step_063304`; a failure BLOCKS the final with no substitution.

## 6. Deviations from the plan

- **Environment pin (torchvision).** The osler_v4 Y5 command's `--with "torchvision==0.29.*"` now resolves 0.29.2, which pulls
  torch 2.15.0 + triton 3.9.0; the authors' fast path then fails at import (fla 0.5.2 raises `ValueError` inside `triton.autotune`
  on `chunk_delta_h.py`). Z1 uses `--with "torchvision==0.29.0"` (torch 2.14.0, triton 3.8.0, fla 0.5.2), under which the 40-item
  parity reproduces osler_v4's numbers exactly (C007, C008). The anchor job runs under the same overlay. No design number changes.
- **Z1 script is an adapter, not a copy.** `scripts/osler_v6/z1_anchor_scoring.py` loads `scripts/osler_v4/y5_soft_labels.py` by
  path and overrides its path constants; the scoring code is byte-for-byte osler_v4's. The Z1-specific code is the pool rebuild and
  the verified reuse seed only.
- **Bench set grew.** 73 bench files at Z1 vs 55 in the Y5 manifest; 13 of them were created after Y5. The rebuild shows they add
  no exclusion to the eligible anchor pool (C004), so the reuse is exact rather than a re-mapping.

## 8. Next commands (after Z4; amended by A1)

- Z5: `uv run --frozen python scripts/osler_v6/z5_choose.py` (reads both arms' step-13,992 predictions, writes
  `outputs/osler_v6/Z5/choice.json`; the seed-1 arm is reported beside seed 0 and never replaces it).
- Z6 (only if Z5 chose a recipe): `bash scripts/osler_v6/launch_arm.sh G1final configs/osler_v5/arm_F2.yaml
  data/train/osler_v6/mix_f2_distill.jsonl outputs/osler_v6/Z2/mix_manifest.json outputs/osler_v6/Z6/final 63304 10`,
  then `uv run --frozen python scripts/osler_v6/z6_select.py --arm-dir outputs/osler_v6/Z6/final`.
  **A1 (2026-11-10):** the only candidate is the step-63,304 evaluation; it is eligible only on a positional pass in all
  three orders. On a pass the script writes `checkpoints/chosen_step_063304`; on a failure it writes BLOCKED and exits
  non-zero. The intermediate evaluations are reported in the table, never chosen.
- Z7: `uv run --frozen python scripts/osler_v6/z7_score.py --checkpoint
  outputs/osler_v6/Z6/final/checkpoints/chosen_step_063304`, then `uv run --frozen python scripts/osler_v6/z7_gate.py`.
  If Z6 is BLOCKED, Z7 reports the final as BLOCKED against Gate O6 and does not substitute another checkpoint.

## 7. Questions for the operator

- None blocking. For review: the fill-order clarification (2026-11-08T08:27Z, above and §6) — D57's quota, labelled draw
  and seeding are unchanged, but the unlabelled fill order was made length-bucketed before any arm result existed, and the
  λ probe was re-run on the final sampler. The random-fill probe and the aborted arm's logs are kept as evidence.

