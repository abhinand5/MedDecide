# STATE — osler_v6

Loop status: `RUNNING`
Branch: `loop/osler_v6`. Last updated (UTC): `2026-11-08T07:14Z` (Z0, Z1 DONE; Z2 IN_PROGRESS).
Task in flight: Z2 — the 200-step λ probe (GPU) on the built mix; detached, uv PID `667812`, python PID `667815`, log
`outputs/osler_v6/Z2/logs/probe.log`; started 2026-11-08T07:22Z.

## 1. Task board

| id | task | depends | status |
|---|---|---|---|
| Z0 | Setup: GO recorded, teacher revision checked, Z8 closure noted, AGENTS current loop | — | DONE 2026-11-07T17:40Z |
| Z1 | Anchor scoring to 400,000 items with the osler_v4 Y5 path (resume from 112,128; parity 40 of 40 re-run); outputs to `outputs/osler_v6/Z1/` | Z0 | DONE 2026-11-08T07:13Z — 385,536 scored (96.38 %), 14,464 cut by the 12-hour budget, audit PASS |
| Z2 | Dosing sampler (D57, 8 per batch, ≥3 labelled anchors, seeded) and the λ calibration probe (200 steps at λ = 1; λ then frozen; ADVISORY §2.4) | Z1 | IN_PROGRESS 2026-11-08T07:14Z |
| Z3 | Arm G1 (F2 + distillation, seed 0) at the frozen λ, 13,992 steps; check the 30–50 % band over its first 200 steps; G0 is the existing F2 seed-0 result C062 (checked, not re-scored) | Z2 | PENDING |
| Z4 | Arm G1 seed 1, only if G1 seed 0 passes positional health | Z3 | PENDING |
| Z5 | Choice rule (ADVISORY §2.6), dev only. None qualifies → outcome (4): report and stop | Z3, Z4 | PENDING |
| Z6 | Final at 63,304 steps, positional health at each evaluation, checkpoint by S (only if Z5 chose a recipe) | Z5 | PENDING |
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

## 4c. Checklist — Z2 (in progress)

- [x] D57 dosing sampler and its tests (committed before Z1 closed): 3 labelled anchors in every formed batch, per-epoch permutation without replacement, seeded, statistics logged.
- [x] Pre-clip share probe in the trainer (`distill_probe`, `distill_probe_steps`), leaving the update unchanged.
- [x] Build `data/train/osler_v6/mix_f2_distill.jsonl`: 1,456,438 rows, 368,221 labelled by text (25.28 %; anchor 855,532), composition 78d7e547…, sha `91ff5362…`; the augmentation matches osler_v5's (37,844 + 37,843).
- [ ] 200-step probe at λ = 1 on the mix, budget: every step carries ≥3 labelled anchors, finite; s0 = mean per-step share; λ = (0.4/0.6)·(1-s0)/s0 frozen (no re-tune).
- [ ] `SELF_AUDIT.md` (fresh-process re-derivation, `z2_audit.py`), CLAIMS rows, STATE exact, commit and push.

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

## 7. Questions for the operator

- None at Z1 start.

