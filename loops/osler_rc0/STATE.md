# STATE — osler_rc0

Loop status: IN_PROGRESS — R0, R1 DONE; R2 scoring (started 2026-11-24T21:33Z)

## 1. Task board

| id | task | depends | approval | status |
|---|---|---|---|---|
| R0 | Preflight, SHA-256 hashes, pplx snapshot identity check, `medconceptsqa` overlap check, on-disk prediction inventory, baseline commit (CPU) | — | **Loop GO** | DONE 2026-11-24T07:05Z |
| R1 | Train `Osler-4B-rc0` (63,304 steps, Option A token-capped plan, `4b_full_run.py`, ~12–14 h GPU) | R0 | **GPU approval** | DONE 2026-11-24T21:32Z (S 0.839238, positional PASS) |
| R2 | Dev-only calibration for `Osler-9B-rc0` + score `Osler-9B-rc0` & `d3-flash` base with full `probs` on all missing/prob-less sets (~1.5 h GPU) | R0 | **GPU approval** | IN_PROGRESS 2026-11-24T21:33Z |
| R3 | Score frozen `Osler-4B-rc0` on all 9 frozen sets with full `probs` (~1 h GPU) | R1 | **GPU approval** | PENDING |
| R4 | Score fresh tier-2 `confirm/general.jsonl` on competitors (`pplx-27B`, `MedDecider-9B`, `MedDecider-4B`, `Clef-Flash`, optional `JEV-9B` 30-min cap; ~1.5 h GPU) | R0 | **GPU approval** | PENDING |
| R5 | Standardized latency, throughput & VRAM benchmark (500-item stratified sample, batch 1 & 16; ~20 min GPU) | R1, R2 | **GPU approval** | PENDING |
| R6 | Mandatory local MIMIC PHI & memorisation audit (D28): visited-member + Y10 membership AUC for 9B & 4B; 4B verbatim probe; 9B non-generative check (~1 h GPU) | R1, R2 | **GPU approval** | PENDING |
| R7 | Unified 12-model leaderboard (`rc0_leaderboard.py` + unit tests), `raw↔recomp` ID join, Brier/ECE, paired CIs, per-category tables, formal Pareto frontier (CPU) | R2, R3, R4, R5 | Loop GO | PENDING |
| R8 | Release staging verification (offline reload parity <1e-4, 4B base generation byte-identity, 9B no-`lm_head` check) + honest model cards (NO upload) | R6, R7 | GPU approval (short) | PENDING |
| R9 | Final lint/test check, 5-claim fresh-process spot-check, `FINDINGS.md`, `Loop status: STOPPED`, commit (CPU) | R8 | Loop GO | PENDING |

## 2. Operator decisions and approvals
- **Loop GO:** **GIVEN — 2026-11-24** (operator: full autonomous end-to-end execution of R0–R9 without intervention).
- **GPU approval (R1–R6, R8):** **GIVEN — 2026-11-24** (operator, blanket: R1, R2, R3, R4, R5, R6, and R8 may run autonomously, one GPU job at a time, detached, logs under `outputs/osler_rc0/<task>/logs/`).
- **4B batching (Option A — token-capped plan, no D57 quota, matched to 9B):** **APPROVED — 2026-11-24** (operator confirmed Option A for R1).
- **Git push policy:** **APPROVED — 2026-11-24** for `loop/osler_rc0` only (commit and `git push -u origin loop/osler_rc0` after each task so progress travels through git; **never** commit to or push `main`, `dev`, or any other branch).
- **Release upload policy:** **FORBIDDEN** in this loop. R8 stages locally under gitignored `outputs/osler_rc0/R8/release_staging/` only; never upload to Hugging Face or make external posts.
- **Autonomous fault recovery:** If R1 is interrupted or OOMs, resume cleanly with `--resume` from the latest 2,500-step full checkpoint. If any optional secondary competitor check (`JEV-9B` on `confirm_general` or a competitor latency wrapper) fails past its 30-min timebox, record `NOT MEASURED — <reason>` and proceed without stopping. Never stop to ask the operator unless a hard guardrail triggers (e.g., projected R1 > 72 h, missing required data/disk space that cannot be safely cleared in `/workspace/tmp`, or unexpected data-rule violation).

## 3. Active checklist — R2 (in progress)

- [x] R1 DONE: all five dev evaluations, final S 0.839238, positional all-orders PASS, dev temperatures saved in `final_step_063304`; self-audit `outputs/osler_rc0/R1/SELF_AUDIT.md`; CLAIMS C006–C009.
- [x] R2 launched 2026-11-24T21:33:00Z (detached chain `scripts/osler_rc0/r2_score_chain.sh`): fastdev_v4 scoring for the 9B fit, then 8 sets × {9B rc0, d3 base}, then the fit, then calibrated copies of all 9 sets.
- [ ] R2 acceptance: fastdev extraction 14,597 rows (verified); `9b_calibration.json` with three temperatures, dev pre/post NLL/Brier/ECE and 0 argmax changes; 8 sets × 2 models scored with finite normalised probs; calibrated copies have 0 argmax flips and identical accuracy to the pre-existing T10 files on knowledge, mr_verdict, mimic_suite, confirm_general.
- [ ] `verify_predictions.py --task r2` PASS; `outputs/osler_rc0/R2/SELF_AUDIT.md`; CLAIMS rows; commit `loop(osler_rc0): R2 DONE — <headline>`.

## 4. Log
- 2026-11-24T21:33Z — **R1 DONE.** `Osler-4B-rc0` frozen at step 63,304: dev **S 0.839238** (MR 0.946701, MIMIC 0.783854, CG v2 micro 0.787160), knowledge 0.762364, **all-orders positional PASS** (two-letter 0.488978 vs single-letter 0.852295, n=499/501; both permuted seeds pass). Trajectory 0.8016 → 0.7919 → 0.8350 → 0.8380 → 0.8392 (the 9B final at the same steps: 0.8067 / 0.8292 / 0.8368 / 0.8529 / 0.8538). Wall clock 14.07 h, 0 non-finite losses, 0 tripwires, GPU peak 57.1 GB. Dev temperatures saved in the checkpoint (choice 1.522233, noul 2.374354, score 1.538844). Self-audit + CLAIMS C006–C009. R2 chain launched immediately (21:33Z).
- 2026-11-24T19:12Z — R1 interim: step 55,968 dev evaluation: **S 0.8380** (MR 0.9503, MIMIC 0.7805, CG v2 micro 0.7833), knowledge dev 0.7708, positional all-orders **PASS** (its second PASS of the run). The 9B final at the same step scored S 0.852858, so the 4B trails the 9B by ~0.015 at 88 % of training. Remaining: the final 63,304 evaluation and save (~1.5 h).
- 2026-11-24T16:07Z — R1 interim: step 41,976 dev evaluation: **S 0.8350** (MR 0.9488, MIMIC 0.7771, CG v2 micro 0.7792), knowledge dev 0.7768; positional all-orders FAIL at this checkpoint (third eval: PASS at 13,992, FAIL at 27,984 and 41,976). At matched steps the 9B final scored S 0.836843, so the 4B is within 0.002 of the 9B here. Honest-recording rule applies; the final 63,304 checkpoint stays frozen.
- 2026-11-24T13:02Z — R1 interim: step 27,984 dev evaluation: S 0.7919 (MR 0.9035, MIMIC 0.7408, CG v2 micro 0.7316), knowledge 0.7394, positional all-orders **FAIL** at this intermediate checkpoint (13,992 passed). The CG-v2 micro part moved 0.7656 → 0.7316 while MR/MIMIC held, so the dip is the general CG partition. The final checkpoint is frozen at 63,304 and will be reported as measured; no checkpoint swap.
- 2026-11-24T09:52Z — R1 interim: step 13,992 dev evaluation done. **S 0.8016** (MR 0.8986, MIMIC 0.7406, CG v2 micro 0.7656), knowledge dev 0.7358, all-orders positional PASS. The 9B final at the same step scored S 0.806665, so the 4B is tracking the 9B closely at matched steps. Training healthy (0 non-finite losses). Also pre-registered the R6 non-member check: 2,000 late-anchor `mimic_suite` non-members have **0** content-hash matches in the mix.
- 2026-11-24T07:23Z — R1 interim: step 2,500/63,304 (first full checkpoint saved), median 0.655 s/step, GPU peak 57.1 GB of 97.8 GB, grad norms finite. Projected optimiser time ≈ 11.1 h remaining; five dev evals at 13,992/27,984/41,976/55,968/63,304 add ~1.5 h. Code for R2–R9 (calibration, leaderboard + tests, latency, PHI audit, release staging, spot-check) is written and lint-clean in the working tree; full pytest is deferred until R1 releases the only GPU (it loads an 89 GB model), per the one-GPU-job rule.
- 2026-11-24T07:05Z — **R0 DONE.** Branch/GPU/disk preflight clean; 14 input SHA-256 digests stored (9 frozen sets sum 35,769 items, `mix_v7` 1,526,405); pplx snapshots `6d771ad7…`/`9ff7cf98…` byte-identical (24 files, 0 differences); `medconceptsqa` 0/1,000 item-level overlap (the 1,000 exact question matches are one generic prompt string); prediction inventory complete with zero unmapped/duplicate ids and the expected coverage; ruff clean; pytest one known strict-xpass failure (documented, test untouched). Self-audit `outputs/osler_rc0/R0/SELF_AUDIT.md`; CLAIMS C001–C005; commit + push.
- 2026-11-24T06:50Z — **R1 launched** (detached, GPU): 63,304 steps, token-capped plan, 407,250 batches, KL dose 3.63 %, distillation 32.64 %, seed 0, pplx base `9ff7cf98…`. Cache loaded (1,526,405 rows / 2,100,632,536 tokens), cache verify 32/32 identical, training at ~0.66 s/step median (step 900) → ~11.5 h optimiser + 5 dev evals.
- 2026-11-24T06:15Z — Branch `loop/osler_rc0` created from `loop/osler_v7` (`95d84b7`). Plan written (`GOAL.md`, `ADVISORY.md`, `STATE.md`, `CLAIMS.md`, `KICKOFF.md`) and `AGENTS.md` updated. All 12 comparison model revisions, prediction schemas, `raw↔recomp` ID bijections, and pplx snapshot diffs verified against disk prior to kickoff. Waiting for operator GO + GPU approval in §2.

## 5. Questions for the operator
- None. All approvals recorded in §2; execute R0–R9 autonomously and stop at R9.
