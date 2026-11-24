# STATE — osler_rc0

Loop status: IN_PROGRESS — R0 DONE; R1 training (started 2026-11-24T06:50Z)

## 1. Task board

| id | task | depends | approval | status |
|---|---|---|---|---|
| R0 | Preflight, SHA-256 hashes, pplx snapshot identity check, `medconceptsqa` overlap check, on-disk prediction inventory, baseline commit (CPU) | — | **Loop GO** | DONE 2026-11-24T07:05Z |
| R1 | Train `Osler-4B-rc0` (63,304 steps, Option A token-capped plan, `4b_full_run.py`, ~12–14 h GPU) | R0 | **GPU approval** | IN_PROGRESS 2026-11-24T06:50Z |
| R2 | Dev-only calibration for `Osler-9B-rc0` + score `Osler-9B-rc0` & `d3-flash` base with full `probs` on all missing/prob-less sets (~1.5 h GPU) | R0 | **GPU approval** | PENDING |
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

## 3. Active checklist — R1 (in progress)

- [x] Launch exactly the ADVISORY §4 R1 command (detached, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, `--batching token --steps 63304 --eval-steps 13992,27984,41976,55968,63304`), log `outputs/osler_rc0/R1/arm/logs/arm.log`; launched 2026-11-24T06:50:12Z, provenance `outputs/osler_rc0/R1/arm/run_provenance.json` (checkpoint `pplx-decider-flash-v1.2@9ff7cf98…`, mix sha `b8b97031…`, plan 407,250 batches, KL dose 3.63 %, distillation 32.64 % — matches the parity report).
- [ ] Step-500 wall-clock guard: median s/step and projected wall clock; if projected > 72 h, stop and mark BLOCKED. (Median at step 900: 0.655 s/step → ~11.5 h optimiser time, guard PASS.)
- [ ] `arm_result.json`: `steps == 63304`, `stopped_early == false`, 0 non-finite losses in `logs/train.jsonl`.
- [ ] All 5 dev evaluations complete; record dev S trajectory (MR, MIMIC, CG v2 micro, knowledge dev).
- [ ] D52b positional health at 63,304: `positional_passes_all_orders == true`; record honestly if false (no checkpoint swap without operator approval).
- [ ] Dev temperature fits (`choice`, `noul`, `score`) saved inside `checkpoints/final_step_063304`; record SHA-256 of the checkpoint files.
- [ ] `outputs/osler_rc0/R1/SELF_AUDIT.md`, CLAIMS rows, commit `loop(osler_rc0): R1 DONE — <headline>`.

### Notes carried from R0

- Pytest: the only failure is the pre-existing strict-xfail `tests/test_padding_trained.py::test_trained_checkpoint_logits_are_padding_invariant`; the test is not modified (KICKOFF §5). 1,223 tests were collected in the R0 run; the two files added afterwards (`tests/test_calibration.py`) pass separately (7/7).
- The `medconceptsqa` question-text match is a generic prompt string, not overlap; item-level checks are 0/1,000 on both exact state+question and 13-grams.

## 4. Log
- 2026-11-24T07:05Z — **R0 DONE.** Branch/GPU/disk preflight clean; 14 input SHA-256 digests stored (9 frozen sets sum 35,769 items, `mix_v7` 1,526,405); pplx snapshots `6d771ad7…`/`9ff7cf98…` byte-identical (24 files, 0 differences); `medconceptsqa` 0/1,000 item-level overlap (the 1,000 exact question matches are one generic prompt string); prediction inventory complete with zero unmapped/duplicate ids and the expected coverage; ruff clean; pytest one known strict-xpass failure (documented, test untouched). Self-audit `outputs/osler_rc0/R0/SELF_AUDIT.md`; CLAIMS C001–C005; commit + push.
- 2026-11-24T06:50Z — **R1 launched** (detached, GPU): 63,304 steps, token-capped plan, 407,250 batches, KL dose 3.63 %, distillation 32.64 %, seed 0, pplx base `9ff7cf98…`. Cache loaded (1,526,405 rows / 2,100,632,536 tokens), cache verify 32/32 identical, training at ~0.66 s/step median (step 900) → ~11.5 h optimiser + 5 dev evals.
- 2026-11-24T06:15Z — Branch `loop/osler_rc0` created from `loop/osler_v7` (`95d84b7`). Plan written (`GOAL.md`, `ADVISORY.md`, `STATE.md`, `CLAIMS.md`, `KICKOFF.md`) and `AGENTS.md` updated. All 12 comparison model revisions, prediction schemas, `raw↔recomp` ID bijections, and pplx snapshot diffs verified against disk prior to kickoff. Waiting for operator GO + GPU approval in §2.

## 5. Questions for the operator
- None. All approvals recorded in §2; execute R0–R9 autonomously and stop at R9.
