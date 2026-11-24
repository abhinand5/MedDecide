# STATE — osler_rc0

Loop status: APPROVED — autonomous execution R0–R9 (blanket GO + GPU approval given 2026-11-24; R0 next)

## 1. Task board

| id | task | depends | approval | status |
|---|---|---|---|---|
| R0 | Preflight, SHA-256 hashes, pplx snapshot identity check, `medconceptsqa` overlap check, on-disk prediction inventory, baseline commit (CPU) | — | **Loop GO** | PENDING |
| R1 | Train `Osler-4B-rc0` (63,304 steps, Option A token-capped plan, `4b_full_run.py`, ~12–14 h GPU) | R0 | **GPU approval** | PENDING |
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

## 3. Active checklist — R0 (pending)
- [ ] Verify branch `loop/osler_rc0`, GPU idle (`nvidia-smi`), 2 GB disk write test in `/workspace/tmp`.
- [ ] Hash all 9 frozen eval files, `dev_format_matched.jsonl`, `mix_v7.jsonl`, 4B prepared cache/plan, 9B `step_063304.pt` (`outputs/osler_rc0/R0/hashes.json`).
- [ ] Verify `pplx-decider-flash-v1.2` snapshot `6d771ad7...` vs `9ff7cf98...` weight/config byte identity (`pplx_snapshot_check.json`).
- [ ] Audit `medconceptsqa_all` (1,000 panel items) vs `mix_v7.jsonl` exact + 13-gram overlap (`medconceptsqa_overlap.json`, aggregates only).
- [ ] Inventory all on-disk competitor prediction files (row counts, `probs`, `raw`/`recomp` ID match; `inventory.json`).
- [ ] Run `uv run ruff check .` and `uv run pytest`; write `outputs/osler_rc0/R0/SELF_AUDIT.md`; append CLAIMS; commit baseline on `loop/osler_rc0`.

## 4. Log
- 2026-11-24T06:15Z — Branch `loop/osler_rc0` created from `loop/osler_v7` (`95d84b7`). Plan written (`GOAL.md`, `ADVISORY.md`, `STATE.md`, `CLAIMS.md`, `KICKOFF.md`) and `AGENTS.md` updated. All 12 comparison model revisions, prediction schemas, `raw↔recomp` ID bijections, and pplx snapshot diffs verified against disk prior to kickoff. Waiting for operator GO + GPU approval in §2.

## 5. Questions for the operator
- None. All approvals recorded in §2; execute R0–R9 autonomously and stop at R9.
