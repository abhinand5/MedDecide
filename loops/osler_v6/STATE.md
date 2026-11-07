# STATE — osler_v6

Loop status: `RUNNING`
Branch: `loop/osler_v6`. Last updated (UTC): `2026-11-07T19:08Z` (Z0 DONE; Z1 IN_PROGRESS, anchor job running).
Task in flight: Z1 anchor scoring; detached, uv PID `301442`, python PID `301463`, log
`outputs/osler_v6/Z1/logs/anchor.log`; started 2026-11-07T19:02Z, `--max-hours 12`.

## 1. Task board

| id | task | depends | status |
|---|---|---|---|
| Z0 | Setup: GO recorded, teacher revision checked, Z8 closure noted, AGENTS current loop | — | DONE 2026-11-07T17:40Z |
| Z1 | Anchor scoring to 400,000 items with the osler_v4 Y5 path (resume from 112,128; parity 40 of 40 re-run); outputs to `outputs/osler_v6/Z1/` | Z0 | IN_PROGRESS 2026-11-07T18:42Z |
| Z2 | Dosing sampler (D57, 8 per batch, ≥3 labelled anchors, seeded) and the λ calibration probe (200 steps at λ = 1; λ then frozen; ADVISORY §2.4) | Z1 | PENDING |
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

## 4b. Checklist — Z1 (in progress)

- [x] Script `scripts/osler_v6/z1_anchor_scoring.py`: the osler_v4 Y5 path, Z1 output dir, reuse by id + text hash; `uv run --frozen ruff check .` and the pytest suite pass before its commit (C009).
- [x] Pool rebuilt with the current bench exclusions (all 73 `data/bench/**/*.jsonl`, including the osler_v4 y9 panel/knowledge/MIMIC sets created after Y5); its sha256 is osler_v4's `a9fab387…`, so no old scored row is excluded (C004).
- [x] Reuse: all 112,128 osler_v4 records verified against their pool row (item id, template, qtype, option count, probability sum); 0 dropped, positions preserved; written to `outputs/osler_v6/Z1/anchor_probs.jsonl` (C006).
- [x] Parity re-run: 40 of 40 argmax agreement on both batched paths; chosen path `reference_batched`; numbers bit-identical to osler_v4's (C007).
- [ ] Anchor scoring detached under `--max-hours 12`; log and PID at the top; polled every 20+ minutes.
- [ ] Join `data/train/osler_v6/z1_anchor_soft.jsonl`; overlap with every bench row re-checked by text hash = 0.
- [ ] Summary numbers, `SELF_AUDIT.md`, CLAIMS rows, STATE exact, commit and push.

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

