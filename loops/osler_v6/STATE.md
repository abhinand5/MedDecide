# STATE — osler_v6

Loop status: `RUNNING`
Branch: `loop/osler_v6`. Last updated (UTC): `2026-11-07T17:40Z` (Z0 DONE; Z1 PENDING).
Task in flight: none.

## 1. Task board

| id | task | depends | status |
|---|---|---|---|
| Z0 | Setup: GO recorded, teacher revision checked, Z8 closure noted, AGENTS current loop | — | DONE 2026-11-07T17:40Z |
| Z1 | Anchor scoring to 400,000 items with the osler_v4 Y5 path (resume from 112,128; parity 40 of 40 re-run); outputs to `outputs/osler_v6/Z1/` | Z0 | PENDING |
| Z2 | Dosing sampler (D57): 8 items per batch, ≥3 labelled anchors, seeded; λ calibration on first 200 steps (30–50 % gradient share), then frozen | Z1 | PENDING |
| Z3 | Arm G1 (F2 + distillation, seed 0), 13,992 steps; G0 is the existing F2 seed-0 result C062 (checked, not re-scored) | Z2 | PENDING |
| Z4 | Arm G1 seed 1, only if G1 seed 0 passes positional health | Z3 | PENDING |
| Z5 | Choice rule (ADVISORY §2.6), dev only. None qualifies → outcome (4): report and stop | Z3, Z4 | PENDING |
| Z6 | Final at 63,304 steps, positional health at each evaluation, checkpoint by S (only if Z5 chose a recipe) | Z5 | PENDING |
| Z7 | Gate O6 verdict, audit, FINDINGS.md, five-claim spot-check, closure | Z6 or Z5 | PENDING |

## 2. Operator decisions

- **GO for D57, D58 and the ADVISORY §2 design:** `yes` — 2026-11-07, operator instruction "start osler_v6 with everything intact".
- **Osler_v5 Z8:** not run; closed by the operator on 2026-11-07 (see osler_v5 FINDINGS addendum).
- **Base (ADVISORY §6, D59, Clef-Flash vs pplx-4B):** OPEN. Every run in this loop uses the pplx-decider-flash-v1.2 base (the same as
  osler_v5 F2), because §2 was written for it. §6 is not used until the operator decides.
- **Teacher:** `perplexity-ai/pplx-decider-v1.2-27b@cce0f3f80033e9d89d80831d5281fd72124aacce` (osler_v4 Y5; parity C072). The
  ADVISORY's earlier "v1.1" was a typo, corrected at Z0.

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

## 5. Log

- 2026-11-07T17:40Z — Z0 DONE. Z1 is the next task. It needs its own output directory, so the Y5 script is run from a copy that
  writes to `outputs/osler_v6/Z1/` (no append to osler_v4 outputs). That copy is a code change: ruff and the suite run before its
  commit, not during the GPU run.
