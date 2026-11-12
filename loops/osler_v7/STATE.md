# STATE — osler_v7

Loop status: NOT STARTED (planned 2026-11-12; waiting for the operator's GO in §2)

## 1. Task board

| id | task | depends | approval | status |
|---|---|---|---|---|
| T0 | Setup: merge to dev (osler_v4–v6 fast-forward; d3_distill merge; lint fix), branch `loop/osler_v7`, AGENTS Current loop, this plan | — | operator instruction 2026-11-12 | DONE 2026-11-12 (dev at 64de99a; gate ruff clean, pytest exit 0 with data linked) |
| T1 | Composition and prior-shift checks: osler_v6 mix and 385,536 anchors vs the panel's item shapes (CPU); predicted-yes and option-position bias on losing sources | T0 | GO (given 2026-11-12) | PENDING |
| T2 | Build the confirmation set (ADVISORY §8) and the format-matched dev set (§6); manifests with SHA-256 and counts | T1 | GO (given) + confirmation-source decision (§0.3) | PENDING |
| T3 | Port check for the d3 harness: `score` items, long MIMIC notes, 255-row readout at the last prompt position (CPU where possible) | T0 | GO (given 2026-11-12) | PENDING |
| T4 | Lever-3 grid on the osler_v6 final (inference only): LoRA {0.5, 0.75, 1} × readout {0, 0.5, 1} on the dev set | T2 | **GPU approval** | PENDING |
| T5 | Label the pool with pplx-27B (C1–C4 and the unlabelled KL pool); measure throughput; audit counts | T2 | **GPU approval** | PENDING |
| T6 | 4B arm on pplx-flash: v6 mix + C1–C4 + KL to base (λ_KL §0.2), 13,992 steps; selection on dev (§7) | T4, T5 | **GPU approval**; §0.1–0.4 settled | PENDING |
| T7 | Interpolation and selection for the 4B arm (§4, §7); calibrate | T6 | GPU approval (inference) | PENDING |
| T8 | 9B arm on d3-flash: same mix; same rules | T3, T5, T7-4B-track-decision | **GPU approval** | PENDING |
| T9 | Interpolation and selection for the 9B arm | T8 | GPU approval (inference) | PENDING |
| T10 | Gate v7 for each eligible model against its own zero-shot base on the confirmation set (once each) | T7 or T9, T2 | GPU approval (inference) | PENDING |
| T11 | 63,304-step final for a passing arm only (54 hours) | T10 passing | **GPU approval** | PENDING (conditional) |
| T12 | FINDINGS.md, SELF_AUDIT, five-claim spot-check, closure | all | — | PENDING |

## 2. Operator decisions and approvals
- **Loop GO:** GIVEN — 2026-11-12 (operator). Unlocks T1–T3 (CPU only).
- **GPU approvals:** NOT GIVEN. Needed before T4, T5, T6, T8, T10, T11.
- **Open decisions (ADVISORY §0):** 1 gate limits; 2 KL weight; 3 confirmation-set source; 4 declared `peft` dependency; 5 teacher
  (proposed: pplx-27B; confirm).
- **Bases (settled by the operator in chat, 2026-11-12):** 4B = pplx-decider-flash-v1.2; 9B = d3-flash.
- Shared decision (not part of this loop): what to do with the local-only branches and dirty worktrees (see §5).

## 3. Checklist — T0 (done)
- [x] Fast-forward dev to `loop/osler_v6` (e875834), pushed.
- [x] Merge `distill/d3_distill` into dev (`bbde9ab`, no conflicts by merge-tree).
- [x] Lint fix in two d3 scripts (ruff I001), pushed as `64de99a`.
- [x] Full gate on the merged tree: ruff clean; pytest exit 0 (run with the `data/` folder linked into the worktree; without it,
      `test_overfit_64_items` falls back to synthetic items and fails by design, which is an environment artifact, not a regression).
- [x] Branch `loop/osler_v7` created from dev; this loop's files and the AGENTS Current loop section.
- Declared dependency gap: `peft` is missing from `pyproject.toml` (ADVISORY §0.4).

## 4. Log
- 2026-11-12 — the plan written from the advisor review. No run started. Lever diagnostics and arms wait for approvals.
- 2026-11-12 — operator recorded Loop GO (§2). T1–T3 are approved; GPU tasks still wait for their own approval lines.

## 5. Cleanup (not done; needs the operator's decision)
- Local-only branches with work not on GitHub: `bench/d1-2b`, `bench/d3-ultra`, `loop/knowledge_v1`, `loop/synth_v1`,
  `synth/driver`, `synth/driver-fix`, `synth/generator-code`, `synth/generator-fixed`, `synth/mimic-grounded`, `synth/train-v1`,
  and the `worktree-agent-*` branches.
- **Do not push `synth/*` or `loop/synth_v1` without a MIMIC content check.** The repo is public.
- Worktrees with uncommitted changes: `synth-mimic-wt` (4 entries, untracked `data` and `outputs`, plus deleted tracked MIMIC
  outputs) and `.claude/worktrees/agent-a1d038ff24c8d733c` (23 entries). Leave both alone until checked.
- No branch or worktree has been deleted.
