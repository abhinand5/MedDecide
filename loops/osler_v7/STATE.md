# STATE — osler_v7

Loop status: IN PROGRESS — T2 DONE 2026-11-12; T4 and T5 are next (GPU approved); T1 and T3 DONE

## 1. Task board

| id | task | depends | approval | status |
|---|---|---|---|---|
| T0 | Setup: merge to dev (osler_v4–v6 fast-forward; d3_distill merge; lint fix), branch `loop/osler_v7`, AGENTS Current loop, this plan | — | operator instruction 2026-11-12 | DONE 2026-11-12 (dev at 64de99a; gate ruff clean, pytest exit 0 with data linked) |
| T1 | Composition and prior-shift checks: osler_v6 mix and 385,536 anchors vs the panel's item shapes (CPU); predicted-yes and option-position bias on losing sources | T0 | GO (given 2026-11-12) | DONE 2026-11-12 (6 of 7 panel sources lose, macro −0.024486 repeats Gate O6; yes and last-position bias measured; audit PASS, C001–C007) |
| T2 | Build the confirmation set (ADVISORY §8) and the format-matched dev set (§6); manifests with SHA-256 and counts | T1 | GO (given) + confirmation-source decision (§0.3) | DONE 2026-11-12 (dev 22,934 rows / 7 shapes; confirmation 5,988 fresh general + 13,103+4,253 audit + 5,456 knowledge; all hashes, 0 training overlap; audit PASS 39 checks; peft declared; C014–C019) |
| T3 | Port check for the d3 harness: `score` items, long MIMIC notes, 255-row readout at the last prompt position (CPU where possible) | T0 | GO (given 2026-11-12) | DONE 2026-11-12 (parity PASS, 9 rows exact 0.0; 3 longest MIMIC forwards NOT MEASURED over the CPU timebox; d3_batch scoring helper not portable to this runtime; audit PASS, C008–C013) |
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
- **GPU approvals:** GIVEN — 2026-11-12 (operator, blanket): T4, T5, T6, T8, T10, T11 and the inference steps may run, one GPU
  job at a time, detached, logs under `outputs/osler_v7/<task>/logs/`. Hard rules still apply: no panel or MIMIC held-out
  items in training, selection or calibration; no MIMIC text to hosted services; no raw data, keys or checkpoints committed;
  no pod or volume stop/delete.
- **ADVISORY §0 decisions (operator, 2026-11-12, "feel free"):** §0.1 gate margins: accepted as proposed (non-inferiority
  −0.01, audit floor 0, knowledge −0.02). §0.2 KL weight: 0.1 (0.5 only under the kill rule). §0.3 confirmation set: accepted as
  proposed (post-2026-03-01 records, structured gold, built and manifested before any arm trains; operator reviews the
  manifest before first use). §0.4 `peft`: declare it in `pyproject.toml` (updates `uv.lock`). §0.5 teacher: pplx-27B, local.
  ADVISORY.md itself is unchanged; these values govern here.
- **Open decisions (ADVISORY §0):** 1 gate limits; 2 KL weight; 3 confirmation-set source; 4 declared `peft` dependency; 5 teacher
  (proposed: pplx-27B; confirm).
- **Bases (settled by the operator in chat, 2026-11-12):** 4B = pplx-decider-flash-v1.2; 9B = d3-flash.
- Shared decision (not part of this loop): what to do with the local-only branches and dirty worktrees (see §6).

## 3. Checklist — T0 (done)
- [x] Fast-forward dev to `loop/osler_v6` (e875834), pushed.
- [x] Merge `distill/d3_distill` into dev (`bbde9ab`, no conflicts by merge-tree).
- [x] Lint fix in two d3 scripts (ruff I001), pushed as `64de99a`.
- [x] Full gate on the merged tree: ruff clean; pytest exit 0 (run with the `data/` folder linked into the worktree; without it,
      `test_overfit_64_items` falls back to synthetic items and fails by design, which is an environment artifact, not a regression).
- [x] Branch `loop/osler_v7` created from dev; this loop's files and the AGENTS Current loop section.
- Declared dependency gap: `peft` is missing from `pyproject.toml` (ADVISORY §0.4).

## 3b. Checklist — T1 (done)
- [x] Verify inputs and counts: v6 mix 1,456,438 rows (sha `91ff5362…`), anchors 385,536 rows (sha `edd4c207…`), panel 4,353 rows
      (sha from Y9 manifest), v6 final and zero-shot pplx panel predictions (4,353 scored each), join by recomputed item id.
- [x] Composition: qtype, template/source, option-count, label-balance (noul yes share, choice gold-position prior, score level)
      and text-length distributions for mix, anchors and panel; share gaps and total-variation distance; panel date fields absent
      (record-window comparison NOT MEASURED and labelled).
- [x] Prior shift on the panel: per-dataset final vs zero-shot accuracy/Brier; predicted-yes rate and yes/no confusion on the noul
      items; predicted option-position and gold-position distributions, accuracy by gold position, choice items; list the losing
      sources (point estimate < 0).
- [x] Aggregate-only outputs in `outputs/osler_v7/T1/` (`composition.json`, `prior_shift.json`, `run_provenance.json`); no item
      text or predictions committed; `SELF_AUDIT.md` with fresh-process re-derivation.
- [x] CLAIMS rows; note in §4 that T1 is diagnostic only and feeds no v7 selection or gate.
- [x] `uv run ruff check .` and `uv run pytest` (code added: `scripts/osler_v7/t1_diagnose.py`, `scripts/osler_v7/t1_audit.py` + unit tests); commit and push.

## 3c. Checklist — T3 (done)
- [x] Acquire the pinned d3-flash snapshot (`vllm-sr/d3-flash@67e03e1b…`, 16.8 GB) into `$HF_HOME`; record the path,
      manifest/config hash, readout shape and context length. If the runtime cannot run on CPU, record the exact
      failure and mark the model-dependent parity NOT MEASURED rather than substituting a number.
- [x] Port check, CPU where possible: `score` items (2–10 ordered levels), long MIMIC notes (longest mix states),
      and the 255-row readout at the last prompt position; harness (`D3Student.forward_logits`) vs the runtime's own
      `D3.probabilities` on the same prepared rows, 1e-4 bar as d3_distill step 2.
- [x] Structural checks that need no model forward: `render_order`/`render_targets` and `align_probabilities` for
      score; mask/left-padding last-position selection; option counts above and below 255; length vs context window.
- [x] Aggregate-only output in `outputs/osler_v7/T3/` + logs; SELF_AUDIT.md; CLAIMS rows; no item text committed.
- [x] `uv run ruff check .` and `uv run pytest`; commit and push.

## 3d. Checklist — T2 (done)
- [x] Format-matched dev set (§6): compose from the frozen dev assets (fastdev_v4, cg_v2, knowledge_dev_v2), dedupe,
      exclude the code-generated / teacher-prose templates, record per-shape coverage (many-option 10–50, two-option
      pairs, code items, 4- and 10-option MCQ, MR verdict family, MIMIC audit templates), and verify 0 item-id and
      0 normalised-text overlap with the v6 mix and anchors; record the MIMIC patient-split provenance.
- [x] Fresh tier-2 general component (§8): build a new post-window set (2026-10-06…2026-11-12) from ClinicalTrials,
      openFDA and PubMed with structured gold by construction; keep the test split only; record counts, SHA, the
      strict-slice share (≥2026-09-10) and 0 training overlap. Fallback if the build fails: the frozen v0.2 fresh
      strict slice, recorded as a deviation.
- [x] Confirmation audit and knowledge components: reference the frozen Y9 mr_verdict, mimic_suite and knowledge
      files by path + SHA-256 + counts; document that MIMIC/knowledge have no post-window records and are test
      splits never used for training or selection; the operator reviews the manifest before first use.
- [x] Manifests (`dev_manifest.json`, `confirmation_manifest.json`, `provenance.json`) with SHA-256 and counts;
      SELF_AUDIT.md; CLAIMS rows; no raw items committed.
- [x] Declare `peft` in `pyproject.toml` (§0.4) and update `uv.lock`; `uv run ruff check .` and `uv run pytest`;
      commit and push.

## 4. Log
- 2026-11-12 — the plan written from the advisor review. No run started. Lever diagnostics and arms wait for approvals.
- 2026-11-12 — operator recorded Loop GO (§2). T1–T3 are approved; GPU tasks still wait for their own approval lines.
- 2026-11-12 — operator gave blanket GPU approval and accepted the ADVISORY §0 proposals (§2). T2 is unblocked; T4 onward
  may start in order when their dependencies are DONE.
- 2026-11-12T12:52Z — **T1 DONE.** Composition and prior shift measured on CPU. Composition: the mix and anchors
  differ from the panel sharply in question type (TVD 0.332 / 0.234), option count (TVD 0.705 / 0.633; the panel's
  10- and 22-option items are rare in training) and text length (+4,177 / +2,050 mean state chars), with 0 shared
  sources or templates; label balance is close (noul yes 0.522 mix / 0.525 panel; gold relative position 0.491 /
  0.518). Prior shift: the v6 final loses 6 of 7 panel sources (macro −0.024486, reproducing Gate O6); on the noul
  source it over-predicts yes (argmax 0.580 vs gold 0.525 and base 0.498; FP 25 vs 6) and on the 4-option
  medconceptsqa source it over-predicts the last position (0.359 vs gold 0.238). Diagnostic only: no v7 recipe,
  selection or gate input reads T1 (ADVISORY §7 selects on the format-matched dev set).
  SELF_AUDIT PASS (96 checks, 0 problems); CLAIMS C001–C007. Item-mean Brier on the panel also matches Gate O6
  (−0.033914): five sources better, medical_question_pairs and symptom_to_diagnosis worse.
- 2026-11-12T13:36Z — **T3 DONE.** Pinned d3-flash (`67e03e1b…`, 8.39B loaded) fetched and manifest-verified; the
  runtime answers CPU with torch-reference kernels (no fla/causal-conv1d). Structural checks pass on 12 rows
  (score/noul/choice order, gold mapping, criteria, ≤255 options). Parity on CPU: 9 rows (score, noul, choice,
  long MIMIC) are **bit-identical** to the runtime's own probabilities (max diff 0.0, argmax 9/9; 1e-4 gate PASS);
  readout [255, 4096] at the last prompt position. Three longest MIMIC forwards (6,176–6,182 tokens) are NOT
  MEASURED over the 6,000-token CPU timebox — reported, not substituted; tokenisation/prepare accepted them
  untruncated. Finding: `d3_batch.prepare_request`/`score_batch` target a runtime with a `states` block and a
  three-argument `probabilities`; d3-flash exposes neither, so the 9B path must use `D3Student` (as `d3_train`
  does). SELF_AUDIT PASS (24 checks); CLAIMS C008–C013.
- 2026-11-12T16:55Z — **T2 DONE.** Frozen the format-matched dev set and the confirmation set, with manifests.
  Dev: 22,934 rows, sha `3d6dc6f8…`, seven required shapes present (many-option 1,916; pairs 3,869; code
  1,698; 4-option MCQ 9,940; 10-option 73; MR family 1,597; MIMIC templates 5,562); 0 overlap with the v6
  mix/anchors. Confirmation: new fresh tier-2 general build from records 2026-10-06…2026-11-12 (CT 5,660 /
  openFDA 2,340 / PubMed 8,254 before filtering), test split capped at 1,000/template and decontaminated
  (530 rows/432 records whose text recurs in training), 5,988 rows, sha `bf6614a6…`, strict slice 100 %;
  audit Y9 `mr_verdict` 13,103 + `mimic_suite` 4,253 and knowledge 5,456 referenced with SHAs, all test,
  0 training overlap. The fresh build's only failed check (one openFDA record across dev/test) was handled by
  excluding the record, not by editing the check; the manifest records the §8 date-rule deviation for the
  audit/knowledge components and flags operator review before first use. SELF_AUDIT PASS (39 checks);
  CLAIMS C014–C019. `peft>=0.21` declared; `uv.lock` updated (0.21.2).
- 2026-11-12T13:39Z — T3 code-gate arithmetic correction: 1,185 tests collected; 2 deselected and 4 CUDA
  skipped leave 1,179 passed of 1,183 run. (The T3 commit message's '1,179 run' was the passed count, not the
  run count; the exit code was 0 either way.)

## 5. Questions for the operator
- **T2 confirmation source (§0.3):** T2's approval row requires the confirmation-set source decision, which is
  still open in §2. ADVISORY §8 proposes records dated after 2026-03-01 with gold from structured source fields,
  built and manifested before any arm trains, reviewed by the operator before first use. Confirm or name another
  source; T2 does not start until then.
- **GPU approvals:** still NOT GIVEN for T4, T5, T6, T8, T10, T11.

## 6. Cleanup (not done; needs the operator's decision)
- Local-only branches with work not on GitHub: `bench/d1-2b`, `bench/d3-ultra`, `loop/knowledge_v1`, `loop/synth_v1`,
  `synth/driver`, `synth/driver-fix`, `synth/generator-code`, `synth/generator-fixed`, `synth/mimic-grounded`, `synth/train-v1`,
  and the `worktree-agent-*` branches.
- **Do not push `synth/*` or `loop/synth_v1` without a MIMIC content check.** The repo is public.
- Worktrees with uncommitted changes: `synth-mimic-wt` (4 entries, untracked `data` and `outputs`, plus deleted tracked MIMIC
  outputs) and `.claude/worktrees/agent-a1d038ff24c8d733c` (23 entries). Leave both alone until checked.
- No branch or worktree has been deleted.
