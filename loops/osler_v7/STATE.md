# STATE — osler_v7

Loop status: IN PROGRESS — T8 9B arm on d3-flash training (13,992 steps); T7 DONE

## 1. Task board

| id | task | depends | approval | status |
|---|---|---|---|---|
| T0 | Setup: merge to dev (osler_v4–v6 fast-forward; d3_distill merge; lint fix), branch `loop/osler_v7`, AGENTS Current loop, this plan | — | operator instruction 2026-11-12 | DONE 2026-11-12 (dev at 64de99a; gate ruff clean, pytest exit 0 with data linked) |
| T1 | Composition and prior-shift checks: osler_v6 mix and 385,536 anchors vs the panel's item shapes (CPU); predicted-yes and option-position bias on losing sources | T0 | GO (given 2026-11-12) | DONE 2026-11-12 (6 of 7 panel sources lose, macro −0.024486 repeats Gate O6; yes and last-position bias measured; audit PASS, C001–C007) |
| T2 | Build the confirmation set (ADVISORY §8) and the format-matched dev set (§6); manifests with SHA-256 and counts | T1 | GO (given) + confirmation-source decision (§0.3) | DONE 2026-11-12 (dev 22,934 rows / 7 shapes; confirmation 5,988 fresh general + 13,103+4,253 audit + 5,456 knowledge; all hashes, 0 training overlap; audit PASS 39 checks; peft declared; C014–C019) |
| T3 | Port check for the d3 harness: `score` items, long MIMIC notes, 255-row readout at the last prompt position (CPU where possible) | T0 | GO (given 2026-11-12) | DONE 2026-11-12 (parity PASS, 9 rows exact 0.0; 3 longest MIMIC forwards NOT MEASURED over the CPU timebox; d3_batch scoring helper not portable to this runtime; audit PASS, C008–C013) |
| T4 | Lever-3 grid on the osler_v6 final (inference only): LoRA {0.5, 0.75, 1} × readout {0, 0.5, 1} on the dev set | T2 | **GPU approval** | DONE 2026-11-12 (9 cells, 9,404.4 s; base readout + trained LoRA does not recover two-letter 0.519 vs 0.651; 0.75-scale cells best macro ~0.8086; audit PASS 139 checks; C020–C022) |
| T5 | Label the pool with pplx-27B (C1–C4 and the unlabelled KL pool); measure throughput; audit counts | T2 | **GPU approval** | DONE 2026-11-13 (pool 69,967; parity 40/40; 69,967 scored at 5.22 items/s / 4,602 tok/s; join max sum error 1.1e-16, 8 constant; audit PASS 14 checks; C023–C027) |
| T6 | 4B arm on pplx-flash: v6 mix + C1–C4 + KL to base (λ_KL §0.2), 13,992 steps; selection on dev (§7) | T4, T5 | **GPU approval**; §0.1–0.4 settled | DONE 2026-11-13 (13,992 steps; eval S 0.797548 → 0.811190; positional PASS; audit PASS 26 checks; C029–C031) |
| T7 | Interpolation and selection for the 4B arm (§4, §7); calibrate | T6 | GPU approval (inference) | DONE 2026-11-13 (all four blends eligible; chosen s=1.00, S 0.810521; positional PASS all orders at every scale; audit PASS 59 checks; C032–C035) |
| T8 | 9B arm on d3-flash: same mix; same rules | T3, T5, T7-4B-track-decision | **GPU approval** | IN_PROGRESS 2026-11-13T17:10Z (d3 pool 1,526,405 rows sha `bb826320…`, 438,188 labelled; dev 22,934 rows sha `5d8d9361…`; driver smoke-tested; 13,992-step run launching) |
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

## 3e. Checklist — T4 (done)
- [x] Verify the v6 63,304-step checkpoint and the T2 dev file hashes; record provenance (GPU, dtype, batch).
- [x] Run the 9-cell LoRA × readout grid on the frozen dev set, one detached GPU job, log under
      `outputs/osler_v7/T4/logs/`; check the output file completeness, not only the exit code.
- [x] Analyse the grid: item/macro accuracy overall and in the MR, MIMIC, CG and knowledge partitions, per qtype
      and option-count group, and the two-letter positional check; state whether the base readout + trained LoRA
      cell recovers the losses. Diagnostic only — no selection, no gate.
- [x] Aggregate-only results committed (`grid.json`); per-item predictions stay gitignored; SELF_AUDIT.md; CLAIMS.
- [x] `uv run ruff check .` and `uv run pytest`; commit and push.

## 3f. Checklist — T5 (done)
- [x] Build C1–C4 cells from the structured sources and merge the pool: C1 20,000, C2 10,000, C3 20,000,
      C4 20,000; merged 69,967 rows; 0 text overlap with the T2 dev/confirmation sets (CLAIMS pending).
- [x] Teacher parity: 40 CG v2 dev items through the 27B's single-request reference, batched reference and fast
      path; choose the path by the Y5 rule (argmax agreement 40/40); record it.
- [x] Label the 69,967-prompt pool with pplx-27B (detached, resumable, logs under `outputs/osler_v7/T5/logs/`);
      measure items/s and input tokens/s; check the output completeness rather than the exit code alone.
- [x] Join the labels (probability length/sum checks, invalid and constant counts), write the labelled file and
      its manifest; SELF_AUDIT.md; CLAIMS rows.
- [x] `uv run ruff check .` and `uv run pytest`; commit and push.

## 3g. Checklist — T6 (done)
- [x] v7 mix built (1,526,405 rows; C1–C4 69,967; labelled 438,188) and dry-run through the arm driver.
- [x] Arm config (`kl_to_base` 0.1 on the v7_c pool, v6-frozen distill weight 3.243598 with the 3-labelled quota,
      per-item weighting, no double augmentation).
- [x] Run 13,992 steps detached, one GPU job; check completeness of the log, evaluations and checkpoints, not the
      exit code alone; record the first-200-step distillation share.
- [x] SELF_AUDIT (R6), CLAIMS rows; commit and push.

## 3h. Checklist — T7 (done)
- [x] Score the zero-shot base and the four blend scales (0.25/0.5/0.75/1.0) on the frozen dev set (GPU, detached).
- [x] Apply §7: paired template-macro difference vs the base (lower bound > -0.01) and choose the highest dev S among
      eligible; calibrate the chosen blend on fast dev and export it for the gate.
- [x] SELF_AUDIT/CLAIMS; commit and push.

### T7 results — 4B blend table (operator-requested; frozen 22,934-item dev set)

| blend s | S | MR | MIMIC | CG v2 micro | knowledge dev | positional (3 orders) | paired diff vs base (point, lower) |
|---|---|---|---|---|---|---|---|
| 0.00 (zero-shot base) | 0.762977 | 0.828916 | 0.672190 | 0.787826 | 0.775633 | PASS | reference |
| 0.25 | 0.785320 | 0.855478 | 0.712923 | 0.787560 | 0.763571 | PASS | +0.018789, +0.012943 |
| 0.50 | 0.804976 | 0.886725 | 0.736513 | 0.791689 | 0.773221 | PASS | +0.034296, +0.027109 |
| 0.75 | 0.808884 | 0.890830 | 0.745996 | 0.789824 | 0.773221 | PASS | +0.038255, +0.029297 |
| **1.00 (chosen)** | **0.810521** | 0.887582 | 0.752425 | 0.791556 | 0.763571 | PASS | +0.041984, +0.032131 |

- Selection: all four scales are eligible (paired template-macro lower bound > -0.01); the **chosen blend is s = 1.00**
  (no interpolation toward the base) because it has the highest dev S. Paired differences are positive at every scale, so
  the trained arm is non-inferior to its zero-shot base on dev; interpolation does not add S here.
- Positional health passes in all three orders for every scale, including the base (`positional.json`).
- Calibration of the chosen blend on fast dev (14,597 items): choice 1.206404, noul 1.351652, score 1.135541; exported
  checkpoint `outputs/osler_v7/T7/chosen/`.
- The s = 1.00 MIMIC value here (0.752425) differs from the T6 evaluation's (0.754032) by 0.0016 because the blend path
  reconstructs the readout in FP32 from the base snapshot; both are reported as measured.
- Source files: `outputs/osler_v7/T7/selection.json`, `outputs/osler_v7/T7/positional.json`,
  `outputs/osler_v7/T7/predictions/scale_{0,0.25,0.5,0.75,1}.jsonl` (gitignored), `outputs/osler_v7/T7/chosen/`,
  `outputs/osler_v7/T7/audit.json`, `outputs/osler_v7/T7/SELF_AUDIT.md`.

## 3i. Checklist — T8 (in progress)
- [x] Convert the v7 mix and the T2 dev set to the d3 harness's row format (d3_pool.jsonl sha `bb826320…`,
      1,526,405 rows with 438,188 teacher-labelled; d3_dev.jsonl 22,934 rows sha `5d8d9361…`).
- [x] Driver `t8_run_arm.py`: CE + teacher KL on labelled rows + KL-to-base on `v7_c*` rows (adapters disabled,
      checkpoint's base readout, no gradients), dev evaluations at 6,996 and 13,992 (S, knowledge, positional
      original order), full checkpoints, resume. Smoke-tested on 512 rows / 5 steps.
- [ ] Run 13,992 steps detached; verify the evaluations and checkpoints; audit; commit.

## 4. Log
- 2026-11-12T17:38Z — T4 grid launched as one detached GPU job (9 cells, batch 32 / 32,768 padded tokens, log
  `outputs/osler_v7/T4/logs/t4_grid.log`). Three earlier attempts were stopped before a cell completed while batching was
  tuned (length-sorted dev order; logs `t4_grid_attempt1..3.log`); no partial results are used. T4 finishes when `grid.json`
  exists and `scripts/osler_v7/t4_audit.py` passes.
- 2026-11-12T18:26Z — T4 first cell done (lora 0.5 / readout 0: item accuracy 0.7800, template macro 0.7799,
  two-letter 0.6750, 1,038.8 s); the remaining eight cells run in the same detached job. While the GPU job runs,
  the T5 pool builders were prepared and run on CPU: C1 20,000 rows (sha `a27a26f4…`), C3 20,000 rows
  (sha `5e2e50ab…`), C4 20,000 rows (sha `fcde88fa…`); C2 and the pplx-27B labelling are still to come.
- 2026-11-14T02:40Z — T8 in-run evaluation note: the step-6,996 evaluation was written with empty composite partitions
  because `d3_dev.jsonl` carried the raw file ids instead of the trainer's recomputed item ids and did not carry
  `dev_origin`. The prediction file is intact; aggregating it against the T2 origin map gives (additional,
  derived) S 0.807386, MR 0.868521, MIMIC 0.748496, CG v2 micro 0.805141, knowledge dev 0.826297, and the
  original-order positional check two-letter 0.328657 vs single 0.792415 → **FAIL** (the threshold is half the
  single-letter rate). The converter now writes recomputed ids and `dev_origin`; the running arm continues and its
  final evaluation will be aggregated the same way, with all-orders positional measured before any selection. This
  is a driver bookkeeping defect, not a change to a check.
- 2026-11-13T19:55Z — T8 operational notes: the first 9B attempt OOM'd at step 1 on a batch of 8 long MIMIC rows after
  the 3,980 s preparation; batches are now capped by 8 items and 8,192 padded tokens, prepared TrainRows are
  cached (prepared.pt, gitignored), the KL-to-base reference escapes `torch.inference_mode` tensors, and the 32
  PromptEncoder layers run under non-reentrant activation checkpointing (peak GPU 38 GB instead of 95 GB). The
  arm restarted from the cache at 19:50Z and is training; the token-capped batching is a memory-forced deviation
  from the fixed batch-8 rule and is recorded as such. No results yet. (The train log contains steps from the
  failed attempts as well; the audit will read only the final contiguous run.)
- 2026-11-13T17:10Z — T8 started. The v7 mix and the T2 dev set were converted to the d3 harness's row format
  (d3_pool 1,526,405 rows, sha `bb826320…`, 438,188 teacher-labelled; d3_dev 22,934 rows, sha `5d8d9361…`), the
  driver (CE + teacher KL + KL-to-base on `v7_c*`, dev evaluations, checkpoints, resume) passed a 5-step smoke test,
  and the 13,992-step run is launching. No results yet.
- 2026-11-13T16:45Z — **T7 DONE.** The blend table (operator request) is written above: all four scales are
  eligible, and s = 1.00 is chosen with dev S 0.810521 (no interpolation gain); positional health passes in all
  three orders for every scale including the base. The chosen blend is calibrated on fast dev (choice 1.206404,
  noul 1.351652, score 1.135541) and exported to `outputs/osler_v7/T7/chosen/`. SELF_AUDIT PASS (59 checks);
  CLAIMS C032–C035.
- 2026-11-13T14:47Z — **T6 DONE.** The 4B arm finished 13,992 steps (14.0 h over the interrupted+resumed run; the
  step-2,500 allocator mitigation is recorded). Dev: S 0.797548 at 6,996 and **0.811190 at 13,992** (MR 0.887582,
  MIMIC 0.754032, CG 0.791955, knowledge 0.763571; positional PASS in all three orders). Versus osler_v6 G1s0 at the
  same step (S 0.825891), the v7 arm is ~1.5 points lower, almost all on MR. Temperatures choice 1.206404,
  noul 1.351652, score 1.135541. SELF_AUDIT PASS (26 checks); dose: first-200 distillation share 0.352292, KL-to-base
  active on 40.5 % of steps (v7_c is 4.6 % of the mix), run-mean KL 0.023079. CLAIMS C029–C031. T7 selection
  (base + four scales on the 22,934-item dev set) started at 14:50Z.
- 2026-11-13T03:05Z — T6 operational note: the arm reached step 2,525 with the CUDA allocator grown to 95.7 of 97.9 GB
  (OOM risk). It was stopped at the step-2,500 full checkpoint and resumed with
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (the v6 C040 precedent; an allocator setting, not a design
  change). After the resume GPU memory was 57.3 GB and training continued from step 2,500. The first-200-step
  dose rows are preserved; the resumed run is the statistical replica v6 accepted on the same setting.
- 2026-11-13T00:47Z — **T5 DONE.** The 27B labelled the whole 69,967-prompt pool: 3.72 h, 5.22 items/s,
  4,602 input tokens/s, 0 errors; join max probability-sum error 1.11e-16, 8 constant rows; SELF_AUDIT PASS
  (14 checks); CLAIMS C023–C027. The v7 mix is built: 1,526,405 rows (v6 1,456,438 + C1–C4 69,967), 438,188
  labelled (28.7 %), sha `b8b97031…`; the v6 mix itself carries 22,108 duplicate rows / 5,232 ids, a pre-existing
  property recorded and kept as trained (C028). Family shares: C0 95.4 %, C1 1.31 %, C2 0.66 %, C3 1.31 %,
  C4 1.31 % — reported against the 25 % cap, not reweighted. The 4B arm started (13,992 steps, detached,
  log `outputs/osler_v7/T6/logs/arm.log`).
- 2026-11-12T21:05Z — T6 driver dry-run on a synthetic 25-row mix (20 v6 + 5 C1 rows) passed: the dev file partitions
  into 14,597 fast / 7,508 CG / 829 knowledge items with 1,000 positional items, and conversion/composition work
  before the model. The smoke files were removed. T5 labelling continues (3,072/69,967 at 5.25 items/s,
  ETA ~3.4 h).
- 2026-11-12T21:00Z — T5 labelling is running on the GPU (reference_batched path, 512 prompts in 95.3 s =
  5.375 items/s, 4,520 input tokens/s; ETA ~3.6 h for the 69,967 prompts). T6 code prepared while it runs: the
  trainer/config gain `kl_include_template_prefixes`, so the KL-to-base penalty applies only to the v7_c* pool;
  `configs/osler_v7/arm_4b.yaml` sets kl_to_base 0.1, the v6-frozen distill weight 3.243598 with the 3-labelled
  quota, per-item weighting and no double augmentation; `scripts/osler_v7/t6_build_mix.py` will append the
  labelled C1-C4 pool to the v6 mix once T5 joins. The v7 arm driver is still to be written.
- 2026-11-12T20:40Z — **T4 DONE.** Nine cells, 9,404.4 s on one GPU. Trained cell (1,1): item 0.788872, macro
  0.807105, two-letter 0.651. Base readout + trained LoRA (1,0): item 0.772739, two-letter 0.519 — the base readout
  does not recover the two-letter loss, so the damage sits in the LoRA, not the readout; the 0.75-scale cells give
  the best macro (0.808647 at readout 1, 0.807665 at readout 0.5) and recover two-letter to 0.696. Diagnostic only:
  no selection, no gate. SELF_AUDIT PASS (139 checks); CLAIMS C020–C022.
- 2026-11-12T18:40Z — T5 pool built on CPU while the T4 GPU job runs: C1 20,000 rows (sha `a27a26f4…`),
  C2 10,000 rows (sha `8f5cfe14…`, 5,000 yes / 5,000 no), C3 20,000 rows (sha `5e2e50ab…`; ATC and procedure
  descriptions are NOT MEASURED, no dictionary on disk), C4 20,000 rows (sha `fcde88fa…`); merged pool 69,967
  rows (sha `1767bc85…`), 0 text overlap with the T2 dev/confirmation sets. `t5_label_pool.py` is ready; its
  GPU parity + labelling run is queued behind T4 (at the Y5 rate, ~3.2 h for the 70k prompts).
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
- **T6 distillation weight:** ADVISORY §3 writes the loss as CE + λ_KL·KL-to-base only, while §2 attaches pplx-27B
  teacher labels to C1–C4. The v7 arm config reuses osler_v6's frozen distillation weight 3.243598 with the
  3-labelled quota (the "v6 mix + C1–C4" reading). If another teacher weight is intended, it must be named
  before T6 starts.
- **KL direction:** ADVISORY §3 writes KL(student ‖ zero-shot base); the existing D41 `kl_to_base` term computes
  KL(base ‖ model) (base as the distillation target), which is where the proposed λ = 0.1 comes from (osler_v3).
  T6 uses the existing term and records this interpretation; naming the opposite direction would need a new loss.
- **Family-share cap:** ADVISORY §2 caps each family's loss-gradient share at 25 %. The mix manifest reports the
  measured row shares (C0 is ~95 % of rows); no per-family reweighting is applied because the plan does not say
  which grouping the cap refers to. Clarify before T6 if a reweight is required.

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
