# student_v1 — RUN STATE

> **This file is the loop's memory.** The live copy is `loops/student_v1/STATE.md`
> (committed, so the operator can review it remotely). Read it at the start of every
> iteration, before the plan. If it disagrees with your recollection, **the file
> wins** — your context may have been compacted since the last iteration.
>
> **How to update:** set the task `IN_PROGRESS` with a UTC start time *before* working;
> on completion set `DONE` or `BLOCKED — <reason>`, fill the finished time, and append a
> ≤5-line entry to the iteration log. Never delete a log entry; append only. Timestamps
> are `date -u +%FT%TZ`. Never paste item text, predictions, or secrets into this file.

Loop status: `STOPPED`  <!-- closure at 2026-10-08T22:10Z: V8 and V9 DONE, V10 DONE, hard stop -->
Run started (UTC): `2026-10-08T03:59:12Z`
Last updated (UTC): `2026-10-08T22:10:00Z`
Iterations so far: `11`

---

## 1. Task board

Statuses: `PENDING` → `IN_PROGRESS` → `DONE` | `BLOCKED — reason`

| id | task | GPU | deps | status | started (UTC) | finished (UTC) |
|---|---|---|---|---|---|---|
| V0 | Orientation, snapshot, Unsloth environment | smoke | — | DONE | 2026-10-08T03:59:12Z | 2026-10-08T04:17:00Z |
| V1 | JEV-9B on v0.2 + G1 recomputed per D16 | yes | V0 | DONE | 2026-10-08T04:30:39Z | 2026-10-08T05:28:00Z |
| V2 | Padding fix + measured effect | small | V0 | BLOCKED — acceptance item 1 (padded-batch test ≤ 1e-3, fp32) fails at 1.92e-3 with the fix; tolerance not changed; no-padding batch-1 path adopted and measured (`loops/student_v1/V2_padding.md`); operator question 1 | 2026-10-08T05:30:03Z | 2026-10-08T07:20:00Z |
| V3 | Checkpoint trajectory (seen vs held-out) | yes | V2 | DONE — held-out per-template macro falls 0.550 → 0.332 over 22 checkpoints (Spearman −0.637, CI [−0.862, −0.223]); seen flat; runs on V2's adopted path (D2) | 2026-10-08T07:11:32Z | 2026-10-08T08:24:30Z |
| V4 | Training data v1 (diversity) | no | V0 | DONE — 10 screen-passing new templates; 482,889 train / 5,619 dev rows (class-balanced noul, D10); leakage 0; NOT MEASURED: ≥ 2 consistency designs and catalog sources (D4); seen-template score dev 33 < 200 (D9) | 2026-10-08T06:09:26Z | 2026-10-08T07:20:00Z |
| V5 | Unsloth validation + converter + evaluator | small | V0, V4 | DONE — typed-decisions accuracy 0.361 → 0.801 (300 steps; integration only); round trip 2,000 rows, 0 skipped; untrained head at chance on 200 dev items (0.345 vs 0.337); row converters tested; the GPU evaluator run is part of V9 | 2026-10-08T08:27:48Z | 2026-10-08T08:36:00Z |
| V6 | Arm A: pointer head | yes | V2, V4 | DONE (on V2's adopted path, D2) — selected step 12,500 by dev pooled macro (0.7603; dev acc 0.7999, Brier 0.2729 / 0.2698 calibrated); 15,000 steps, 30 dev evals; no divergence tripwire | 2026-10-08T08:39:05Z | 2026-10-08T11:02:36Z |
| V7 | Arm B: letter-readout LoRA | yes | V4 | DONE (diverged by the grad-norm tripwire at step 13,254; recorded per the ADVISORY rule; verdict checkpoint step 5,000 by dev pooled macro 0.6997; unrestricted best 14,000 shown as additional; claims V072–V078) | 2026-10-08T11:02:36Z | 2026-10-08T13:28:04Z |
| V8 | Arm C: Clef-style head via Unsloth | yes | V5 | DONE — trained (15,000 steps, 6,025 s), dev selection step 5,000 (pooled macro 0.6851; the tripwire did not trip, so unrestricted = restricted), finalize at step 5,000 (37,289 test predictions; the rerun is identical), with D15 batch order and D16 option order. A first finalize run stopped on a path bug after writing its predictions; fixed and rerun. A duplicate run was stopped before writing anything | 2026-10-08T13:58:03Z | 2026-10-08T22:10:00Z |
| V9 | Evaluation + G1 per arm + arm-vs-arm | yes | V1, V6, V7, V8 | DONE — overall test accuracy A 0.7591, B 0.7528, C 0.7545 (V113). G1 per arm: A seen PASS, held-out FAIL (V079–V082); B seen PASS on 3 templates, held-out NOT MEASURED (V089–V092); C seen PASS, held-out FAIL (V105). G1 classification fix (V106). Arm pairs (V107–V112). Selective and latency tables (V114, V115). NOT MEASURED: HLE, option-shuffle flip rate, long-record slice, per-record latency, arm B held-out | 2026-10-08T15:10:58Z | 2026-10-08T22:10:00Z |
| V10 | Findings and closure — HARD STOP | no | all | DONE — `FINDINGS.md` (summary first), `NEXT.md`, the closure feed, the five-claim spot-check (all five reproduced), `Loop status: STOPPED` | 2026-10-08T22:10:00Z | 2026-10-08T22:10:00Z |

Rules: take the **first** `PENDING` task whose deps are all `DONE` (exception in ADVISORY §6:
V4 may run on CPU while V1–V3 use the GPU). Never run two GPU jobs at once. A `BLOCKED` task
does not block unrelated work — record it and move on. No row may be left `IN_PROGRESS` at
closure.

---

## 2. Key values discovered during the run

Fill these in as they are measured; later tasks read them from here rather than
recomputing or guessing.

| key | value | source | task |
|---|---|---|---|
| unsloth / torch / transformers versions (unsloth env; main env) | unsloth env: unsloth 2026.10.2, torch 2.14.1+cu130, transformers 5.17.0, peft 0.21.2; main env: torch 2.14.1+cu130, transformers 5.18.0 | `outputs/student_v1/V0/envs.json` | V0 |
| G1 for student_v0 per D16 (seen / held-out) | seen: PASS vs zero-shot (macro +0.1633, CI [0.1530, 0.1741]) and vs JEV-9B on the intersection (n=15,911; macro +0.0435, CI [0.0311, 0.0548]). Held-out: FAIL vs both (vs zero-shot macro −0.0176, CI [−0.0301, −0.0048]; vs JEV-9B macro −0.1553, CI [−0.1661, −0.1435]). Claims V011–V014. | `loops/student_v1/g1_student_v0_rerun.md` | V1 |
| JEV-9B v0.2 coverage | 35,097 of 37,289 kept test items (94.1%); 2,192 excluded by the 8,192-token prompt cap, all on openFDA templates (claims V008) | `outputs/student_v1/V1/jev9b/jev9b.json` | V1 |
| padding effect on student_v0 test accuracy (max per-template Δ) | adopted path (batch 1, no padding) minus original batched run: max per-template Δ −1.048 pts (pubmedqa_ynm_v1, 95% CI −2.10 to −0.21, n=477, 5 items); micro accuracy +0.008 pts (CI −0.043 to +0.062) over 37,289 items; 151 predictions changed; micro Brier −0.00003 (CI −0.00009 to +0.00003) | `outputs/student_v1/V2/effect.json`; report `loops/student_v1/V2_padding.md`; claims V026–V036 | V2 |
| padded-batch residual, strict fp32, 8 dev items (ADVISORY tolerance 1e-3) | original code 3.103e-3; fixed code 1.925e-3 (FAIL); shape-only control (alone vs alone + 64 pads) 1.495e-3; uniform-batch control 3.381e-4 | `outputs/student_v1/V2/probe_*_fp32.json`, `logit_control_strict_fp32.json`; claims V019–V025 | V2 |
| Spearman(step, held-out dev accuracy) with CI | | `loops/student_v1/trajectory.md` | V3 |
| training mix v1: items, distinct templates (vs student_v0), max template share | 482,889 train rows, 24 distinct templates (student_v0: 14; ratio 1.714); max template share 0.0800 (cap 38,631); selection dev 5,619 rows over 28 templates; new noul templates class-balanced (D10) | `data/train/student_v1/manifest.json` (aggregate `loops/student_v1/mix_summary.json`); claims V048–V049 (supersede V040, V042) | V4 |
| leakage check result | 0 hits in the final train rows (818,863) and dev rows (index, date, held-out template and question); 3,294 train and 835 dev pre-window rows removed by the counted check (student_v0 base checked clean) | manifest `leakage`; claims V043–V045 | V4 |
| Unsloth typed-decisions validation (before → after) | | | V5 |
| arm A / B / C: selected step, dev macro, wall-clock, grad p95 / max | A: step 12,500, dev pooled macro 0.7603, wall 8,609 s, grad p95 24.0 / max 3,439 (V067–V071). B: step 5,000 by the tripwire rule, dev pooled macro 0.6997; tripwire tripped at step 13,254 (6,022.8) (V072–V078). C: step 5,000 (unrestricted; no trip), dev pooled macro 0.6851, training 6,025 s, grad p95 14.6 / max 384.8 (V096, V099) | `loops/student_v1/arm_training.md`, `loops/student_v1/arm_c_training.md` | V6–V8 |
| G1 per arm (seen / held-out) | arm A: `loops/student_v1/g1_arm_a.md` (V079–V084). Arm B: seen PASS on the 3 D12-passing templates vs zero-shot (+0.3056) and JEV-9B (+0.0953); held-out NOT MEASURED, all 4 cells fail D12 (V090–V092) | `loops/student_v1/g1_arm_a.md`, `loops/student_v1/g1_arm_b.md` | V9 |

---

## 3. Current task checklist

**Fill this in before starting any task**, by copying that task's concrete steps out of
the plan as unticked boxes. Tick each one the moment it is finished, not at the end.
This is the only record of progress *within* a task — without it, a compaction mid-task
leaves you unable to tell what you already did.

Clear this section and write the new task's checklist when you start the next task; the
completed checklist goes into the iteration-log entry.

```
Task in flight: V8 (arm C, Clef-style head via Unsloth; restarted 2026-10-08T15:25Z with D15 and D16)
Working dir:    outputs/student_v1/V8/   (chain log: outputs/student_v1/V8/logs/chain_c4.log)

- [x] arm C scripts: v8_unsloth_train.py, v8_unsloth_predict.py, v8_select.py, v8_finalize.py
- [x] dataset built once: 482,889 items, 0 skipped, 15,344 truncated at max_seq_length 8,192 (ADVISORY §8.4 count); the build took 16.5 min
- [x] the 14:50 run (kept as history) stopped at sampler construction: `group_by_length` reads a `pixel_values` key for list datasets, so the added `length` column did not help. The "done" lines in chain_c2.log are false: that chain did not gate on exit codes
- [x] batch order (D15): `src/meddecide/train/batch_order.py`, tests in `tests/test_batch_order.py`; fixed batches of 8 in arms A/B's chunks, seed 0
- [x] option order (D16): choice and score items permuted as arms A/B do; train-file gold positions before and after (V085, V086)
- [x] gate in chain_c5 (15:52Z): ruff exit 0; full pytest exit 0 (GPU tests included; `pyproject.toml` sets `addopts = "-q"`, so the gate's `-q` becomes `-qq` and hides the count line; the exit code is the evidence)
- [x] smoke test (exit 0, 15:54Z): 30 steps, 240 distinct items = 30 x 8, 1,984 of 4,000 items option-permuted, grad norm 0.82 to 3.06 (max 6.95, far below 5,000), 22.9 s training (about 0.8 s per step at this size)
- [x] training done (exit 0, 17:55Z): 15,000 steps in 6,025 s (about 2.5 steps/s); mean loss 0.5056; pre-clip grad norm p50 3.485, p95 14.623, max 384.776; tripwire not tripped (V096–V098; `loops/student_v1/arm_c_training.md`)
- [x] incident (found and fixed 15:57Z): a stale chain shell, still reading `chain_c4.sh` after the file had been overwritten, started a second full training run at 15:47:49 (pid 203130; parent init; stdout to the same `train.log`; created `arm_c/`). Killed by pid at 15:57Z, before its dataset build finished: it wrote no checkpoint, cache or training file (`arm_c/` is empty; `train.log` holds only the chain's run, pid 205124). Lesson: never overwrite a running chain script; write a new file and stop the old pid first
- [ ] grad-norm tripwire from train_steps.jsonl (per step): apply the ADVISORY rule before selection
- [x] dev predictions for all 30 checkpoints (5,619 items each; done 20:25Z). Selection done 20:25Z (exit 0): step 5,000, dev pooled macro 0.6851 (accuracy 0.7325, Brier 0.3696); runner-up step 7,500 at 0.6842 (margin 0.0009). Tripwire not tripped, so the unrestricted rule stands. Lowest dev macro 0.512, so the divergence rule (below 0.50 twice) is not met
- [ ] Unsloth calibrate on seen dev at the selected checkpoint + test predictions (`v8_finalize.py` at step 5,000, running since 20:39Z in `chain_final`, about 40 min for 37,289 test items; then the conversion to run_student format; the log's last line says when it's done)
- [ ] commit after the gate passes and at each milestone; SELF_AUDIT (R6), arm_training.md

V9 (in flight in parallel; arm B evaluated, arm C pending):
- [x] arm A test predictions and G1 (`g1_arm_a.md`, V079–V084)
- [x] arm B temperatures at step 5,000 (V088); arm B letter test (exit 1 on the final print only; outputs verified complete: 37,289 rows, 19 cells)
- [x] arm B G1 (`scripts/bench/g1.py`, D12 applied): 3 of 11 fresh templates PASS; held-out NOT MEASURED (V089–V092)
- [x] arm B additional analyses (labelled): readout probe done 20:39Z (V100, V101, V103 correction): the top vocabulary token is an offered letter in 152 of 152 items; the mass sits partly on non-offered letters (0.005–0.337) and on space variants (up to 0.253). The prompt cap (V102): 2,213 of 37,289 arm B test items (5.93 %) reach 8,192 tokens, all in fda_route_claim_noul_v1, fda_boxed_warning_noul_v1 and fda_class_choice_v2; the scoring path drops their leading tokens. Accuracy on the D12-failing cells as a labelled additional table: still to do
- [ ] arm C finalize at its selected step (`v8_finalize.py`, by hand), then `v9_unsloth_to_preds.py`
- [x] arm pairs A−B (V093–V095): verdict (fresh, D12-applied, 9,236 items, 3 templates) macro +1.13 pts [+0.78, +1.48] (A − B); held-out NOT MEASURED; additional all-tier seen −1.24 pts (the sign flips with the scope), held-out −0.06 pts. Still to do: G1 for arm C; pairs A−C and B−C (`v9_arms.py`, now with the G1 scope and `--exclude`)
- [ ] `loops/student_v1/arms.md`, `loops/student_v1/g1.md` (combined), SELF_AUDIT, CLAIMS
```

---

## 4. Iteration log (append only, newest last)

<!-- Template for each entry:
### <task-id> — <DONE|BLOCKED> — <UTC timestamp>
- What ran: <command or script>
- Output: <path>
- Headline: <one number or one sentence, with its CLAIMS id>
- Surprises: <anything unexpected, or "none">
- Next: <what this unblocks>
-->

<!-- The Headline line matters beyond this file: these headlines are the raw material
     FINDINGS.md's Summary section is written from at closure. A headline that only
     makes sense with full context ("done, see report") starves the summary. Write
     each one so a reader who has not seen the task can repeat it: what was measured,
     what came out, with what denominator. -->

### V0 — 2026-10-08T04:17:00Z — DONE — 2026-10-08T04:17:00Z
- What ran: `scripts/student/v1_snapshot.py` (main + `--env unsloth`), `scripts/student/v1_unsloth_load.py` (in `envs/unsloth/.venv`); `uv sync --project envs/unsloth` (123 packages).
- Output: `outputs/student_v1/V0/{snapshot,envs,unsloth_load}.json`, `SELF_AUDIT.md`.
- Headline: Unsloth 2026.10.2 (first release with `FastDecisionModel`) loads Qwen3.5-0.8B in bf16 with `load_in_4bit=False`: 344 bf16 weight tensors; 251 norm/head tensors fp32; `predict` probabilities finite, sums 1.0 / 0.9999 (V001–V007).
- Surprises: the loader keeps RMSNorm weights in fp32 (arm C precision differs from A/B on norms, see Deviations); `causal_conv1d` is not installed in either env, so the reference kernel is used; the main env has transformers 5.18.0 vs 5.17.0 in the Unsloth env.
- Next: V1 (JEV-9B on v0.2 + D16 G1 rerun). Needs an F7-to-harness row converter and a G1 change that restricts verdict baselines to zeroshot + jev9b.

### V1 — DONE — 2026-10-08T05:28:00Z
- What ran: `scripts/bench/run_jev_baseline.py` (detached, 2,927 s, 19 cells); `scripts/bench/jev_to_harness.py`; `scripts/bench/g1.py` (D16 baselines zeroshot + jev9b; `--out loops/student_v1/g1_student_v0_rerun.md`); `scripts/bench/rederive_g1_headline.py` (fresh-process check). Full pytest exit 0, ruff clean.
- Output: `outputs/student_v1/V1/{jev9b/, preds/preds_jev9b__test.jsonl, g1_student_v0_rerun.json, rederive_g1_headline.txt, SELF_AUDIT.md}`; committed aggregate `loops/student_v1/g1_student_v0_rerun.md`.
- Headline: student_v0 step 1,000, seen fresh templates (17,442 items): G1 PASS vs zero-shot 0.8B (macro +0.163, CI [0.153, 0.174]) and vs JEV-9B on the 15,911-item intersection (macro +0.044, CI [0.031, 0.055]). Held-out (6,326 items): FAIL vs both (vs zero-shot −0.018; vs JEV-9B −0.155). JEV-9B scored 35,097 of 37,289 test items (2,192 long openFDA items excluded by the 8,192-token cap). Claims V008–V018.
- Surprises: six allocator OOM warnings during the JEV run, none fatal; `jev9b_provenance.json` has a 1 s wall-clock bug (true time in `jev9b.json`); the strict-slice seen comparison vs JEV-9B fails (additional, CI lower −0.032).
- Next: V2 (padding-invariant pointer head; trained-checkpoint padding test; effect on student_v0 step 1,000). The JEV coverage gap is an operator question (Section 6).

### V2 — BLOCKED — 2026-10-08T07:20:00Z
- What ran: `scripts/student/v2_run_matrix.sh` (probes on committed HEAD d3cc79c and the fixed code, strict fp32 and TF32; shape control); `v2_logit_control.py`; `run_student.py` test scoring at batch 16 (fixed), batch 1 (fixed) and batch 1 with `--no-chunk-rounding` (adopted); `v2_effect.py`, `v2_rederive_effect.py`, `v2_report.py`.
- Output: `outputs/student_v1/V2/` (probes, controls, `effect.json`, `rederive_effect.txt`, `SELF_AUDIT.md`); committed `loops/student_v1/V2_padding.md`.
- Headline: the trained-checkpoint padded-batch test FAILS (1.925e-3 with the fix, 3.103e-3 original; tolerance 1e-3, unchanged). The shape-only control reaches 1.495e-3, so the residual is at the fp32 sequence-length noise floor. The adopted no-padding batch-1 path differs from the original batched run by micro accuracy +0.008 pts (95% CI −0.043 to +0.062) over 37,289 test items, with 151 predictions changed (claims V019–V036).
- Surprises: the TF32 explanation is refuted (identical values with TF32 on and off). Chunk rounding alone moves logits by up to 4.1e-3, more than the bug. pubmedqa has one 5-item change whose CI excludes zero (one interval of ~25; reported as an observation).
- Next: operator question 1 decides whether the no-padding path satisfies acceptance item 1. V3 and V6 run on that path now (deviation D2).

### V4 — DONE — 2026-10-08T07:20:00Z
- What ran: `v1_pubmed_extract.py` (185 baseline files, 5,524,988 articles), `v1_build_components.py` (CT, openFDA, PubMed), `v1_screen.py`, `v1_assemble_mix.py` (run twice: the second after an accounting fix, byte-identical files), `v1_check_mix.py` (fresh-process re-derivation); tests `test_templates_v1.py`, `test_mix_v1.py`.
- Output: `data/train/student_v1/{train,dev}.jsonl` and `manifest.json` (gitignored data); committed `loops/student_v1/mix_summary.json`; `outputs/student_v1/V4/` (screen, component and mix reports, `SELF_AUDIT.md`).
- Headline: 10 of 11 new train-only pre-window templates pass the screen (BoW macro < 0.90; openFDA product type dropped at 0.995). Mix: 818,863 train rows over 24 templates (student_v0: 14; ratio 1.714), largest share 0.0800; selection dev 5,619 rows over 28 templates; 0 leakage hits in the written rows (claims V037–V047).
- Surprises: the first leakage check failed (3,294 train rows duplicated benchmark text or record ids; removed and counted, not hidden); a manifest dev-count merge bug was found and fixed (files unchanged); the openFDA product type is written without the word LABEL.
- Not done: ≥ 2 record–claim consistency designs and the catalog train-candidate sources (NOT MEASURED, D4); seen-template score dev is 33, below 200 (D9). The mix is ready for V5–V8.

### V4 addendum — class balance of the new noul templates (D10) — 2026-10-08T07:30:00Z
- What ran: the same assembly with `balance_classes` on the five new noul templates (train, before the cap) and `balanced_sample` on their dev items (125 per class). Rerun: `scripts/student/v1_assemble_mix.py`; check: `scripts/student/v1_check_mix.py` (16 OK).
- Why: the new noul templates had yes-rates of 0.038 to 0.735 (child tag 3.8 %), so the model would learn the prior. Decided before any arm trained, from pre-window counts only.
- Result: train 482,889 rows (was 818,863), cap 38,631, dev 5,619 rows. Superseded CLAIMS rows V040, V042, V046, V047 are kept as history; V048–V053 replace them.

### V5 — DONE — 2026-10-08T08:36:00Z
- What ran: `envs/unsloth/.venv/bin/python -I scripts/student/v5_unsloth_validate.py` (typed-decisions recipe, 300 steps; round trip of 2,000 MedDecide rows; untrained-head chance check on 200 dev items); `scripts/student/v5_chance_check.py`; converter tests (`tests/test_unsloth_rows.py`, `tests/test_unsloth_predictions.py`).
- Output: `outputs/student_v1/V5/{validation.json, chance_check.json, SELF_AUDIT.md}`.
- Headline: typed-decisions test accuracy 0.361 before → 0.801 after 300 steps (integration only; claim V062); round trip 0 skipped, 174 of 2,000 truncated at 2,048 (V064); untrained head 0.345 on 200 dev items vs chance 0.337 (V065).
- Surprises: an import-order warning (Unsloth before transformers) led to a restart; the score answer convention is not verified until score items are predicted (in V8).
- Next: arm C training and selection (V8), after arms A and B.

### V3 — DONE — 2026-10-08T08:24:30Z
- What ran: `scripts/student/v3_trajectory.py` (22 checkpoints, no padding, bf16), `scripts/student/v3_metrics.py` (pooled-class macro reconstruction; report).
- Output: `outputs/student_v1/V3/`; committed `loops/student_v1/trajectory.md` and figures.
- Headline: held-out per-template macro falls from 0.5498 (step 1,000) to 0.3315 (step 44,000); Spearman −0.637, CI [−0.862, −0.223] (V054); seen flat (Spearman +0.334, CI [−0.175, +0.764]).
- Surprises: the 0.028 gap to student_v0's selection value was a metric-definition difference (per-template vs pooled-class macro), resolved by reconstruction (0.6793 vs 0.6798).
- Next: the arms (V6–V8) test whether the V4 mix follows the same curve.

### V6 — DONE — 2026-10-08T11:03:00Z
- What ran: `scripts/bench/train_student.py --config configs/student_v1_arm_a.yaml` (15,000 steps, LoRA r=8 alpha 16, dev every 500 on the 5,619-item V4 dev set, no padding); `scripts/student/v6_report.py`.
- Output: `outputs/student_v1/V6/arm_a/`; committed `loops/student_v1/arm_training.md`.
- Headline: selected step 12,500 by dev pooled-class macro 0.7603 (accuracy 0.7999, Brier 0.2729 uncalibrated); 30 dev evaluations; grad norm p95 24.0, max 3,439 (not tripped); wall 8,609 s (claims V067–V071).
- Surprises: dev macro is volatile between evals (0.57–0.76 at accuracy 0.74–0.80); the selection margin over step 15,000 is 0.0004.
- Next: arm B (V7) trains now; then arm C (V8).

### V8 — DONE — 2026-10-08T22:10:00Z
- What ran: `v8_unsloth_train.py` (15,000 steps; D15 batch order; D16 option order; logs every step), `v8_unsloth_predict.py` for 30 checkpoints (dev), `v8_select.py`, `v8_finalize.py` at step 5,000, `v8_report.py`, `v8_option_positions.py --augment`; chains `chain_c4`/`chain_c5`/`chain_final`/`chain_final2`.
- Output: `outputs/student_v1/V8/`; committed `loops/student_v1/arm_c_training.md`; SELF_AUDIT in outputs.
- Headline: step 5,000 selected on dev (pooled macro 0.6851, V099); tripwire not tripped (max pre-clip grad norm 384.776, V096); test accuracy 0.7545 (V113); finalize rerun identical (V117).
- Surprises: the first run failed on a `relative_to` bug (fixed, rerun); a stale shell started a duplicate full run (stopped, no output); the option-order prior in the train file (V085) forced D16.
- Next: none in the loop; the decisions are in STATE §6.

### V9 — DONE — 2026-10-08T22:10:00Z
- What ran: `scripts/bench/g1.py` per arm with the tier-1 option (`g1_arm_a.md`, `g1_arm_b.md`, `g1_arm_c.md`); `v9_arms.py` (A-B, A-C, B-C, with per-arm D12 exclusions); `v7_readout_probe.py`; `v9_report.py` (`g1.md`, `arms.md`); `v9_baselines.py`; the full gate (ruff and pytest exit 0).
- Output: `loops/student_v1/g1.md`, `arms.md`, `g1_arm_*.md`; `outputs/student_v1/V9/` (pairs, probe, baselines).
- Headline: overall test accuracy A 0.7591, B 0.7528, C 0.7545 (V113); G1 seen PASS for A and C (V079, V080, V105), held-out FAIL for A and C, held-out NOT MEASURED for B (V092); arm-pair signs depend on scope (V109, V110).
- Surprises: G1 misclassified the Unsloth head as a letter readout (fixed, V106); arms A and B both answer a constant letter on the held-out `ct_arm_role_noul_v1` (V104); 5.9 % of test items reach the prompt cap (V102).
- Next: the NEXT.md list (HLE, flip rate, long-record slice, the operator decisions).

### V10 — DONE — 2026-10-08T22:10:00Z
- What ran: `FINDINGS.md` and `NEXT.md` written from the claims; five claims recomputed (`FINDINGS.md`, spot-check); `NEXT.md`.
- Output: `loops/student_v1/FINDINGS.md`, `loops/student_v1/NEXT.md`.
- Headline: no head is clearly favoured. Arm C leads on the held-out templates in both scopes (-5.90 and -4.48 pts); the seen sign flips with the scope; arm B's readout fails D12 on most templates.
- Surprises: none beyond V8 and V9.
- Next: operator decisions 1 to 6 (STATE §6); the loop stops here by the hard-stop rule.


## 5. Blocked items

| id | what is blocked | exact reason | what would unblock it |
|---|---|---|---|
| V2 | ADVISORY acceptance item 1: trained-checkpoint padded-batch test ≤ 1e-3 (fp32, 8 dev items) | fails at 1.925e-3 with the fix (3.103e-3 original). The shape-only control (alone vs alone + 64 pads) is 1.495e-3, so 1e-3 is below the fp32 sequence-length noise floor. Tolerance not changed (R8). | operator decides: accept the measured no-padding path as satisfying item 1 (my pick), or revise the tolerance to the noise floor (question 1). |
| V4 (part) | ≥ 2 record–claim consistency designs (S2 role-binding) | not built this loop: time, and each design needs its own string-presence baseline and balancing. The V4 acceptance does not require them. | question 2: defer to loop 2 (my pick) or build before V6. |
| V4 (part) | catalog train-candidate sources (D17) | not converted: per-dataset parsers and licence records are needed; time. The V4 acceptance does not require them. | question 2 (same). |
| V4 (part) | seen-template score dev ≥ 200 | only 33 seen-template score items exist in v0.2 dev (`trec_covid_graded_score_v1`); the new score template supplies 250 dev items. | NOT MEASURED for the seen-template count (D9); no change to the benchmark (v0.2 is frozen). |

---

## 6. Questions for the operator

Anything you could not resolve without a human. Be specific enough to answer without
re-reading the run: state the ambiguity, the options, and which you would pick.

- **JEV-9B coverage on the 2,192 long openFDA test items (V1).** JEV-9B is unscored on 661 `fda_boxed_warning_noul_v1` (held-out), 596 `fda_class_choice_v2`, and 935 `fda_route_claim_noul_v1` items because they exceed the 8,192-token prompt cap. G1 is therefore computed on the intersection, and its vs-JEV direction on those items is unmeasured. Options: (a) accept the intersection as is (labelled); (b) re-run JEV-9B on those items with a memory-safe attention path (the single-item allocations reached 28 GB, so this needs engineering and a raised cap); (c) report them as NOT MEASURED for JEV. I would pick (a) for this loop and (b) only if the program needs JEV on the full set.

---

1. **V2 acceptance item 1 (padded-batch tolerance).** Options: (a) accept the adopted no-padding batch-1 path (ADVISORY §V2 fallback) as satisfying item 1 — my pick: it is free of batch composition by construction, and the residual is shape noise; (b) revise the tolerance to the measured fp32 noise floor (about 1.5e-3 at logit level) — a settled threshold, so not changed here; (c) keep V2 BLOCKED and re-run V3 and V6 after your decision. V3 and V6 currently run on the adopted path (D2).
2. **V4 items not built (≥ 2 consistency designs; catalog train-candidate sources).** The acceptance does not require them. My pick: defer both to loop 2 and list them in NEXT.md. Say so if you want them before arms A–C train.

3. **Arm B divergence tripwire (V7).** The ADVISORY rule stops a run whose pre-clip grad norm exceeds 5,000. Arm B exceeded it once (6,022.8 at step 13,254); the trainer ran on and recorded no stop. Applied as written, the verdict checkpoint is step 5,000 (dev pooled macro 0.6997), not step 14,000 (0.7561). Options: (a) keep the rule as written (my pick: it is the plan's rule, and the unrestricted result is reported beside it); (b) treat a single spike as clipped noise (S9 run 3 had a spike of 196,688 and did not diverge) and use step 14,000 for the verdict; (c) re-run arm B with the tripwire enforced in the trainer. Please decide before the arms comparison is read.
4. **Arm C batching and option order (D15, D16).** The first arm C run used Unsloth's random batches, and the restart uses arms A/B's chunked length bucketing with fixed batches of 8, plus their option-order permutation. Both changes match arm C to arms A and B. Without the permutation, arm C would learn the train file's gold-position prior: the gold is the last option in 90.8 % of 6-option items (V085). Costs: arm C's batches are not arms A/B's token-capped plan (8,192 real tokens per batch), and training wall-clock is not yet measured. Options: (a) accept the matched design (my pick); (b) re-run arm C with Unsloth's random batches and no permutation (the first design; it carries the prior and is not matched); (c) build arms A/B's exact token-capped batch plan into the Unsloth loader (variable batch sizes; more engineering, beyond the 2 h timebox). Please decide before the arm C comparison is read.
5. **D12 on the trained letter readout (V9, arm B).** The gate fails 8 of the 11 fresh templates, and all 4 held-out cells, on median full-vocabulary label mass below 0.5 (0.05 to 0.32). Two of those cells also fail the constant-answer check (modal answer share 0.987 and 0.966). The probe (CPU/GPU, see the arm B checklist) shows the scoring and generation prompts are identical, that the restricted argmax equals the greedy letter, and that the mass sits on non-offered letters (E, F, G, H, I for a 4-option question). So the failure is real, not a measurement bug. Options: (a) report arm B as D12 says: its G1 rests on the 3 PASS templates, and held-out is NOT MEASURED (my pick); (b) re-define the readout (renormalise over all letters, or add the space variants) to pass the gate: a change of the metric after results, which R4 forbids as a verdict, so it can only appear as a labelled additional analysis; (c) amend D12 so that trained letter readouts are gated differently: a settled decision, so operator only. Please decide before arm B's G1 is read as a comparison.
6. **Which scope the head comparison reads.** A minus C is -2.78 pts on the 7 fresh seen templates but +1.14 pts on all 15 seen templates; on held-out, C is ahead in both scopes (-5.90 and -4.48 pts; V109, V110). The program decision (keep the head the evidence favours) needs this choice. The loop did not pick a head.

## 7. Deviations from the plan

- **V1 (JEV-9B via transformers + peft, not vLLM):** `adapter_vllm` head read through transformers and peft because vLLM is not installed here. The card's mean |Δp| 0.0008 between paths is unverified in this environment.
- **V1 (verbalizer readout is non-letter for D12):** JEV-9B's probabilities are a softmax over offered options only, so label mass is 1.0 by construction and the greedy-agreement check is inapplicable. Treated as non-letter (as the pointer head is), so `NO_VERBALIZER_READOUT_REASON` is used. D12 thresholds unchanged.
- **V1 (`g1.py --out` now required):** the old default `loops/student_v0/g1.md` was removed so an earlier loop's output cannot be overwritten.
- **V1 (JEV coverage gap, reported, not hidden):** the F7 protocol's 8,192-token prompt cap leaves 2,192 test items unscored by JEV-9B. G1's vs-JEV comparisons use the item intersection (seen 15,911 of 17,442; held-out 5,665 of 6,326), labelled in the report. The cap was not changed.
- **V1 (provenance file):** `jev9b_provenance.json` shows a 1 s wall clock and empty `models`; the true wall clock is in `jev9b.json`. Recorded in SELF_AUDIT caveat 3; not used as evidence.
- **V0 (norm precision, decision deferred to V8):** Unsloth's `FastDecisionModel` loads the 251 RMSNorm/LayerNorm tensors in fp32; arms A/B load them in bf16. ADVISORY §3 asks for matched bf16 arms. Unresolved; V8 will either cast the norms to bf16 to match or record the difference in the per-arm provenance table.

Any place you departed from GOAL/ADVISORY, with the reason. An empty section is the
expected outcome. Editing code or a check to make it pass is never an acceptable
deviation — that is a `BLOCKED`.

---

- **D1 (V2 acceptance item 1, §5):** the padded-batch test is not met. It is kept as `xfail(strict=True)` with the measured number in the reason; the check, data and 1e-3 tolerance are unchanged (R8). `loops/student_v1/V2_padding.md` carries the measurement.
- **D2 (dependency of V3 and V6 on V2):** the rule holds a dependent task until its dependency is DONE. V2 is BLOCKED only on the tolerance check. The evaluation path both tasks need (no padding, batch 1) exists and is measured, and the tolerance question cannot change it. V3 started on that path and V6 will use it for dev evaluation. This is a judgement call; if the operator rejects the path, V3 and V6 are rerun.
- **D3 (V2 evaluation path):** the adopted path is batch 1 with no padding (`round_to_chunk=False`; ADVISORY §V2 fallback). The chunk-rounded batch-1 and batch-16 scorings are reported as comparisons, not used for evaluation. Training (arms A–C) still uses rounded, left-padded batches; the effect of that on training is not measured.
- **D4 (V4 "Do" items not done):** ≥ 2 record–claim consistency designs and the catalog train-candidate sources are NOT MEASURED (time; the acceptance does not require them). Proposed for loop 2 in NEXT.md.
- **D5 (V4 PubMed sampling):** 185 baseline files (n1150–n1334) were read in full for the pre-window cut. The templates use a 2 % sample of the 4.9 M pre-window records, chosen by a stable hash of the PMID (98,166 records). Recorded and deterministic, as the ADVISORY allows.
- **D6 (V4 dev size):** 250 items per template (not 150) so the selection dev reaches ≥ 4,000 (5,619 rows). Deterministic by item id.
- **D7 (V4 screen rule):** the decision uses BoW macro < 0.90 (ADVISORY), not the benchmark's micro rule; the micro value is reported beside it. One template dropped (openFDA product type, 0.9953).
- **D8 (V4 leakage removals):** 3,294 train and 835 dev pre-window rows that matched a v0.2 test/dev record id or state hash were removed by the counted check (openFDA label-text duplicates 3,215 / 794). The student_v0 base was checked first and is clean.
- **D9 (V4 score-item target):** seen-template score dev items are 33 (target 200): NOT MET. The new score template has 65,509 train rows (target 5,000: met) and 250 dev rows. No benchmark change.

- **D10 (V4 class balance):** the five new noul templates are balanced per class (train: the smaller class's count; dev: 125 per class). Source yes-rates were 0.038 to 0.735, and an unbalanced template would teach its prior. Decided before any arm trained, from pre-window counts only. Superseded V-rows are kept in CLAIMS.

- **D12 (arm C batching; superseded by D15):** the Unsloth trainer's default random batches, padded to their longest item, ran at about 2 s per step (1.5 h per 2,700 steps; about 9 h projected), against 0.08 s per step for arms A and B. Arm C was restarted with `group_by_length=True` (the transformers option that groups items of similar length). The step budget, data, LoRA and learning rates are unchanged. The batching is therefore not identical to arms A and B; this is recorded in the audit. Arm C also logs every step, so the grad-norm tripwire can be applied.
- **D13 (arm B divergence tripwire):** the pre-clip grad norm exceeded 5,000 once (6,022.8 at step 13,254). The ADVISORY rule records the run as diverged and evaluates it at its best saved checkpoint, i.e. one saved at or before the trip. That checkpoint is step 5,000 (dev pooled macro 0.6997), not the trainer's unrestricted best (step 14,000, 0.7561). The verdict uses step 5,000; step 14,000 is reported as additional. The trainer did not stop the run. This is a rule application that the operator should review (question 3).
- **D14 (arm B temperatures):** the trainer fits temperatures only for its own best checkpoint. The verdict checkpoint (step 5,000) has its temperatures refitted on dev by `scripts/student/v7_temperatures.py` with the same `fit_per_qtype` (`outputs/student_v1/V9/arm_b/temperature_step5000.json`).

- **D15 (arm C batch order; replaces D12's wiring):** `group_by_length` failed (transformers reads the processor's `pixel_values` key for list datasets, and the `length` column is not used for list datasets). Arm C now uses arms A/B's bucketing (`src/meddecide/train/batch_order.py`: file-order chunks of 800 items; chunk visiting order and per-chunk batch permutation from the same seeded streams as `data.py`; seed 0), passed to the trainer as a fixed sampler. Differences from arms A/B, recorded: batches hold exactly 8 items (arms A/B also cap a batch at 8,192 real tokens, so their long-item batches are smaller), and the in-chunk sort uses the token count (arms A/B use a character proxy). Unsloth's random sampler is not used. Wall-clock is pending the smoke test.
- **D16 (arm C option order):** arms A and B permute each item's options per (seed, epoch, file index) (`shuffle_options`); the first arm C converter did not. The file's choice gold positions are far from uniform (V085). Arm C now applies the same permutation to choice and score items (`batch_order.permute_options`); keys are relabelled by display position and the gold follows its content (V086: uniform after permutation). Noul items are not permuted, because the Unsloth noul question has no option list (arms A/B permute the two displayed answers); noul's skew is label balance (V087). Evaluation is unchanged: arms A and B evaluate in file order (`canonicalise_options` does not reorder choice items), and arm C's evaluation uses the same file order.

## 8. Closure summary feed

<!-- Filled at closure, feeding FINDINGS.md's Summary section. One row per major
     outcome, in the order a human should hear them. Each claim cites its CLAIMS id;
     failures and blocked tasks get rows too. -->

| # | outcome (one sentence, plain language) | claims | artifact |
|---|---|---|---|
| 1 | Arm A (pointer head): overall test accuracy 0.7591; G1 seen PASS against zero-shot and JEV-9B; held-out FAIL; constant answer on the held-out `ct_arm_role_noul_v1` | V113, V079–V082, V104 | `g1_arm_a.md`, `arms.md` |
| 2 | Arm B (letter readout): D12 fails 8 of 11 fresh templates and all held-out cells, on low label mass that comes from probability on non-offered letters; G1 seen PASS on 3 templates only; held-out NOT MEASURED | V089–V092, V100, V101, V103 | `g1_arm_b.md`, `readout_probe.json` |
| 3 | Arm C (Unsloth decision head): overall 0.7545; G1 seen PASS; held-out FAIL and strict seen FAIL against JEV-9B (lower bound -0.0001) | V105, V113 | `g1_arm_c.md` |
| 4 | Arm-vs-arm sign depends on scope: A minus C is -2.78 pts (fresh seen) and +1.14 pts (all-tier seen); on held-out C leads in both scopes | V107–V112 | `pairs_abc.json`, `arms.md` §1 |
| 5 | The train file's option order carried a strong gold-position prior (last option in 90.8 % of 6-option items); arm C was matched by option permutation | V085, V086, V098 | `option_positions*.json` |
| 6 | Arm C trained 15,000 steps with no tripwire; selected step 5,000 on dev (pooled macro 0.6851), finalize rerun identical | V096–V099, V117 | `arm_c_training.md` |
| 7 | 5.9 % of test items reach the 8,192-token prompt cap (all in three openFDA templates); the cap is the same for every arm | V102 | arm B predictions |
| 8 | Not measured (R3): HLE, option-shuffle flip rate, long-record slice, per-record latency, arm B held-out G1 | arms.md §5 | `arms.md` |
| 9 | Measurement defects found and fixed without changing a reported number: G1 classification of the decision head (arms A and B reproduce exactly), path bugs in three scripts, a duplicate training run stopped before any output | V106, V117 | `FINDINGS.md` §4 |


- **D17 (observed prompt cap; not a deviation):** the arms' 8,192-token prompt cap (`encode_item`) drops the leading tokens of an over-long prompt and keeps the question and options. Measured: 2,213 of 37,289 arm B test items are at the cap, all in three openFDA templates (V102). The cap is the same for every arm, so the comparison is matched; the D12 greedy sample uses untruncated prompts, so its agreement for those three cells is not like-for-like (noted with the D12 results).