# O5 — readout checks, export and fp32 numerics (osler_v0)

Status: **DONE with a recorded failure.** The pre-registered run (attempt 4) passes 10 of 11 checks. The one failure is the
merged-adapter export, recorded as `READOUT_FAIL` under the as-run configuration and not edited. The separate-adapter
export passes in every run. The failure traces to the precision of the fp32 kernel path, not to the merge arithmetic (§3).
Two decisions are with the operator (STATE §6, Question 14).

Every number below has a CLAIMS row (`loops/osler_v0/CLAIMS.md`, O094–O121) with its recompute command. Raw outputs are in
`outputs/osler_v0/O5/` (gitignored). Model: `Qwen/Qwen3.5-4B` at revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, fp32,
50 items from `data/train/student_v1/dev.jsonl` (147 offered options), GPU NVIDIA RTX PRO 6000 Blackwell.

## 1. The pre-registered run (attempt 4): the verdict

| check | measured | tolerance | result | claims |
|---|---|---|---|---|
| initialised option-code logits vs zero-shot letter logits (50 items) | 5.72e-06 | 1e-4 | pass | O095 |
| initialised option-code probabilities vs letter readout | 1.43e-06 | 1e-4 | pass | O096 |
| causal path unchanged with the non-causal flag off (3 items) | 0 | 1e-5 | pass | O097 |
| non-causal flag on: change at earlier positions (3 items) | 19.07 | at least 1e-4 | pass | O098 |
| padding invariance, causal, option-code (4 pairs) | 2.74e-04 | 1e-3 | pass | O099 |
| padding invariance, bidirectional, option-code (4 pairs) | 2.83e-04 | 1e-3 | pass | O100 |
| padding invariance, pointer (4 pairs) | 2.75e-05 | 1e-3 | pass | O101 |
| greedy generation, adapter off, byte-identical to the base (2 prompts) | True | exact | pass | O102 |
| control: adapter on changes next-token logits | 8.48 | at least 1e-4 | pass | O103 |
| exported causal LM vs native, adapter separate (50 items) | 2.15e-06 | 1e-3 | pass | O104 |
| exported causal LM vs native, adapter **merged** (50 items) | **1.04e-02** | 1e-3 | **FAIL** | O105 |

Verdict: 10 of 11 pass (O094). The merged-adapter artefact is `READOUT_FAIL` under this configuration.

Provenance note: attempt 4 ran before the driver gained its `--precision` and `--seed` options, and its adapter's `lora_A`
was drawn from the unseeded global RNG (§3). The verdict is recorded as measured. No check, tolerance or ordering was
changed by the later edits (the diff is the two options and the lazy imports that let the precision setting apply first).

## 2. Seeded re-run and the additional precision settings

The same configuration was re-run with a recorded seed (seed 0, so the adapter is reproducible). The two additional settings
are run on that same seed. They are labelled as additional analyses. They are **not** the verdict.

| run | role | checks passed | merged export | bidirectional padding | claims |
|---|---|---|---|---|---|
| attempt 4 | verdict (unseeded draw) | 10 of 11 | 1.04e-02 FAIL | 2.83e-04 pass | O094, O100, O105 |
| as-run, seed 0 | seeded re-run of the verdict configuration | 9 of 11 | 4.17e-03 FAIL | 1.03e-03 FAIL (tolerance 1e-3) | O106–O109 |
| IEEE kernel precision, seed 0 | additional analysis | 11 of 11 | 5.05e-05 pass | 1.19e-05 pass | O112–O113 |
| reference (kernels blocked), seed 0 | additional analysis | 11 of 11 | 1.01e-04 pass | 1.72e-05 pass | O110–O111 |

The as-run merged export fails in both adapter draws, and it is roughly 4 to 10 times the tolerance. The bidirectional
padding check fails in seed 0, just over its tolerance, and passes in attempt 4. So the as-run padding result depends on the
adapter draw and sits at the tolerance. Under the IEEE and reference settings every check passes with a wide margin.

## 3. Mechanism: the fp32 kernel path (diagnostic)

The model's linear-attention layers run flash-linear-attention's Triton kernels (fla 0.5.2, triton 3.8.0) and causal-conv1d
1.7.0 (O122 records the kernel path and package versions). On this GPU (Blackwell, sm_120) fla's chunked triangular solve requests TF32, and
Triton's default precision for fp32 dot products is TF32. The default "fp32" forward is therefore not full fp32. The
controls below use one synthetic item (no benchmark, training or teacher text) of 128 positions, with the same adapter
draw (seed 0) under each setting:

| setting | separate forward vs pure fp32 reference (last hidden, max abs) | merged vs separate gap | change under a 1e-7 relative weight perturbation | claims |
|---|---|---|---|---|
| as-run (default kernels) | **0.2385** (about 0.45 % of max abs h = 52.6) | 0.0416 (seed 0); 0.0489 and 0.0549 (seeds 1, 2) | 0.0435 (seeds 0–2: 0.0435, 0.0533, 0.0443) | O114, O116, O120 |
| IEEE dots (diagnostic override) | 3.1e-04 | 3.6e-04 (seed 0); 3.5e-04 (seed 1) | 2.2e-04 (seeds 0, 1: 2.2e-04, 3.2e-04) | O115, O116, O120 |
| reference (kernels blocked) | — (it is the reference) | 4.5e-04 (seed 0) | 2.4e-04 (seed 0) | O115, O116 |

Readings:

- The merge arithmetic is not the cause. With fp32-faithful dots (IEEE or reference) the merged and separate forwards agree
  to about 4e-04 on values near 52. The large as-run merge gap is the as-run kernel path's sensitivity: a 1e-7 relative
  perturbation of the weights moves its output by about as much as the merge does (O116 against O114).
- The as-run kernel path is not a faithful fp32 computation of this model. Its separate forward differs from the pure fp32
  reference by about 0.24 on the last hidden state. The IEEE-dot path differs from the reference by 3e-04 (O120).
- Reproducibility. Within one process, repeated forwards are identical (O118). With a fixed seed, two processes are
  bit-identical under both the as-run and IEEE settings (O119). The large cross-process differences in the unseeded
  records (separate forward 35.2 and 25.8 apart) came from the unseeded adapter draw, not from the kernels (O121). The driver
  now seeds the RNG before the adapter is built, and every seeded record stores its seed.
- What was not measured. The ADVISORY's training and evaluation dtype for O6–O10 is bf16. The bf16 path was not measured,
  and neither was whether fla's solve request affects bf16 operands. The IEEE setting is a diagnostic override of a module
  constant in fla (`SOLVE_TRIL_DOT_PRECISION`) plus Triton's `TRITON_F32_DEFAULT`, not a supported configuration.
- Earlier scratch runs (not recorded in this loop) pointed the same way. The numbers above are the recorded ones and are the
  only ones to cite.

## 4. Consequences and open decisions

1. **The verdict stands as measured.** The pre-registered configuration fails the merged-adapter export check (O094, O105). It
   is not re-labelled with the additional settings.
2. **The separate-adapter export passes** in every run (2.15e-06 in attempt 4; 1.91e-06 seeded; O104, O109). The arms train
   and evaluate with separate adapters, so O6–O8 are not affected by the merged-artefact failure.
3. **The merged artefact must not be used** (including the quantised ≤ 4 GB export in O11) until Question 14 is decided. Its
   status is `READOUT_FAIL` under the as-run configuration.
4. **Operator decisions (STATE §6):**
   - Question 14: which numerical setting is "fp32" for the O5 verdict. Options: (a) keep the as-run setting (current; the
     merged export fails); (b) approve a new pre-registered O5 run under IEEE dots, with its own record (all 11 pass on seed 0
     as an additional analysis, not a verdict); (c) use the reference path for the checks.
   - Whether O9–O11 scoring uses IEEE dots. As-run padding results show probabilities that depend on batch composition at
     about 1e-3 (O108), which matters for paired comparisons at that scale.
   - Question 15: whether the bf16 training and evaluation path needs its own TF32 control before the arms run. Recommendation:
     proceed with the pre-registered bf16 recipe, since all arms share the same kernels, and measure the bf16 path before
     any bf16 number is reported as final.

## 5. Code and reproduction

- Library: `src/meddecide/precision.py` (precision settings, kernel resolution, weight perturbation helpers);
  `src/meddecide/train/osler_arm.py` (`dev_prediction_rows`); `src/meddecide/train/trainer.py` (`Trainer.score`).
- Drivers: `scripts/osler/o5_checks.py` (`--precision`, `--seed`; attempt 4 is the verdict record), `scripts/osler/o5_kernel_control.py`
  (`--precision`, `--tag`, `--seed`), `scripts/osler/o5_summary.py` (all tables in this report).
- Tests: `tests/test_precision_o5.py`, `tests/test_dev_predictions_o6.py`, `tests/test_arm_run_o6.py` (extended).
- Reproduce every table: `uv run --frozen python scripts/osler/o5_summary.py`, then the recompute commands in CLAIMS O094–O121.
