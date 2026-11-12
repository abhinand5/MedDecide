# FINDINGS — osler_v6 (distillation with a dosing fix)

## Summary

1. **Gate O6 FAILS — on the external-panel check only.** The A1-selected final at 63,304 steps beats zero-shot
   pplx-decider-flash-v1.2 on the MR verdict (accuracy **+0.148622 [0.133258, 0.163411]**, Brier better) and the MIMIC audit
   suite (accuracy **+0.136241 [0.121585, 0.149646]**, Brier better), and is non-inferior on knowledge
   (**−0.001270 [−0.011213, 0.008543]**), but the external panel is **−0.024486 [−0.038358, −0.011460]**, below the −0.02
   limit (C052).
2. **The loop qualified a recipe and produced a final.** G1s0 (F2 + D57 distillation at λ = 3.243598) passes all four
   pre-registered §2.6 checks — knowledge paired lower +0.006244, MR lower −0.020692, MIMIC lower −0.010395, CG v2 micro
   0.794511 ≥ G0's 0.784399, positional health PASS — and is the chosen recipe (C034). The seed-1 stability arm is close
   (knowledge +0.01125, MR −0.00670, MIMIC +0.00125) and is reported beside seed 0, never chosen.
3. **The D57 dose is on target.** The probe gives s0 = 0.170491 → λ = **3.243598** (C019); the arm's own first-200-step mean
   share is **0.355474** (C024) and the final's is 0.351422 (C035), both inside the pre-registered 0.30–0.50 band.
4. **Against osler_v4's final, osler_v6 improves the panel (+1.77 macro points), knowledge (+0.54) and MR (+0.99) but loses
   2.9 points on the MIMIC audit suite** (C056). The panel gap to zero-shot is smaller than osler_v4's but still present, so the
   gate verdict is the same as Gate O4: panel FAIL, the other three checks PASS.
5. **The panel is the binding constraint.** Distillation at the pre-registered dose improved general knowledge to
   non-inferiority against the zero-shot model and preserved the MR lead, but did not close the 2.4-point panel deficit (the
   pre-registration required the lower bound above −2 points). The MIMIC regression against osler_v4's final is measured but
   not diagnosed here.

## 1. What ran

| task | outcome |
|---|---|
| Z0 | GO recorded; teacher pinned at `cce0f3f8…`; ADVISORY teacher typo fixed; loop artifacts committed. |
| Z1 | 385,536 of the 400,000 anchor items scored with the osler_v4 Y5 path (marginally identical pool `a9fab387…`, parity 40/40 bit-identical to Y5); 112,128 records reused and verified; 14,464 items cut by the pre-registered 12-hour budget (rate 6.32 items/s < 10/s); soft labels overlap 0; audit PASS (C004–C014). |
| Z2 | D57 sampler + λ probe. The first probe's random fill order exposed a 3.5× padding defect; the unlabelled fill order was made length-bucketed **before any arm result existed** (D57's quota, labelled draw and seeding unchanged), the probe was re-run, and λ = 3.243598 was frozen (C019–C023). |
| Z3 | G1 seed 0, 13,992 steps at the frozen λ: first-200-step band 0.355474 PASS, final dev S 0.825891, knowledge 0.775000, positional PASS; audit PASS (C024, C027, C028). |
| Z4 | G1 seed 1 (stability), 13,992 steps: band 0.365937 PASS, final dev S 0.826311, knowledge 0.770750, positional PASS; audit PASS (C029, C032, C033). |
| Z5 | The pre-registered §2.6 rule chose **G1s0** (knowledge lower +0.006244, MR −0.020692, MIMIC −0.010395, CG 0.794511, positional PASS); the seed-1 arm is reported beside it (C034). |
| Z6 | The 63,304-step final on G1s0's recipe. It crashed once (CUDA OOM at step 22,317) and was resumed from the step-20,000 full checkpoint; it finished 2026-11-12T04:25Z. Ten evaluations; **A1** selected the step-63,304 evaluation (positional PASS 0.448898 against the 0.428144 threshold) and calibrated it on dev only (C049–C051). |
| Z7 | Gate O6 scoring (four frozen Y9 sets, hashes verified) and the verdict: **FAIL on the panel check** (C052); audit and five-claim spot-check PASS (C055). |

## 2. Gate O6 (the verdict)

The four checks are Gate O4's, unchanged (ADVISORY §2.8), against zero-shot pplx-decider-flash-v1.2. Accuracy is the
template macro of the paired differences; Brier is the item mean; 1,000 bootstrap resamples within templates, seed 0.

| check | set | verdict | accuracy difference (final − zero-shot) | Brier difference | items |
|---|---|---|---|---|---|
| i | mr_verdict | **PASS** | +0.148622 [+0.133258, +0.163411] | −0.232221 [−0.240635, −0.223873] | 13,103 |
| ii | mimic_suite | **PASS** | +0.136241 [+0.121585, +0.149646] | −0.161882 [−0.175678, −0.147930] | 4,253 |
| iii | panel | **FAIL** | −0.024486 [−0.038358, −0.011460] | −0.033914 [−0.043932, −0.023870] | 4,353 |
| iv | knowledge | **PASS** | −0.001270 [−0.011213, +0.008543] | −0.037352 [−0.046373, −0.028035] | 5,435 paired of 5,456 |

**Gate O6: FAIL**, because check iii's lower bound is below −0.02. No readout failure (D12) appears on any set; the
knowledge set has 21 items (0.38 %) the baseline did not score, under the 1 % unpaired limit.

Accuracy levels (template macro; micro in parentheses): panel final 0.614487 (0.535952) against zero-shot 0.638972 (0.5615),
MedDecider-4B 0.615903, MedDecider-9B 0.630129; MR final 0.853005 against 0.704384; MIMIC final 0.770347 against 0.634106;
knowledge final 0.707840 against 0.709669 (MedDecider-9B 0.726511). Additional knowledge rows: vs MedDecider-4B +0.011469
[−0.000075, +0.023782], vs MedDecider-9B −0.018113 [−0.031772, −0.003857] (C053, C054).

## 3. The final and the A1 selection

The ten evaluations (dev only) trace a rising S; the positional health check on the DDXPlus two-letter readout alternates.

| step | S | MR | MIMIC | CG micro | knowledge | positional |
|---|---|---|---|---|---|---|
| 6,330 | 0.796776 | 0.899795 | 0.720461 | 0.770073 | 0.761500 | PASS |
| 12,661 | 0.791281 | 0.894419 | 0.704173 | 0.775250 | 0.758500 | PASS |
| 18,991 | 0.818523 | 0.947022 | 0.740762 | 0.767786 | 0.744750 | FAIL |
| 25,322 | 0.822039 | 0.941014 | 0.750818 | 0.774287 | 0.751750 | PASS |
| 31,652 | 0.827817 | 0.956085 | 0.747782 | 0.779583 | 0.754750 | PASS |
| 37,982 | 0.837467 | 0.963514 | 0.769906 | 0.778982 | 0.764000 | FAIL |
| 44,313 | 0.840188 | 0.966014 | 0.780263 | 0.774287 | 0.765250 | FAIL |
| 50,643 | 0.841495 | 0.959522 | 0.781889 | 0.783075 | 0.763750 | PASS |
| 56,974 | 0.845382 | 0.963445 | 0.787942 | 0.784760 | 0.767250 | PASS |
| **63,304** | **0.844190** | 0.964278 | 0.785818 | 0.782473 | 0.763500 | **PASS** |

Under **A1** the checkpoint is the step-63,304 evaluation only, positional-gated. It passes (original-order two-letter
0.448898 against 0.5 × 0.856287 = 0.428144; both permutations pass), so `z6_select.py` calibrated it on dev fast dev v4
(choice 1.227721, noul 1.384522, score 1.217728) and saved `chosen_step_063304`. All ten evaluations are reported; none was
dropped, and no best-S selection was made (C050).

## 4. The dose (D57)

The probe at λ = 1 (200 steps, the final sampler) gives s0 = 0.170491 → λ = (0.4/0.6)·(1−s0)/s0 = **3.243598**, frozen. The
arm's own first 200 steps at that λ give a mean pre-clip share of 0.355474 (C024) — inside the pre-registered band
[0.30, 0.50] — with exactly 3 labelled anchors on every step; the final's first 200 steps give 0.351422 (C035, a health
check). The distillation term was therefore applied at the intended share throughout.

## 5. Defects found and how they were handled

- **Padding defect in the sampler (Z2, before any arm result).** Drawing the unlabelled fill uniformly at random padded every
  batch to its longest record: probe #1 averaged 24,198 padded against 6,879 real tokens (3.54×) and 7.76 s/step, implying an
  arm of ~17 h and a final of ~75 h. D57 pins the quota (≥3 labelled), the without-replacement labelled draw, the seeding and
  the logged statistics, but not the fill order; the fill stream was therefore length-bucketed inside blocks of 32 batches
  and the batches are shuffled inside the block (the same anti-padding, anti-curriculum mechanism as `iter_batches`). The
  probe was re-run on the final sampler (s0 0.170491, padding 2.39×, 3.53 s/step) and λ was frozen from it. Probe #1 and the
  42-step aborted arm are preserved (`outputs/osler_v6/Z2/probe_random_fill/`, `outputs/osler_v6/Z3/G1s0_aborted_padding/`);
  no arm result existed when the change was made (C022, C023).
- **CUDA OOM at step 22,317 (Z6).** A Triton autotune benchmark inside the `fla` l2norm backward ran out of memory after a
  4-item, 54,272-padded-token batch. Full checkpoints existed at 17,500 and 20,000; the run was resumed from 20,000 with
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (an allocator setting, not a design change) and replayed straight through
  the former crash point (C040). A resume-path defect was found and fixed on the way: a resumed driver would have overwritten
  `evaluations.json` with only its post-resume evaluations (commit 53691b4); the pre-crash evaluations and log are preserved.
- **Positional-health oscillation (Z6).** The original-order DDXPlus check failed at 3 of 10 evaluations while both permuted
  orders passed. Investigation: on the 499 two-letter-gold items the model predicts two-letter keys for 407 of them but
  concentrates on BJ (128), BM (42), BC (38) — a key/position bias, not a constant answer; the permutations break it (C046).
  This is the D52b check working as intended; the final evaluation passes and A1 therefore permits the final.

## 6. Five-claim spot-check

Each was re-derived from raw artifacts in a fresh process (`outputs/osler_v6/Z7/spot_check.json`, C055):

| claim | recorded | re-derived | match |
|---|---|---|---|
| C012 — Z1 scored total | 385,536 | 385,536 (counted from `anchor_probs.jsonl`) | yes |
| C019 — probe s0 / λ | 0.17049145545775424 / 3.243597720934029 | identical (from `steps.jsonl`) | yes |
| C034 — Z5 choice / knowledge lower | G1s0 / 0.00624375 | identical (from the two arms' raw predictions) | yes |
| C049 — final-eval S | 0.8441896136161665 | identical (from the step-63,304 prediction files) | yes |
| C052 — Gate O6 verdict | FAIL, panel lower −0.038358 | identical (fresh gate re-run) | yes |

## 7. Provenance

Teacher `perplexity-ai/pplx-decider-v1.2-27b@cce0f3f80033e9d89d80831d5281fd72124aacce` (Y5 path, parity re-run 40/40 and
bit-identical to osler_v4's). Base `perplexity-ai/pplx-decider-flash-v1.2` (the same base as osler_v5 F2; §6/D59 is open and
unused). Mix: osler_v5's corrected F2 composition `78d7e547…` with 368,221 teacher-labelled rows attached by normalised text
(C015). λ 3.243598. Final checkpoint `chosen_step_063304`, calibrated on dev fast dev v4 only. Gate sets are osler_v4's
frozen Y9 files, each SHA-256 checked against osler_v4's manifest before scoring; the gate is osler_v4's `y9_gate.py`
unmodified, validated by reproducing Gate O4 exactly. Test sets, the panel, the MIMIC audit suite and held-out items never
influenced any training or selection decision.

## 8. Not measured / open

- The MIMIC-suite regression against osler_v4's final (−2.918 macro points) is measured but not diagnosed; the final is still
  +13.6 over zero-shot.
- The panel's per-subset breakdown versus osler_v4's final is not reported here.
- The 9B ladder, §6/the base decision (D59), the held-out-template list (§2.9), and adaptive thinking were out of scope.
- The final's training was interrupted once (C040); the resumed run is a statistical replica of the intended recipe (the
  deterministic sampler stream matches G1s0's on all common steps — C037 — while bf16 trajectories diverge numerically).

## 9. What this means

The loop's pre-registered design delivered what it promised on the dose and on general knowledge: the distillation term ran
at the calibrated share, knowledge is now non-inferior to the zero-shot pplx-flash (−0.13 points, interval spanning zero) and
far above MedDecider-4B, and the MR/MIMIC leads are intact. It did not close the external-panel gap: the panel is 2.45 points
below zero-shot with the lower bound at −3.84, so **Gate O6 fails** and the release/mission question is unchanged from Gate
O4. Against osler_v4's final the panel improved by 1.77 points and knowledge by 0.54, but the MIMIC suite lost 2.9 points.
The next levers named by the program are the base decision (ADVISORY §6, D59) and the 9B ladder; a further 4B distillation
loop at a larger dose is not supported by this evidence.
