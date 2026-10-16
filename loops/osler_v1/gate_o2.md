# osler_v1 — Gate O2 (PROGRAM D31), Osler-4B

Osler rows: both-order averages for choice items, noul and score items scored once (the O2 convention). Accuracy is the
template macro, bootstrapped within templates (1000 resamples, seed 0); Brier is the item mean. An interval
is shown as point [lower, upper]. Items in scope: mr_test 19294, external_panel 4353, knowledge 5456, mr_heldout_test 584.
pmc_similar_patient_v1 is 8100 of 19294 MR test items (42.0%); its rows are additional only.

| check | baseline | items (paired) | accuracy diff [lower, upper] | Brier diff [lower, upper] | pass | skipped / missing (baseline), Osler missing |
|---|---|---|---|---|---|---|
| (i) MR suite | MedDecider-4B | 12992 | +0.0805 [+0.0700, +0.0916] | +0.0351 [+0.0287, +0.0414] | no | 6302 / 0, 0 |
| (i) MR suite | zero-shot Qwen3.5-4B | 19294 | +0.0943 [+0.0829, +0.1054] | -0.0449 [-0.0496, -0.0404] | yes | 0 / 0, 0 |
| (ii) external panel | MedDecider-4B | 4141 | -0.0223 [-0.0365, -0.0085] | +0.0332 [+0.0261, +0.0409] | no | 212 / 0, 0 |
| (ii) external panel | zero-shot Qwen3.5-4B | 4353 | +0.0568 [+0.0410, +0.0741] | -0.0924 [-0.1047, -0.0806] | yes | 0 / 0, 0 |

(iii) knowledge guard vs zero-shot Qwen3.5-4B (MedQA + MedMCQA test, micro accuracy): 5456 paired items, +0.0440 [+0.0324, +0.0548]; guard passes (lower bound must exceed -0.02).

**Osler-4B verdict: FAIL.** failed: MR suite vs MedDecider-4B; external panel vs MedDecider-4B (accuracy lower bound must exceed -0.02).

**Osler-9B verdict: NOT MEASURED — deferred to the next loop by the operator (V9-D1).**

## Additional rows (labelled; never in the verdict)

| check | baseline | items (paired) | accuracy diff [lower, upper] | Brier diff [lower, upper] | pass | skipped / missing (baseline), Osler missing |
|---|---|---|---|---|---|---|
| held-out MR, test rows (readout pass cells) | MedDecider-4B | 70 | +0.2654 [+0.1485, +0.3913] | -0.2584 [-0.3964, -0.1239] | yes | 0 / 0, 0 |
| held-out MR, test rows (readout pass cells) | zero-shot Qwen3.5-4B | 584 | +0.1389 [+0.0532, +0.2233] | +0.1965 [+0.1616, +0.2309] | no | 0 / 0, 0 |
| d1-3B, MR test | d1-3B | 19294 | +0.1436 [+0.1311, +0.1554] | -0.1986 [-0.2058, -0.1920] | yes | 0 / 0, 0 |
| MR test without pmc_similar_patient_v1 (additional; never in the verdict) | MedDecider-4B | 4892 | +0.0888 [+0.0759, +0.1020] | +0.1597 [+0.1445, +0.1738] | no | 6302 / 0, 0 |
| MR test without pmc_similar_patient_v1 (additional; never in the verdict) | zero-shot Qwen3.5-4B | 11194 | +0.0970 [+0.0841, +0.1092] | -0.0020 [-0.0083, +0.0034] | no | 0 / 0, 0 |
| external panel superiority (additional; never in the verdict) | MedDecider-4B | 4141 | -0.0223 [-0.0365, -0.0085] | +0.0332 [+0.0261, +0.0409] | no | 212 / 0, 0 |
| external panel superiority (additional; never in the verdict) | zero-shot Qwen3.5-4B | 4353 | +0.0568 [+0.0410, +0.0741] | -0.0924 [-0.1047, -0.0806] | yes | 0 / 0, 0 |

Single-order rows (the original option order only; reported beside the gate, never in it):

| check | baseline | items (paired) | accuracy diff [lower, upper] | Brier diff [lower, upper] | pass | skipped / missing (baseline), Osler missing |
|---|---|---|---|---|---|---|
| single order, MR test | MedDecider-4B | 12992 | +0.0713 [+0.0607, +0.0824] | +0.0772 [+0.0705, +0.0842] | no | 6302 / 0, 0 |
| single order, MR test | zero-shot Qwen3.5-4B | 19294 | +0.0831 [+0.0719, +0.0945] | -0.0160 [-0.0208, -0.0113] | yes | 0 / 0, 0 |
| single order, external panel | MedDecider-4B | 4141 | -0.0296 [-0.0442, -0.0143] | +0.0623 [+0.0517, +0.0730] | no | 212 / 0, 0 |
| single order, external panel | zero-shot Qwen3.5-4B | 4353 | +0.0378 [+0.0201, +0.0553] | -0.0589 [-0.0716, -0.0459] | yes | 0 / 0, 0 |

- MDACE: NOT MEASURED — the dataset is not on the volume.
- The MIMIC part of the MR suite: NOT MEASURED — MIMIC is absent (V5c). The MR test suite holds generated and real-text items (V5b); its parts are reported as additional rows (v10_additional.py).
