# osler_v1 — results (V10)

Gate O2 (PROGRAM D31) as written, for Osler-4B. Every number has a CLAIMS row (V165 to V187). Intervals are the bootstrap
(1,000 resamples, seed 0; within templates) and are shown as point [lower, upper]. Accuracy is the template macro, except the
knowledge guard, which is micro (D24). Brier is the item mean. Differences are Osler-4B minus the baseline: a positive
accuracy difference favours Osler-4B; a negative Brier difference favours Osler-4B.

## 1. Verdicts

| model | verdict | what failed |
|---|---|---|
| Osler-4B (B1; the kept block) | **FAIL** (V165) | (i) MR suite against MedDecider-4B: the Brier upper bound is not below zero; (ii) external panel against MedDecider-4B: the accuracy lower bound is below -0.02 |
| Osler-9B | NOT MEASURED (deferred to the next loop by the operator, V9-D1) | — |

## 2. The primary checks (D31)

| check | baseline | items (paired) | accuracy diff [lower, upper] | Brier diff [lower, upper] | pass | CLAIMS |
|---|---|---|---|---|---|---|
| (i) MR suite (19,294) | MedDecider-4B | 12,992 (6,302 skipped) | +0.0805 [+0.0700, +0.0916] | +0.0351 [+0.0287, +0.0414] | no (Brier) | V167 |
| (i) MR suite (19,294) | zero-shot Qwen3.5-4B | 19,294 | +0.0943 [+0.0829, +0.1054] | -0.0449 [-0.0496, -0.0404] | yes | V166 |
| (ii) external panel (4,353) | MedDecider-4B | 4,141 (212 skipped) | -0.0223 [-0.0365, -0.0085] | +0.0332 [+0.0261, +0.0409] | no (accuracy) | V169 |
| (ii) external panel (4,353) | zero-shot Qwen3.5-4B | 4,353 | +0.0568 [+0.0410, +0.0741] | -0.0924 [-0.1047, -0.0806] | yes | V168 |
| (iii) knowledge guard (5,456; MedQA 1,273 and MedMCQA 4,183) | zero-shot Qwen3.5-4B | 5,456 | +0.0440 [+0.0324, +0.0548] (micro) | — | yes | V170 |

The superiority bound on the panel (lower bound above zero) is reported separately and holds against zero-shot (+0.0410)
and does not hold against MedDecider-4B.

Skipped items: MedDecider-4B answers at most 10 options (letters A to J). It skips 6,302 MR items, all choice items with more
than 10 options (synthea_page_category_v1 5,891; mr_page_category_tabular_v1 300; mtsamples_page_category_v1 111), and 212
panel items (symptom_to_diagnosis). The gate excludes the baseline's skipped items from the paired comparison, as the O1 and
O2 verdicts did, and reports the counts (V172).

## 3. Parts of the MR suite (additional; never in a verdict; ADVISORY V10)

The MR test suite holds 7,939 generated items (V1 code-generated records 1,430; V5a Synthea pages 6,509) and 11,355 real-text
items (V5b: PMC-Patients 8,100, CodiEsp 1,654, mkurman 1,490, MTSamples 111). MIMIC is absent: its part is NOT MEASURED (V187).

| part | baseline | items (paired) | accuracy diff [lower, upper] | Brier diff [lower, upper] | CLAIMS |
|---|---|---|---|---|---|
| generated (7,939) | MedDecider-4B | 1,748 (6,191 skipped) | +0.2351 [+0.2148, +0.2563] | -0.3399 [-0.3678, -0.3127] | V184 |
| generated (7,939) | zero-shot Qwen3.5-4B | 7,939 | +0.1588 [+0.1458, +0.1712] | -0.0871 [-0.0928, -0.0815] | V184 |
| real text (11,355) | MedDecider-4B | 11,244 (111 skipped) | -0.0742 [-0.0837, -0.0658] | +0.0934 [+0.0876, +0.0997] | V184 |
| real text (11,355) | zero-shot Qwen3.5-4B | 11,355 | +0.0136 [-0.0049, +0.0333] | -0.0154 [-0.0216, -0.0090] | V184 |
| MIMIC | — | — | NOT MEASURED (absent) | — | V187 |

Per template against MedDecider-4B on the 12,992 paired MR items (ADVISORY §2, outcome 3; V185):

| template | paired items | accuracy diff | Brier diff |
|---|---|---|---|
| codiesp_icd_support_v1 (real text) | 1,654 | -0.0992 | +0.4738 |
| mkurman_icd_support_v1 (real text) | 1,490 | -0.1624 | +0.3972 |
| pmc_similar_patient_v1 (real text) | 8,100 | +0.0390 | -0.0402 |
| synthea_indicator_evidence_v1 | 618 | +0.1650 | -0.2336 |
| mr_indicator_evidence_v1 | 576 | +0.2569 | -0.3725 |
| mr_icd_support_template_v2 | 554 | +0.2834 | -0.4246 |

The six accuracy differences average to +0.0805, the gate's point estimate. The Brier loss against MedDecider-4B sits on the two
real-text ICD templates.

## 4. Held-out templates (additional; never in a verdict; ADVISORY §2, outcome 5)

The held-out indicator is aki_creatinine_rise: 538 test items (synthetic indicator template 514; MR indicator template 24).
Every item names creatinine. The held-out ICD family is heart_failure: 46 test items. Both are excluded from training (V028,
V031). MedDecider-4B's readout fails on the synthetic indicator template in the held-out set (V089): that cell reports no
accuracy, so MedDecider-4B is compared on its pass cells only (V188). Zero-shot Qwen3.5-4B passes every held-out cell.

| held-out check | baseline | items | accuracy diff [lower, upper] | Brier diff [lower, upper] | CLAIMS |
|---|---|---|---|---|---|
| indicator, MR template only | MedDecider-4B | 24 (pass cells) | +0.2917 [+0.1250, +0.4583] | -0.1784 [-0.3212, -0.0533] | V188 |
| indicator, both templates | zero-shot Qwen3.5-4B | 538 | +0.0344 [-0.0554, +0.1356] | +0.2499 [+0.2170, +0.2832] | V173 |
| indicator, synthetic template | MedDecider-4B | 514 | READOUT_FAIL: no accuracy reported | — | V188 |
| ICD family (heart_failure) | MedDecider-4B | 46 | +0.2391 [+0.0652, +0.4130] | -0.3001 [-0.4815, -0.1026] | V174 |
| ICD family (heart_failure) | zero-shot Qwen3.5-4B | 46 | +0.3478 [+0.1951, +0.5000] | -0.4282 [-0.6147, -0.2491] | V174 |
| held-out test rows, combined (the gate's row) | MedDecider-4B | 70 (pass cells) | +0.2654 [+0.1485, +0.3913] | -0.2584 [-0.3964, -0.1239] | V188 |
| held-out test rows, combined | zero-shot Qwen3.5-4B | 584 | +0.1389 [+0.0532, +0.2233] | +0.1965 [+0.1616, +0.2309] | V175 |

The first versions of the MedDecider-4B indicator and combined figures (accuracy +0.2013 and +0.2139) included the
READOUT_FAIL cell; they are withdrawn (V173 and V175, in part, by V188). The 46-item ICD family and the 24-item MR indicator have
wide intervals. Against zero-shot Qwen3.5-4B the held-out indicator shows no accuracy gain (its interval includes zero) and a much
worse Brier.

## 5. Other additional rows

- **pmc_similar_patient_v1** is 8,100 of 19,294 MR test items (41.98 %; the gate report shows 42.0 %). The MR suite without it
  (V177): against MedDecider-4B (4,892 paired; 6,302 skipped) accuracy +0.0888 [+0.0759, +0.1020], Brier +0.1597
  [+0.1445, +0.1738]; against zero-shot (11,194 items) accuracy +0.0970 [+0.0841, +0.1092], Brier -0.0020 [-0.0083, +0.0034].
- **Single original option order** (V178): MR suite against MedDecider-4B (12,992 paired) accuracy +0.0713 [+0.0607, +0.0824],
  Brier +0.0772 [+0.0705, +0.0842]; against zero-shot (19,294) accuracy +0.0831 [+0.0719, +0.0945], Brier -0.0160
  [-0.0208, -0.0113]. External panel against MedDecider-4B (4,141) accuracy -0.0296 [-0.0442, -0.0143], Brier +0.0623
  [+0.0517, +0.0730]; against zero-shot (4,353) accuracy +0.0378 [+0.0201, +0.0553], Brier -0.0589 [-0.0716, -0.0459].
- **d1-3B** on the MR suite (19,294; V176): accuracy +0.1436 [+0.1311, +0.1554], Brier -0.1986 [-0.2058, -0.1920].
- **osler_v0 Osler-4B, arm L** (the option-code head that osler_v0 kept; outputs/osler_v0/O11; V179), against Osler-4B (v1),
  positive accuracy favouring Osler-4B (v1): external panel (4,353) accuracy +0.0358 [+0.0186, +0.0520], Brier -0.0260
  [-0.0358, -0.0159]; knowledge (5,456) accuracy +0.0764 [+0.0619, +0.0913], Brier -0.0753 [-0.0858, -0.0651]. The osler_v0
  MR suite is NOT MEASURED (no osler_v0 rows on the MR test items). The osler_v0 Osler-9B is NOT MEASURED (no per-item O2
  predictions found).

## 6. Defects screened on the V10 rows (V180, V183)

Osler-4B on all scored scopes (MR test, external panel, knowledge; 29,103 items): accuracy 0.7255 against a mean chance of
0.2729; on the MR test alone (19,294): accuracy 0.7908 against 0.2966. No below-chance flag, no constant-answer flag, and no
divergence flag (no training losses). Too good on three templates at 1.0000: mr_icd_support_template_v2 (554 items),
mr_page_category_tabular_v1 (300) and synthea_page_category_v1 (5,891). The two page-category templates are the lexical failures
accepted at build (Naive Bayes 1.000; V1-D2, V5a-D1). The ICD template's Naive Bayes screen is 0.804 (a pass), so the lexical
screen does not explain its 100 % accuracy; this is open (V183).

Post hoc (additional; the three templates were chosen after the screen flagged them; V181): without them (12,549 of 19,294
items), against MedDecider-4B (12,438 paired; 111 skipped) accuracy +0.0399 [+0.0290, +0.0505], Brier +0.0555
[+0.0493, +0.0615]; against zero-shot (12,549) accuracy +0.0916 [+0.0752, +0.1068], Brier -0.0463 [-0.0531, -0.0394].

## 7. Completeness and checks

- Osler-4B scored all 30,621 selected items in both orders and in the original order; no row is missing, duplicated or
  non-finite; the calibration equals the dev fits (V171).
- An independent re-derivation in a fresh process matches every primary point estimate to 0.0005 (V182;
  `scripts/osler_v1/v10_check.py`).
- Gate wording: the gate's text said the MR test suite is the generated part. It is not (V186); the text was corrected, and the
  verdict and every number were unchanged.
- The Osler-9B verdict stays NOT MEASURED (V9-D1). MDACE is NOT MEASURED (the dataset is not on the volume).
