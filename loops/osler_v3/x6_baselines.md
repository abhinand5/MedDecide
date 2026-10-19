# osler_v3 — X6 record: fast dev v3, the clinical-general proxy, the new baselines (aggregates only)

Written at the X6 closure (2026-10-19, UTC). No item text. Every number is a row of `CLAIMS.md` (C072–C080), with its recompute command.
Partitions: `loops/osler_v3/partitions.md`. Charts: `loops/osler_v3/charts.md`.

## Partitions (fixed before any Stage A run)

- Clinical-general (CG) proxy: 4,865 items = knowledge dev v2 (4,000) + 865 kept fast-dev-v2 clinical-family items (C074). The
  12 items with repeated option labels cannot be scored by the pplx reference (C075).
- Fast dev v3: 12,903 items = fast dev v2 (11,323) + 1,580 items of the new templates (C072): RAGTruth grounded 400, RAGTruth sentence
  400, RAGTruth type 50, DrugProt 400, chart indicator 330. Blocked chart templates and the X3 templates contribute nothing (D-X4-4, D-X3-*).
- New test pool (additional rows only): 9,016 items = X1 test 8,832 + chart indicator test 184 (C073). X3 contributes 0.

## Zero-shot pplx-decider-flash-v1.2 on the CG proxy (the Stage A reference)

- 4,865 items; 4,853 scored; 12 skipped (repeated option labels; D44(a), 0.25 %) (C075).
- Accuracy on the 4,853 scored items: 0.7781 (C076). Dev proxy only; never a verdict.

## Baselines on the new test pool (additional rows only)

Coverage (C077): every model has all 9,016 pool items as rows. MedDecider 4B and 9B score 5,372 items; they skip the 3,644 DrugProt
items, which have more than 10 options (the authors' letter set is A to J). For those items the MedDecider comparison is NOT
MEASURED (D44(a)); the DrugProt rows of MedDecider are not in any comparison below.

Accuracy on the scored items (C078):

| model | scored | accuracy |
|---|---|---|
| zero-shot pplx-decider-flash-v1.2 | 9,016 | 0.6818 |
| zero-shot Qwen3.5-4B | 9,016 | 0.6226 |
| zero-shot Qwen3.5-9B | 9,016 | 0.6654 |
| MedDecider-4B | 5,372 | 0.6322 |
| MedDecider-9B | 5,372 | 0.7364 |
| JEV-9B | 9,016 | 0.6064 |

The MedDecider rows cover the RAGTruth and chart items only, so they are not comparable with the rows above them on the full pool.

Per template (C079; the values are in `outputs/osler_v3/X6/baselines_summary.json`): the RAGTruth halluc type template has 40 items,
so its values are noisy.

## 'Too good' check on the chart indicator items (C080)

The indicator template reads well for zero-shot pplx (0.8152 on its 184 test items). Splitting by whether a narrative page states a unit
value (C068) gives 0.8378 on 37 items with a stated value and 0.8095 on 147 without. The high accuracy is therefore not explained by the
stated values. For Qwen 9B the split is 0.6486 (37) and 0.6735 (147). The check is additional and does not change the template's status.

## Deviations recorded during X6

- D-X6-1: referenced on the board since the CG build (commit 1668b36) but its entry was never written. The CG build and its exclusion counts are in `partitions.md`; this record asserts nothing further under that label.
- D-X6-2 (zero-shot pplx output path; fixed before any use).
- D-X6-3 (the 865 fast-dev-v2 CG rows lacked the O2 keys `benchmark` and `set_name`; set to `mix_v2` and the item source; membership
  unchanged).
- D-X6-4 (the zero-shot pplx CG step failed twice at those rows; resumed from its saved rows after the fix; the queue went on to the
  new-test-pool steps, which are additional rows).

## Not measured

- MedDecider 4B and 9B on the 3,644 DrugProt items (more than 10 options; D44(a)).
- X3 template items in any X6 table (blocked; 0 rows).
- Osler-4B on any X6 table: NOT MEASURED until the final model (X9).
