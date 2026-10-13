# mr_data.md — osler_v1 medical-record audit data (aggregates only)

Written at V1. Nothing here is item text: counts, hashes, screen numbers and the held-out lists only. The files are gitignored
(`data/gen/osler_v1/`). Recompute: `scripts/osler_v1/v1_build_mr_generators.py` (build), `scripts/osler_v1/v1_audit.py`
(counts, twins, splits, held-out checks, chance and majority; writes `outputs/osler_v1/V1/SELF_AUDIT.md`).

## 1. Generators built in V1

| generator | family | held out | items train / dev / test / heldout_test | twins | O3 screen (dev) |
|---|---|---|---|---|---|
| `mr_page_category_tabular_v1` | MR_page_category | no | 6,000 / 300 / 300 / 0 | none (one item per patient id) | **FAIL, expected (ADVISORY §3.6):** string presence 0.276; Naive Bayes 1.000; gold label appears in the record text for 23 dev items |
| `mr_indicator_evidence_v1` | MR_indicator_evidence | `aki_creatinine_rise` | 11,130 / 550 / 576 / 74 | 6,165 of 6,165 pairs complete; gold flips in all | PASS: Naive Bayes 0.796 (threshold 0.90) |
| `mr_icd_support_template_v2` | MR_icd_support | `heart_failure` | 10,902 / 544 / 554 / 102 | 6,051 of 6,051 pairs complete; gold flips in all | PASS: Naive Bayes 0.804; string presence 0.026 |

Totals: **train 28,032; dev 1,394; test 1,430; heldout_test 176.** Held-out rows removed from the training splits: 870 indicator
rows and 1,098 ICD rows (the whole pair, counted by row; `manifest.json`, `held_out_train_rows_dropped`).

Patients (from `outputs/osler_v1/V1/self_audit.json`): train 17,016; dev 847; test 865; heldout_test 88; total 18,816; patients in
two splits 0; rows whose split disagrees with the patient's hash 0.

## 2. Design change inside V1 (recorded as V1-D1)

The first build used the single-diagnosis ICD template (`mr_icd_support_template_v1`). At full size it **failed** the O3 screen:
Naive Bayes 0.938 on dev 550 (threshold 0.90), and the Naive Bayes solved both twins of 54.5 % of pairs. The planning-time probe
(1,500 patients per generator) had passed, so the Naive Bayes gains with data. The attribute words of one diagnosis name its code.
Evidence: `outputs/osler_v1/V1/screen_build1_icd_v1design.json` and `outputs/osler_v1/V1/logs/build1_icd_v1design.log`.

The v2 template keeps the same draw for the target diagnosis, its gold and its one-fact twin. It adds two distractor diagnoses from
other families, names the target condition in the question, and redraws whenever a distractor is a held-out family, shares a content
word with the target condition, or contains the target condition's name. Naive Bayes falls to 0.804 (dev 544), with a margin.
The v1 function stays in `generators.py` as the recorded failing design; it is not in the registry.

## 3. Label balance and chance (dev, from the written files)

| generator | uniform chance | majority gold share | notes |
|---|---|---|---|
| `mr_page_category_tabular_v1` | 0.083 | 0.233 | five classes generated of twelve (the rest come from MIMIC, teacher pages and Synthea); train classes F 1,180, G 1,226, I 1,256, J 1,142, K 1,196 |
| `mr_indicator_evidence_v1` | 0.500 | 0.500 | yes and no exactly balanced in every split (train 5,565 each) |
| `mr_icd_support_template_v2` | 0.299 | 0.283 | gold "none" is 41.9 % of base items (train: 2,283 of 5,451); gold letters A and B are most frequent in train (2,816 and 2,844 of 10,902) because the codes sort first. Training shuffles option order each epoch (`shuffle_options: true`), which removes that letter prior from the training signal |

## 4. Held-out checks (fixed lists; never used for any choice)

- Indicator `aki_creatinine_rise` and ICD family `heart_failure` are held out. The text search finds the heart-failure name in
  `heldout_test` only (102 rows), not in train, dev or test. The held-out indicator key is absent from train.
- `heldout_test` holds 176 rows (74 indicator, 102 ICD). Reported separately in every later result; never in a verdict or a keep decision.

## 5. Reading check (ADVISORY V1; ten items per generator, read in a fresh process, not committed)

Defects by type, none changes a label:
1. Page-category headers and titles name the class ("EMERGENCY DEPT", "LABORATORY RESULTS", "CLAIMS REVIEW"). Expected lexical
   cue; this is the accepted screen failure.
2. OCR-style noise in some labels and facility names (letter-for-digit swaps, transpositions). Intended; the numbers are untouched.
3. The ICD template's laterality wording repeats the noun ("of the knee of the right knee"). A wording defect in the template,
   recorded for the teacher variant to avoid; the label is unaffected.
4. Fax banners, page footers and continuation pages appear as designed.

## 6. Deviations recorded in V1

- **V1-D1** (above): the ICD template is v2 with distractors because v1 failed the screen at full size. The v2 design is the one used.
- **V1-D2**: the page-category screen failure is accepted as the ADVISORY §3.6 expects. Its share of any mix is capped (ADVISORY V7).
- **V1-D3**: the page-category generator emits one item per patient (no twins); the twin audit does not apply to it.

## 7. Not in V1

MIMIC (V5c), Synthea (V5a), MTSamples, CodiEsp and PMC-Patients (V5b), and the teacher prose (V4) are later tasks. No external data
is read in V1.
