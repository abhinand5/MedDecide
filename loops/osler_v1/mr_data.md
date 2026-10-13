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

## 8. V5a — Synthea encounters (CPU; `richardyoung/synthea-575k-patients`, MIT, streamed)

Extract (40,000 emergency and inpatient encounters; 33,879 emergency and 6,121 inpatient): observation rows cover 37.1 % of the
encounters (14,855; 127 of 11,424 row groups read, the patience rule stops the scan after matches end); medications 63.6 %,
procedures 64.4 %, conditions 53.6 %. Coverage is recorded, never assumed (CLAIMS V042).

| template | train | dev | test | heldout_test |
|---|---|---|---|---|
| `synthea_page_category_v1` (tabular pages rendered from Synthea facts) | 53,153 | 6,962 | 5,891 | 0 |
| `synthea_indicator_evidence_v1` | 6,696 | 740 | 618 | 1,342 (aki only) |

Patients: train 3,270; dev 383; test 412; held-out pool 225. No training patient appears in any other split; dev and test share
no patient. The held-out pool shares 225 patients with dev or test (deviation V5a-D2). Those patients' other items stay in dev and
test; no training item belongs to them.

Page-category classes in train: MAR 20,554, bill 20,220, laboratory 5,793, flowsheet 3,321, emergency department 3,265 (the
renderers are the V1 tabular renderers; the class mix is heavy in MAR and bill).

Indicator balance (yes / no, per indicator, after down-sampling in train, dev and test):

| indicator | train | dev | test |
|---|---|---|---|
| bilirubin_ge_1_2 | 812 / 812 | 84 / 84 | 62 / 62 |
| sirs_wbc | 988 / 988 | 109 / 109 | 72 / 72 |
| sirs_heart_rate | 822 / 822 | 107 / 107 | 92 / 92 |
| platelets_lt_150 | 314 / 314 | 38 / 38 | 35 / 35 |
| sirs_temperature | 303 / 303 | 25 / 25 | 39 / 39 |
| pao2_lt_60 | 37 / 37 | **2 / 2** | **4 / 4** |
| map_lt_65 | 44 / 44 | **2 / 2** | **1 / 1** |
| lactate_gt_2 | 28 / 28 | **3 / 3** | **4 / 4** |
| aki_creatinine_rise (held out; `heldout_test` only, not balanced) | — | — | 60 yes / 1,282 no (majority 0.955) |

Limits, recorded: three indicators have at most four positives per class in dev and test, so their template accuracies are not
measurable at useful precision; the held-out indicator is 4.5 % yes. A larger Synthea extract would be needed for those indicators
(the NEXT proposal); this loop keeps the ADVISORY's 40,000.

O3 screen on dev (ADVISORY §3.6): `synthea_indicator_evidence_v1` PASS (Naive Bayes 0.576; string presence not applicable).
`synthea_page_category_v1` FAIL, expected (lexical page headers): string presence 0.278; Naive Bayes 1.000; gold label in the
record text for 261 dev items. Accepted under the same rule as the tabular page category (V5a-D1, V1-D2); its share of any mix is
capped in V7.

Read check (ten items per template, fresh process, terminal only): page headers and titles name the class (lexical, expected);
OCR-style noise and fax banners appear as designed; the indicator pages carry the trap structure (prior-encounter lines, room-air
versus oxygen, held versus given medications) and the gold follows the values. Dropped rows, counted: held-out aki items of train
patients 5,652; observations with non-numeric values 124,502; temperatures in °F 8 (not converted; counted).

## 9. V5b — real clinical text (CPU and network)

| template | role | train | dev | test | source and licence | notes |
|---|---|---|---|---|---|---|
| `mtsamples_page_category_v1` | training (page category) | 668 | 97 | 111 | MTSamples (`harishnair04/mtsamples`); licence uncertain, accepted by the operator (D32) | pages per class (train and dev and test together): radiology 269, consult 255, progress 153, discharge 103, ED 63, admission 25, laboratory 8. Rows read 4,999: unmapped 3,971 (by reason in the manifest); under 200 characters 97; mapped 931 rows, which merge to 876 items after duplicates are merged and 23 transcriptions with conflicting classes are dropped |
| `codiesp_icd_support_v1` | training (ICD support) | 3,374 | none | 1,654 | CodiEsp (`Biomedical-TeMU/CodiEsp_corpus`, CC BY 4.0); the English text is **machine-translated**; corpus train is our train, corpus dev is our test; the corpus test set has no codes and is unused | dropped: code not billable in FY2026 673; category with a single billable code 914; category with several coded codes 787 |
| `pmc_similar_patient_v1` | training (similar-patient pairs) | 52,549 | 7,990 | 8,100 | PMC-Patients (`zhengyun21/PMC-Patients`, CC BY-NC-SA 4.0 overall, operator choice; subset per row: all 24,144 rows are `comm`) | anchors: up to 40,000 patients (file order) that link to another patient in the file, out of 167,034 read; a partner may be any patient in the file. Train pairs whose other patient is a dev or test anchor or in a dev or test pair: 11,361 dropped |
| `mkurman_icd_support_v1` | **evaluation only** (test) | — | — | 1,490 | mkurman (`mkurman/clinical-case-icd10-diagnosis`, commercial config; CC BY 4.0 rows; the 3 CC BY-ND 4.0 rows dropped) | dropped: single-billable-code category 37 |
| NHS synthetic notes | **NOT MEASURED** | | | | `NHSEDataScience/synthetic_clinical_notes` (MIT) | `note_type` takes 26 department or care-setting values (for example "Medicine Inpatients", "Physiotherapy Documentation", "Pre-op Consent"); none maps cleanly to the page classes (timebox used) |

Leakage: no real-text item's text key matches the v0.2 test, dev, panel, robustness or held-out sets, nor the osler_v1 MR dev, test or
held-out items (zero for every template; `outputs/osler_v1/V5b/audit.json`). Problems found by the audit: none.

Chance and majority on dev or test (the audit file): MTSamples dev 0.083 and 0.361, test 0.083 and 0.288; CodiEsp test 0.130 and
0.177; PMC dev 0.5 and 0.5, test 0.5 and 0.5; mkurman test chance and majority in CLAIMS V056.

Informational Naive Bayes on dev (not a gate; real text): MTSamples 0.590; PMC 0.520 (near chance: the lexical overlap of the two
case summaries does not predict the citation link). CodiEsp has no dev split, so its Naive Bayes is NOT MEASURED.

Limits, recorded: MTSamples has very few laboratory (8) and admission (25) pages; CodiEsp is machine-translated; PMC-Patients
pairs all come from the `comm` subset in this build (68,639 of 68,639); the NHS set is not measured.
