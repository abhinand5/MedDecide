# osler_v4 Y4 — baselines (aggregates only)

Readout checks per (set, model, template) follow D12 and D21 as written in the report script; label mass is not applied
to the authors' rows. Cells that fail a check read READOUT_FAIL and are not reported as results; a set-level figure
is reported only when every template cell in it passes. The macro is the template macro (D24).

## suite_flash

overall: 4253 items; micro 0.624; template macro 0.6341; chance 0.2898; status PASS

| template | items | micro | majority | modal share | status |
|---|---|---|---|---|---|
| mimic_drg_cc_level_v1 | 498 | 0.5562 | 0.3333 | 0.494 | PASS |
| mimic_ed_acuity_v1 | 494 | 0.3502 | 0.5567 | 0.3441 | PASS |
| mimic_ed_disposition_v1 | 497 | 0.6358 | 0.5312 | 0.5714 | PASS |
| mimic_icd_specificity_v1 | 488 | 0.7336 | 0.1947 | 0.1906 | PASS |
| mimic_icd_support_v1 | 400 | 0.665 | 0.2325 | 0.2475 | PASS |
| mimic_lab_validation_v1 | 376 | 0.9601 | 0.5957 | 0.5718 | PASS |
| mimic_med_given_v1 | 500 | 0.626 | 0.5 | 0.55 | PASS |
| mimic_msdrg_v1 | 500 | 0.436 | 0.2 | 0.192 | PASS |
| mimic_principal_dx_v1 | 500 | 0.744 | 0.2 | 0.19 | PASS |

## suite_meddecider4b

overall: 4253 items; chance 0.2898; status READOUT_FAIL — 1 of 9 template cells fail the gate (mimic_drg_cc_level_v1); the set-level accuracy is not reported as a result (D12)

additional, not a result: micro accuracy over the template cells that pass the gate: 3755 items; micro 0.6333

| template | items | micro | majority | modal share | status |
|---|---|---|---|---|---|
| mimic_drg_cc_level_v1 | 498 | 0.3655 | 0.3333 | 0.9518 | READOUT_FAIL — constant answer (A) on 0.952 of items |
| mimic_ed_acuity_v1 | 494 | 0.3725 | 0.5567 | 0.4555 | PASS |
| mimic_ed_disposition_v1 | 497 | 0.6962 | 0.5312 | 0.5835 | PASS |
| mimic_icd_specificity_v1 | 488 | 0.7213 | 0.1947 | 0.1783 | PASS |
| mimic_icd_support_v1 | 400 | 0.62 | 0.2325 | 0.2425 | PASS |
| mimic_lab_validation_v1 | 376 | 0.9362 | 0.5957 | 0.5585 | PASS |
| mimic_med_given_v1 | 500 | 0.68 | 0.5 | 0.696 | PASS |
| mimic_msdrg_v1 | 500 | 0.39 | 0.2 | 0.202 | PASS |
| mimic_principal_dx_v1 | 500 | 0.722 | 0.2 | 0.18 | PASS |

## suite_meddecider9b

overall: 4253 items; micro 0.6189; template macro 0.6278; chance 0.2898; status PASS

| template | items | micro | majority | modal share | status |
|---|---|---|---|---|---|
| mimic_drg_cc_level_v1 | 498 | 0.4217 | 0.3333 | 0.7229 | PASS |
| mimic_ed_acuity_v1 | 494 | 0.4474 | 0.5567 | 0.4291 | PASS |
| mimic_ed_disposition_v1 | 497 | 0.6801 | 0.5312 | 0.507 | PASS |
| mimic_icd_specificity_v1 | 488 | 0.709 | 0.1947 | 0.1824 | PASS |
| mimic_icd_support_v1 | 400 | 0.66 | 0.2325 | 0.2525 | PASS |
| mimic_lab_validation_v1 | 376 | 0.9122 | 0.5957 | 0.5186 | PASS |
| mimic_med_given_v1 | 500 | 0.69 | 0.5 | 0.646 | PASS |
| mimic_msdrg_v1 | 500 | 0.438 | 0.2 | 0.206 | PASS |
| mimic_principal_dx_v1 | 500 | 0.692 | 0.2 | 0.184 | PASS |

## suite_osler_v2

overall: 4253 items; micro 0.5551; template macro 0.5651; chance 0.2898; status PASS

| template | items | micro | majority | modal share | status |
|---|---|---|---|---|---|
| mimic_drg_cc_level_v1 | 498 | 0.4639 | 0.3333 | 0.747 | PASS |
| mimic_ed_acuity_v1 | 494 | 0.2571 | 0.5567 | 0.3482 | PASS |
| mimic_ed_disposition_v1 | 497 | 0.6519 | 0.5312 | 0.5614 | PASS |
| mimic_icd_specificity_v1 | 488 | 0.7008 | 0.1947 | 0.1967 | PASS |
| mimic_icd_support_v1 | 400 | 0.505 | 0.2325 | 0.3275 | PASS |
| mimic_lab_validation_v1 | 376 | 0.9548 | 0.5957 | 0.5505 | PASS |
| mimic_med_given_v1 | 500 | 0.594 | 0.5 | 0.514 | PASS |
| mimic_msdrg_v1 | 500 | 0.36 | 0.2 | 0.2 | PASS |
| mimic_principal_dx_v1 | 500 | 0.598 | 0.2 | 0.198 | PASS |

## heldout_flash

overall: 110 items; micro 0.7; template macro 0.7199; chance 0.2636; status PASS

| template | items | micro | majority | modal share | status |
|---|---|---|---|---|---|
| mimic_icd_specificity_v1 | 12 | 0.75 | 0.4167 | 0.3333 | PASS |
| mimic_icd_support_v1 | 80 | 0.6875 | 0.2625 | 0.275 | PASS |
| mimic_principal_dx_v1 | 18 | 0.7222 | 0.2222 | 0.3333 | PASS |

## heldout_meddecider4b

overall: 110 items; micro 0.6636; template macro 0.7176; chance 0.2636; status PASS

| template | items | micro | majority | modal share | status |
|---|---|---|---|---|---|
| mimic_icd_specificity_v1 | 12 | 0.75 | 0.4167 | 0.3333 | PASS |
| mimic_icd_support_v1 | 80 | 0.625 | 0.2625 | 0.4875 | PASS |
| mimic_principal_dx_v1 | 18 | 0.7778 | 0.2222 | 0.2778 | PASS |

## heldout_osler_v2

overall: 110 items; micro 0.5091; template macro 0.6704; chance 0.2636; status PASS

| template | items | micro | majority | modal share | status |
|---|---|---|---|---|---|
| mimic_icd_specificity_v1 | 12 | 0.8333 | 0.4167 | 0.4167 | PASS |
| mimic_icd_support_v1 | 80 | 0.4 | 0.2625 | 0.425 | PASS |
| mimic_principal_dx_v1 | 18 | 0.7778 | 0.2222 | 0.2778 | PASS |

## fastdev_mimic_flash

overall: 5562 items; micro 0.6638; template macro 0.6659; chance 0.3099; status PASS

| template | items | micro | majority | modal share | status |
|---|---|---|---|---|---|
| mimic_apr_mortality_v1 | 400 | 0.525 | 0.3675 | 0.4775 | PASS |
| mimic_apr_severity_v1 | 400 | 0.485 | 0.3375 | 0.665 | PASS |
| mimic_doc_support_v1 | 400 | 0.88 | 0.5025 | 0.5875 | PASS |
| mimic_drg_cc_level_v1 | 400 | 0.56 | 0.355 | 0.4825 | PASS |
| mimic_ed_acuity_v1 | 400 | 0.405 | 0.5275 | 0.37 | PASS |
| mimic_ed_disposition_v1 | 400 | 0.665 | 0.61 | 0.6625 | PASS |
| mimic_icd_specificity_v1 | 400 | 0.7475 | 0.1875 | 0.1925 | PASS |
| mimic_icd_support_v1 | 400 | 0.6625 | 0.2375 | 0.27 | PASS |
| mimic_lab_validation_v1 | 362 | 0.9807 | 0.5994 | 0.5856 | PASS |
| mimic_med_given_v1 | 400 | 0.6775 | 0.505 | 0.5725 | PASS |
| mimic_msdrg_v1 | 400 | 0.4075 | 0.19 | 0.2125 | PASS |
| mimic_pcs_support_v1 | 400 | 0.66 | 0.505 | 0.795 | PASS |
| mimic_principal_dx_v1 | 400 | 0.72 | 0.1975 | 0.1975 | PASS |
| mimic_rad_exam_v1 | 400 | 0.9475 | 0.22 | 0.225 | PASS |

## fastdev_mimic_meddecider4b

overall: 5562 items; chance 0.3099; status READOUT_FAIL — 1 of 14 template cells fail the gate (mimic_drg_cc_level_v1); the set-level accuracy is not reported as a result (D12)

additional, not a result: micro accuracy over the template cells that pass the gate: 5162 items; micro 0.6606

| template | items | micro | majority | modal share | status |
|---|---|---|---|---|---|
| mimic_apr_mortality_v1 | 400 | 0.53 | 0.3675 | 0.4575 | PASS |
| mimic_apr_severity_v1 | 400 | 0.535 | 0.3375 | 0.61 | PASS |
| mimic_doc_support_v1 | 400 | 0.6725 | 0.5025 | 0.82 | PASS |
| mimic_drg_cc_level_v1 | 400 | 0.3775 | 0.355 | 0.945 | READOUT_FAIL — constant answer (A) on 0.945 of items |
| mimic_ed_acuity_v1 | 400 | 0.4225 | 0.5275 | 0.4925 | PASS |
| mimic_ed_disposition_v1 | 400 | 0.6725 | 0.61 | 0.6175 | PASS |
| mimic_icd_specificity_v1 | 400 | 0.7275 | 0.1875 | 0.2075 | PASS |
| mimic_icd_support_v1 | 400 | 0.6375 | 0.2375 | 0.255 | PASS |
| mimic_lab_validation_v1 | 362 | 0.9558 | 0.5994 | 0.5829 | PASS |
| mimic_med_given_v1 | 400 | 0.6975 | 0.505 | 0.6475 | PASS |
| mimic_msdrg_v1 | 400 | 0.39 | 0.19 | 0.22 | PASS |
| mimic_pcs_support_v1 | 400 | 0.685 | 0.505 | 0.73 | PASS |
| mimic_principal_dx_v1 | 400 | 0.7275 | 0.1975 | 0.19 | PASS |
| mimic_rad_exam_v1 | 400 | 0.9625 | 0.22 | 0.225 | PASS |

## fastdev_mimic_meddecider9b

overall: 5562 items; micro 0.6352; template macro 0.6372; chance 0.3099; status PASS

| template | items | micro | majority | modal share | status |
|---|---|---|---|---|---|
| mimic_apr_mortality_v1 | 400 | 0.4625 | 0.3675 | 0.635 | PASS |
| mimic_apr_severity_v1 | 400 | 0.5025 | 0.3375 | 0.435 | PASS |
| mimic_doc_support_v1 | 400 | 0.7175 | 0.5025 | 0.765 | PASS |
| mimic_drg_cc_level_v1 | 400 | 0.4375 | 0.355 | 0.7525 | PASS |
| mimic_ed_acuity_v1 | 400 | 0.4475 | 0.5275 | 0.455 | PASS |
| mimic_ed_disposition_v1 | 400 | 0.6625 | 0.61 | 0.5825 | PASS |
| mimic_icd_specificity_v1 | 400 | 0.72 | 0.1875 | 0.2 | PASS |
| mimic_icd_support_v1 | 400 | 0.6225 | 0.2375 | 0.26 | PASS |
| mimic_lab_validation_v1 | 362 | 0.9227 | 0.5994 | 0.5276 | PASS |
| mimic_med_given_v1 | 400 | 0.6625 | 0.505 | 0.5825 | PASS |
| mimic_msdrg_v1 | 400 | 0.46 | 0.19 | 0.185 | PASS |
| mimic_pcs_support_v1 | 400 | 0.625 | 0.505 | 0.825 | PASS |
| mimic_principal_dx_v1 | 400 | 0.705 | 0.1975 | 0.19 | PASS |
| mimic_rad_exam_v1 | 400 | 0.9725 | 0.22 | 0.225 | PASS |

## fastdev_mimic_osler_v2

overall: 5562 items; micro 0.5951; template macro 0.5976; chance 0.3099; status PASS

| template | items | micro | majority | modal share | status |
|---|---|---|---|---|---|
| mimic_apr_mortality_v1 | 400 | 0.44 | 0.3675 | 0.7125 | PASS |
| mimic_apr_severity_v1 | 400 | 0.315 | 0.3375 | 0.4575 | PASS |
| mimic_doc_support_v1 | 400 | 0.9225 | 0.5025 | 0.53 | PASS |
| mimic_drg_cc_level_v1 | 400 | 0.4475 | 0.355 | 0.79 | PASS |
| mimic_ed_acuity_v1 | 400 | 0.25 | 0.5275 | 0.4225 | PASS |
| mimic_ed_disposition_v1 | 400 | 0.62 | 0.61 | 0.5075 | PASS |
| mimic_icd_specificity_v1 | 400 | 0.6875 | 0.1875 | 0.195 | PASS |
| mimic_icd_support_v1 | 400 | 0.48 | 0.2375 | 0.3175 | PASS |
| mimic_lab_validation_v1 | 362 | 0.9558 | 0.5994 | 0.5552 | PASS |
| mimic_med_given_v1 | 400 | 0.6325 | 0.505 | 0.5325 | PASS |
| mimic_msdrg_v1 | 400 | 0.3175 | 0.19 | 0.2 | PASS |
| mimic_pcs_support_v1 | 400 | 0.7 | 0.505 | 0.66 | PASS |
| mimic_principal_dx_v1 | 400 | 0.6275 | 0.1975 | 0.2175 | PASS |
| mimic_rad_exam_v1 | 400 | 0.97 | 0.22 | 0.23 | PASS |

## cg_v2_flash

overall: 8296 items; chance 0.2494; status READOUT_FAIL — 1 of 13 template cells fail the gate (nfcorpus_relevant_noul_v1); the set-level accuracy is not reported as a result (D12)

additional, not a result: micro accuracy over the template cells that pass the gate: 8209 items; micro 0.7908

| template | items | micro | majority | modal share | status |
|---|---|---|---|---|---|
| cg_v2_ddxplus_validate_v1 | 1000 | 0.704 | 0.061 | 0.071 | PASS |
| cg_v2_headqa_dev_v1 | 1340 | 0.8515 | 0.2769 | 0.2754 | PASS |
| cg_v2_mednli_dev_v1 | 995 | 0.8714 | 0.3457 | 0.3839 | PASS |
| cg_v2_pubmedqa_labelled_v1 | 1000 | 0.76 | 0.552 | 0.564 | PASS |
| medmcqa_4opt_v1 | 2744 | 0.7434 | 0.3032 | 0.2861 | PASS |
| medqa_usmle4_v1 | 747 | 0.8782 | 0.2744 | 0.2892 | PASS |
| medquad_routing_v1 | 87 | 1.0 | 0.2874 | 0.2874 | PASS |
| mmlu_mc_v1 | 68 | 0.8971 | 0.3088 | 0.3235 | PASS |
| nfcorpus_relevant_noul_v1 | 87 | 0.5977 | 0.5057 | 0.908 | READOUT_FAIL — constant answer (no) on 0.908 of items |
| pubhealth_claim_true_noul_v1 | 86 | 0.8953 | 0.593 | 0.6047 | PASS |
| pubmedqa_ynm_v1 | 87 | 0.8046 | 0.4598 | 0.5057 | PASS |
| trec_covid_graded_score_v1 | 33 | 0.3939 | 0.3333 | 0.5455 | PASS |
| trec_covid_relevant_noul_v1 | 22 | 0.7273 | 0.5 | 0.7727 | PASS |

## cg_v2_meddecider4b

overall: 7296 items; micro 0.8117; template macro 0.8233; chance 0.2808; status PASS; templates not scored by this run: cg_v2_ddxplus_validate_v1 (C086)

| template | items | micro | majority | modal share | status |
|---|---|---|---|---|---|
| cg_v2_headqa_dev_v1 | 1340 | 0.8216 | 0.2769 | 0.2687 | PASS |
| cg_v2_mednli_dev_v1 | 995 | 0.8653 | 0.3457 | 0.3648 | PASS |
| cg_v2_pubmedqa_labelled_v1 | 1000 | 0.814 | 0.552 | 0.622 | PASS |
| medmcqa_4opt_v1 | 2744 | 0.7569 | 0.3032 | 0.2915 | PASS |
| medqa_usmle4_v1 | 747 | 0.913 | 0.2744 | 0.2838 | PASS |
| medquad_routing_v1 | 87 | 1.0 | 0.2874 | 0.2874 | PASS |
| mmlu_mc_v1 | 68 | 0.8971 | 0.3088 | 0.3382 | PASS |
| nfcorpus_relevant_noul_v1 | 87 | 0.6552 | 0.5057 | 0.8276 | PASS |
| pubhealth_claim_true_noul_v1 | 86 | 0.814 | 0.593 | 0.593 | PASS |
| pubmedqa_ynm_v1 | 87 | 0.8276 | 0.4598 | 0.5287 | PASS |
| trec_covid_graded_score_v1 | 33 | 0.6061 | 0.3333 | 0.4848 | PASS |
| trec_covid_relevant_noul_v1 | 22 | 0.9091 | 0.5 | 0.5909 | PASS |

## cg_v2_osler_v2

overall: 8307 items; micro 0.7565; template macro 0.7982; chance 0.2494; status PASS

| template | items | micro | majority | modal share | status |
|---|---|---|---|---|---|
| cg_v2_ddxplus_validate_v1 | 1000 | 0.431 | 0.061 | 0.166 | PASS |
| cg_v2_headqa_dev_v1 | 1342 | 0.8323 | 0.2772 | 0.275 | PASS |
| cg_v2_mednli_dev_v1 | 995 | 0.8744 | 0.3457 | 0.3367 | PASS |
| cg_v2_pubmedqa_labelled_v1 | 1000 | 0.779 | 0.552 | 0.583 | PASS |
| medmcqa_4opt_v1 | 2753 | 0.7356 | 0.3037 | 0.2957 | PASS |
| medqa_usmle4_v1 | 747 | 0.8728 | 0.2744 | 0.2892 | PASS |
| medquad_routing_v1 | 87 | 1.0 | 0.2874 | 0.2874 | PASS |
| mmlu_mc_v1 | 68 | 0.8529 | 0.3088 | 0.3088 | PASS |
| nfcorpus_relevant_noul_v1 | 87 | 0.8506 | 0.5057 | 0.5402 | PASS |
| pubhealth_claim_true_noul_v1 | 86 | 0.9419 | 0.593 | 0.6047 | PASS |
| pubmedqa_ynm_v1 | 87 | 0.8276 | 0.4598 | 0.5287 | PASS |
| trec_covid_graded_score_v1 | 33 | 0.697 | 0.3333 | 0.5758 | PASS |
| trec_covid_relevant_noul_v1 | 22 | 0.6818 | 0.5 | 0.8182 | PASS |

