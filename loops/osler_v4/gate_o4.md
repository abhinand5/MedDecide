# osler_v4 — Gate O4 (D49): the final Osler-4B against the two gate baselines

Generated 2026-10-25T06:35:08Z by `scripts/osler_v4/y9_gate.py` from `outputs/osler_v4/Y9/gate.json` (aggregates only).
Pre-registration: STATE §7, Y9 pre-registration (recorded before the final model was scored). Pairing per D44(a) as pre-registered; readout per D12; the MIMIC audit suite against MedDecider-4B on the eight passing templates (STATE §6 question 6). Accuracy is the template macro of the paired differences, final minus baseline; Brier is the item mean, final minus baseline (a negative value favours the final model). Bounds: paired bootstrap within templates, 1,000 resamples, seed 0.

## Verdict

| baseline | Gate O4 | (i) MR verdict | (ii) MIMIC audit suite | (iii) external panel | (iv) knowledge |
|---|---|---|---|---|---|
| zero-shot pplx-decider-flash-v1.2 | **FAIL** | PASS (accuracy +0.1387 [+0.1243; +0.1530]; Brier -0.2178 [-0.2261; -0.2095]) | PASS (accuracy +0.1654 [+0.1502; +0.1795]; Brier -0.1805 [-0.1978; -0.1641]) | FAIL (accuracy -0.0421 [-0.0584; -0.0269]; Brier -0.0215 [-0.0335; -0.0099]) | PASS (accuracy -0.0067 [-0.0172; +0.0048]; Brier -0.0363 [-0.0463; -0.0257]) |
| MedDecider-4B | **NOT MEASURED** | PASS (accuracy +0.2449 [+0.2328; +0.2568]; Brier -0.2981 [-0.3069; -0.2895]) | PASS (accuracy +0.1651 [+0.1490; +0.1804]; Brier -0.1915 [-0.2088; -0.1729]) | NOT MEASURED | PASS (accuracy +0.0060 [-0.0071; +0.0194]; Brier +0.0062 [-0.0024; +0.0147]) |

## Checks in detail

### Against zero-shot pplx-decider-flash-v1.2

- (i) i mr_verdict: **PASS**. accuracy lower > 0 and Brier upper < 0.
  items 13103; paired 13103; unpaired none (share 0.00%); dropped templates none; readout failures final none, baseline none.
  accuracy +0.1387 [+0.1243; +0.1530]; Brier -0.2178 [-0.2261; -0.2095]; items 13103; templates ['codiesp_icd_support_v1', 'mkurman_icd_support_v1', 'mr_icd_support_template_v2', 'mr_indicator_evidence_v1', 'mtsamples_page_category_v1', 'pmc_similar_patient_v1', 'synthea_indicator_evidence_v1'].
- (ii) ii mimic_suite: **PASS**. accuracy lower > 0 and Brier upper < 0.
  items 4253; paired 4253; unpaired none (share 0.00%); dropped templates none; readout failures final none, baseline none.
  accuracy +0.1654 [+0.1502; +0.1795]; Brier -0.1805 [-0.1978; -0.1641]; items 4253; templates ['mimic_drg_cc_level_v1', 'mimic_ed_acuity_v1', 'mimic_ed_disposition_v1', 'mimic_icd_specificity_v1', 'mimic_icd_support_v1', 'mimic_lab_validation_v1', 'mimic_med_given_v1', 'mimic_msdrg_v1', 'mimic_principal_dx_v1'].
- (iii) iii panel: **FAIL**. accuracy lower > -0.02.
  items 4353; paired 4353; unpaired none (share 0.00%); dropped templates none; readout failures final none, baseline none.
  accuracy -0.0421 [-0.0584; -0.0269]; Brier -0.0215 [-0.0335; -0.0099]; items 4353; templates ['ext_medconceptsqa_choice_v1', 'ext_medexpqa_en_choice_v1', 'ext_medexqa_choice_v1', 'ext_medical_question_pairs_noul_v1', 'ext_medxpertqa_text_choice_v1', 'ext_mmlu_pro_health_choice_v1', 'ext_symptom_to_diagnosis_choice_v1'].
- (iv) iv knowledge: **PASS**. accuracy lower > -0.02.
  items 5456; paired 5435; unpaired {'not scored by the baseline': 21} (share 0.38%); dropped templates none; readout failures final none, baseline none.
  accuracy -0.0067 [-0.0172; +0.0048]; Brier -0.0363 [-0.0463; -0.0257]; items 5435; templates ['medmcqa_4opt_v1', 'medqa_usmle4_v1'].

### Against MedDecider-4B

- (i) i mr_verdict: **PASS**. accuracy lower > 0 and Brier upper < 0.
  items 13103; paired 12992; unpaired {'more than 10 options': 111} (share 0.85%); dropped templates ['mtsamples_page_category_v1']; readout failures final none, baseline none.
  accuracy +0.2449 [+0.2328; +0.2568]; Brier -0.2981 [-0.3069; -0.2895]; items 12992; templates ['codiesp_icd_support_v1', 'mkurman_icd_support_v1', 'mr_icd_support_template_v2', 'mr_indicator_evidence_v1', 'pmc_similar_patient_v1', 'synthea_indicator_evidence_v1'].
- (ii) ii mimic_suite: **PASS**. accuracy lower > 0 and Brier upper < 0; restricted to the 8 templates that pass for both models (STATE §6 question 6: the eight templates that pass for MedDecider-4B on the MIMIC audit suite).
  items 4253; paired 4253; unpaired none (share 0.00%); dropped templates none; readout failures final none, baseline ['mimic_drg_cc_level_v1']; restricted to 8 templates.
  accuracy +0.1651 [+0.1490; +0.1804]; Brier -0.1915 [-0.2088; -0.1729]; items 3755; templates ['mimic_ed_acuity_v1', 'mimic_ed_disposition_v1', 'mimic_icd_specificity_v1', 'mimic_icd_support_v1', 'mimic_lab_validation_v1', 'mimic_med_given_v1', 'mimic_msdrg_v1', 'mimic_principal_dx_v1'].
- (iii) iii panel: **NOT MEASURED**. 212 of 4353 items unpaired (4.87%, above 1 %; D44(a)).
  items 4353; paired 4141; unpaired {'more than 10 options': 212} (share 4.87%); dropped templates ['ext_symptom_to_diagnosis_choice_v1']; readout failures final none, baseline none.
- (iv) iv knowledge: **PASS**. accuracy lower > -0.02.
  items 5456; paired 5435; unpaired {'not scored by the baseline': 21} (share 0.38%); dropped templates none; readout failures final none, baseline none.
  accuracy +0.0060 [-0.0071; +0.0194]; Brier +0.0062 [-0.0024; +0.0147]; items 5435; templates ['medmcqa_4opt_v1', 'medqa_usmle4_v1'].

## Additional rows (labelled; no verdict)

### Against zero-shot pplx-decider-flash-v1.2
- mr_reported: accuracy +0.0040 [+0.0005; +0.0091]; Brier -0.0039 [-0.0049; -0.0029]; items 6191; unpaired none.
- mr_heldout_test: accuracy +0.0731 [+0.0150; +0.1383]; Brier +0.1616 [+0.1250; +0.1976]; items 584; unpaired none.
- mimic_heldout: accuracy -0.0815 [-0.2236; +0.0565]; Brier +0.1700 [+0.0733; +0.2792]; items 110; unpaired none.
- mednli_test: accuracy +0.0042 [-0.0120; +0.0211]; Brier -0.0197 [-0.0357; -0.0040]; items 1422; unpaired none.
- pairs_test: accuracy +0.0920 [+0.0640; +0.1240]; Brier -0.1208 [-0.1551; -0.0891]; items 500; unpaired none.
- panel: accuracy -0.0421 [-0.0584; -0.0269]; Brier -0.0215 [-0.0335; -0.0099]; items 4353; unpaired none.
- mr_verdict_without_osler_v1_flags (post hoc; additional): accuracy +0.1257 [+0.1066; +0.1459]; Brier -0.2384 [-0.2544; -0.2229]; items 4449; unpaired none.
- panel per dataset (accuracy, final minus baseline):
  - ext_medconceptsqa_choice_v1: -0.0550 [-0.0850; -0.0230] on 1000 items
  - ext_medexpqa_en_choice_v1: +0.0080 [-0.0400; +0.0560] on 125 items
  - ext_medexqa_choice_v1: -0.0128 [-0.0287; +0.0021] on 940 items
  - ext_medical_question_pairs_noul_v1: -0.0847 [-0.1288; -0.0406] on 295 items
  - ext_medxpertqa_text_choice_v1: -0.0220 [-0.0460; +0.0020] on 1000 items
  - ext_mmlu_pro_health_choice_v1: -0.0294 [-0.0563; -0.0038] on 781 items
  - ext_symptom_to_diagnosis_choice_v1: -0.0991 [-0.1604; -0.0377] on 212 items

### Against MedDecider-4B
- mr_reported: NOT MEASURED (no paired items)
- mr_heldout_test: accuracy +0.1742 [+0.1037; +0.2541]; Brier +0.1319 [+0.0978; +0.1662]; items 584; unpaired none.
- mimic_heldout: accuracy -0.0792 [-0.2047; +0.0380]; Brier +0.1012 [-0.0214; +0.2328]; items 110; unpaired none.
- mednli_test: accuracy +0.0274 [+0.0098; +0.0450]; Brier -0.0511 [-0.0673; -0.0337]; items 1422; unpaired none.
- pairs_test: accuracy +0.2740 [+0.2320; +0.3160]; Brier -0.3071 [-0.3516; -0.2594]; items 500; unpaired none.
- panel: accuracy -0.0163 [-0.0332; +0.0001]; Brier +0.0412 [+0.0313; +0.0521]; items 4141; unpaired {'more than 10 options': 212}.
- mr_verdict_without_osler_v1_flags (post hoc; additional): accuracy +0.2545 [+0.2394; +0.2688]; Brier -0.3974 [-0.4120; -0.3810]; items 4338; unpaired none.
- panel per dataset (accuracy, final minus baseline):
  - ext_medconceptsqa_choice_v1: -0.0460 [-0.0810; -0.0130] on 1000 items
  - ext_medexpqa_en_choice_v1: +0.0560 [-0.0160; +0.1202] on 125 items
  - ext_medexqa_choice_v1: -0.0085 [-0.0266; +0.0096] on 940 items
  - ext_medical_question_pairs_noul_v1: -0.0746 [-0.1186; -0.0304] on 295 items
  - ext_medxpertqa_text_choice_v1: +0.0060 [-0.0220; +0.0340] on 1000 items
  - ext_mmlu_pro_health_choice_v1: -0.0307 [-0.0563; -0.0026] on 781 items

