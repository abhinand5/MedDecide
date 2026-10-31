# d1-2B local benchmark (bench/d1-2b): aggregates only

Generated 2026-10-31T08:42:38Z by `scripts/bench/d1_report.py` from `outputs/d1_bench/<set>/predictions.jsonl` (gitignored; private). Aggregates only: no item, note, prediction or label text is in this file.

- Model: `LiquidAI/d1-2B` at revision `10e7671d44e4b37700020b1eed3f594c5f2af0b4` (LFM Open License v1.0; the license terms are in `loops/d1_bench/STATE.md`).
- Software: transformers 5.18.0 (the card asks for >=5.19; see STATE), torch 2.14.1+cu130, GPU NVIDIA RTX PRO 6000 Blackwell Server Edition, 595.91.07. Calibration: the checkpoint's `calibration.json` is applied by the card's engine.
- Definitions (as `loops/osler_v4/results.md`): template macro accuracy = mean over templates of each template's mean correctness; item micro accuracy = mean correctness over all scored items. The readout status is the D12 gate (`meddecide.eval.readout_gate`, D21 variant for a trained readout that normalises over the offered options): a cell fails on a constant answer or when its 95 % upper bound is below chance.
- Prompt protocol: one state and one question per call, via the card's `system_one` (noul: P(yes); choice: a distribution over the item's option keys; score: a distribution over its levels, lowest first). No text is generated.

## 1. d1-2B on the frozen sets

| set | items | scored | templates | template macro accuracy | item micro accuracy | readout gate (D12/D21) |
|---|---:|---:|---:|---:|---:|---|
| Knowledge test (MedQA + MedMCQA) | 5456 | 5456 | 2 | 0.4705 | 0.4555 | PASS |
| MedNLI test | 1422 | 1422 | 1 | 0.8080 | 0.8080 | PASS |
| External clinical panel (7 datasets) | 4353 | 4353 | 7 | 0.4810 | 0.4259 | PASS |
| MR verdict suite v1 | 13103 | 13103 | 7 | 0.6085 | 0.7069 | PASS |
| MIMIC audit suite v1 (local only) | 4253 | 4253 | 9 | 0.5704 | 0.5587 | READOUT_FAIL on 1 template(s): mimic_drg_cc_level_v1 |

## 2. Readout-gate cells (every template of every set)

| set | template | items | accuracy | chance (mean 1/n) | modal answer share | 95 % upper bound | status |
|---|---|---:|---:|---:|---:|---:|---|
| knowledge | medmcqa_4opt_v1 | 4183 | 0.4423 | 0.2500 | 0.2649 | 0.4571 | PASS |
| knowledge | medqa_usmle4_v1 | 1273 | 0.4988 | 0.2500 | 0.2718 | 0.5255 | PASS |
| mednli_test | mednli_v1 | 1422 | 0.8080 | 0.3333 | 0.3748 | 0.8277 | PASS |
| panel | ext_medconceptsqa_choice_v1 | 1000 | 0.3240 | 0.2500 | 0.2740 | 0.3530 | PASS |
| panel | ext_medexpqa_en_choice_v1 | 125 | 0.4640 | 0.2000 | 0.2240 | 0.5440 | PASS |
| panel | ext_medexqa_choice_v1 | 940 | 0.6819 | 0.2500 | 0.3415 | 0.7117 | PASS |
| panel | ext_medical_question_pairs_noul_v1 | 295 | 0.8441 | 0.5000 | 0.6169 | 0.8847 | PASS |
| panel | ext_medxpertqa_text_choice_v1 | 1000 | 0.1650 | 0.1000 | 0.1340 | 0.1890 | PASS |
| panel | ext_mmlu_pro_health_choice_v1 | 781 | 0.4020 | 0.1129 | 0.1383 | 0.4379 | PASS |
| panel | ext_symptom_to_diagnosis_choice_v1 | 212 | 0.4858 | 0.0455 | 0.1509 | 0.5472 | PASS |
| mr_verdict | codiesp_icd_support_v1 | 1654 | 0.3059 | 0.1296 | 0.3174 | 0.3283 | PASS |
| mr_verdict | mkurman_icd_support_v1 | 1490 | 0.2651 | 0.1226 | 0.3685 | 0.2866 | PASS |
| mr_verdict | mr_icd_support_template_v2 | 554 | 0.7274 | 0.2788 | 0.3917 | 0.7635 | PASS |
| mr_verdict | mr_indicator_evidence_v1 | 576 | 0.6892 | 0.5000 | 0.6753 | 0.7274 | PASS |
| mr_verdict | mtsamples_page_category_v1 | 111 | 0.5946 | 0.0833 | 0.2973 | 0.6847 | PASS |
| mr_verdict | pmc_similar_patient_v1 | 8100 | 0.8632 | 0.5000 | 0.6089 | 0.8702 | PASS |
| mr_verdict | synthea_indicator_evidence_v1 | 618 | 0.8139 | 0.5000 | 0.5793 | 0.8431 | PASS |
| mimic_suite | mimic_drg_cc_level_v1 | 498 | 0.3434 | 0.3333 | 0.9799 | 0.3855 | READOUT_FAIL — constant answer (C) on 0.980 of items |
| mimic_suite | mimic_ed_acuity_v1 | 494 | 0.5121 | 0.2000 | 0.5526 | 0.5526 | PASS |
| mimic_suite | mimic_ed_disposition_v1 | 497 | 0.5050 | 0.3333 | 0.7082 | 0.5473 | PASS |
| mimic_suite | mimic_icd_specificity_v1 | 488 | 0.6189 | 0.1753 | 0.2131 | 0.6619 | PASS |
| mimic_suite | mimic_icd_support_v1 | 400 | 0.5975 | 0.2728 | 0.2550 | 0.6450 | PASS |
| mimic_suite | mimic_lab_validation_v1 | 376 | 0.9548 | 0.5000 | 0.5824 | 0.9734 | PASS |
| mimic_suite | mimic_med_given_v1 | 500 | 0.7160 | 0.5000 | 0.5880 | 0.7540 | PASS |
| mimic_suite | mimic_msdrg_v1 | 500 | 0.2940 | 0.1679 | 0.2140 | 0.3340 | PASS |
| mimic_suite | mimic_principal_dx_v1 | 500 | 0.5920 | 0.1712 | 0.1740 | 0.6360 | PASS |

## 3. Comparison with zero-shot comparators (additional; no verdict)

Comparator values are transcribed from the claim rows named in the last column (`loops/osler_v5/CLAIMS.md` on `loop/osler_v5`, and `docs/paper/EVIDENCE.md` §5.2 on the same branch). The readout status of Qwen3.5-4B and Clef-Flash is PASS for every cell (C089). `NOT MEASURED` means the permitted sources give no such figure.

| set | model | template macro accuracy | item micro accuracy | source |
|---|---|---:|---:|---|
| Knowledge test (MedQA + MedMCQA) | d1-2B (this run) | 0.4705 | 0.4555 | this run (section 1) |
| Knowledge test (MedQA + MedMCQA) | Qwen3.5-4B zero-shot | 0.6450 | 0.6162 | C095 |
| Knowledge test (MedQA + MedMCQA) | Clef-Flash | 0.7105 | 0.6864 | C095 |
| Knowledge test (MedQA + MedMCQA) | pplx-decider-flash-v1.2 zero-shot | NOT MEASURED | 0.6835 | C096 (micro on the 5,435 items it scored; macro not stated) |
| MedNLI test | d1-2B (this run) | 0.8080 | 0.8080 | this run (section 1) |
| MedNLI test | Qwen3.5-4B zero-shot | 0.7862 | 0.7862 | C092 (one template; macro = micro) |
| MedNLI test | Clef-Flash | 0.8636 | 0.8636 | C092 (one template; macro = micro) |
| MedNLI test | pplx-decider-flash-v1.2 zero-shot | NOT MEASURED | NOT MEASURED | NOT MEASURED in the permitted sources |
| External clinical panel (7 datasets) | d1-2B (this run) | 0.4810 | 0.4259 | this run (section 1) |
| External clinical panel (7 datasets) | Qwen3.5-4B zero-shot | 0.5483 | 0.4831 | C093 |
| External clinical panel (7 datasets) | Clef-Flash | 0.6502 | 0.5711 | C093 |
| External clinical panel (7 datasets) | pplx-decider-flash-v1.2 zero-shot | NOT MEASURED | NOT MEASURED | NOT MEASURED in the permitted sources |
| MR verdict suite v1 | d1-2B (this run) | 0.6085 | 0.7069 | this run (section 1) |
| MR verdict suite v1 | Qwen3.5-4B zero-shot | 0.5574 | 0.6176 | C090 |
| MR verdict suite v1 | Clef-Flash | 0.6830 | 0.6830 | C090 |
| MR verdict suite v1 | pplx-decider-flash-v1.2 zero-shot | 0.7044 | NOT MEASURED | macro: EVIDENCE.md 5.2 (osler_v4 scoreboard, C228); micro NOT MEASURED |
| MIMIC audit suite v1 (local only) | d1-2B (this run) | 0.5704 | 0.5587 | this run (section 1) |
| MIMIC audit suite v1 (local only) | Qwen3.5-4B zero-shot | 0.5697 | 0.5603 | C091 |
| MIMIC audit suite v1 (local only) | Clef-Flash | 0.6391 | 0.6306 | C091 |
| MIMIC audit suite v1 (local only) | pplx-decider-flash-v1.2 zero-shot | 0.6341 | NOT MEASURED | macro: EVIDENCE.md 5.2 (osler_v4 scoreboard, C228); micro NOT MEASURED |

## 4. Additional diagnostics (labelled; no verdict)

- Knowledge test (MedQA + MedMCQA): answer types {'choice': 5456}.
- MedNLI test: answer types {'choice': 1422}.
- External clinical panel (7 datasets): answer types {'choice': 4058, 'noul': 295}.
- MR verdict suite v1: answer types {'noul': 9294, 'choice': 3809}.
- MIMIC audit suite v1 (local only): answer types {'choice': 2883, 'score': 494, 'noul': 876}; score items 494; argmax of the distribution differs from the card's median level on 172.

## 5. Claims (R1)

| id | claim | value | artifact | recompute |
|---|---|---|---|---|
| D1-1 | Knowledge test (MedQA + MedMCQA): template macro, item micro, items | macro 0.4705; micro 0.4555; 5456 items | `outputs/d1_bench/knowledge/predictions.jsonl` | `uv run --frozen --no-sync python scripts/bench/d1_report.py` |
| D1-2 | MedNLI test: template macro, item micro, items | macro 0.8080; micro 0.8080; 1422 items | `outputs/d1_bench/mednli_test/predictions.jsonl` | `uv run --frozen --no-sync python scripts/bench/d1_report.py` |
| D1-3 | External clinical panel (7 datasets): template macro, item micro, items | macro 0.4810; micro 0.4259; 4353 items | `outputs/d1_bench/panel/predictions.jsonl` | `uv run --frozen --no-sync python scripts/bench/d1_report.py` |
| D1-4 | MR verdict suite v1: template macro, item micro, items | macro 0.6085; micro 0.7069; 13103 items | `outputs/d1_bench/mr_verdict/predictions.jsonl` | `uv run --frozen --no-sync python scripts/bench/d1_report.py` |
| D1-5 | MIMIC audit suite v1 (local only): template macro, item micro, items | macro 0.5704; micro 0.5587; 4253 items | `outputs/d1_bench/mimic_suite/predictions.jsonl` | `uv run --frozen --no-sync python scripts/bench/d1_report.py` |

Re-derivation and self-audit: see `loops/d1_bench/STATE.md`.
