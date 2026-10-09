# Zero-shot ladder baselines on MedDecide-Bench v0.1 (F6)

Generated: `2026-10-06T15:39:07Z`  
Models: `LiquidAI/LFM2.5-350M`, `google/medgemma-1.5-4b-it`, `Qwen/Qwen3.5-0.8B-Base`, `Qwen/Qwen3.5-0.8B`, `Qwen/Qwen3.5-4B`, `Qwen/Qwen3.5-9B`  
Cells: 96 (91 PASS, 5 READOUT_FAIL, 0 NOT MEASURED)  
Prediction rows: 181464

Protocol: the fixed T6 protocol (chat template, thinking disabled, single-letter
instruction) with the **F2 readout**, which renders every question type with letter labels
and reads the option-letter tokens. The `choice` path is byte-identical to the path
validated against lm-evaluation-harness in loop bench_v0; `noul`/`score` now go through the
same path (see `readout_validation.md`).

**Reporting rule (D12).** An accuracy appears below only when its cell passes the
readout-health gate: median label mass >= 0.5, greedy agreement >= 0.9 on a seeded sample,
and an accuracy CI whose upper bound is not below chance. A failing cell reads
`READOUT_FAIL — <check>` **and no accuracy is shown for it**. A cell that could not be
measured reads `NOT MEASURED — <reason>`. `coverage` is the share of *this template's* test
items the model could take (it was the share of the whole source in bench_v0).

Batch size 8, max batch 32k tokens, greedy sample 50 items per cell, option-shuffle flip
rate on up to 500 items per template. Predictions stay in the gitignored `outputs/`.

## Tier 1 (established public test sets)

| model | source | template | qtype | n | coverage | accuracy (95% CI) | majority | chance | label mass | greedy | flip |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `LiquidAI/LFM2.5-350M` | medmcqa | `medmcqa_4opt_v1` | choice | 4183/4183 | 1.000 | 0.3246 (0.3105-0.3390) | 0.322 | 0.250 | 1.0000 | 1.000 | 0.706 |
| `LiquidAI/LFM2.5-350M` | medqa | `medqa_usmle4_v1` | choice | 1273/1273 | 1.000 | 0.2914 (0.2663-0.3158) | 0.277 | 0.250 | 1.0000 | 1.000 | 0.696 |
| `LiquidAI/LFM2.5-350M` | medquad | `medquad_routing_v1` | choice | 5000/5000 | 1.000 | 0.4870 (0.4730-0.5008) | 0.252 | 0.250 | 1.0000 | 0.980 | 0.668 |
| `LiquidAI/LFM2.5-350M` | mmlu_medical | `mmlu_mc_v1` | choice | 1103/1103 | 1.000 | 0.2412 (0.2149-0.2675) | 0.325 | 0.250 | 1.0000 | 0.980 | 0.738 |
| `LiquidAI/LFM2.5-350M` | nfcorpus | `nfcorpus_graded_score_v1` | score | 429/429 | 1.000 | 0.7226 (0.6806-0.7646) | 0.723 | 0.333 | 1.0000 | 1.000 | 0.326 |
| `LiquidAI/LFM2.5-350M` | nfcorpus | `nfcorpus_relevant_noul_v1` | noul | 647/647 | 1.000 | 0.5023 (0.4621-0.5395) | 0.501 | 0.500 | 1.0000 | 1.000 | 0.196 |
| `LiquidAI/LFM2.5-350M` | pubmedqa | `pubmedqa_ynm_v1` | choice | 477/477 | 1.000 | 0.4256 (0.3816-0.4717) | 0.551 | 0.333 | 0.9995 | 0.940 | 0.681 |
| `LiquidAI/LFM2.5-350M` | scifact | `scifact_relevant_noul_v1` | noul | 600/600 | 1.000 | 0.5033 (0.4633-0.5434) | 0.500 | 0.500 | 1.0000 | 1.000 | 0.140 |
| `Qwen/Qwen3.5-0.8B` | medmcqa | `medmcqa_4opt_v1` | choice | 4183/4183 | 1.000 | 0.3792 (0.3648-0.3938) | 0.322 | 0.250 | 0.9967 | 0.940 | 0.514 |
| `Qwen/Qwen3.5-0.8B` | medqa | `medqa_usmle4_v1` | choice | 1273/1273 | 1.000 | 0.4140 (0.3865-0.4415) | 0.277 | 0.250 | 0.9959 | 1.000 | 0.378 |
| `Qwen/Qwen3.5-0.8B` | medquad | `medquad_routing_v1` | choice | 5000/5000 | 1.000 | 0.9346 (0.9276-0.9412) | 0.252 | 0.250 | 0.9988 | 1.000 | 0.078 |
| `Qwen/Qwen3.5-0.8B` | mmlu_medical | `mmlu_mc_v1` | choice | 1103/1103 | 1.000 | 0.4606 (0.4333-0.4905) | 0.325 | 0.250 | 0.9968 | 0.920 | 0.394 |
| `Qwen/Qwen3.5-0.8B` | nfcorpus | `nfcorpus_graded_score_v1` | score | 429/429 | 1.000 | 0.7226 (0.6806-0.7646) | 0.723 | 0.333 | 0.9979 | 1.000 | 0.238 |
| `Qwen/Qwen3.5-0.8B` | nfcorpus | `nfcorpus_relevant_noul_v1` | noul | 647/647 | 1.000 | 0.5440 (0.5070-0.5811) | 0.501 | 0.500 | 0.9964 | 0.980 | 0.040 |
| `Qwen/Qwen3.5-0.8B` | pubmedqa | `pubmedqa_ynm_v1` | choice | 477/477 | 1.000 | 0.6876 (0.6478-0.7275) | 0.551 | 0.333 | 0.9923 | 1.000 | 0.254 |
| `Qwen/Qwen3.5-0.8B` | scifact | `scifact_relevant_noul_v1` | noul | 600/600 | 1.000 | 0.6100 (0.5733-0.6500) | 0.500 | 0.500 | 0.9952 | 0.960 | 0.084 |
| `Qwen/Qwen3.5-0.8B-Base` | medmcqa | `medmcqa_4opt_v1` | choice | 4183/4183 | 1.000 | READOUT_FAIL — greedy agreement 0.880 < 0.9 (n=50) | 0.322 | 0.250 | 0.9326 | 0.880 | 0.522 |
| `Qwen/Qwen3.5-0.8B-Base` | medqa | `medqa_usmle4_v1` | choice | 1273/1273 | 1.000 | 0.3716 (0.3440-0.3983) | 0.277 | 0.250 | 0.8284 | 1.000 | 0.438 |
| `Qwen/Qwen3.5-0.8B-Base` | medquad | `medquad_routing_v1` | choice | 5000/5000 | 1.000 | 0.9420 (0.9354-0.9484) | 0.252 | 0.250 | 0.9236 | 1.000 | 0.050 |
| `Qwen/Qwen3.5-0.8B-Base` | mmlu_medical | `mmlu_mc_v1` | choice | 1103/1103 | 1.000 | 0.4569 (0.4288-0.4869) | 0.325 | 0.250 | 0.9155 | 1.000 | 0.408 |
| `Qwen/Qwen3.5-0.8B-Base` | nfcorpus | `nfcorpus_graded_score_v1` | score | 429/429 | 1.000 | 0.7226 (0.6806-0.7646) | 0.723 | 0.333 | 0.9179 | 1.000 | 0.520 |
| `Qwen/Qwen3.5-0.8B-Base` | nfcorpus | `nfcorpus_relevant_noul_v1` | noul | 647/647 | 1.000 | 0.5023 (0.4621-0.5394) | 0.501 | 0.500 | 0.8927 | 1.000 | 0.000 |
| `Qwen/Qwen3.5-0.8B-Base` | pubmedqa | `pubmedqa_ynm_v1` | choice | 477/477 | 1.000 | 0.6247 (0.5828-0.6667) | 0.551 | 0.333 | 0.9052 | 0.960 | 0.491 |
| `Qwen/Qwen3.5-0.8B-Base` | scifact | `scifact_relevant_noul_v1` | noul | 600/600 | 1.000 | 0.5017 (0.4617-0.5433) | 0.500 | 0.500 | 0.8805 | 1.000 | 0.002 |
| `Qwen/Qwen3.5-4B` | medmcqa | `medmcqa_4opt_v1` | choice | 4183/4183 | 1.000 | 0.5893 (0.5740-0.6029) | 0.322 | 0.250 | 0.9937 | 0.940 | 0.246 |
| `Qwen/Qwen3.5-4B` | medqa | `medqa_usmle4_v1` | choice | 1273/1273 | 1.000 | 0.7031 (0.6779-0.7259) | 0.277 | 0.250 | 0.9964 | 1.000 | 0.210 |
| `Qwen/Qwen3.5-4B` | medquad | `medquad_routing_v1` | choice | 5000/5000 | 1.000 | 0.9848 (0.9818-0.9878) | 0.252 | 0.250 | 0.9999 | 1.000 | 0.006 |
| `Qwen/Qwen3.5-4B` | mmlu_medical | `mmlu_mc_v1` | choice | 1103/1103 | 1.000 | 0.8169 (0.7915-0.8422) | 0.325 | 0.250 | 0.9958 | 0.980 | 0.114 |
| `Qwen/Qwen3.5-4B` | nfcorpus | `nfcorpus_graded_score_v1` | score | 429/429 | 1.000 | READOUT_FAIL — accuracy CI upper bound 0.2868 < chance 0.3333 (below-chance cell) | 0.723 | 0.333 | 0.9965 | 1.000 | 0.198 |
| `Qwen/Qwen3.5-4B` | nfcorpus | `nfcorpus_relevant_noul_v1` | noul | 647/647 | 1.000 | 0.6445 (0.6090-0.6832) | 0.501 | 0.500 | 0.9995 | 1.000 | 0.016 |
| `Qwen/Qwen3.5-4B` | pubmedqa | `pubmedqa_ynm_v1` | choice | 477/477 | 1.000 | 0.7358 (0.6960-0.7757) | 0.551 | 0.333 | 0.9992 | 1.000 | 0.059 |
| `Qwen/Qwen3.5-4B` | scifact | `scifact_relevant_noul_v1` | noul | 600/600 | 1.000 | 0.8200 (0.7883-0.8483) | 0.500 | 0.500 | 0.9995 | 1.000 | 0.022 |
| `Qwen/Qwen3.5-9B` | medmcqa | `medmcqa_4opt_v1` | choice | 4183/4183 | 1.000 | 0.6562 (0.6421-0.6706) | 0.322 | 0.250 | 0.9962 | 1.000 | 0.204 |
| `Qwen/Qwen3.5-9B` | medqa | `medqa_usmle4_v1` | choice | 1273/1273 | 1.000 | 0.7573 (0.7329-0.7800) | 0.277 | 0.250 | 0.9986 | 0.980 | 0.144 |
| `Qwen/Qwen3.5-9B` | medquad | `medquad_routing_v1` | choice | 5000/5000 | 1.000 | 0.9894 (0.9868-0.9924) | 0.252 | 0.250 | 0.9984 | 1.000 | 0.006 |
| `Qwen/Qwen3.5-9B` | mmlu_medical | `mmlu_mc_v1` | choice | 1103/1103 | 1.000 | 0.8586 (0.8377-0.8776) | 0.325 | 0.250 | 0.9973 | 0.980 | 0.092 |
| `Qwen/Qwen3.5-9B` | nfcorpus | `nfcorpus_graded_score_v1` | score | 429/429 | 1.000 | READOUT_FAIL — accuracy CI upper bound 0.2308 < chance 0.3333 (below-chance cell) | 0.723 | 0.333 | 0.9967 | 0.980 | 0.061 |
| `Qwen/Qwen3.5-9B` | nfcorpus | `nfcorpus_relevant_noul_v1` | noul | 647/647 | 1.000 | 0.6414 (0.6058-0.6801) | 0.501 | 0.500 | 0.9985 | 1.000 | 0.012 |
| `Qwen/Qwen3.5-9B` | pubmedqa | `pubmedqa_ynm_v1` | choice | 477/477 | 1.000 | 0.7002 (0.6583-0.7380) | 0.551 | 0.333 | 0.9954 | 1.000 | 0.115 |
| `Qwen/Qwen3.5-9B` | scifact | `scifact_relevant_noul_v1` | noul | 600/600 | 1.000 | 0.8433 (0.8133-0.8717) | 0.500 | 0.500 | 0.9980 | 1.000 | 0.026 |
| `google/medgemma-1.5-4b-it` | medmcqa | `medmcqa_4opt_v1` | choice | 4183/4183 | 1.000 | 0.4210 (0.4071-0.4363) | 0.322 | 0.250 | 0.8037 | 0.960 | 0.570 |
| `google/medgemma-1.5-4b-it` | medqa | `medqa_usmle4_v1` | choice | 1273/1273 | 1.000 | READOUT_FAIL — greedy agreement 0.860 < 0.9 (n=50) | 0.277 | 0.250 | 0.6811 | 0.860 | 0.456 |
| `google/medgemma-1.5-4b-it` | medquad | `medquad_routing_v1` | choice | 5000/5000 | 1.000 | 0.9350 (0.9280-0.9418) | 0.252 | 0.250 | 0.9275 | 0.980 | 0.088 |
| `google/medgemma-1.5-4b-it` | mmlu_medical | `mmlu_mc_v1` | choice | 1103/1103 | 1.000 | 0.4769 (0.4479-0.5059) | 0.325 | 0.250 | 0.7924 | 0.900 | 0.464 |
| `google/medgemma-1.5-4b-it` | nfcorpus | `nfcorpus_graded_score_v1` | score | 429/429 | 1.000 | 0.7226 (0.6806-0.7646) | 0.723 | 0.333 | 0.9666 | 1.000 | 0.193 |
| `google/medgemma-1.5-4b-it` | nfcorpus | `nfcorpus_relevant_noul_v1` | noul | 647/647 | 1.000 | 0.5008 (0.4621-0.5379) | 0.501 | 0.500 | 0.9703 | 1.000 | 0.000 |
| `google/medgemma-1.5-4b-it` | pubmedqa | `pubmedqa_ynm_v1` | choice | 477/477 | 1.000 | 0.5094 (0.4654-0.5514) | 0.551 | 0.333 | 0.8958 | 1.000 | 0.147 |
| `google/medgemma-1.5-4b-it` | scifact | `scifact_relevant_noul_v1` | noul | 600/600 | 1.000 | 0.5000 (0.4600-0.5417) | 0.500 | 0.500 | 0.9541 | 1.000 | 0.000 |

## Tier 2 (fresh, 2026-03-01 window)

| model | source | template | qtype | n | coverage | accuracy (95% CI) | majority | chance | label mass | greedy | flip |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `LiquidAI/LFM2.5-350M` | clinicaltrials | `ct_healthy_volunteers_noul_v1` | noul | 2000/2000 | 1.000 | 0.4995 (0.4775-0.5225) | 0.500 | 0.500 | 0.9999 | 1.000 | 0.322 |
| `LiquidAI/LFM2.5-350M` | clinicaltrials | `ct_phase_choice_v1` | choice | 1100/1100 | 1.000 | 0.2182 (0.1936-0.2436) | 0.200 | 0.143 | 1.0000 | 0.900 | 0.842 |
| `LiquidAI/LFM2.5-350M` | clinicaltrials | `ct_randomised_noul_v1` | noul | 2000/2000 | 1.000 | 0.5030 (0.4815-0.5260) | 0.500 | 0.500 | 0.9999 | 1.000 | 0.388 |
| `LiquidAI/LFM2.5-350M` | openfda | `fda_boxed_warning_noul_v1` | noul | 1910/1910 | 1.000 | 0.5136 (0.4916-0.5351) | 0.500 | 0.500 | 0.9999 | 0.940 | 0.456 |
| `LiquidAI/LFM2.5-350M` | openfda | `fda_class_choice_v1` | choice | 4000/4000 | 1.000 | 0.5152 (0.4995-0.5302) | 0.250 | 0.250 | 1.0000 | 0.940 | 0.512 |
| `LiquidAI/LFM2.5-350M` | pubmed | `pubmed_humans_noul_v1` | noul | 1316/1316 | 1.000 | 0.4970 (0.4704-0.5258) | 0.500 | 0.500 | 1.0000 | 1.000 | 0.496 |
| `LiquidAI/LFM2.5-350M` | pubmed | `pubmed_mesh_major_choice_v1` | choice | 4000/4000 | 1.000 | 0.5477 (0.5332-0.5623) | 0.250 | 0.250 | 1.0000 | 1.000 | 0.556 |
| `LiquidAI/LFM2.5-350M` | pubmed | `pubmed_observational_noul_v1` | noul | 206/206 | 1.000 | 0.5000 (0.4320-0.5631) | 0.500 | 0.500 | 0.9999 | 1.000 | 0.534 |
| `Qwen/Qwen3.5-0.8B` | clinicaltrials | `ct_healthy_volunteers_noul_v1` | noul | 2000/2000 | 1.000 | 0.6035 (0.5815-0.6235) | 0.500 | 0.500 | 0.9879 | 1.000 | 0.410 |
| `Qwen/Qwen3.5-0.8B` | clinicaltrials | `ct_phase_choice_v1` | choice | 1100/1100 | 1.000 | 0.3455 (0.3182-0.3736) | 0.200 | 0.143 | 0.9981 | 1.000 | 0.636 |
| `Qwen/Qwen3.5-0.8B` | clinicaltrials | `ct_randomised_noul_v1` | noul | 2000/2000 | 1.000 | 0.7955 (0.7775-0.8140) | 0.500 | 0.500 | 0.9902 | 0.980 | 0.288 |
| `Qwen/Qwen3.5-0.8B` | openfda | `fda_boxed_warning_noul_v1` | noul | 1910/1910 | 1.000 | 0.6157 (0.5927-0.6393) | 0.500 | 0.500 | 0.9920 | 0.940 | 0.496 |
| `Qwen/Qwen3.5-0.8B` | openfda | `fda_class_choice_v1` | choice | 4000/4000 | 1.000 | 0.8400 (0.8283-0.8508) | 0.250 | 0.250 | 0.9984 | 0.960 | 0.130 |
| `Qwen/Qwen3.5-0.8B` | pubmed | `pubmed_humans_noul_v1` | noul | 1316/1316 | 1.000 | 0.6345 (0.6064-0.6649) | 0.500 | 0.500 | 0.9964 | 1.000 | 0.454 |
| `Qwen/Qwen3.5-0.8B` | pubmed | `pubmed_mesh_major_choice_v1` | choice | 4000/4000 | 1.000 | 0.9872 (0.9838-0.9905) | 0.250 | 0.250 | 0.9992 | 1.000 | 0.008 |
| `Qwen/Qwen3.5-0.8B` | pubmed | `pubmed_observational_noul_v1` | noul | 206/206 | 1.000 | 0.7087 (0.6456-0.7718) | 0.500 | 0.500 | 0.9970 | 0.940 | 0.403 |
| `Qwen/Qwen3.5-0.8B-Base` | clinicaltrials | `ct_healthy_volunteers_noul_v1` | noul | 2000/2000 | 1.000 | 0.6420 (0.6220-0.6635) | 0.500 | 0.500 | 0.9176 | 1.000 | 0.296 |
| `Qwen/Qwen3.5-0.8B-Base` | clinicaltrials | `ct_phase_choice_v1` | choice | 1100/1100 | 1.000 | 0.3155 (0.2900-0.3427) | 0.200 | 0.143 | 0.9448 | 1.000 | 0.754 |
| `Qwen/Qwen3.5-0.8B-Base` | clinicaltrials | `ct_randomised_noul_v1` | noul | 2000/2000 | 1.000 | 0.5525 (0.5315-0.5740) | 0.500 | 0.500 | 0.9378 | 0.940 | 0.044 |
| `Qwen/Qwen3.5-0.8B-Base` | openfda | `fda_boxed_warning_noul_v1` | noul | 1910/1910 | 1.000 | READOUT_FAIL — greedy agreement 0.880 < 0.9 (n=50) | 0.500 | 0.500 | 0.9239 | 0.880 | 0.368 |
| `Qwen/Qwen3.5-0.8B-Base` | openfda | `fda_class_choice_v1` | choice | 4000/4000 | 1.000 | 0.8442 (0.8340-0.8555) | 0.250 | 0.250 | 0.9347 | 0.980 | 0.114 |
| `Qwen/Qwen3.5-0.8B-Base` | pubmed | `pubmed_humans_noul_v1` | noul | 1316/1316 | 1.000 | 0.7895 (0.7690-0.8131) | 0.500 | 0.500 | 0.9332 | 1.000 | 0.320 |
| `Qwen/Qwen3.5-0.8B-Base` | pubmed | `pubmed_mesh_major_choice_v1` | choice | 4000/4000 | 1.000 | 0.9850 (0.9812-0.9885) | 0.250 | 0.250 | 0.9490 | 1.000 | 0.014 |
| `Qwen/Qwen3.5-0.8B-Base` | pubmed | `pubmed_observational_noul_v1` | noul | 206/206 | 1.000 | 0.7913 (0.7330-0.8447) | 0.500 | 0.500 | 0.9382 | 0.980 | 0.291 |
| `Qwen/Qwen3.5-4B` | clinicaltrials | `ct_healthy_volunteers_noul_v1` | noul | 2000/2000 | 1.000 | 0.7115 (0.6910-0.7315) | 0.500 | 0.500 | 0.9995 | 1.000 | 0.030 |
| `Qwen/Qwen3.5-4B` | clinicaltrials | `ct_phase_choice_v1` | choice | 1100/1100 | 1.000 | 0.4391 (0.4109-0.4691) | 0.200 | 0.143 | 0.9982 | 0.980 | 0.344 |
| `Qwen/Qwen3.5-4B` | clinicaltrials | `ct_randomised_noul_v1` | noul | 2000/2000 | 1.000 | 0.8620 (0.8460-0.8775) | 0.500 | 0.500 | 0.9995 | 1.000 | 0.018 |
| `Qwen/Qwen3.5-4B` | openfda | `fda_boxed_warning_noul_v1` | noul | 1910/1910 | 1.000 | 0.6639 (0.6408-0.6859) | 0.500 | 0.500 | 0.9996 | 0.940 | 0.064 |
| `Qwen/Qwen3.5-4B` | openfda | `fda_class_choice_v1` | choice | 4000/4000 | 1.000 | 0.8962 (0.8865-0.9055) | 0.250 | 0.250 | 0.9998 | 0.960 | 0.074 |
| `Qwen/Qwen3.5-4B` | pubmed | `pubmed_humans_noul_v1` | noul | 1316/1316 | 1.000 | 0.8404 (0.8207-0.8617) | 0.500 | 0.500 | 0.9995 | 1.000 | 0.016 |
| `Qwen/Qwen3.5-4B` | pubmed | `pubmed_mesh_major_choice_v1` | choice | 4000/4000 | 1.000 | 0.9928 (0.9900-0.9950) | 0.250 | 0.250 | 0.9999 | 1.000 | 0.008 |
| `Qwen/Qwen3.5-4B` | pubmed | `pubmed_observational_noul_v1` | noul | 206/206 | 1.000 | 0.6699 (0.6068-0.7330) | 0.500 | 0.500 | 0.9997 | 1.000 | 0.034 |
| `Qwen/Qwen3.5-9B` | clinicaltrials | `ct_healthy_volunteers_noul_v1` | noul | 2000/2000 | 1.000 | 0.7355 (0.7175-0.7545) | 0.500 | 0.500 | 0.9951 | 1.000 | 0.008 |
| `Qwen/Qwen3.5-9B` | clinicaltrials | `ct_phase_choice_v1` | choice | 1100/1100 | 1.000 | 0.4327 (0.4027-0.4618) | 0.200 | 0.143 | 0.9989 | 0.960 | 0.278 |
| `Qwen/Qwen3.5-9B` | clinicaltrials | `ct_randomised_noul_v1` | noul | 2000/2000 | 1.000 | 0.8840 (0.8690-0.8975) | 0.500 | 0.500 | 0.9940 | 1.000 | 0.012 |
| `Qwen/Qwen3.5-9B` | openfda | `fda_boxed_warning_noul_v1` | noul | 1910/1910 | 1.000 | 0.7618 (0.7419-0.7812) | 0.500 | 0.500 | 0.9943 | 0.940 | 0.058 |
| `Qwen/Qwen3.5-9B` | openfda | `fda_class_choice_v1` | choice | 4000/4000 | 1.000 | 0.9005 (0.8915-0.9093) | 0.250 | 0.250 | 0.9993 | 0.960 | 0.074 |
| `Qwen/Qwen3.5-9B` | pubmed | `pubmed_humans_noul_v1` | noul | 1316/1316 | 1.000 | 0.8678 (0.8488-0.8875) | 0.500 | 0.500 | 0.9964 | 1.000 | 0.014 |
| `Qwen/Qwen3.5-9B` | pubmed | `pubmed_mesh_major_choice_v1` | choice | 4000/4000 | 1.000 | 0.9950 (0.9928-0.9970) | 0.250 | 0.250 | 0.9998 | 1.000 | 0.002 |
| `Qwen/Qwen3.5-9B` | pubmed | `pubmed_observational_noul_v1` | noul | 206/206 | 1.000 | 0.7379 (0.6796-0.7961) | 0.500 | 0.500 | 0.9977 | 1.000 | 0.019 |
| `google/medgemma-1.5-4b-it` | clinicaltrials | `ct_healthy_volunteers_noul_v1` | noul | 2000/2000 | 1.000 | 0.7230 (0.7035-0.7420) | 0.500 | 0.500 | 0.9860 | 0.980 | 0.222 |
| `google/medgemma-1.5-4b-it` | clinicaltrials | `ct_phase_choice_v1` | choice | 1100/1100 | 1.000 | 0.3109 (0.2836-0.3400) | 0.200 | 0.143 | 0.9017 | 1.000 | 0.666 |
| `google/medgemma-1.5-4b-it` | clinicaltrials | `ct_randomised_noul_v1` | noul | 2000/2000 | 1.000 | 0.6230 (0.6015-0.6445) | 0.500 | 0.500 | 0.9903 | 1.000 | 0.072 |
| `google/medgemma-1.5-4b-it` | openfda | `fda_boxed_warning_noul_v1` | noul | 1910/1910 | 1.000 | 0.5063 (0.4832-0.5298) | 0.500 | 0.500 | 0.9544 | 0.980 | 0.066 |
| `google/medgemma-1.5-4b-it` | openfda | `fda_class_choice_v1` | choice | 4000/4000 | 1.000 | 0.8725 (0.8622-0.8828) | 0.250 | 0.250 | 0.9674 | 0.960 | 0.090 |
| `google/medgemma-1.5-4b-it` | pubmed | `pubmed_humans_noul_v1` | noul | 1316/1316 | 1.000 | 0.8777 (0.8594-0.8944) | 0.500 | 0.500 | 0.9855 | 0.960 | 0.258 |
| `google/medgemma-1.5-4b-it` | pubmed | `pubmed_mesh_major_choice_v1` | choice | 4000/4000 | 1.000 | 0.9822 (0.9782-0.9860) | 0.250 | 0.250 | 0.9892 | 1.000 | 0.012 |
| `google/medgemma-1.5-4b-it` | pubmed | `pubmed_observational_noul_v1` | noul | 206/206 | 1.000 | 0.5291 (0.4612-0.5922) | 0.500 | 0.500 | 0.9843 | 1.000 | 0.015 |

## Templates screened out (F3) — not measured here

| template | reasons |
|---|---|
| `ct_intervention_type_choice_v1` | n_test<200 |
| `ct_primary_purpose_choice_v1` | n_test<200 |
| `fda_route_choice_v1` | n_test<200 |
| `pubmed_pubtype_choice_v1` | n_test<200 |
| `trec_covid_graded_score_v1` | n_test<200 |
| `trec_covid_relevant_noul_v1` | n_test<200 |

## Notes

* `majority` is the largest gold class's share of the cell's items; every v0.1 template is
  class-balanced (F4), so it is well below 1.0 and an accuracy can be read against it.
* `flip` is the share of sampled items whose chosen option *content* changed when the
  options were reordered — a positional-guessing signal, not an accuracy.
* Cells whose `n` is below the template's item count are visible in the `n` column as
  `n/total`; the difference is items the model could not take and is reported, never
  silently dropped.

