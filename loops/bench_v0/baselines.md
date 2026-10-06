# Zero-shot baseline table (T9)

**Generated:** 2026-10-06T00:17:45Z  
**Prediction rows:** 264478 across 132 (model, source, template, split) groups  
**Models in this table:** `LiquidAI/LFM2.5-350M`, `Qwen/Qwen3.5-0.8B`, `Qwen/Qwen3.5-0.8B-Base`, `Qwen/Qwen3.5-4B`, `Qwen/Qwen3.5-9B`, `google/medgemma-1.5-4b-it`

Protocol: the fixed T6 protocol (chat template, thinking disabled, single-letter
instruction, option-letter readout) — applied unchanged to every model. The harness was
validated against lm-evaluation-harness before any number here was reported
(`loops/bench_v0/harness_validation.md`: agreement -0.50 points, tolerance ±2.0).

Every accuracy is shown with its majority baseline; calibration is raw (no temperature
fitting — that belongs on dev items, not here). `coverage` is the share of the source's
test items the model could take.

## Per (model, source, template)

| model | source | template | split | qtype | n | coverage | accuracy (95% CI) | majority | macro acc | Brier | ECE | label mass | p50 s | shuffle flip | abstention |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `LiquidAI/LFM2.5-350M` | clinicaltrials | `ct_intervention_type_choice_v1` | test | choice | 2588 | 0.243 | 0.261 (0.245-0.278) | 0.273 | 0.178 | 1.201 | 0.519 | 1.000 | 0.002 | NOT MEASURED | NOT MEASURED |
| `LiquidAI/LFM2.5-350M` | clinicaltrials | `ct_phase_choice_v1` | test | choice | 701 | 0.066 | 0.245 (0.214-0.277) | 0.345 | 0.227 | 1.268 | 0.566 | 1.000 | 0.001 | NOT MEASURED | NOT MEASURED |
| `LiquidAI/LFM2.5-350M` | clinicaltrials | `ct_primary_purpose_choice_v1` | test | choice | 2221 | 0.209 | 0.593 (0.573-0.614) | 0.596 | 0.176 | 0.712 | 0.308 | 1.000 | 0.002 | NOT MEASURED | NOT MEASURED |
| `LiquidAI/LFM2.5-350M` | clinicaltrials | `ct_randomised_noul_v1` | test | noul | 2221 | 0.209 | 0.688 (0.670-0.707) | 0.693 | 0.503 | 0.571 | 0.267 | 0.000 | 0.002 | NOT MEASURED | NOT MEASURED |
| `LiquidAI/LFM2.5-350M` | medmcqa | `medmcqa_4opt_v1` | test | choice | 2835 | 1.000 | 0.368 (0.349-0.386) | 0.383 | 0.328 | 1.112 | 0.518 | 1.000 | 0.001 | NOT MEASURED | NOT MEASURED |
| `LiquidAI/LFM2.5-350M` | medqa | `medqa_usmle4_v1` | test | choice | 1273 | 1.000 | 0.291 (0.266-0.315) | 0.277 | 0.271 | 1.226 | 0.566 | 1.000 | 0.001 | NOT MEASURED | NOT MEASURED |
| `LiquidAI/LFM2.5-350M` | medquad | `medquad_routing_v1` | test | choice | 5000 | 1.000 | 0.493 (0.480-0.508) | 0.255 | 0.489 | 0.878 | 0.412 | 1.000 | 0.001 | NOT MEASURED | NOT MEASURED |
| `LiquidAI/LFM2.5-350M` | mmlu_medical | `mmlu_mc_v1` | test | choice | 1103 | 1.000 | 0.241 (0.216-0.267) | 0.325 | 0.276 | 1.368 | 0.661 | 1.000 | 0.001 | NOT MEASURED | NOT MEASURED |
| `LiquidAI/LFM2.5-350M` | nfcorpus | `nfcorpus_graded_score_v1` | test | score | 969 | 0.600 | 0.333 (0.305-0.363) | 0.333 | 0.333 | 1.325 | 0.663 | 0.000 | 0.001 | NOT MEASURED | NOT MEASURED |
| `LiquidAI/LFM2.5-350M` | nfcorpus | `nfcorpus_relevant_noul_v1` | test | noul | 647 | 0.400 | 0.501 (0.462-0.538) | 0.501 | 0.500 | 0.993 | 0.496 | 0.000 | 0.001 | NOT MEASURED | NOT MEASURED |
| `LiquidAI/LFM2.5-350M` | openfda | `fda_boxed_warning_noul_v1` | test | noul | 94 | 0.514 | 0.191 (0.117-0.277) | 0.787 | 0.432 | 1.547 | 0.783 | 0.175 | 0.003 | NOT MEASURED | NOT MEASURED |
| `LiquidAI/LFM2.5-350M` | openfda | `fda_class_choice_v1` | test | choice | 16 | 0.087 | 0.625 (0.375-0.875) | 0.438 | 0.548 | 0.657 | 0.353 | 1.000 | 0.005 | NOT MEASURED | NOT MEASURED |
| `LiquidAI/LFM2.5-350M` | openfda | `fda_route_choice_v1` | test | choice | 73 | 0.399 | 0.247 (0.151-0.342) | 0.630 | 0.117 | 1.199 | 0.546 | 0.959 | 0.009 | NOT MEASURED | NOT MEASURED |
| `LiquidAI/LFM2.5-350M` | pubmed | `pubmed_humans_noul_v1` | test | noul | 5000 | 0.250 | 0.883 (0.874-0.892) | 0.883 | 0.500 | 0.225 | 0.096 | 0.000 | 0.002 | NOT MEASURED | NOT MEASURED |
| `LiquidAI/LFM2.5-350M` | pubmed | `pubmed_mesh_major_choice_v1` | test | choice | 5000 | 0.250 | 0.546 (0.532-0.560) | 0.259 | 0.541 | 0.760 | 0.367 | 1.000 | 0.002 | NOT MEASURED | NOT MEASURED |
| `LiquidAI/LFM2.5-350M` | pubmed | `pubmed_pubtype_choice_v1` | test | choice | 5000 | 0.250 | 0.029 (0.025-0.034) | 0.926 | 0.306 | 1.843 | 0.917 | 1.000 | 0.002 | NOT MEASURED | NOT MEASURED |
| `LiquidAI/LFM2.5-350M` | pubmedqa | `pubmedqa_ynm_v1` | test | choice | 477 | 1.000 | 0.417 (0.371-0.461) | 0.551 | 0.353 | 0.816 | 0.357 | 0.999 | 0.002 | NOT MEASURED | NOT MEASURED |
| `LiquidAI/LFM2.5-350M` | scifact | `scifact_relevant_noul_v1` | test | noul | 600 | 1.000 | 0.500 (0.460-0.542) | 0.500 | 0.500 | 0.994 | 0.497 | 0.000 | 0.001 | NOT MEASURED | NOT MEASURED |
| `LiquidAI/LFM2.5-350M` | trec_covid | `trec_covid_graded_score_v1` | test | score | 117 | 0.600 | 0.333 (0.248-0.419) | 0.333 | 0.333 | 1.322 | 0.661 | 0.000 | 0.001 | NOT MEASURED | NOT MEASURED |
| `LiquidAI/LFM2.5-350M` | trec_covid | `trec_covid_relevant_noul_v1` | test | noul | 78 | 0.400 | 0.500 (0.385-0.603) | 0.500 | 0.500 | 0.997 | 0.498 | 0.000 | 0.001 | NOT MEASURED | NOT MEASURED |
| `Qwen/Qwen3.5-0.8B` | clinicaltrials | `ct_intervention_type_choice_v1` | test | choice | 2588 | 0.243 | 0.527 (0.508-0.545) | 0.273 | 0.426 | 0.704 | 0.154 | 0.999 | 0.010 | 0.300 | 0.133 |
| `Qwen/Qwen3.5-0.8B` | clinicaltrials | `ct_phase_choice_v1` | test | choice | 701 | 0.066 | 0.456 (0.421-0.492) | 0.345 | 0.344 | 0.724 | 0.149 | 0.998 | 0.007 | 0.300 | 0.133 |
| `Qwen/Qwen3.5-0.8B` | clinicaltrials | `ct_primary_purpose_choice_v1` | test | choice | 2221 | 0.209 | 0.614 (0.594-0.633) | 0.596 | 0.398 | 0.551 | 0.069 | 0.998 | 0.009 | 0.300 | 0.133 |
| `Qwen/Qwen3.5-0.8B` | clinicaltrials | `ct_randomised_noul_v1` | test | noul | 2221 | 0.209 | 0.710 (0.691-0.728) | 0.693 | 0.527 | 0.436 | 0.202 | 0.056 | 0.009 | 0.300 | 0.133 |
| `Qwen/Qwen3.5-0.8B` | medmcqa | `medmcqa_4opt_v1` | test | choice | 2835 | 1.000 | 0.295 (0.278-0.310) | 0.383 | 0.277 | 0.843 | 0.194 | 0.996 | 0.003 | 0.300 | 0.133 |
| `Qwen/Qwen3.5-0.8B` | medqa | `medqa_usmle4_v1` | test | choice | 1273 | 1.000 | 0.416 (0.389-0.443) | 0.277 | 0.411 | 0.737 | 0.141 | 0.996 | 0.010 | 0.300 | 0.133 |
| `Qwen/Qwen3.5-0.8B` | medquad | `medquad_routing_v1` | test | choice | 5000 | 1.000 | 0.926 (0.919-0.933) | 0.255 | 0.925 | 0.121 | 0.088 | 0.999 | 0.002 | 0.300 | 0.133 |
| `Qwen/Qwen3.5-0.8B` | mmlu_medical | `mmlu_mc_v1` | test | choice | 1103 | 1.000 | 0.459 (0.431-0.490) | 0.325 | 0.473 | 0.658 | 0.138 | 0.996 | 0.005 | 0.300 | 0.133 |
| `Qwen/Qwen3.5-0.8B` | nfcorpus | `nfcorpus_graded_score_v1` | test | score | 969 | 0.600 | 0.313 (0.284-0.342) | 0.333 | 0.313 | 0.821 | 0.260 | 0.224 | 0.009 | 0.300 | 0.133 |
| `Qwen/Qwen3.5-0.8B` | nfcorpus | `nfcorpus_relevant_noul_v1` | test | noul | 647 | 0.400 | 0.501 (0.462-0.538) | 0.501 | 0.500 | 0.842 | 0.421 | 0.155 | 0.009 | 0.300 | 0.133 |
| `Qwen/Qwen3.5-0.8B` | openfda | `fda_boxed_warning_noul_v1` | test | noul | 94 | 0.514 | 0.202 (0.128-0.287) | 0.787 | 0.457 | 0.955 | 0.571 | 0.114 | 0.016 | 0.300 | 0.133 |
| `Qwen/Qwen3.5-0.8B` | openfda | `fda_class_choice_v1` | test | choice | 16 | 0.087 | 0.875 (0.688-1.000) | 0.438 | 0.854 | 0.183 | 0.185 | 0.998 | 0.025 | 0.300 | 0.133 |
| `Qwen/Qwen3.5-0.8B` | openfda | `fda_route_choice_v1` | test | choice | 73 | 0.399 | 0.671 (0.562-0.781) | 0.630 | 0.664 | 0.332 | 0.184 | 0.957 | 0.045 | 0.300 | 0.133 |
| `Qwen/Qwen3.5-0.8B` | pubmed | `pubmed_humans_noul_v1` | test | noul | 5400 | 0.265 | 0.839 (0.829-0.849) | 0.881 | 0.787 | 0.225 | 0.043 | 0.063 | 0.010 | 0.300 | 0.133 |
| `Qwen/Qwen3.5-0.8B` | pubmed | `pubmed_mesh_major_choice_v1` | test | choice | 5000 | 0.245 | 0.985 (0.981-0.988) | 0.259 | 0.985 | 0.032 | 0.052 | 0.999 | 0.010 | 0.300 | 0.133 |
| `Qwen/Qwen3.5-0.8B` | pubmed | `pubmed_pubtype_choice_v1` | test | choice | 5000 | 0.245 | 0.102 (0.094-0.111) | 0.926 | 0.720 | 1.291 | 0.589 | 0.998 | 0.010 | 0.300 | 0.133 |
| `Qwen/Qwen3.5-0.8B` | pubmedqa | `pubmedqa_ynm_v1` | test | choice | 477 | 1.000 | 0.690 (0.648-0.730) | 0.551 | 0.510 | 0.468 | 0.121 | 0.992 | 0.014 | 0.300 | 0.133 |
| `Qwen/Qwen3.5-0.8B` | scifact | `scifact_relevant_noul_v1` | test | noul | 600 | 1.000 | 0.503 (0.463-0.543) | 0.500 | 0.503 | 0.705 | 0.376 | 0.227 | 0.009 | 0.300 | 0.133 |
| `Qwen/Qwen3.5-0.8B` | trec_covid | `trec_covid_graded_score_v1` | test | score | 117 | 0.600 | 0.410 (0.316-0.504) | 0.333 | 0.410 | 0.775 | 0.176 | 0.218 | 0.009 | 0.300 | 0.133 |
| `Qwen/Qwen3.5-0.8B` | trec_covid | `trec_covid_relevant_noul_v1` | test | noul | 78 | 0.400 | 0.500 (0.385-0.603) | 0.500 | 0.500 | 0.777 | 0.394 | 0.139 | 0.009 | 0.300 | 0.133 |
| `Qwen/Qwen3.5-0.8B-Base` | clinicaltrials | `ct_intervention_type_choice_v1` | test | choice | 2588 | 0.243 | 0.472 (0.452-0.491) | 0.273 | 0.376 | 0.757 | 0.193 | 0.952 | 0.010 | NOT MEASURED | NOT MEASURED |
| `Qwen/Qwen3.5-0.8B-Base` | clinicaltrials | `ct_phase_choice_v1` | test | choice | 701 | 0.066 | 0.447 (0.411-0.482) | 0.345 | 0.324 | 0.745 | 0.136 | 0.942 | 0.007 | NOT MEASURED | NOT MEASURED |
| `Qwen/Qwen3.5-0.8B-Base` | clinicaltrials | `ct_primary_purpose_choice_v1` | test | choice | 2221 | 0.209 | 0.602 (0.582-0.623) | 0.596 | 0.370 | 0.559 | 0.053 | 0.945 | 0.009 | NOT MEASURED | NOT MEASURED |
| `Qwen/Qwen3.5-0.8B-Base` | clinicaltrials | `ct_randomised_noul_v1` | test | noul | 2221 | 0.209 | 0.703 (0.684-0.721) | 0.693 | 0.515 | 0.406 | 0.186 | 0.035 | 0.008 | NOT MEASURED | NOT MEASURED |
| `Qwen/Qwen3.5-0.8B-Base` | medmcqa | `medmcqa_4opt_v1` | test | choice | 2835 | 1.000 | 0.284 (0.267-0.300) | 0.383 | 0.275 | 0.846 | 0.196 | 0.921 | 0.002 | NOT MEASURED | NOT MEASURED |
| `Qwen/Qwen3.5-0.8B-Base` | medqa | `medqa_usmle4_v1` | test | choice | 1273 | 1.000 | 0.377 (0.350-0.404) | 0.277 | 0.370 | 0.744 | 0.141 | 0.826 | 0.005 | NOT MEASURED | NOT MEASURED |
| `Qwen/Qwen3.5-0.8B-Base` | medquad | `medquad_routing_v1` | test | choice | 5000 | 1.000 | 0.943 (0.936-0.949) | 0.255 | 0.942 | 0.119 | 0.132 | 0.921 | 0.002 | NOT MEASURED | NOT MEASURED |
| `Qwen/Qwen3.5-0.8B-Base` | mmlu_medical | `mmlu_mc_v1` | test | choice | 1103 | 1.000 | 0.461 (0.434-0.490) | 0.325 | 0.483 | 0.631 | 0.099 | 0.891 | 0.003 | NOT MEASURED | NOT MEASURED |
| `Qwen/Qwen3.5-0.8B-Base` | nfcorpus | `nfcorpus_graded_score_v1` | test | score | 969 | 0.600 | 0.331 (0.303-0.360) | 0.333 | 0.331 | 0.819 | 0.308 | 0.030 | 0.007 | NOT MEASURED | NOT MEASURED |
| `Qwen/Qwen3.5-0.8B-Base` | nfcorpus | `nfcorpus_relevant_noul_v1` | test | noul | 647 | 0.400 | 0.501 (0.462-0.538) | 0.501 | 0.500 | 0.876 | 0.437 | 0.028 | 0.007 | NOT MEASURED | NOT MEASURED |
| `Qwen/Qwen3.5-0.8B-Base` | openfda | `fda_boxed_warning_noul_v1` | test | noul | 94 | 0.514 | 0.191 (0.117-0.277) | 0.787 | 0.450 | 0.807 | 0.535 | 0.079 | 0.016 | NOT MEASURED | NOT MEASURED |
| `Qwen/Qwen3.5-0.8B-Base` | openfda | `fda_class_choice_v1` | test | choice | 16 | 0.087 | 0.875 (0.688-1.000) | 0.438 | 0.854 | 0.225 | 0.196 | 0.914 | 0.025 | NOT MEASURED | NOT MEASURED |
| `Qwen/Qwen3.5-0.8B-Base` | openfda | `fda_route_choice_v1` | test | choice | 73 | 0.399 | 0.699 (0.589-0.808) | 0.630 | 0.517 | 0.337 | 0.170 | 0.880 | 0.047 | NOT MEASURED | NOT MEASURED |
| `Qwen/Qwen3.5-0.8B-Base` | pubmed | `pubmed_humans_noul_v1` | test | noul | 5000 | 0.250 | 0.882 (0.873-0.891) | 0.883 | 0.682 | 0.177 | 0.059 | 0.065 | 0.010 | NOT MEASURED | NOT MEASURED |
| `Qwen/Qwen3.5-0.8B-Base` | pubmed | `pubmed_mesh_major_choice_v1` | test | choice | 5000 | 0.250 | 0.980 (0.976-0.984) | 0.259 | 0.980 | 0.047 | 0.081 | 0.950 | 0.010 | NOT MEASURED | NOT MEASURED |
| `Qwen/Qwen3.5-0.8B-Base` | pubmed | `pubmed_pubtype_choice_v1` | test | choice | 5000 | 0.250 | 0.069 (0.062-0.076) | 0.926 | 0.737 | 1.186 | 0.604 | 0.919 | 0.010 | NOT MEASURED | NOT MEASURED |
| `Qwen/Qwen3.5-0.8B-Base` | pubmedqa | `pubmedqa_ynm_v1` | test | choice | 477 | 1.000 | 0.629 (0.587-0.671) | 0.551 | 0.488 | 0.532 | 0.119 | 0.903 | 0.008 | NOT MEASURED | NOT MEASURED |
| `Qwen/Qwen3.5-0.8B-Base` | scifact | `scifact_relevant_noul_v1` | test | noul | 600 | 1.000 | 0.500 (0.460-0.542) | 0.500 | 0.500 | 0.870 | 0.438 | 0.027 | 0.007 | NOT MEASURED | NOT MEASURED |
| `Qwen/Qwen3.5-0.8B-Base` | trec_covid | `trec_covid_graded_score_v1` | test | score | 117 | 0.600 | 0.333 (0.248-0.419) | 0.333 | 0.333 | 0.810 | 0.288 | 0.021 | 0.007 | NOT MEASURED | NOT MEASURED |
| `Qwen/Qwen3.5-0.8B-Base` | trec_covid | `trec_covid_relevant_noul_v1` | test | noul | 78 | 0.400 | 0.500 (0.385-0.603) | 0.500 | 0.500 | 0.875 | 0.439 | 0.017 | 0.007 | NOT MEASURED | NOT MEASURED |
| `Qwen/Qwen3.5-4B` | clinicaltrials | `ct_intervention_type_choice_v1` | test | choice | 2588 | 0.243 | 0.588 (0.568-0.607) | 0.273 | 0.617 | 0.610 | 0.235 | 0.999 | 0.031 | NOT MEASURED | NOT MEASURED |
| `Qwen/Qwen3.5-4B` | clinicaltrials | `ct_phase_choice_v1` | test | choice | 701 | 0.066 | 0.576 (0.539-0.611) | 0.345 | 0.453 | 0.605 | 0.137 | 0.995 | 0.023 | NOT MEASURED | NOT MEASURED |
| `Qwen/Qwen3.5-4B` | clinicaltrials | `ct_primary_purpose_choice_v1` | test | choice | 2221 | 0.209 | 0.506 (0.485-0.528) | 0.596 | 0.509 | 0.685 | 0.158 | 0.999 | 0.030 | NOT MEASURED | NOT MEASURED |
| `Qwen/Qwen3.5-4B` | clinicaltrials | `ct_randomised_noul_v1` | test | noul | 2221 | 0.209 | 0.718 (0.698-0.735) | 0.693 | 0.796 | 0.543 | 0.283 | 0.004 | 0.027 | NOT MEASURED | NOT MEASURED |
| `Qwen/Qwen3.5-4B` | medmcqa | `medmcqa_4opt_v1` | test | choice | 2835 | 1.000 | 0.122 (0.110-0.134) | 0.383 | 0.122 | 1.309 | 0.579 | 0.982 | 0.007 | NOT MEASURED | NOT MEASURED |
| `Qwen/Qwen3.5-4B` | medqa | `medqa_usmle4_v1` | test | choice | 1273 | 1.000 | 0.701 (0.676-0.724) | 0.277 | 0.709 | 0.423 | 0.044 | 0.992 | 0.017 | NOT MEASURED | NOT MEASURED |
| `Qwen/Qwen3.5-4B` | medquad | `medquad_routing_v1` | test | choice | 5000 | 1.000 | 0.984 (0.981-0.987) | 0.255 | 0.984 | 0.026 | 0.008 | 1.000 | 0.006 | NOT MEASURED | NOT MEASURED |
| `Qwen/Qwen3.5-4B` | mmlu_medical | `mmlu_mc_v1` | test | choice | 1103 | 1.000 | 0.818 (0.793-0.842) | 0.325 | 0.813 | 0.264 | 0.037 | 0.983 | 0.009 | NOT MEASURED | NOT MEASURED |
| `Qwen/Qwen3.5-4B` | nfcorpus | `nfcorpus_graded_score_v1` | test | score | 969 | 0.600 | 0.416 (0.384-0.448) | 0.333 | 0.416 | 1.036 | 0.487 | 0.023 | 0.022 | NOT MEASURED | NOT MEASURED |
| `Qwen/Qwen3.5-4B` | nfcorpus | `nfcorpus_relevant_noul_v1` | test | noul | 647 | 0.400 | 0.567 (0.530-0.607) | 0.501 | 0.568 | 0.782 | 0.376 | 0.003 | 0.021 | NOT MEASURED | NOT MEASURED |
| `Qwen/Qwen3.5-4B` | openfda | `fda_boxed_warning_noul_v1` | test | noul | 94 | 0.514 | 0.777 (0.691-0.862) | 0.787 | 0.493 | 0.381 | 0.202 | 0.013 | 0.051 | NOT MEASURED | NOT MEASURED |
| `Qwen/Qwen3.5-4B` | openfda | `fda_class_choice_v1` | test | choice | 16 | 0.087 | 0.938 (0.812-1.000) | 0.438 | 0.938 | 0.093 | 0.085 | 0.998 | 0.078 | NOT MEASURED | NOT MEASURED |
| `Qwen/Qwen3.5-4B` | openfda | `fda_route_choice_v1` | test | choice | 73 | 0.399 | 0.890 (0.808-0.959) | 0.630 | 0.799 | 0.155 | 0.051 | 0.958 | 0.139 | NOT MEASURED | NOT MEASURED |
| `Qwen/Qwen3.5-4B` | pubmed | `pubmed_humans_noul_v1` | test | noul | 5600 | 0.272 | 0.482 (0.469-0.496) | 0.880 | 0.706 | 0.826 | 0.340 | 0.003 | 0.031 | NOT MEASURED | NOT MEASURED |
| `Qwen/Qwen3.5-4B` | pubmed | `pubmed_mesh_major_choice_v1` | test | choice | 5000 | 0.243 | 0.991 (0.989-0.994) | 0.259 | 0.991 | 0.013 | 0.007 | 1.000 | 0.031 | NOT MEASURED | NOT MEASURED |
| `Qwen/Qwen3.5-4B` | pubmed | `pubmed_pubtype_choice_v1` | test | choice | 5000 | 0.243 | 0.467 (0.453-0.483) | 0.926 | 0.789 | 0.854 | 0.376 | 0.999 | 0.031 | NOT MEASURED | NOT MEASURED |
| `Qwen/Qwen3.5-4B` | pubmedqa | `pubmedqa_ynm_v1` | test | choice | 477 | 1.000 | 0.736 (0.696-0.776) | 0.551 | 0.562 | 0.380 | 0.094 | 0.998 | 0.027 | NOT MEASURED | NOT MEASURED |
| `Qwen/Qwen3.5-4B` | scifact | `scifact_relevant_noul_v1` | test | noul | 600 | 1.000 | 0.632 (0.592-0.670) | 0.500 | 0.632 | 0.615 | 0.293 | 0.004 | 0.022 | NOT MEASURED | NOT MEASURED |
| `Qwen/Qwen3.5-4B` | trec_covid | `trec_covid_graded_score_v1` | test | score | 117 | 0.600 | 0.462 (0.368-0.547) | 0.333 | 0.462 | 0.826 | 0.370 | 0.018 | 0.021 | NOT MEASURED | NOT MEASURED |
| `Qwen/Qwen3.5-4B` | trec_covid | `trec_covid_relevant_noul_v1` | test | noul | 78 | 0.400 | 0.641 (0.538-0.744) | 0.500 | 0.641 | 0.656 | 0.349 | 0.004 | 0.020 | NOT MEASURED | NOT MEASURED |
| `Qwen/Qwen3.5-9B` | clinicaltrials | `ct_intervention_type_choice_v1` | test | choice | 2588 | 0.243 | 0.600 (0.580-0.619) | 0.273 | 0.591 | 0.654 | 0.274 | 0.999 | 0.037 | NOT MEASURED | NOT MEASURED |
| `Qwen/Qwen3.5-9B` | clinicaltrials | `ct_phase_choice_v1` | test | choice | 701 | 0.066 | 0.563 (0.526-0.601) | 0.345 | 0.439 | 0.642 | 0.174 | 0.999 | 0.030 | NOT MEASURED | NOT MEASURED |
| `Qwen/Qwen3.5-9B` | clinicaltrials | `ct_primary_purpose_choice_v1` | test | choice | 2221 | 0.209 | 0.615 (0.593-0.635) | 0.596 | 0.582 | 0.585 | 0.185 | 0.999 | 0.037 | NOT MEASURED | NOT MEASURED |
| `Qwen/Qwen3.5-9B` | clinicaltrials | `ct_randomised_noul_v1` | test | noul | 2221 | 0.209 | 0.793 (0.776-0.809) | 0.693 | 0.847 | 0.390 | 0.180 | 0.001 | 0.033 | NOT MEASURED | NOT MEASURED |
| `Qwen/Qwen3.5-9B` | medmcqa | `medmcqa_4opt_v1` | test | choice | 2835 | 1.000 | 0.121 (0.109-0.134) | 0.383 | 0.123 | 1.424 | 0.654 | 0.988 | 0.010 | NOT MEASURED | NOT MEASURED |
| `Qwen/Qwen3.5-9B` | medqa | `medqa_usmle4_v1` | test | choice | 1273 | 1.000 | 0.755 (0.731-0.778) | 0.277 | 0.757 | 0.348 | 0.074 | 0.997 | 0.022 | NOT MEASURED | NOT MEASURED |
| `Qwen/Qwen3.5-9B` | medquad | `medquad_routing_v1` | test | choice | 5000 | 1.000 | 0.988 (0.984-0.991) | 0.255 | 0.988 | 0.019 | 0.003 | 0.997 | 0.009 | NOT MEASURED | NOT MEASURED |
| `Qwen/Qwen3.5-9B` | mmlu_medical | `mmlu_mc_v1` | test | choice | 1103 | 1.000 | 0.860 (0.840-0.880) | 0.325 | 0.858 | 0.208 | 0.047 | 0.986 | 0.012 | NOT MEASURED | NOT MEASURED |
| `Qwen/Qwen3.5-9B` | nfcorpus | `nfcorpus_graded_score_v1` | test | score | 969 | 0.600 | 0.418 (0.387-0.447) | 0.333 | 0.418 | 1.012 | 0.468 | 0.002 | 0.027 | NOT MEASURED | NOT MEASURED |
| `Qwen/Qwen3.5-9B` | nfcorpus | `nfcorpus_relevant_noul_v1` | test | noul | 647 | 0.400 | 0.524 (0.485-0.566) | 0.501 | 0.525 | 0.784 | 0.402 | 0.002 | 0.027 | NOT MEASURED | NOT MEASURED |
| `Qwen/Qwen3.5-9B` | openfda | `fda_boxed_warning_noul_v1` | test | noul | 94 | 0.514 | 0.787 (0.702-0.862) | 0.787 | 0.500 | 0.342 | 0.179 | 0.002 | 0.066 | NOT MEASURED | NOT MEASURED |
| `Qwen/Qwen3.5-9B` | openfda | `fda_class_choice_v1` | test | choice | 16 | 0.087 | 0.938 (0.812-1.000) | 0.438 | 0.938 | 0.072 | 0.056 | 0.999 | 0.102 | NOT MEASURED | NOT MEASURED |
| `Qwen/Qwen3.5-9B` | openfda | `fda_route_choice_v1` | test | choice | 73 | 0.399 | 0.877 (0.795-0.945) | 0.630 | 0.796 | 0.184 | 0.084 | 0.959 | 0.185 | NOT MEASURED | NOT MEASURED |
| `Qwen/Qwen3.5-9B` | pubmed | `pubmed_humans_noul_v1` | test | noul | 5000 | 0.250 | 0.685 (0.672-0.698) | 0.883 | 0.819 | 0.592 | 0.290 | 0.001 | 0.037 | NOT MEASURED | NOT MEASURED |
| `Qwen/Qwen3.5-9B` | pubmed | `pubmed_mesh_major_choice_v1` | test | choice | 5000 | 0.250 | 0.994 (0.991-0.996) | 0.259 | 0.994 | 0.011 | 0.003 | 1.000 | 0.039 | NOT MEASURED | NOT MEASURED |
| `Qwen/Qwen3.5-9B` | pubmed | `pubmed_pubtype_choice_v1` | test | choice | 5000 | 0.250 | 0.595 (0.581-0.609) | 0.926 | 0.813 | 0.688 | 0.320 | 0.999 | 0.038 | NOT MEASURED | NOT MEASURED |
| `Qwen/Qwen3.5-9B` | pubmedqa | `pubmedqa_ynm_v1` | test | choice | 477 | 1.000 | 0.706 (0.667-0.742) | 0.551 | 0.597 | 0.409 | 0.098 | 0.995 | 0.033 | NOT MEASURED | NOT MEASURED |
| `Qwen/Qwen3.5-9B` | scifact | `scifact_relevant_noul_v1` | test | noul | 600 | 1.000 | 0.610 (0.572-0.650) | 0.500 | 0.610 | 0.541 | 0.264 | 0.001 | 0.027 | NOT MEASURED | NOT MEASURED |
| `Qwen/Qwen3.5-9B` | trec_covid | `trec_covid_graded_score_v1` | test | score | 117 | 0.600 | 0.496 (0.410-0.581) | 0.333 | 0.496 | 0.723 | 0.299 | 0.004 | 0.027 | NOT MEASURED | NOT MEASURED |
| `Qwen/Qwen3.5-9B` | trec_covid | `trec_covid_relevant_noul_v1` | test | noul | 78 | 0.400 | 0.564 (0.462-0.679) | 0.500 | 0.564 | 0.621 | 0.293 | 0.002 | 0.027 | NOT MEASURED | NOT MEASURED |
| `google/medgemma-1.5-4b-it` | clinicaltrials | `ct_intervention_type_choice_v1` | test | choice | 2588 | 0.243 | 0.577 (0.559-0.596) | 0.273 | 0.579 | 0.774 | 0.365 | 0.947 | 0.015 | NOT MEASURED | NOT MEASURED |
| `google/medgemma-1.5-4b-it` | clinicaltrials | `ct_phase_choice_v1` | test | choice | 701 | 0.066 | 0.374 (0.338-0.408) | 0.345 | 0.326 | 1.019 | 0.444 | 0.893 | 0.011 | NOT MEASURED | NOT MEASURED |
| `google/medgemma-1.5-4b-it` | clinicaltrials | `ct_primary_purpose_choice_v1` | test | choice | 2221 | 0.209 | 0.645 (0.625-0.664) | 0.596 | 0.504 | 0.608 | 0.251 | 0.928 | 0.014 | NOT MEASURED | NOT MEASURED |
| `google/medgemma-1.5-4b-it` | clinicaltrials | `ct_randomised_noul_v1` | test | noul | 2221 | 0.209 | 0.816 (0.801-0.831) | 0.693 | 0.796 | 0.271 | 0.067 | 0.003 | 0.013 | NOT MEASURED | NOT MEASURED |
| `google/medgemma-1.5-4b-it` | medmcqa | `medmcqa_4opt_v1` | test | choice | 2835 | 1.000 | 0.292 (0.276-0.309) | 0.383 | 0.267 | 1.101 | 0.471 | 0.773 | 0.003 | NOT MEASURED | NOT MEASURED |
| `google/medgemma-1.5-4b-it` | medqa | `medqa_usmle4_v1` | test | choice | 1273 | 1.000 | 0.481 (0.454-0.510) | 0.277 | 0.464 | 0.808 | 0.343 | 0.647 | 0.008 | NOT MEASURED | NOT MEASURED |
| `google/medgemma-1.5-4b-it` | medquad | `medquad_routing_v1` | test | choice | 5000 | 1.000 | 0.929 (0.922-0.936) | 0.255 | 0.928 | 0.108 | 0.013 | 0.906 | 0.003 | NOT MEASURED | NOT MEASURED |
| `google/medgemma-1.5-4b-it` | mmlu_medical | `mmlu_mc_v1` | test | choice | 1103 | 1.000 | 0.476 (0.448-0.504) | 0.325 | 0.505 | 0.783 | 0.328 | 0.747 | 0.004 | NOT MEASURED | NOT MEASURED |
| `google/medgemma-1.5-4b-it` | nfcorpus | `nfcorpus_graded_score_v1` | test | score | 969 | 0.600 | 0.355 (0.324-0.385) | 0.333 | 0.355 | 0.722 | 0.154 | 0.024 | 0.010 | NOT MEASURED | NOT MEASURED |
| `google/medgemma-1.5-4b-it` | nfcorpus | `nfcorpus_relevant_noul_v1` | test | noul | 647 | 0.400 | 0.512 (0.474-0.550) | 0.501 | 0.512 | 0.518 | 0.131 | 0.006 | 0.010 | NOT MEASURED | NOT MEASURED |
| `google/medgemma-1.5-4b-it` | openfda | `fda_boxed_warning_noul_v1` | test | noul | 94 | 0.514 | 0.606 (0.500-0.702) | 0.787 | 0.568 | 0.493 | 0.121 | 0.002 | 0.028 | NOT MEASURED | NOT MEASURED |
| `google/medgemma-1.5-4b-it` | openfda | `fda_class_choice_v1` | test | choice | 16 | 0.087 | 0.812 (0.625-1.000) | 0.438 | 0.792 | 0.303 | 0.150 | 0.940 | 0.042 | NOT MEASURED | NOT MEASURED |
| `google/medgemma-1.5-4b-it` | openfda | `fda_route_choice_v1` | test | choice | 73 | 0.399 | 0.890 (0.822-0.959) | 0.630 | 0.795 | 0.185 | 0.081 | 0.928 | 0.077 | NOT MEASURED | NOT MEASURED |
| `google/medgemma-1.5-4b-it` | pubmed | `pubmed_humans_noul_v1` | test | noul | 5000 | 0.250 | 0.662 (0.649-0.675) | 0.883 | 0.809 | 0.532 | 0.191 | 0.003 | 0.014 | NOT MEASURED | NOT MEASURED |
| `google/medgemma-1.5-4b-it` | pubmed | `pubmed_mesh_major_choice_v1` | test | choice | 5000 | 0.250 | 0.984 (0.981-0.988) | 0.259 | 0.985 | 0.025 | 0.006 | 0.987 | 0.015 | NOT MEASURED | NOT MEASURED |
| `google/medgemma-1.5-4b-it` | pubmed | `pubmed_pubtype_choice_v1` | test | choice | 5000 | 0.250 | 0.067 (0.059-0.073) | 0.926 | 0.713 | 1.803 | 0.893 | 0.964 | 0.015 | NOT MEASURED | NOT MEASURED |
| `google/medgemma-1.5-4b-it` | pubmedqa | `pubmedqa_ynm_v1` | test | choice | 477 | 1.000 | 0.507 (0.461-0.549) | 0.551 | 0.376 | 0.740 | 0.299 | 0.882 | 0.013 | NOT MEASURED | NOT MEASURED |
| `google/medgemma-1.5-4b-it` | scifact | `scifact_relevant_noul_v1` | test | noul | 600 | 1.000 | 0.550 (0.508-0.590) | 0.500 | 0.550 | 0.468 | 0.192 | 0.007 | 0.010 | NOT MEASURED | NOT MEASURED |
| `google/medgemma-1.5-4b-it` | trec_covid | `trec_covid_graded_score_v1` | test | score | 117 | 0.600 | 0.342 (0.256-0.427) | 0.333 | 0.342 | 0.805 | 0.289 | 0.043 | 0.010 | NOT MEASURED | NOT MEASURED |
| `google/medgemma-1.5-4b-it` | trec_covid | `trec_covid_relevant_noul_v1` | test | noul | 78 | 0.400 | 0.526 (0.423-0.641) | 0.500 | 0.526 | 0.538 | 0.204 | 0.006 | 0.010 | NOT MEASURED | NOT MEASURED |

## Templates dropped by the T8 screen (measured, but not headline rows)

These are scored for traceability; the screen dropped them before T9, so they are not part
of the baseline claim.

| model | source | template | split | qtype | n | coverage | accuracy (95% CI) | majority | macro acc | Brier | ECE | label mass | p50 s | shuffle flip | abstention |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `LiquidAI/LFM2.5-350M` | clinicaltrials | `ct_healthy_volunteers_noul_v1` | test | noul | 2900 | 0.273 | 0.015 (0.011-0.019) | 1.000 | 0.015 | 1.816 | 0.936 | 0.000 | 0.002 | NOT MEASURED | NOT MEASURED |
| `LiquidAI/LFM2.5-350M` | pubmed | `pubmed_observational_noul_v1` | test | noul | 5000 | 0.250 | 0.015 (0.012-0.019) | 0.985 | 0.500 | 1.895 | 0.966 | 0.000 | 0.002 | NOT MEASURED | NOT MEASURED |
| `Qwen/Qwen3.5-0.8B` | clinicaltrials | `ct_healthy_volunteers_noul_v1` | test | noul | 2900 | 0.273 | 0.440 (0.422-0.459) | 1.000 | 0.440 | 0.683 | 0.239 | 0.043 | 0.009 | 0.300 | 0.133 |
| `Qwen/Qwen3.5-0.8B` | pubmed | `pubmed_observational_noul_v1` | test | noul | 5000 | 0.245 | 0.041 (0.036-0.047) | 0.985 | 0.513 | 1.229 | 0.746 | 0.021 | 0.009 | 0.300 | 0.133 |
| `Qwen/Qwen3.5-0.8B-Base` | clinicaltrials | `ct_healthy_volunteers_noul_v1` | test | noul | 2900 | 0.273 | 0.625 (0.608-0.642) | 1.000 | 0.625 | 0.514 | 0.132 | 0.037 | 0.009 | NOT MEASURED | NOT MEASURED |
| `Qwen/Qwen3.5-0.8B-Base` | pubmed | `pubmed_observational_noul_v1` | test | noul | 5000 | 0.250 | 0.198 (0.188-0.210) | 0.985 | 0.593 | 0.813 | 0.462 | 0.061 | 0.009 | NOT MEASURED | NOT MEASURED |
| `Qwen/Qwen3.5-4B` | clinicaltrials | `ct_healthy_volunteers_noul_v1` | test | noul | 2900 | 0.273 | 0.933 (0.923-0.942) | 1.000 | 0.933 | 0.070 | 0.033 | 0.009 | 0.027 | NOT MEASURED | NOT MEASURED |
| `Qwen/Qwen3.5-4B` | pubmed | `pubmed_observational_noul_v1` | test | noul | 5000 | 0.243 | 0.752 (0.740-0.764) | 0.985 | 0.776 | 0.255 | 0.105 | 0.003 | 0.030 | NOT MEASURED | NOT MEASURED |
| `Qwen/Qwen3.5-9B` | clinicaltrials | `ct_healthy_volunteers_noul_v1` | test | noul | 2900 | 0.273 | 0.913 (0.903-0.922) | 1.000 | 0.913 | 0.101 | 0.047 | 0.011 | 0.034 | NOT MEASURED | NOT MEASURED |
| `Qwen/Qwen3.5-9B` | pubmed | `pubmed_observational_noul_v1` | test | noul | 5000 | 0.250 | 0.820 (0.809-0.831) | 0.985 | 0.817 | 0.208 | 0.093 | 0.001 | 0.037 | NOT MEASURED | NOT MEASURED |
| `google/medgemma-1.5-4b-it` | clinicaltrials | `ct_healthy_volunteers_noul_v1` | test | noul | 2900 | 0.273 | 0.850 (0.838-0.863) | 1.000 | 0.850 | 0.223 | 0.065 | 0.005 | 0.013 | NOT MEASURED | NOT MEASURED |
| `google/medgemma-1.5-4b-it` | pubmed | `pubmed_observational_noul_v1` | test | noul | 5000 | 0.250 | 0.965 (0.959-0.970) | 0.985 | 0.647 | 0.116 | 0.167 | 0.003 | 0.014 | NOT MEASURED | NOT MEASURED |

## Cells that are not measured

- `shuffle flip` and `abstention` are `NOT MEASURED` for every model except the one the T6
  probe ran on (Qwen3.5-0.8B on 30 MedQA items): the probes are per-model runs, and a
  model with no probe run gets no number rather than an inferred one.
- `coverage` < 1 means the model could not take some items (option count or prompt
  length); the runner records those per item, and no item is silently skipped.
- Decision-model baselines (Laya, Julia-1, GLiNER2.5-Decide, JEV-9B, open-jev) are
  **NOT MEASURED — integration not started**: each needs its own published inference path,
  which is a separate 2-hour timebox per model (ADVISORY T9) and was not reached in this
  session. The ladder models are the priority and are complete.

## Reading these numbers

- A model below its majority baseline is reported as such, not adjusted.
- ECE is computed on the raw option softmax; it is large for every model here, which is the
  measurement (a 0.8B model's letter distribution is not calibrated) and not a defect of
  the harness — the T6 readout diagnostics (`share_vocab_argmax_is_option`) are reported in
  `results.json` for that reason.
- The fresh tier is the headline tier; tier-1 numbers are for comparability with published
  work and carry a contamination caveat until T4's probe is run.
