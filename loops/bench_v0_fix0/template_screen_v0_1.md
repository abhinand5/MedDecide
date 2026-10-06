# Template screen v0.1 (F3)

Generated: `2026-10-06T08:57:09Z`  
Benchmark: `data/bench/v0.1/tier1 + data/bench/v0.1/fresh`  
Drop rule: drop when macro accuracy >= 0.9, or micro >= 0.9 and >= 0.1 above the majority baseline with macro >= 0.50, or n_test < 200, or a single gold class, or a flagged gold-in-state leak

Every number below is recomputed from the written benchmark files. `noul`/`score` regex and
BoW baselines are reported as `NOT MEASURED` where they do not apply — a 0.000 there would
read as "at chance" when the truth is "this baseline cannot be computed".

## tier1

| template | qtype | n test | classes | majority | chance | BoW micro | BoW macro | regex | gold-in-state | drop |
|---|---|---|---|---|---|---|---|---|---|---|
| `medmcqa_4opt_v1` | choice | 4183 | 4 | 0.322 | 0.250 | 0.285 | 0.263 | NOT MEASURED | 0.005 | keep |
| `medqa_usmle4_v1` | choice | 1273 | 4 | 0.277 | 0.250 | 0.229 | 0.224 | NOT MEASURED | 0.009 | keep |
| `medquad_routing_v1` | choice | 5000 | 4 | 0.252 | 0.250 | 0.256 | 0.256 | 0.000 | 0.593 | keep |
| `mmlu_mc_v1` | choice | 1103 | 4 | 0.325 | 0.250 | 0.298 | 0.247 | NOT MEASURED | 0.005 | keep |
| `nfcorpus_graded_score_v1` | score | 429 | 2 | 0.723 | 0.333 | NOT MEASURED | NOT MEASURED | NOT MEASURED | 0.014 | keep |
| `nfcorpus_relevant_noul_v1` | noul | 647 | 2 | 0.501 | 0.500 | 0.526 | 0.525 | NOT MEASURED | 0.414 | keep |
| `pubmedqa_ynm_v1` | choice | 477 | 3 | 0.551 | 0.333 | 0.556 | 0.339 | NOT MEASURED | 0.323 | keep |
| `scifact_relevant_noul_v1` | noul | 600 | 2 | 0.500 | 0.500 | NOT MEASURED | NOT MEASURED | NOT MEASURED | 0.453 | keep |
| `trec_covid_graded_score_v1` | score | 117 | 3 | 0.333 | 0.333 | 0.385 | 0.385 | NOT MEASURED | 0.000 | **DROP** n_test<200 |
| `trec_covid_relevant_noul_v1` | noul | 78 | 2 | 0.500 | 0.500 | 0.603 | 0.603 | NOT MEASURED | 0.372 | **DROP** n_test<200 |

8 kept, 2 dropped in tier1.

Dropped templates and reasons:

* `trec_covid_graded_score_v1` — n_test<200
* `trec_covid_relevant_noul_v1` — n_test<200

## fresh

| template | qtype | n test | classes | majority | chance | BoW micro | BoW macro | regex | gold-in-state | drop |
|---|---|---|---|---|---|---|---|---|---|---|
| `ct_healthy_volunteers_noul_v1` | noul | 2000 | 2 | 0.500 | 0.500 | 0.784 | 0.784 | NOT MEASURED | 0.396 | keep |
| `ct_intervention_type_choice_v1` | choice | 99 | 11 | 0.091 | 0.091 | 0.313 | 0.313 | 0.212 | 0.303 | **DROP** n_test<200 |
| `ct_phase_choice_v1` | choice | 1100 | 5 | 0.200 | 0.143 | 0.503 | 0.503 | 0.178 | 0.116 | keep |
| `ct_primary_purpose_choice_v1` | choice | 198 | 9 | 0.111 | 0.111 | 0.288 | 0.288 | 0.192 | 0.253 | **DROP** n_test<200 |
| `ct_randomised_noul_v1` | noul | 2000 | 2 | 0.500 | 0.500 | 0.824 | 0.824 | NOT MEASURED | 0.380 | keep |
| `fda_boxed_warning_noul_v1` | noul | 1910 | 2 | 0.500 | 0.500 | 0.879 | 0.879 | NOT MEASURED | 0.501 | keep |
| `fda_class_choice_v1` | choice | 4000 | 4 | 0.250 | 0.250 | 0.245 | 0.245 | NOT MEASURED | 0.000 | keep |
| `fda_route_choice_v1` | choice | 84 | 12 | 0.083 | 0.059 | 0.345 | 0.345 | 0.405 | 0.750 | **DROP** n_test<200 |
| `pubmed_humans_noul_v1` | noul | 1316 | 2 | 0.500 | 0.500 | 0.856 | 0.856 | NOT MEASURED | 0.462 | keep |
| `pubmed_mesh_major_choice_v1` | choice | 4000 | 4 | 0.250 | 0.250 | 0.246 | 0.246 | NOT MEASURED | 0.463 | keep |
| `pubmed_observational_noul_v1` | noul | 206 | 2 | 0.500 | 0.500 | 0.796 | 0.796 | NOT MEASURED | 0.461 | keep |
| `pubmed_pubtype_choice_v1` | choice | 114 | 6 | 0.167 | 0.167 | 0.675 | 0.675 | 0.553 | 0.588 | **DROP** n_test<200 |

8 kept, 4 dropped in fresh.

Dropped templates and reasons:

* `ct_intervention_type_choice_v1` — n_test<200
* `ct_primary_purpose_choice_v1` — n_test<200
* `fda_route_choice_v1` — n_test<200
* `pubmed_pubtype_choice_v1` — n_test<200

