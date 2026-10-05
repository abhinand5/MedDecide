# Zero-shot baseline table (T9)

**Generated:** 2026-10-05T22:21:30Z  
**Prediction rows:** 44313 across 22 (model, source, template, split) groups  
**Models in this table:** `Qwen/Qwen3.5-0.8B`

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

## Templates dropped by the T8 screen (measured, but not headline rows)

These are scored for traceability; the screen dropped them before T9, so they are not part
of the baseline claim.

| model | source | template | split | qtype | n | coverage | accuracy (95% CI) | majority | macro acc | Brier | ECE | label mass | p50 s | shuffle flip | abstention |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `Qwen/Qwen3.5-0.8B` | clinicaltrials | `ct_healthy_volunteers_noul_v1` | test | noul | 2900 | 0.273 | 0.440 (0.422-0.459) | 1.000 | 0.440 | 0.683 | 0.239 | 0.043 | 0.009 | 0.300 | 0.133 |
| `Qwen/Qwen3.5-0.8B` | pubmed | `pubmed_observational_noul_v1` | test | noul | 5000 | 0.245 | 0.041 (0.036-0.047) | 0.985 | 0.513 | 1.229 | 0.746 | 0.021 | 0.009 | 0.300 | 0.133 |

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
