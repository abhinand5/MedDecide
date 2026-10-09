# Zero-shot ladder baselines on MedDecide-Bench v0.2 consistency templates (S2)

Generated: `2026-10-06T21:47:22Z`  
Models: `Qwen/Qwen3.5-0.8B`, `Qwen/Qwen3.5-9B`  
Cells: 6 (6 PASS, 0 READOUT_FAIL, 0 NOT MEASURED)  
Prediction rows: 16000

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

## Tier 2 (fresh, 2026-03-01 window)

| model | source | template | qtype | n | coverage | accuracy (95% CI) | majority | chance | label mass | greedy | flip |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `Qwen/Qwen3.5-0.8B` | clinicaltrials | `ct_arm_role_noul_v1` | noul | 2000/2000 | 1.000 | 0.6950 (0.6760-0.7155) | 0.500 | 0.500 | 0.9896 | 0.940 | 0.350 |
| `Qwen/Qwen3.5-0.8B` | clinicaltrials | `ct_claim_set_choice_v1` | choice | 4000/4000 | 1.000 | 0.2517 (0.2392-0.2650) | 0.250 | 0.250 | 0.9969 | 0.940 | 0.634 |
| `Qwen/Qwen3.5-0.8B` | openfda | `fda_route_claim_noul_v1` | noul | 2000/2000 | 1.000 | 0.8875 (0.8735-0.9005) | 0.500 | 0.500 | 0.9951 | 0.940 | 0.234 |
| `Qwen/Qwen3.5-9B` | clinicaltrials | `ct_arm_role_noul_v1` | noul | 2000/2000 | 1.000 | 0.8780 (0.8630-0.8915) | 0.500 | 0.500 | 0.9865 | 1.000 | 0.046 |
| `Qwen/Qwen3.5-9B` | clinicaltrials | `ct_claim_set_choice_v1` | choice | 4000/4000 | 1.000 | 0.7778 (0.7652-0.7905) | 0.250 | 0.250 | 0.9980 | 1.000 | 0.100 |
| `Qwen/Qwen3.5-9B` | openfda | `fda_route_claim_noul_v1` | noul | 2000/2000 | 1.000 | 0.9020 (0.8900-0.9150) | 0.500 | 0.500 | 0.9980 | 0.940 | 0.066 |

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

