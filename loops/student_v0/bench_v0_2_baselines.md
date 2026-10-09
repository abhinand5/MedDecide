# Zero-shot ladder baselines on MedDecide-Bench v0.2 (S1: the three `_v2` templates)

Generated: `2026-10-06T20:36:29Z`  
Models: `Qwen/Qwen3.5-0.8B`, `Qwen/Qwen3.5-9B`  
Cells: 6 (5 PASS, 1 READOUT_FAIL, 0 NOT MEASURED)  
Prediction rows: 14948

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
| `Qwen/Qwen3.5-0.8B` | nfcorpus | `nfcorpus_graded_score_v2` | score | 238/238 | 1.000 | 0.5966 (0.5336-0.6597) | 0.500 | 0.500 | 0.9953 | 0.940 | 0.202 |
| `Qwen/Qwen3.5-9B` | nfcorpus | `nfcorpus_graded_score_v2` | score | 238/238 | 1.000 | READOUT_FAIL — greedy agreement 0.880 < 0.9 (n=50) | 0.500 | 0.500 | 0.8025 | 0.880 | 0.282 |

## Tier 2 (fresh, 2026-03-01 window)

| model | source | template | qtype | n | coverage | accuracy (95% CI) | majority | chance | label mass | greedy | flip |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `Qwen/Qwen3.5-0.8B` | openfda | `fda_class_choice_v2` | choice | 3236/3236 | 1.000 | 0.7803 (0.7670-0.7933) | 0.250 | 0.250 | 0.9982 | 1.000 | 0.198 |
| `Qwen/Qwen3.5-0.8B` | pubmed | `pubmed_mesh_major_choice_v2` | choice | 4000/4000 | 1.000 | 0.9580 (0.9515-0.9645) | 0.250 | 0.250 | 0.9991 | 1.000 | 0.050 |
| `Qwen/Qwen3.5-9B` | openfda | `fda_class_choice_v2` | choice | 3236/3236 | 1.000 | 0.8705 (0.8588-0.8820) | 0.250 | 0.250 | 0.9989 | 0.980 | 0.100 |
| `Qwen/Qwen3.5-9B` | pubmed | `pubmed_mesh_major_choice_v2` | choice | 4000/4000 | 1.000 | 0.9830 (0.9788-0.9870) | 0.250 | 0.250 | 0.9996 | 1.000 | 0.020 |

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

