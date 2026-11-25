# MedDecide leaderboard (RC0 snapshot)

Generated 2026-11-25T04:25:20Z at git `1ae8214014f0a2df22cfee0d3c23256d4d3fd277`. This is a pointer to the RC0 board: every number is recomputed from per-item prediction files by `scripts/osler_rc0/rc0_leaderboard.py` (12 models x 9 frozen evaluation sets, accuracy with 95% bootstrap CI, template macro, multi-class Brier, 15-bin ECE, coverage).

- Full board: [loops/osler_rc0/LEADERBOARD.md](loops/osler_rc0/LEADERBOARD.md)
- Per-source and per-template tables: [loops/osler_rc0/PER_CATEGORY.md](loops/osler_rc0/PER_CATEGORY.md)
- Pareto frontier and paired comparisons: [loops/osler_rc0/PARETO.md](loops/osler_rc0/PARETO.md)

## Headline: accuracy on the five headline sets

| model | panel | knowledge | confirm_general | mr_verdict | mimic_suite | 5-set macro |
|---|---|---|---|---|---|---|
| Osler-9B-rc0 | 0.5927 | 0.7322 | 0.9008 | 0.9016 | 0.7733 | 0.7801 |
| pplx-27B | 0.6510 | 0.7823 | 0.8843 | 0.8093 | 0.6729 | 0.7600 |
| Osler-4B-rc0 | 0.5502 | 0.6728 | 0.8883 | 0.8941 | 0.7597 | 0.7530 |
| d3-flash | 0.5899 | 0.7335 | 0.8774 | 0.7351 | 0.6301 | 0.7132 |
| pplx-flash | 0.5615 | 0.6835 | 0.8889 | 0.7466 | 0.6240 | 0.7009 |
| Clef-Flash | 0.5711 | 0.6864 | 0.8933 | 0.6831 | 0.6306 | 0.6929 |
| MedDecider-9B | 0.5603 | 0.6997 | 0.8825 | 0.6653 | 0.6189 | 0.6853 |
| MedDecider-4B | 0.5436 | 0.6703 | 0.8820 | 0.6681 | 0.6019 | 0.6732 |
| JEV-9B | 0.5127 | 0.6448 | NOT MEASURED | 0.6303 | 0.5885 | — |
| d1-2B | 0.4259 | 0.4555 | NOT MEASURED | 0.7069 | 0.5587 | — |
| Qwen3.5-9B | 0.5419 | 0.6774 | NOT MEASURED | 0.6315 | 0.6029 | — |
| Qwen3.5-4B | 0.4831 | 0.6162 | NOT MEASURED | 0.6176 | 0.5603 | — |

## Fronts

- 2-axis Pareto front (OOD General Avg vs Clinical Audit Avg): Osler-9B-rc0, pplx-27B.
- Undominated on all five headline sets (<=10B): Osler-9B-rc0, d3-flash.

Coverage and NOT MEASURED reasons are in the full board; no cell is imputed. The prior loops' boards (`loops/osler_v*/FINDINGS.md`) remain the record for their own recipes.

