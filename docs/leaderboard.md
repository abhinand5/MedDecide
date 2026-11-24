# MedDecide leaderboard (eval0 snapshot, 2026-11-21)

Aggregate numbers only. Recomputed from prediction files by `scripts/eval0/leaderboard.py`
(`outputs/eval0/leaderboard/leaderboard.json`). Accuracy with 95% bootstrap CI (2,000 resamples). Paired differences use
the same items and compare each model with its own zero-shot base.

## External panel (4,353 items; the set v6 failed)

| model | accuracy | 95% CI | macro (template) | coverage |
|---|---|---|---|---|
| pplx-decider-v1.2-27b (zero-shot, ceiling) | 0.6510 | — | — | 4,353 |
| Osler-9B final (63,304 steps) | 0.5927 | — | — | 4,353 |
| d3-flash (zero-shot, 9B base) | 0.5899 | — | — | 4,353 |
| Osler-9B arm (13,992 steps) | 0.5883 | — | — | 4,353 |
| Osler-4B arm (13,992 steps) | 0.5610 | — | — | 4,353 |
| pplx-decider-flash-v1.2 (zero-shot, 4B base) | 0.5615 | — | — | 4,353 |
| MedDecider-9B (lion-ai) | 0.5603 | — | — | 4,141 (212 22-option items not scored) |
| MedDecider-4B (lion-ai) | 0.5436 | — | — | 4,141 (same) |
| Osler-4B v2 | 0.5343 | — | — | 4,353 |
| Osler-4B v4 final | 0.5254 | — | — | 4,353 |
| JEV-9B | NOT MEASURED — no prediction file | | | 0 |

Same-subset check (the 4,141 items MedDecider covers): Osler-9B final 0.5878; d3-flash base 0.5851; Osler-9B arm 0.5854;
Osler-4B arm 0.5576; pplx-flash 0.5554; MedDecider-9B 0.5603; MedDecider-4B 0.5436; pplx-27B 0.6489.

Paired vs own base (panel, all 4,353): Osler-9B final **+0.0028**, 95% CI [−0.0083, +0.0131]; Osler-9B arm −0.0016
[−0.0113, +0.0083]; Osler-4B arm −0.0005 [−0.0106, +0.0099]. No detectable change on the panel from training.

## Knowledge test (5,456 frozen items; out of training distribution)

| model | accuracy | 95% CI | coverage |
|---|---|---|---|
| pplx-decider-v1.2-27b (zero-shot, ceiling) | 0.7823 | — | 5,435 |
| d3-flash (zero-shot, 9B base) | 0.7335 | — | 5,456 |
| Osler-9B final (63,304 steps) | 0.7322 | — | 5,456 |
| Osler-9B arm (13,992 steps) | 0.7258 | — | 5,456 |
| MedDecider-9B (lion-ai) | 0.6997 | — | 5,435 |
| Osler-4B v2 | 0.6752 | — | 5,456 |
| Osler-4B v4 final | 0.6765 | — | 5,456 |
| Osler-4B arm (13,992 steps) | 0.6736 | — | 5,456 |
| pplx-decider-flash-v1.2 (zero-shot, 4B base) | 0.6835 | — | 5,435 |
| MedDecider-4B (lion-ai) | 0.6703 | — | 5,435 |

Paired vs own base: Osler-9B final −0.0013 [−0.0090, +0.0059] (a tie); Osler-9B arm −0.0077 [−0.0152, −0.0006];
Osler-4B arm −0.0090 [−0.0171, −0.0009].

## In-domain training splits (not evidence of general skill)

MR verdict (13,103) and MIMIC suite (4,253) are frozen test splits drawn from the same families the Osler models were
trained on. Read them as in-domain results only.

| model | MR verdict | MIMIC suite |
|---|---|---|
| Osler-9B final | 0.9016 | 0.7733 |
| Osler-4B v4 final | 0.8862 | 0.7947 |
| Osler-9B arm | 0.8548 | 0.7444 |
| Osler-4B arm | 0.8336 | 0.7214 |
| d3-flash (9B base) | 0.7351 | 0.6301 |
| pplx-flash (4B base) | 0.7466 | 0.6240 |
| pplx-27B | 0.8093 | 0.6729 |
| MedDecider-4B | 0.6681 | NOT MEASURED |
| MedDecider-9B | 0.6653 | NOT MEASURED |

## Caveats

- **Calibration (Brier, ECE):** measured on the panel only, and only for the 9B from the re-run with probabilities
  (the earlier 9B knowledge, MR and MIMIC prediction files stored the argmax only). Knowledge, MR and MIMIC calibration for
  the 9B: NOT MEASURED.
- **Osler-4B v2 and v4 final** are earlier lineage models, shown for context.
- **The leaderboard is not a release claim.** The panel is the external check; the Osler-9B does not beat pplx-27B
  (0.593 vs 0.651) and matches its own base on the panel. Release needs operator sign-off and the PHI/memorisation audit.
- **Universe:** the knowledge, MR and MIMIC sets are the frozen Y9 splits; their item counts match the frozen files.
