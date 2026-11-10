# d3_distill — Gate O7

Written `2026-11-10T04:06:26Z` by `scripts/d3_distill/step3_gate.py`. Aggregates only; no item text.

## Rule (PLAN.md step 3, unchanged from distill_v0)

Panel lower bound > 0 against zero-shot pplx-decider-flash-v1.2; knowledge lower bound > -0.02; MR verdict and
MIMIC within -0.02 of the osler_v4 final. The teacher is an additional row; the students are judged on the gate.
Pairing by item id, unpaired above 1 % ⇒ NOT MEASURED (D44(a)); a failing readout cell for either model ⇒ NOT
MEASURED (D12); paired percentile bootstrap, 1,000 resamples, seed 0 (the Gate O4 machinery).

## Results

### mini

Verdict: **NOT MEASURED (missing checks: mr_verdict vs osler_v4 final, MIMIC)**

| check | status | rule | accuracy delta [95 %] | Brier delta | paired / unpaired |
|---|---|---|---|---|---|
| panel | NOT RUN |  | — | — | — |
| knowledge | NOT RUN |  | — | — | — |
| mr_verdict_additional | NOT RUN |  | — | — | — |
| mr_verdict_vs_osler_v4_final | NOT MEASURED | the osler_v4 final's predictions and weights are not on this machine (operator instruction 2026-11-09) | — | — | — |
| mimic | NOT MEASURED | no MIMIC data on this machine (operator instruction: do not touch MIMIC) | — | — | — |

Own figures (template macro accuracy / item-mean Brier):

| set | final student | zero-shot student | pplx-flash |
|---|---|---|---|
| panel | NOT RUN | NOT RUN | 0.6362 / 0.6418 (4353) |
| knowledge | NOT RUN | NOT RUN | 0.7105 / 0.4670 (5435) |
| mr_verdict | NOT RUN | NOT RUN | 0.7050 / 0.3777 (13103) |

### flash

Verdict: **NOT MEASURED (missing checks: mr_verdict vs osler_v4 final, MIMIC)**

| check | status | rule | accuracy delta [95 %] | Brier delta | paired / unpaired |
|---|---|---|---|---|---|
| panel | NOT RUN |  | — | — | — |
| knowledge | NOT RUN |  | — | — | — |
| mr_verdict_additional | NOT RUN |  | — | — | — |
| mr_verdict_vs_osler_v4_final | NOT MEASURED | the osler_v4 final's predictions and weights are not on this machine (operator instruction 2026-11-09) | — | — | — |
| mimic | NOT MEASURED | no MIMIC data on this machine (operator instruction: do not touch MIMIC) | — | — | — |

Own figures (template macro accuracy / item-mean Brier):

| set | final student | zero-shot student | pplx-flash |
|---|---|---|---|
| panel | NOT RUN | NOT RUN | 0.6362 / 0.6418 (4353) |
| knowledge | NOT RUN | NOT RUN | 0.7105 / 0.4670 (5435) |
| mr_verdict | NOT RUN | NOT RUN | 0.7050 / 0.3777 (13103) |

## Notes

- The MR check the PLAN defines is against the osler_v4 final; that model's predictions and weights are not
  on this machine, so it is NOT MEASURED and no other number is substituted. The mr_verdict row against
  pplx-flash is additional (the operator's 2026-11-09 instruction), not the PLAN's MR check.
- MIMIC is NOT MEASURED: the operator forbids MIMIC data on this machine.
- The d3_bench 500/1,000-item subsamples are not used as the gate; results.md says they cannot stand in for it.
- Teacher row (additional, d3_bench seeded subsamples only, not the full sets): d3-ultra knowledge 0.805 (1,000),
  panel 0.638 (500), mr_verdict 0.842 (500); d3_bench also has d1-ultra, Clef-Flash and pplx-flash rows there.
- Majority (most frequent gold) baselines of the full sets (osler_v4 C231): mr_verdict 0.4177, panel 0.1941,
  knowledge 0.3118.
- Baseline reproduction check: our locally scored zero-shot pplx run gives template macro knowledge 0.71051
  (+0.0008 vs the published 0.7097), panel 0.63617 (-0.0028 vs 0.6390), mr_verdict 0.70500 (+0.0006 vs
  0.7044); every template passes the readout gate. The paired gate uses these local rows (same items, same
  code) and records the cross-check deltas under `baseline_reproduces_published`; the published row is a
  provenance check, not the comparator.
