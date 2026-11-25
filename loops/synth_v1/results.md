# synth_v1 — Gate O5 result (aggregates only; 2026-10-31)

Model: the 63,304-step F2 run on the corrected mix plus the MIMIC-grounded synthetic block (16,582 grounded items built; 13,120 kept in the mix after the ICD-specificity exclusion and the held-out drop). Scored calibrated on the frozen sets. Gate O5 = Gate O4 (D55, unchanged). Source: outputs/synth_v1/final/gate.json (gitignored).

| check | vs zero-shot pplx-decider-flash-v1.2 | vs MedDecider-4B |
|---|---|---|
| (i) MR verdict | +0.1534 [+0.1384, +0.1667]: PASS | +0.2621 [+0.2500, +0.2737]: PASS |
| (ii) MIMIC audit suite | +0.1270 [+0.1118, +0.1409]: PASS | +0.1252 [+0.1105, +0.1402] (8 templates): PASS |
| (iii) external panel | −0.0340 [−0.0483, −0.0212]: FAIL (limit −0.02) | NOT MEASURED (212 of 4,353 unpaired) |
| (iv) knowledge | −0.0036 [−0.0136, +0.0058]: PASS | +0.0091 [−0.0035, +0.0224]: PASS |

Overall: vs zero-shot pplx: **FAIL** (the panel check only). vs MedDecider-4B: NOT MEASURED (the panel check).

Comparison with osler_v4's final (Gate O4): panel −0.042 (lower −0.058), knowledge −0.007, MR +0.139, MIMIC +0.165. This model is closer on the panel (−0.034) and lower on MIMIC (+0.127), with the synthetic block in the mix.

Not in this verdict: the membership audit and the blend (not run). Training-run dev figures are in loops/synth_v1/STATE.md.
