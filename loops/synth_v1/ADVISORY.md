# ADVISORY — synth_v1: MIMIC-grounded synthetic audit data, trained into Osler (autonomous)

Decided by the operator in chat on 2026-10-29: build MIMIC-grounded synthetic audit items locally (Nemotron writes text from
code-built facts; labels from structured fields), and train with them. No ablation. Judgement comes from the final model's
results on the public and audit sets (Gate O5, D55 rules, unchanged).

## Plan

1. **Build** (running; subagent branch `synth/mimic-grounded`): grounded items for the osler_v4 structured-label templates, with
   keep rates, leakage probe and aggregates in `outputs/synth_v1/`.
2. **Mix:** the corrected osler_v5 mix (D56a) plus the grounded items as one added block, capped at 20 % of rows and no more than
   8 % per template. The composition hash is recorded.
3. **Recipe:** osler_v5 F2 (per-item weighting, option augmentation; the arm with the best CG micro on the corrected mix, C062).
   63,304 steps, ten evaluations, positional health at each evaluation, checkpoint by S.
4. **Gate O5** (D55): the panel and knowledge checks against zero-shot pplx, MR and MIMIC checks against both baselines, the
   same limits as Gate O4. Reported with the MedDecider-4B panel as NOT MEASURED, as before.
5. **Comparison** (additional, labelled): this final against osler_v5's F2 at 63,304 steps if that exists, and against
   the osler_v4 final, on every set.
6. **Release gates (not generation gates):** memorisation and PHI audit (osler_v4 Y10) before any public weights.

## Risks, stated up front

- Without an ablation, any gain cannot be attributed to the synthetic block. The claim will say so.
- The synthetic block inherits MIMIC's distribution, so it may raise MIMIC scores without transferring. Held-out family
  transfer (MIMIC held-out, 110 items) is the check on that.
