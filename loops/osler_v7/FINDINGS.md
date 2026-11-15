# FINDINGS — osler_v7 (data-engineered Pareto recipe)

## Summary

The loop built the format-matched C1–C4 cells, distilled them from the local pplx-27B, trained one arm per base on the same
v7 mix (CE + teacher distillation + KL-to-base on the C1–C4 pool), selected each arm by the pre-registered dev rule, and
gated each against its own zero-shot base. **Both arms improved on their bases on dev; interpolation (Lever 2) added
nothing for either model — the trained endpoint (s = 1.00) won both selections. The 9B arm passes Gate v7; the 4B gate is
NOT MEASURED because the pplx authors' zero-shot path refuses 21 of 5,456 knowledge rows (repeated option labels), and a
partially covered check does not pass.** The 9B final at 63,304 steps reached dev S 0.853820 and also passes the same gate
as an additional row.

| model | dev S (13,992-step arm) | chosen blend | gate vs own base |
|---|---|---|---|
| 4B (pplx-flash) | 0.811190 | s = 1.00 (S 0.810521) | **NOT MEASURED** (G-a/G-b/G-c measured clear their margins; base coverage 5,435/5,456 on knowledge) |
| 9B (d3-flash) | 0.826048 | s = 1.00 (S 0.826048) | **PASS** (G-a −0.00627, G-b +0.09266, G-c −0.01764 lower bounds) |
| 9B final (63,304 steps) | 0.853820 | — | **PASS** (additional: G-a +0.01540, G-b +0.14685, G-c −0.00842 lower bounds) |

## The five-claim spot-check

Five headline numbers were re-derived from the raw artifacts in a fresh process (details and commands in
`outputs/osler_v7/T12/SELF_AUDIT.md` and `outputs/osler_v7/T12/spotcheck.json`):

| claim | recorded | re-derived |
|---|---|---|
| C030 T6 arm final S | 0.81118996 | 0.81118996 |
| C037 T8 arm final S | 0.82604843 | 0.82604843 |
| C040 T9 chosen S (s=1) | 0.82604843 | 0.82604843 |
| C045 T10 9B G-b lower bound | 0.09266438 | 0.09266438 |
| C049 T11 final S | 0.85381978 | 0.85381978 |

## What each lever measured

- **T1 (diagnosis, CPU):** the v6 mix is dominated by long MIMIC and replay rows; the option-code readout is healthy on
  single-letter templates but not on two-letter ones. (CLAIMS C001–C004.)
- **T4 (Lever 3 grid, 4B):** nine cells at 13,992 steps. The best dev template macro was the LoRA-scale 0.75 / readout-scale
  1.0 cell (0.808647), and the two-letter loss was in the LoRA, not the readout: dropping the readout scale to 0.0 at LoRA
  1.0 left two-letter accuracy at 0.519 versus 0.651. Diagnostic only — it did not select the recipe. (C009–C014.)
- **T5/T6 (data engine + 4B arm):** the C1–C4 pool (69,967 rows) was labelled by the local pplx-27B (3.72 h, 5.22 items/s,
  0 errors); the v7 mix is 1,526,405 rows with 438,188 labelled (28.7 %). The 4B arm reached S 0.811190 at 13,992 steps
  (MR 0.887582, MIMIC 0.754032, CG 0.791955, knowledge 0.763571; positional PASS). Dose: first-200 gradient distillation
  share 0.352292; KL-to-base active on 40.5 % of steps. (C023–C031.)
- **T7 (Lever 2, 4B):** every blend scale was non-inferior to the base on the paired template-macro difference; s = 1.00
  had the highest dev S (0.810521). All scales passed the positional check in all three orders. Calibrated temperatures:
  choice 1.206404, noul 1.351652, score 1.135541. (C032–C035.)
- **T8 (9B arm):** S 0.826048 at 13,992 steps (MR 0.891392, MIMIC 0.767228, CG 0.819526, knowledge 0.834741; positional
  original PASS). Dose: first-200 gradient share 0.152055; KL-to-base active on only 3.36 % of steps because token-capped
  batches put the long `v7_c` rows alone. (C036–C039.)
- **T9 (Lever 2, 9B):** every scale non-inferior; the 0.25 and 0.50 blends **fail** all-orders positional health, so §9
  removed them and s = 1.00 (S 0.826048) was chosen. (C040–C043.)
- **T10 (Gate v7):** the 9B arm passes all three checks. The 4B's three measured lower bounds also clear their margins
  (G-a −0.00852918, G-b +0.06206641, G-c −0.01686213 on the intersection), but the pplx zero-shot path skipped 21
  knowledge rows (`repeated_label`), so the strict verdict is **NOT MEASURED** under ADVISORY §7. (C044–C047.)
- **T11 (63,304-step 9B final):** S rose 0.806665 → 0.829150 → 0.836843 → 0.852858 → **0.853820**, with all-orders
  positional PASS at the end; it also passes Gate v7 against its base as an additional row (G-a +0.01540201,
  G-b +0.14685244, G-c −0.0084196 lower bounds). (C048–C052.)

## Deviations and limitations (recorded, not smoothed)

- **Batching deviation (T8/T11):** the 8.4B backbone OOM'd on fixed batches of eight long MIMIC rows, so batches are
  capped at 8 items and 8,192 padded tokens and all 32 PromptEncoder layers are activation-checkpointed. The recipe's
  item weighting is unchanged; the KL-to-base dose fell to 3.4–3.5 % of steps because the long `v7_c` rows batch alone.
  The ADVISORY sets no KL active-share target, but this is a weaker Lever-1 dose than intended.
- **Evaluation bookkeeping (T8/T9):** the first in-run evaluator and the first T9 run had id/origin and relative-path
  defects; all reported numbers were re-aggregated from the intact prediction files and re-audited. No check or threshold
  was edited.
- **4B gate coverage:** the 21 skipped knowledge rows are a property of the pplx authors' zero-shot path, not of the
  trained model. The question whether to accept the intersection as a PASS is recorded in STATE §5b for the operator.
- **Panel:** the external panel was not scored in v7; ADVISORY reports it only as an additional row, never as a gate input.
  It is **NOT MEASURED** for these models.
- The two bases use different readout protocols (4B per-question marker, 9B last-prompt position); no cross-protocol claim
  is made, and each model is gated only against its own base.

## What comes next

1. Operator decision on the 4B knowledge-row coverage (STATE §5b) — strict NOT MEASURED vs an intersection-based PASS.
2. If the 9B is released, run the memorisation/PHI audit required by D28 and score the external panel as an additional row.
3. If Lever 1 is revisited, mix the `v7_c` rows with a quota (as the teacher anchors already use) so the KL-to-base term is
   active on most steps rather than 3–4 %.
4. A matched-length study of the 4B at 63,304 steps is the natural next comparison (not run in this loop).
