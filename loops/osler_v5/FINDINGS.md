# osler_v5 — findings (closure)

Loop 1h: per-item loss weighting, option-count augmentation with a positional check, four recipe arms, a 63,304-step Osler-4B, a blend
toward the base and Gate O5. Started 2026-10-26T19:24:27Z. Closed at Z7 on 2026-10-28 with **no final model trained**: no recipe
qualified under the pre-registered choice rule (D56c, ADVISORY §8.4; outcome (4)). Branch `loop/osler_v5`. Every number below has its
CLAIMS row (`loops/osler_v5/CLAIMS.md`). Counts and figures only; no item, note or prediction text is in this file.

## Summary

1. **No recipe qualified, so there is no final model.** On the corrected mix (the 155 `fda_route_choice_v1` rows dropped; C059) the D56c
   rule selects nothing at 13,992 steps (C070). F0, osler_v4's recipe, fails positional health in the original order (two-letter 0.3647
   against a limit of 0.3962; C065). F2 and F2s1 (per-item weighting with option augmentation) pass positional health in all three orders,
   but fail the knowledge lower bound against F0 (−0.0150 and −0.0115 against > −0.01) and the MIMIC lower bound (−0.0343 and −0.0420
   against > −0.03) (C066, C068, C069). The 63,304-step final (Z3), the blend (Z4), Gate O5 (Z5) and the membership audit (Z6) are NOT
   MEASURED, because they need a chosen recipe.

2. **Per-item weighting trades MIMIC for positional health.** At 13,992 steps, against F0, F2's MIMIC macro is −0.0264 and F2s1's is
   −0.0327 (C066, C069). Their MR macro is +0.0050 and +0.0117, and their CG v2 micro is 0.7844 and 0.7791 against F0's 0.7740 (C062,
   C068, C065). The MIMIC share of the applied update was 0.588 under per-step weighting and 0.3365 under per-item (C010 to C011). That
   fits the MIMIC drop. The link is a hypothesis, and this loop did not test it.

3. **Knowledge is below the zero-shot reference in point estimates for all three corrected arms; F2 and F2s1 significantly so.** On the same 4,000 knowledge dev items the zero-shot
   reference scores 0.7725 (C075). At 13,992 steps the arms are −0.0068 against it for F0 (the interval includes zero), −0.0130 for F2
   (−0.0213 to −0.0053) and −0.0093 for F2s1 (−0.0183 to −0.0005). The osler_v4 final's α = 1 drop was −0.0140 (C036). The D56c knowledge
   bound is against F0, which is itself below zero-shot in point estimate, so the rule does not test this drop.

4. **The corrected mix does not clearly fix the positional check.** At seed 0 the old-mix recipe passes at 13,992 steps (two-letter 0.4469
   against 0.3642; C044), while the corrected F0 fails (0.3647; C065). The old-mix seed-1 repeat collapses to 0.0000 (C052). No corrected
   arm collapses: two-letter accuracy is 0.3647 to 0.5391 at 13,992 steps (C073). F0 passes at 6,996 and fails at 13,992 (C064, C065).
   Positional health is therefore unstable across steps and seeds. Whether the corrected mix alone fixes the collapse (ADVISORY §8.8) is
   not settled.

5. **No training defect in the corrected arms.** Each arm has 279 logged steps with no non-finite losses or gradients and no exact-zero
   loss. Median gradient norms are 9.6 to 13.0. One single-batch spike (439.0 at step 2,900) was back to loss 0.0019 by step 3,000 (C074).
   No arm answers one key on a majority of the DDXPlus items (modal shares 0.068 to 0.130; C073).

6. **Additional findings, used in no choice.** On the osler_v4 final the blend curve gives S 0.8386 at α = 1 and a maximum of 0.8415 at
   α = 0.7 (C032). Positional health passes at α ≤ 0.5 and fails from 0.7 (C033). DDXPlus two-letter accuracy falls from 0.6112 to 0.0040
   at α = 1 (C034). F2's readout rows 26 to 254 sit 0.330 from pplx's base, against 0.0057 for osler_v4's final (C063). The old-mix arms
   are not candidates: R0 has S 0.8288 at 13,992 steps (C043), R2 has S 0.8311 with positional health passing (C056), and R3 was stopped at
   step 5,850 with no evaluation (C071).

7. **What was not done.** Z3 to Z6 did not run (BLOCKED; STATE §5). The corrected mix keeps a family share 54 rows above the 35 % cap
   (C059). The `fda_route_choice_v1` template is removed from the mix, not repaired, and its defect screen covered option counts only
   (C048).

## 2. Outcomes (ADVISORY §2)

- **(1)** NOT MEASURED: there is no final model.
- **(2)** NOT MEASURED: there is no final model.
- **(3)** NOT MEASURED: the Z4 rule needs a final. The α curve on the osler_v4 final is additional (C032 to C035).
- **(4)** APPLIES, through the §4 Z2 rule: no arm qualifies, so the recipe question stops and nothing is chosen (C070). The positional
  condition in the outcome's text is not the failing check here, because F2 and F2s1 pass positional health. STATE §7.12 records the
  reading.

## 3. Arms (`loops/osler_v5/arms.md`)

The corrected arms at 13,992 steps are the candidates:

| arm | S | MR macro | MIMIC macro | CG v2 micro | knowledge dev | positional, all three orders | D56c |
|---|---|---|---|---|---|---|---|
| F0 (osler_v4's recipe, seed 0) | 0.8296 | 0.9325 | 0.7822 | 0.7740 | 0.76575 | fails (original order) | control; out (C065) |
| F2 (per-item and augmentation, seed 0) | 0.8259 | 0.9375 | 0.7558 | 0.7844 | 0.7595 | passes | out: knowledge, MIMIC (C062, C066) |
| F2s1 (F2 at seed 1) | 0.8243 | 0.9442 | 0.7496 | 0.7791 | 0.76325 | passes | out: knowledge, MIMIC (C068, C069) |

The choice is none (C070). The old-mix arms (defective mix; additional) are R0, R0 prime, R2 and R3 (C041 to C058, C071, C072).

## 4. The final, the blend, Gate O5 and the scoreboard

- **Final (Z3):** NOT MEASURED. BLOCKED: no recipe qualifies (C070), so there is no checkpoint, S trend or positional health per
  evaluation for a final.
- **Blend (Z4):** NOT MEASURED on a final. The additional α curve on the osler_v4 final is in Summary item 6 (C032 to C035).
- **Gate O5 (Z5):** NOT MEASURED. The gate wiring is validated against osler_v4's published figures (C040).
- **Scoreboard (Z5):** NOT MEASURED, because there is no final to score.
- **Membership audit (Z6):** NOT MEASURED, because there is no final to audit.

## 5. Five-claim spot-check (fresh processes, 2026-10-28T03:59Z; repeated at closure, 04:36Z)

The closure repeat reproduced all five. The C070 command had been renamed to `z2_choose_d56c.py` before the repeat.

| claim | recompute command | result |
|---|---|---|
| C065 | `z2_arm_summary.py F0` | S 0.8296; positional all orders False (two-letter 0.3647 against single 0.7924): reproduced |
| C068 | `z2_arm_summary.py F2s1` | S 0.8243; positional all orders True (0.5271 against 0.7585): reproduced |
| C066, C069 | `z2_pairs.py --control F0 F2 F2s1` | F2 knowledge −0.0063 [−0.0150, +0.0030], MIMIC −0.0264; F2s1 knowledge −0.0025, MR +0.0117, MIMIC −0.0327: reproduced |
| C070 | `z2_choose_d56c.py` | choice none; outcome (4): reproduced |
| C075 | the one-line command in CLAIMS.md | F0 −0.00675, F2 −0.0130, F2s1 −0.00925: reproduced |

The Z2 self-audit (`outputs/osler_v5/Z2/SELF_AUDIT.md`, gitignored) re-derived every headline composite, knowledge count and positional
value from the raw rows: maximum difference 0.00.

Closure checks: all three corrected arms completed 13,992 steps with evaluations at 6,996 and 13,992, and their `full/` resume states are
removed (C077). The lint and the test suite pass with the GPU hidden: `ruff check .` is clean, and the suite gives 1107 passed, 4 skipped
(all four need a CUDA device), 2 deselected, exit 0 (C076).

## 6. Limits

- The choice margins are small. F2s1's knowledge lower bound is 0.0015 past its limit, and F2's MIMIC lower bound is 0.0043 past its
  limit (SELF_AUDIT §4.1). The limits are pre-registered and were not changed.
- There is one repeat seed (F2s1). F0 has one seed, so its seed spread is not measured.
- F0's positional verdict changes between evaluations, so a single-evaluation verdict is weak evidence (SELF_AUDIT §4.2).
- The knowledge bound is against F0, which is below zero-shot.
- The defect screen covered option counts only. No other template was screened for multi-true-label gold.
- No final, blend, gate or audit exists, so the release targets (MIMIC suite ≥ +0.10 over zero-shot pplx; MR verdict template macro ≥
  osler_v2's − 0.01; panel non-inferior) are NOT MEASURED here.

## 7. What this loop cannot claim (ADVISORY §8.8)

Whether the corrected mix alone fixes the collapse is not settled. At seed 0 the corrected control F0 still fails the original order at
13,992 steps. The per-item arms pass positional health, but they fail the knowledge and MIMIC bounds. A seed-1 run of osler_v4's recipe on
the corrected mix was not in the plan and was not run.
