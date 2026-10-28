# osler_v5 — next: proposals (not started)

Each proposal needs the operator's decision, or a new loop with its own pre-registration. Nothing here was run, and no proposal changes
a verdict of this loop. Claim ids are in `CLAIMS.md`. The loop stopped at Z7 with no final model: outcome (4) of the D56c rule (C070).

1. **The `fda_route_choice_v1` defect (operator question 1; STATE §6.1).** The template gives gold `routes[0]` while other routes are
   also true (C048). The corrected mix drops its 155 rows (D56a, C059); it does not repair the template. Proposal: (a) rebuild the
   template with multi-route handling in a new loop, which changes the mix hash, and (b) screen every other template in the mix for the
   same defect. The screen in this loop looked only at option counts (C048), so the other templates are unchecked (SELF_AUDIT §4.6).

2. **The MIMIC bound under per-item weighting (C066, C069).** Both per-item arms pass positional health at 13,992 steps and fail the
   MIMIC lower bound against F0 (−0.0343 and −0.0420 against > −0.03). Hypothesis (not tested): per-item weighting lowers MIMIC's share
   of the applied update from 0.588 to 0.3365 (C010 to C011), so the MIMIC macro falls. Proposal: a new loop, pre-registered, that tests a
   per-item variant with an explicit MIMIC share guard. Its bounds are set before the run; the bounds of this loop are not relaxed.

3. **Knowledge below zero-shot, and the rule's control (C075).** All three corrected arms are below the zero-shot reference in point estimate on the same
   4,000 knowledge dev items (0.7725): F0 −0.0068 (interval includes zero), F2 −0.0130 (−0.0213 to −0.0053), F2s1 −0.0093. The D56c
   bound is against F0, which is itself below zero-shot in point estimate, so it does not test the drop. Decision for the operator: whether the next loop
   measures the knowledge bound against zero-shot (a rule change, to be pre-registered there), and whether a knowledge-preserving
   control (KL to base or broader replay, both already tested in osler_v3 and osler_v4) belongs in its arms.

4. **Positional health is unstable across steps and seeds (C041 to C044, C052, C064, C065).** F0 passes at 6,996 and fails at 13,992;
   the old-mix control passes at seed 0 and collapses at seed 1 (C052). One evaluation is weak evidence for the positional test.
   Proposal: a new loop that pre-registers how positional health is judged (two consecutive evaluations, or the mean over the last
   evaluations), and that repeats the control at a second seed.

5. **The readout drift (C063, additional).** F2's rows 26 to 254 sit 0.330 from pplx's base readout at 13,992, against 0.0057 for
   osler_v4's final. Hypothesis (not tested): the augmentation drives that drift, and the drift is part of the knowledge loss. Proposal:
   a readout-regularised variant (rows 26 to 254 pulled toward the base) as an arm in the same new loop, judged on dev only.

6. **The 54-row family-cap excess (C059).** The corrected mix's family share is 0.350037, above the 35 % cap by 54 rows, because the
   drop removed rows from the total. The cap was not re-applied, since ADVISORY §8.2 changes nothing else. Proposal: re-apply the cap
   when the template is rebuilt under item 1, so both changes are made once.

7. **The final, blend, Gate O5 and membership audit (Z3 to Z6; STATE §5).** None ran. They need a recipe chosen by a rule that the
   corrected arms can meet, which is the point of items 2 to 4. Osler-9B and the 63,304-step final wait for that choice.

8. **Item-id scheme (operator question 5; STATE §6.2).** Unchanged by this loop: the prediction files carry the id that
   `mix_row_to_item` derives, and every join here checks that id set against the frozen file.
