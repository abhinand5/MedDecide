# ADVISORY — osler_v6: distillation with a dosing fix (GO recorded 2026-11-07; see STATE §2)

Status: **GO recorded 2026-11-07** (STATE §2). D57 and D58 are in force; §6 (the base) is still open and is not used for any run. Evidence: `docs/paper/EVIDENCE.md` §5.2–5.3 (knowledge and panel scores) and osler_v4 C116 (the A5 dose).

## 1. Goal and honest odds

Close the public-benchmark gap to pplx's 27B teacher on knowledge and on the external panel, without losing the record-audit lead.

Starting point (osler_v4 final, public sets): knowledge 0.702 (zero-shot pplx-4B 0.710, MedDecider-9B 0.727, pplx-27B 0.817);
panel 0.597 (zero-shot pplx 0.639, pplx-27B 0.720).

Odds, judgement (R10: not measured):
- beating MedDecider-9B on knowledge (needs about +2.5 points): about one in three;
- panel non-inferior to zero-shot pplx (needs about +4.2 points): about two in five.
- "Clear and absolute winner" on every public set: unlikely with this recipe. pplx-4B is itself distilled from the 27B and still sits
  11 points below it on knowledge, so 400,000 soft labels are unlikely to close the whole gap. A 9B Osler on Qwen3.5-9B is the route to
  a larger knowledge margin; it is a separate decision (D22 names it) and is out of scope here.

## 2. Pre-registered design (fixed before any run)

1. **Teacher:** pplx-decider-v1.2-27b, revision cce0f3f80033e9d89d80831d5281fd72124aacce (osler_v4 Y5 teacher; C072), scored through the osler_v4 Y5 path (parity 40 of 40 required again).
2. **Targets (D58):** 400,000 general anchor items (no domain prefix), eligibility as the Y5 eligibility steps in `loops/osler_v4/ADVISORY.md` (Y5), with the exclusions checked by
   text hash and recorded (overlap must be 0). Scoring is batched; the rate is recorded. If the rate is under 10 items/s, the target
   is cut to what fits in 12 GPU-hours and the cut is recorded. At osler_v4's measured ~6.2 items/s, 12 GPU-hours on top of the
   112,128 items already scored (Y5) gives about 380,000 in total; that cut is the expected outcome, not a shortfall.
3. **Dosing (D57):** each batch of 8 items has at least 3 labelled anchors; labelled items are drawn without replacement per epoch;
   the sampler is seeded and its batch statistics are logged (labelled items per batch, share of batches meeting the minimum).
4. **Calibration of λ (D57), a separate probe (task Z2; no arm weights kept):** 200 steps of G1's recipe at λ = 1, pre-clip.
   Share s = ‖∇KL‖ / (‖∇CE‖ + ‖∇KL‖), averaged over the 200 steps; s0 is that mean. The gradient share is modelled as λk/(1+λk)
   with k = s0/(1−s0), so the λ that gives 0.40 is λ = (0.4/0.6)·(1−s0)/s0 (Z2 writes it down before use). λ is then frozen.
   The arm (Z3) starts at the frozen λ, so the D57 band applies to the arm's own first 200 steps: the share over those steps must
   lie in 0.30 to 0.50. If it does not, G1 is BLOCKED and reported; λ is not re-tuned inside the arm. The probe log is kept.
5. **Arms (13,992 steps, the corrected mix of osler_v5 D56a, per-item weighting, augmentation as F2):**
   - G0: F2 without distillation (the reference; F2 seed 0, already run: C062);
   - G1: F2 + distillation under D57.
   Both at seed 0; G1 also at seed 1 if the seed-0 result is a positional pass (osler_v5 showed seed instability).
6. **Choice rule (dev only; the panel and the public test sets never select):**
   - a qualifier passes positional health in all three orders at 13,992 steps;
   - knowledge dev v2 non-inferiority: lower bound of (G1 − G0) > −0.01 (the distillation must not reduce knowledge on dev by more
     than one point; a superiority test was not pre-registered);
   - MR and MIMIC lower bounds against G0 > −0.03;
   - CG v2 micro point estimate ≥ G0's.
   The choice is the qualifier with the highest knowledge dev v2; if none qualifies, outcome (4) again: report and stop.
   G0 is the reference, not a candidate: its bounds against itself are zero, so it never qualifies. If no G1 qualifies, no final is
   trained (outcome 4).
   Seed rule: seed 0 is the candidate for G1. Seed 1 is a stability check reported beside it and never replaces it; a qualifier at seed 1
   without a seed-0 qualifier is reported, not chosen.
7. **Final (chosen recipe; 63,304 steps; ten evaluations; positional health per evaluation; checkpoint by S among the evaluations
   that pass positional health).** Only if a recipe qualifies. If no evaluation passes positional health, the final is BLOCKED and reported.
   Causal scope: G1 vs G0 tests the package (dosing sampler plus KL). No claim about the KL term alone is made; a G0′ control
   (sampler without KL) was considered and not pre-registered, because of its cost (about 4.5 GPU-hours).
8. **Gate O6 (pre-registered here; its four checks are Gate O4's, unchanged, with the same limits):** the final against zero-shot pplx-flash on the external panel (lower bound > −0.02, as Gate
   O4 check iii), knowledge test lower bound > −0.02 against zero-shot, and MR and MIMIC lower bounds > 0 against zero-shot (as O4 i–ii).
   Against MedDecider-4B and MedDecider-9B, knowledge test point estimates are additional rows.
9. **Teacher and data guards:** the 27B's soft labels are never used on a dev or test set; the knowledge dev v2 items stay out of the
   anchor pool; MIMIC items are never sent to the teacher beyond the local scoring that osler_v4 already did (local only).

## 3. What would count as success, and what would not

- Success: G1 passes the choice rule, the final passes Gate O6 on the panel and knowledge, and the MIMIC audit lead holds.
- Partial: knowledge up, panel still below zero-shot: report it as a partial result. No second try on the panel without a new
  pre-registration: the panel is then used up as a gate.
- Failure: no arm qualifies, or the final fails Gate O6. Report it with the evidence (§5 of EVIDENCE.md).

## 4. Time and resources (planning estimates; not measured)

Scoring 400,000 anchors: about 12–18 GPU-hours (osler_v4's rate was about 6 items/s). Two arms at 13,992 steps: about 9 hours. Final:
about 24 hours. Gate and reporting: about 6 hours. Total about 50–55 GPU-hours. Osler_v5's Z8 was not run (closed 2026-11-07), so nothing precedes this loop's first GPU job except the Z1 scoring resume (one GPU job at a time).

## 5. Out of scope

- The d1-2B model (LiquidAI): a 1.9B vision-language-based decision model with no medical results and a stated limit on
  knowledge-heavy questions. It cannot close a knowledge gap to the 27B and would change the base (D26, D33 forbid it in a loop).
  Not run. If wanted later, it is a separate additional row, labelled NOT MEASURED until someone runs it.
- Osler-9B; base, attention or readout changes; new data sources; Nemotron.

## 6. Addendum (2026-10-29): the base is Clef-Flash, not pplx-4B (D59; DRAFT, awaiting operator decision)

Clef-Flash (`Cloudflare/clef-flash`, Apache-2.0) is a 9B post-train of Qwen3.5-9B with a joint schema head. It is the strongest
≤10B model on our public sets (Z10: panel 0.650, knowledge 0.711), so it is the bar to beat. Two sub-decisions must be made first:

- **(a) Head.** Either our option-code readout and attention-only LoRA on Clef's backbone (the D23/D26 architecture, with the
  base swapped; needs a port check against Clef's `joint_schema_model.py`), or Clef's joint head fine-tuned through its own protocol.
- **(b) Bar.** Pre-register the margin over Clef-Flash on the panel and knowledge test (proposal: lower bound above 0 on both, and point
  estimates at least 2 points above Clef), with the audit suites kept.

Odds, judgement (not measured): the base raises the floor, but every fine-tune so far has lowered general skill against its base.
A 9B on Clef's backbone is more likely to match Clef than to beat it by a comfortable margin. The 9B training cost is about twice the
4B run (osler_v4's 63,304 steps took about 21 hours). Before any 9B final, a 13,992-step arm (about 6 hours) must show that our recipe
keeps Clef's panel and knowledge, with positional health passing.
