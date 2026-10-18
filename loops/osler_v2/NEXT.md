# osler_v2 — NEXT (proposals only; nothing here was run or decided)

Each proposal names the evidence it rests on (CLAIMS rows) and what would test it. None changes a settled decision (D24, D31,
D35 as frozen, the keep rule, the held-out lists or the ICD table). A proposal that needs a new rule must write the rule down before
any run that uses it.

## 1. Close the too-good ICD flag (`mr_icd_support_template_v2` at 1.0000; the first thing to investigate)

- Evidence: 1.0000 on 554 test items and on 400 dev items of the final model (W061, W063); the rule probe reaches 0.612 (W015); zero
  exact or near-duplicate overlap with B1 (W016); NB 0.804 (W012, below the D35 rule).
- Hypothesis (untested): the model learned the generator's family-and-certainty logic in distribution.
- Test: a probe that varies one generator property at a time (distractor families, certainty words, the question wording) on dev only,
  then the test set once. Report which property moves the accuracy. Do not change the template in place.

## 2. The panel gap to zero-shot pplx (the outcome-2 lever: broader clinical data)

- Evidence: panel vs zero-shot pplx −0.0198, lower −0.0344 (W056); vs MedDecider-4B the lower bound is −0.0199 (W055). The real-text
  block helped the MR verdict (W048–W049) without closing the panel gap.
- Proposal: a dev-only broad clinical block (for example, more real clinical QA from permissive sources, licences recorded) in a new loop,
  with its keep rule written before the block is built. The panel stays test-only.
- Test: the panel gap on the same test set after the new block, with the same margins (no change to the margins).

## 3. The PMC template and the D35 vocabulary rule

- Evidence: PMC is 8,100 of 13,103 verdict items (W013); its NB screen is 0.520, so D35 keeps it (W012), but osler_v1 measured a
  word-overlap AUC of 0.9734 (osler_v1 V151, not re-measured here). PMC accuracy is 0.9791 for Osler-4B (W061).
- Proposal: decide, before any new result, whether the vocabulary rule should use a threshold-free overlap score as well as NB accuracy.
  Write the rule into a new D-entry first. Report PMC both ways until it is decided.
- Test: the overlap-score AUC on dev for the test-matched templates, then the MR verdict with and without PMC under the decided rule.

## 4. The chart templates (W3b): the label cue is the certainty word

- Evidence: "ruled out" in 0.971 (pilot 1) and 0.667 (pilot 2) of gold-none charts and 0.000 of gold-code charts (W026–W027). One prompt
  fix moved the cue into synonyms (W027). The pilot keep rate is 85–90 % (W024–W025).
- Proposal: change the design, not the wording: every chart states one other problem with a certainty, in both gold classes, so that no
  certainty word tracks the gold label. Write the new design and its acceptance test before the pilot.
- Test: the same phrase check on a fresh pilot, and the chart NB screens on dev, before any chart is used in a block.

## 5. Knowledge guard against pplx: the 21 repeated-label items

- Evidence: pplx's reference criteria are keyed by label; 21 test items (12 dev) repeat a label (W057, W040). The guard is NOT MEASURED on
  the full set (W059).
- Proposal: a pre-registered label-disambiguation rule for the reference path (for example, a letter suffix in the criteria labels, applied
  identically to every item), decided before the next gate. Keep the current NOT MEASURED result as the record of this gate.
- Test: the guard on the full set under the rule, and the rule's effect on the 5,435 items already scored (no change to them).

## 6. Missing baseline items

- Evidence: MedDecider-4B has 111 MR-verdict items and 212 panel items with no prediction in the V6/O2 files (W053, W055). The reported-only
  MR templates have no MedDecider-4B pairs at all (gate JSON).
- Proposal: find out why (readout gate, skipped rows, or a file difference) before the next gate, and report the coverage as a baseline
  property. Do not re-score the baseline to fill the gap without a written rule.

## 7. Teacher prose

- Evidence: C2 (prose) is not kept; MR −0.0042 (lower −0.0121), general lower −0.0197 (W051). osler_v1's B2 showed the same pattern.
- Proposal: prose only with a verified sentence-level design, or no prose; the ICD prose v2 (ADVISORY W3) was not run here.

## 8. The 9B and the MIMIC work (both deferred or blocked)

- Osler-9B: deferred by the operator (D39). When it is taken up, its gate needs MedDecider-9B and JEV-9B (both measured in osler_v0/osler_v1)
  and the pplx 9B base if chosen; none was run here.
- MIMIC: the official files are absent (W005). When they arrive, C4 (a MIMIC block) and the memorisation audit (W9) follow the same rules.
  Nothing here waited for them.

## 9. Replay guard for a larger block

- Evidence: the replay share is 22.65 % (C0), 21.04 % (C1), 20.13 % (C2) (W035); a larger block needs more replay than the 8 % cap allows
  from the listed sources (W017; the cap binds on SNLI).
- Proposal: a second permissive source with its licence and leakage check, written into the replay rule before any block uses it.
