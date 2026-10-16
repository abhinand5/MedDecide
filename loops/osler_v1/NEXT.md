# osler_v1 — NEXT (proposals only; nothing here was run or decided)

Each proposal names the evidence it rests on (CLAIMS rows) and what would test it. None changes a settled decision (D24,
D31, the keep rule, the held-out lists or the ICD table). A proposal that needs a new rule must write the rule down before any
run that uses it.

## 1. Osler-9B on the kept mix (the next loop's first task)

- Osler-9B was deferred by the operator (V9-D1). Its gate verdict is NOT MEASURED. The O2 baselines it needs are present:
  MedDecider-9B and JEV-9B (outputs/osler_v0/O2/meddecider-9b, jev-9b; osler_v0 O2 was the protocol for the panel).
- Evidence: the 4B verdict is FAIL on the external panel against MedDecider-4B and on the MR suite's Brier (V165). Whether the
  9B passes is unknown.

## 2. Real text in the kept block (the central question for the next data loop)

- The kept block B1 holds no real clinical text (CLAIMS V071). Against MedDecider-4B the real-text part loses (−0.0742
  accuracy, +0.0934 Brier; V184), and the two real-text ICD templates carry the loss (CodiEsp −0.0992 and +0.4738; mkurman
  −0.1624 and +0.3972; V185).
- B3 added real text and raised the dev MR macro by +0.0539 over B1 (lower bound +0.0382; V148), but failed two margins of the
  pre-registered keep rule: general (lower bound −0.0141 against −0.01) and knowledge (−0.0400, lower bound −0.0860 against
  −0.02). The MR gain and the knowledge loss arrived together.
- Proposal: a real-text block designed to hold the knowledge and general checks (a larger replay share, as below), written with
  its keep rule before it is built, and compared with B1 on dev only. The test sets stay untouched until the rule decides.
- Test: does the real-text ICD loss close on dev? The keep rule decides, not the test set.

## 3. Replay share and the knowledge drop (a hypothesis, untested)

- The replay share fell from B0 (15.31 %; CLAIMS V070) to B1 (14.38 %; V071), and to B2 (13.63 %) and B3 (12.64 %) (the
  blocks row of STATE §2). The knowledge checks fell with it: B2 −0.0220 and B3 −0.0400 against B1 (V140, V148).
- Hypothesis (untested): the falling replay share dilutes the human-labelled replay that carries knowledge. The ADVISORY
  target of 20 % was not reached in any block (the operator accepted 15.3 % at Q2).
- Test: a pre-registered block at a replay share of 20 % or more, same recipe, same keep rule.

## 4. The too-good ICD template (mr_icd_support_template_v2)

- On V10 it is 1.0000 on 554 test items (V180, V183). Its Naive Bayes screen is 0.804 (V023), so the bag-of-words shortcut does
  not explain it. The string-presence screen at build was 0.0256 (V023).
- Proposal: build a rule-level probe that answers the template from the page's codes with no model, and compare its accuracy
  with the model's. If the probe matches the model, the template measures the rule and not reasoning; report it that way.
- Proposal: run the overlap check on the V10 test items against each training block (the V9 and V8 checks covered dev only; V183).

## 5. pmc_similar_patient_v1 (defect candidate; ADVISORY §8.2)

- It is 8,100 of 19,294 MR test items (41.98 %; V177). B1 has no training rows for it, and the model scores 0.8525 on its dev
  items (V163). B3, which trained on it, scored 0.99 on dev, and a word-overlap score alone reaches AUC 0.9734 (V151). So the
  template can be largely solved by vocabulary.
- The MR suite without it gives Brier −0.0020 against zero-shot (V177): the MR Brier advantage over zero-shot rests on this
  template.
- Proposal: run the Naive Bayes O3 screen on the template, as ADVISORY §8.2 asks (not run in osler_v1). If it fails, redesign the
  generator (V8-D9), and do not change the MR suite in place.

## 6. Calibration on the MR test (hypothesis, untested)

- The temperatures are fitted on the fast dev set only (V160). The Brier loss against MedDecider-4B is largest where the model
  is least accurate against it: the non-pmc items (+0.1597; V177) and the real-text ICD templates (V185).
- Hypothesis (untested): the dev temperatures do not transfer to the MR test distribution.
- Test: fit a calibration on a dev subset that matches the test composition (dev only; the test sets are never used to fit), and
  report Brier on dev before any test run.

## 7. The in-run and post-run dev gap (a measurement question, untested)

- The final checkpoint scores 5,193 correct inside training and 5,199 when scored after the run (V159). B3 showed one item (V147).
- Hypotheses (untested): batch composition and padding; bfloat16 checkpoint rounding against the in-memory weights; evaluation
  batch order (SELF_AUDIT of V9, §5).
- Test: one GPU job that scores the saved checkpoint at two batch sizes and in float32 on dev.

## 8. Speed and budget

- The V9 final run took 41,894 s (11.6 h) in total, with 41,386 s of training for 43,136 steps and 250,024 realised items (CLAIMS
  V156). The scoring of 30,621 items took 1,537 s (V171).
- The V0 measurement recorded 4.1 items a second for the batching recipe (STATE §2). The final run's training averaged about
  6.0 items a second, but the two used different item mixes, so this is not a controlled comparison and no speed-up is claimed.
- Proposal: measure the final batching against the V0 recipe on the same block, before the next long run.

## 9. Not available and not measured

- MIMIC (V5c): the files are absent; no MIMIC part, and no V11 memorisation audit (V11-D1). A next loop may run V5c if the files
  appear; it must not wait for them.
- MDACE: not on the volume; NOT MEASURED.
- osler_v0 Osler-9B and the osler_v0 MR suite: no per-item predictions on the V10 MR items (V179); NOT MEASURED.
- The teacher: V4 closed at the OpenCode Go weekly guard (V4-D5, 2026-10-15T07:56:54Z); the DeepSeek continuation was not run
  (operator Q10). The ICD teacher prose was not built (BLOCKED: pilot pass rate 59.8 %).

## 10. Questions for the operator

This loop asks the operator no new question. The operator's answers and this loop's deviations are in `STATE.md` (sections 6
and 7).
