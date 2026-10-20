# NEXT — proposals for the operator (osler_v3 closure, 2026-10-20)

Proposals only. Nothing here was run, and each needs the operator's decision.

1. **Separate run length from KL before any new final model.** Every Stage A and B1 run is 13,992 steps; osler_v2's final is
   42,200 (C131). The MR gap of 0.018–0.025 against osler_v2's Osler-4B holds for the control too (C122–C126). A short
   follow-up: the control recipe (KL 0, osler_v2's mix) at 42,200 steps, scored on the same v10 items. This would show whether the
   gap is run length. It is one GPU job of about 12 hours (ADVISORY X9 scale). Needs the operator's approval (it retrains a model).
2. **The keep rule cannot pass a block whose gain lies outside the MR partition** (FINDINGS §2; C111–C116). Propose, in a new loop
   and with a written pre-registration, a rule that judges a block on its own templates plus a no-harm check on MR and general (not
   an MR gain). This changes a settled check, so it needs the operator's decision and a new loop; it is not made here.
3. **External panel: the `medical_question_pairs` and `symptom_to_diagnosis` losses** (FINDINGS §4; C127). Both sets are small (295
   and 212 items), and the losses are on every run. Propose a look at the item formats (the panel's own option wording) before any
   data is added. No data is added for them here (the panel must not influence training, D42).
4. **KL-to-base at 0.1 or broad replay as the panel lever** (FINDINGS §3). The panel gap narrows by about half at KL 0.1 and with
   broad replay, but the intervals include zero. A repeat with more seeds would say whether that is real.
5. **Claims template (`clinclaim_grounding_v1`)** remains BLOCKED after its second prompt fix (C107). A new loop would have to redesign the
   not-mentioned instruction and re-pilot under D44(b).
7. **The padding test fails on the full suite.** `tests/test_padding_trained.py::test_trained_checkpoint_logits_are_padding_invariant`
   fails against the student_v0 checkpoint; this loop changed no code under test, and the test was deselected at the X0 gate. The gate
   for this closure deselected it and `test_overfit_64_items` (840 passed, 1 skipped, 2 deselected; `outputs/osler_v3/X10/pytest_gate.log`).
   The operator should decide whether to fix it or retire it. The test was not edited (R8).
6. **MIMIC** is still absent (X5 BLOCKED; D37). If the official files arrive, a later loop can add the MIMIC block and the memorisation audit (X11).
