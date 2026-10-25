# osler_v4 — next: proposals (not started)

Each proposal needs the operator's decision, or a new loop with its own pre-registration. Nothing here was run, and no proposal
changes a verdict of this loop. Claim ids are in `CLAIMS.md`.

1. **A memorisation test the release can use (operator question 13).** The verbatim probe is uninformative for the final causal
   export: the true-continuation NLL is 7.673 nats per token against 2.725 for the zero-shot control, and both give 0 of 300 matches
   (C235; `SELF_AUDIT.md` §5.1). Proposal: (a) compare the causal model's NLL on training notes with its NLL on test notes, both
   measurable with the existing probe harness (`scripts/osler_v4/y10_audit.py`); (b) decide whether a plain causal export is part of
   the release. If the release is the decision model only, the membership AUC (C233; no measurable signal) is the measure, and the
   verbatim probe leaves the release checklist.

2. **Readout checks before any further verdict uses CG v2 or the MR verdict.** (a) The DDXPlus two-letter collapse passes the D12 gate
   (2 of 499 correct against chance 0.0204; C210, C212; question 10). Proposal: a constant-answer and two-letter-coverage check, as a
   change to the gate in a new loop with its own pre-registration (the gate is frozen in this loop). (b) The held-out cell
   `synthea_indicator_evidence_v1` is READOUT_FAIL for four models (C225). (c) `mr_reported` is 1.0 for the final model by a lexical
   shortcut recorded at build (osler_v1; C225, C228); the set should stay a report, never a check.

3. **The two flagged MR verdict templates (operator question 11).** `pmc_similar_patient_v1` (8,100 of the 13,103 verdict items) and
   `mr_icd_support_template_v2` (554 items) were flagged too good in osler_v1. Their Naive Bayes screens pass (C218, C219), and the
   verdict without them keeps its direction (C222). Proposal: decide whether they count before any further verdict uses them; a
   repair would regenerate them under the D35 screens in a new loop.

4. **Why the longer final scores below osler_v2 on the MR verdict (C228).** The final (63,304 steps, with B1 to B3 in the mix under
   operator ruling 1) has a template macro of 0.8431 against osler_v2's 0.8621 on the same 13,103 items. The difference is not tested
   for significance. Proposal: a controlled comparison in a new loop: the final recipe without B1 to B3 at 63,304 steps, and osler_v2's
   recipe at 42,200 steps, to separate the step budget from the block dilution.

5. **The panel gap (C220, C224).** The pre-registered levers (KL 0.1 with broad replay; distillation from the 27B) did not close it on
   the dev proxies (C118). The largest drops are on symptom_to_diagnosis, medical_question_pairs_noul and medconceptsqa. Proposal: a
   dev-only error analysis of those three dataset types, using dev items from non-panel sources with licence checks. Any training change
   is selected on dev only; the panel is the final gate of the next loop, never a selection set.

6. **Transfer to held-out families (C228).** On the 110 MIMIC held-out items (the heart-failure family) the final scores 0.6384 against
   zero-shot 0.7199, below the baseline. Proposal: a held-out-family analysis with dev-only selection. The held-out lists stay fixed
   unless the operator decides otherwise.

7. **MedDecider's ten option letters (operator question 8; C221).** The panel check against MedDecider-4B is NOT MEASURED because 212
   items have more options than the released prompt takes. Options: (a) a runner that uses more letters, which changes the released
   interface (the operator decides); (b) leave it NOT MEASURED.

8. **Carried-over operator decisions.** Question 4 (dev-proxy contamination in earlier loops), question 5 (the item id scheme), and
   question 12 (the account's disk quota). The pytest temporary directories under `/workspace/tmp` (`pytest-gate8-b2` to `pytest-gate13`
   and `pytest-of-root`) remain, and their removal is the operator's decision.

9. **Pre-registered margins.** A4 qualified on Stage A by 0.0001 on its MR lower bound (C118). A next loop that reuses the choice rule
   should pre-register a margin or a larger run before the choice, so that a tie is not decided by rounding.

Do not, in any next loop: select on the panel, the MIMIC audit suite, the held-out lists or any test split; change a frozen suite, a
gate or a keep rule after a run; or treat a NOT MEASURED or BLOCKED result as a pass.
