# osler_v0 — next steps (proposals for the operator)

Written at closure (O12). These are proposals, not decisions: the operator decides which apply, following ADVISORY §2. The numbers below are the measured ones, with their CLAIMS ids.

1. **Knowledge fix first (ADVISORY §2 outcome 3).** Osler-4B fails the knowledge guard: −0.0156 [−0.0291, −0.0018] against a floor of −0.02 [O304]. The ADVISORY names three levers: replay ratio, LoRA scope and early stopping. A next loop should test them with the knowledge guard as its first gate, before any scaling. The pointer arm fails the guard more strongly (−0.0220 [−0.0350, −0.0077]) and the non-causal arm passes it (+0.0051 [−0.0084, +0.0180]) [O314]. The non-causal arm's tier-1 retention is worth a closer look in that loop.

2. **Data breadth for the external panel (outcome 2).** Both Osler models lose the external panel to the best competitor of their size: the 4B to MedDecider-4B (−0.0526 [O304]) and the 9B to MedDecider-9B (−0.0169 [O305]). The next loop's data work is breadth — more clinical generators and more human-labelled sources — with the licence, leakage and held-out rules kept. The leakage check (O307) is the pattern to repeat for each new source.

3. **Save the calibration with the checkpoint.** The arm pipeline fits the per-question-type temperatures after the best checkpoint is saved, so every saved checkpoint is uncalibrated [O293]. The next loop's training script should write the fit into the checkpoint and verify it on reload. The fit itself reproduces the recorded NLL [O294].

4. **Held-out template transfer as its own gate.** The constant-answer behaviour sits on two held-out templates for every model [O324] and on an unseen template for the 0.8B [O317]. A next loop should report transfer to held-out templates separately and treat a collapse to one answer as a defect, not as a score.

5. **The pointer head's confidence.** The pointer arm's probabilities saturate: 7,256 uncalibrated choice items are exactly one-hot [O323], and 227 calibrated choice items are above 1 − 1e-9 [O310]. A calibration step or a confidence penalty should be tested before the pointer readout is considered again.

6. **The non-causal arm's serving cost (ADVISORY §2 outcome 5).** The non-causal arm is ahead of the option-code arm on the common set by +0.0050 [+0.0011, +0.0089] and on the panel by +0.0152 [+0.0006, +0.0305] [O313]. The gain is small, and the head was fixed on dev before it appeared [O241]. Deciding whether the custom attention mask (no stock GGUF path) is worth it is the operator's decision.

7. **Time budget.** The O11 stage took about twelve hours of wall clock against six estimated (deviation 56). The calibrated GPU scoring took 5,951 s for the 4B option-code arm [O298], 5,949 s for the pointer arm [O308], 5,972 s for the non-causal arm [O311], 8,469 s for the 9B [O301] and 1,771 s for the 0.8B [O315]. A next loop should budget about 1.7 hours per 4B model and 2.4 hours per 9B on the full common set, and schedule the additional rows as separate work.

8. **Baselines on the held-out generator items** (deviation 58). Wiring the four D24 baselines to the held-out items would let the held-out verdict be computed. Until then the held-out comparisons are NOT MEASURED [O319].

9. **The quantised export (not measured).** The merged-adapter export failed the 1e-3 check at O5 (CLAIMS O094, O105). The ≤ 4 GB export should wait until Question 14 (the precision of the O5 verdict) is settled.

10. **The dev–test gap on `nfcorpus_relevant_noul_v1`** (CLAIMS O317): the 0.8B reaches 0.948 on dev and 0.600 on test. Audit the template's dev and test items for the shortcut the student loop found before using the template in any gate.
