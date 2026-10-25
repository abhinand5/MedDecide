# osler_v4 — findings (closure)

Loop 1g: MIMIC at scale, Nemotron-crafted minimal pairs, the panel lever, a longer final. Started 2026-10-21T18:26:40Z; closed with
Y11 (the hard stop) on 2026-10-25. Branch `loop/osler_v4`. Every number below has its CLAIMS row (`loops/osler_v4/CLAIMS.md`).
Counts and figures only; no item, note or prediction text is in this file.

## Summary

1. **Gate O4 (D49) fails against zero-shot pplx-decider-flash-v1.2, and is not measured against MedDecider-4B.** The final
   Osler-4B (the Y8 checkpoint at 63,304 steps, scored calibrated) fails the external-panel check against zero-shot pplx: accuracy
   −0.0421 [−0.0584, −0.0269], against the pre-registered limit of −2.0 points on the lower bound (C220). Its MR verdict
   (+0.1387 [+0.1243, +0.1530]), its MIMIC audit suite (+0.1654 [+0.1502, +0.1795]) and its knowledge guard
   (−0.0067 [−0.0172, +0.0048]) pass against that baseline (C220). Against MedDecider-4B the panel check is NOT MEASURED: 212 of
   4,353 panel items (4.87 %) have more than ten options, above the 1 % pairing limit (C221). The other three checks pass against
   MedDecider-4B (MR verdict +0.2449; MIMIC suite +0.1651 on eight templates; knowledge +0.0060; C221).

2. **The panel lever did not close the gap.** Stage A's pre-registered choice is the control, A0. KL 0.1 with broad replay (A4)
   qualifies by 0.0001 on its MR lower bound and trails A0 on CG v2 (0.7836 against 0.7879). Distillation from pplx's 27B
   (A5, D48 yes) is below A4 on both partitions and on CG v2 (C106, C115, C117, C118). The panel is reported per dataset as an
   additional table, and no choice used it (C224).

3. **No Stage B block is kept under keep rule v2.** B1, B2 and B3 fail the MR gate; B4 fails the CG v2 gate (C129, C139, C152,
   C162). Under operator ruling item 1, B1, B2 and B3 enter the final mix (their own-template lower bounds +0.1581, +0.0475 and
   +0.1625 against +0.02; knowledge lower bounds −0.0134, −0.0163 and −0.0082 against −0.02). B4 does not enter (`blocks.md`).

4. **The longer final scores below osler_v2 on the MR verdict, and its CG v2 accuracy is flat.** On the 13,103 MR verdict items
   the template macro is 0.8431 for the final and 0.8621 for osler_v2 (scoreboard, additional; C228). Over the last four
   evaluations CG v2 accuracy stays in a 0.73–0.74 band (0.7407, 0.7314, 0.7324, 0.7393; C173–C200), while the fast dev v4
   template macro is 0.8454, 0.8458, 0.8447 and 0.8504 (selected at 63,304; C200). The CG trend is flat rather than falling.

5. **The final is the best model of 10B or fewer parameters on the MIMIC audit suite and MedNLI, but not on every scope.**
   Additional scoreboard, labelled, no verdict (C228): MIMIC suite 0.7995 (zero-shot 0.6341; MedDecider-9B 0.6278); MedNLI test
   0.8762 (zero-shot 0.8720). It is second to osler_v2 on the MR verdict, fifth of six on the panel (0.5968 against 0.6390 for
   zero-shot), and third on knowledge (0.7024 against 0.7265 for MedDecider-9B and 0.7097 for zero-shot). The mission's "best open
   decision model of its size on every scope" does not hold (ADVISORY §2, question 5).

6. **No measurable membership signal; the verbatim probe cannot test memorisation for the final export.** The membership AUC is
   0.504202 [0.485310, 0.522114] for the final model and 0.508969 [0.492595, 0.527421] for the zero-shot control (C233); both
   intervals contain 0.5. The verbatim probe gives 0 of 300 matches for both models (C235). But the causal export of the final model
   generates poorly on these notes: the true-continuation NLL is 7.673 nats per token, against 2.725 for the control (C235). The
   zero count therefore says little about memorisation (C236; operator question 13).

7. **Defects carried forward; none changes a verdict.** (a) The MR `mr_reported` set is at 1.0 for the final model because of a
   lexical shortcut recorded at build in osler_v1; it is reported, not a PASS check (C225, C228). (b) The held-out cell
   `synthea_indicator_evidence_v1` fails the D12 readout gate for the final model and three baselines (C225). (c) The DDXPlus cell
   passes the D12 gate although its two-letter codes collapse: 2 of 499 two-letter gold items are correct, against a chance rate
   of 0.0204 (C210, C212; operator question 10). (d) Two MR verdict templates carry open too-good flags from osler_v1; the verdict
   with them keeps its direction without them (C218, C219, C222; question 11).

8. **Spot-check.** Five claims were re-run in fresh processes from their recorded commands; all five reproduce their recorded
   figures (C118, C213, C229, C230, C237; §8).

## 1. ADVISORY §2 outcomes

| outcome | applies? | why |
|---|---|---|
| (1) Gate O4 passes: the recipe goes to the 9B next | **No** | FAIL against zero-shot pplx-flash on the external panel (C220); NOT MEASURED against MedDecider-4B (C221). The 9B is not run (out of scope). |
| (2) MIMIC helps MIMIC but not the MR suite v1 | **No** | MIMIC helps MIMIC (+0.1654 against zero-shot; C220) and the MR verdict also improves (+0.1387; C220). The MR gain is measured on templates the final was trained on (osler_v2 C1 is its base); the flagged templates are reported, and the verdict without them keeps its direction (C222). |
| (3) No Stage A arm beats the control on CG v2 | **Yes** | A0 0.7879 is the highest CG v2 accuracy; A1 0.7856, A3 0.7843, A4 0.7836 and A5 0.7791 are below it (C118). The panel is reported per dataset (C224) and no choice used it. |
| (4) A block is not kept | **Yes** | B1 to B4 are NOT KEPT under keep rule v2 (C129, C139, C152, C162). B1 to B3 enter the final mix under operator ruling item 1, and their dev numbers are reported as they are. |
| (5) The final run's CG falls as MR rises | **Partly** | The CG v2 accuracy is flat in a 0.73–0.74 band at the last four evaluations, while the fast dev v4 template macro rises at 63,304 (C173–C200). The trend is reported; the selection rule stands. |

## 2. Gate O4 in detail (D49; the pre-registration of STATE §7, 2026-10-24T06:03:11Z)

Verdicts against the two gate baselines (`gate_o4.md`; C220, C221). Accuracy is the template macro of the paired differences;
intervals are paired bootstrap within templates, 1,000 resamples, seed 0.

| baseline | Gate O4 | (i) MR verdict | (ii) MIMIC audit suite | (iii) external panel | (iv) knowledge |
|---|---|---|---|---|---|
| zero-shot pplx-decider-flash-v1.2 | **FAIL** | PASS: +0.1387 [+0.1243, +0.1530]; Brier −0.2178 | PASS: +0.1654 [+0.1502, +0.1795]; Brier −0.1805 | **FAIL**: −0.0421 [−0.0584, −0.0269]; Brier −0.0215 | PASS: −0.0067 [−0.0172, +0.0048] |
| MedDecider-4B | **NOT MEASURED** | PASS: +0.2449 [+0.2328, +0.2568]; Brier −0.2981 | PASS: +0.1651 [+0.1490, +0.1804] (eight templates; question 6) | NOT MEASURED: 212 of 4,353 unpaired (4.87 %) | PASS: +0.0060 [−0.0071, +0.0194] |

The point estimates were re-derived in a fresh process from the raw predictions with 0 mismatches (C230). Per dataset (additional,
C224), the panel drops against zero-shot are largest on symptom_to_diagnosis (−0.0991 [−0.1604, −0.0377], 212 items),
medical_question_pairs_noul (−0.0847 [−0.1288, −0.0406], 295) and medconceptsqa (−0.0550 [−0.0850, −0.0230], 1,000); medexpqa_en is
+0.0080 (125 items). The MR verdict without the two flagged templates is +0.1257 [+0.1066, +0.1459] against zero-shot and +0.2545
[+0.2394, +0.2688] against MedDecider-4B (additional, post hoc, labelled; C222).

Readout health (D12; additional, C225): `mr_heldout_test` is READOUT_FAIL for the final model, osler_v2, MedDecider-9B and JEV-9B
(the `synthea_indicator_evidence_v1` cell, 514 items, below chance); `mimic_suite` is READOUT_FAIL for MedDecider-4B on one template,
which is why its comparison uses eight templates. The zero-shot pplx-flash cells all pass.

## 3. Stage A (Y6): the recipe

The control A0 is the pre-registered choice (C118). Re-scored on fast dev v4: A0 0.7675, A4 0.7805, A5 0.7743 (C106, C115, C117).
On CG v2, A0 is 0.7879, A1 0.7856, A3 0.7843, A4 0.7836 and A5 0.7791 (C118). A4 qualifies: its MR lower bound is −0.0099 against
the −0.01 limit (margin 0.0001) and its knowledge lower bound is −0.0082. A1 and A3 fail their knowledge lower bounds (−0.0104 and
−0.0111), and A5 fails both MR and knowledge (C118). No readout cell fails for any arm (C107, C111, C113, C115, C117). The
bfloat16 batch dependence is measured: one probability moves by 0.22 with batch composition (C119); it is a limitation of every
figure in this loop, and no choice depends on it.

## 4. Stage B (Y7): the blocks

Keep rule v2 (D47) against the last kept block (C129, C139, C152, C162):

| block | adds | MR partition | general | CG v2 | own (new templates) | knowledge | keep rule v2 | final mix (ruling 1) |
|---|---|---|---|---|---|---|---|---|
| B1 | Y2 MIMIC templates | fails | fails | fails | +0.1581 (against +0.02) | −0.0134 (against −0.02) | NOT KEPT | enters |
| B2 | Y3 minimal pairs | fails | — | fails | +0.0475 | −0.0163 | NOT KEPT | enters |
| B3 | Y2b late sources | fails | — | fails | +0.1625 | −0.0082 | NOT KEPT | enters |
| B4 | osler_v3 X1 real sources | passes | — | fails | — | — | NOT KEPT | does not enter |

A dash means no statistic of that kind is reported in this table for that block; the full statistics are in `blocks.md`. B3's readout cell
`ragtruth_sentence_supported_v1` is a constant answer on 90.5 % of its items; the base shows the same bias, and the cell is excluded
from B3's comparisons under D12 (C150, C151). The realised MIMIC shares of the blocks are 0.3135, 0.0242, 0.0866 and 0.0000 for B1 to B4
(C171), each within the 35 % cap.

## 5. The final run (Y8): selection and the CG and MIMIC trends

- 63,304 steps (the budget); selected step 63,304 with fast dev v4 template macro 0.8504 and accuracy 0.8272 (C200), above 50,643
  (0.8458). The end record agrees (C202). Temperatures are fitted and saved with the checkpoint (C203); the calibrated re-scores keep
  every argmax (C206, C207).
- The last four of the ten evaluations, recorded for the report and not for selection (C173–C200):

| step | fast dev v4 macro | CG v2 accuracy | MIMIC partition accuracy |
|---|---|---|---|
| 44,313 | 0.8454 | 0.7407 | 0.7938 |
| 50,643 | 0.8458 | 0.7314 | 0.8013 |
| 56,974 | 0.8447 | 0.7324 | 0.7965 |
| 63,304 (selected) | 0.8504 | 0.7393 | 0.8087 |

- The CG v2 accuracy of the selected checkpoint (0.7393) is below A0's 0.7879 (C212). DDXPlus is most of the gap: its two-letter option
  codes collapse to a near-constant `A` (35 % of the cell's items); 2 of 499 two-letter gold items are correct against a chance rate
  of 0.0204 (C210, C212, C214). The D12 gate passes that cell; the question of flagging it is operator question 10.
- The MIMIC partition rises at each evaluation except at 56,974, where it dips and recovers at 63,304 (C196, C200).
- Step-log health: 1,266 logged rows, none missing, duplicated or non-finite; the largest per-step CE is 14.5872 at step 22,450, on a
  one-item batch (C213, C214).

## 6. Size-class scoreboard (additional; no verdict; C228)

Template macro accuracy on each scored set. Rows are the final Osler-4B, osler_v2, and the baselines of 10B or fewer parameters,
with pplx-decider-v1.2-27b as the ceiling. A figure marked READOUT_FAIL is not a result (D12). Sets differ in item counts where a
baseline cannot take every option count (MedDecider and JEV-9B on the panel).

| model | MR verdict (13,103) | MIMIC suite (4,253) | panel | knowledge | MedNLI test (1,422) | MIMIC held-out (110) |
|---|---|---|---|---|---|---|
| Osler-4B final (Y8) | 0.8431 | **0.7995** | 0.5968 | 0.7024 | **0.8762** | 0.6384 |
| osler_v2 Osler-4B | **0.8621** | 0.5651 | 0.6041 | 0.7005 | 0.8692 | 0.6704 |
| zero-shot pplx-decider-flash-v1.2 | 0.7044 | 0.6341 | **0.6390** | 0.7097 | 0.8720 | 0.7199 |
| MedDecider-4B | 0.6051 | 0.6115 (READOUT_FAIL) | 0.6159 (4,141) | 0.6969 | 0.8488 | 0.7176 |
| MedDecider-9B | 0.6223 | 0.6278 | 0.6301 (4,141) | **0.7265** | 0.8425 | 0.6713 |
| JEV-9B | 0.6583 | 0.5992 | 0.5840 (4,141) | 0.6692 | 0.8221 | **0.7343** |
| pplx-decider-v1.2-27b (ceiling, additional) | 0.7745 | 0.6819 | 0.7203 | 0.8173 | 0.8748 | 0.7718 |

Two further rows, with the failing cells named: the MR held-out test (584 items) is READOUT_FAIL for the final model (the set figure
0.7679 is not a result), and the MR reported-only set (6,191 items) is 1.0000 for the final model by the documented lexical shortcut
(C225, C228). MedDecider-4B and -9B score none of the reported-only set (more than ten options).

Reading: among the models of 10B or fewer parameters, the final is first on the MIMIC suite and MedNLI, second on the MR verdict and
third on knowledge, and fifth of six on the panel. The 27B ceiling is above the final on the panel, on knowledge and on the MIMIC
held-out items, and below it on the MR verdict, the MIMIC suite and MedNLI. On the MIMIC held-out items (110), the final is 0.6384
against zero-shot 0.7199: the final does not transfer to the held-out heart-failure family, and this is a 110-item figure.

## 7. Membership and verbatim audit (Y10; `audit.md`; a report, not a gate)

- Membership (C233): 2,000 members (training rows of the final mix, allocated per template) against 2,000 non-members (the frozen MIMIC
  audit suite v1, the same allocation); the loss is the negative log of the gold option's probability. The final model's AUC is
  0.504202 [0.485310, 0.522114], and the zero-shot control's is 0.508969 [0.492595, 0.527421]. Both intervals contain 0.5: no measurable
  membership signal. The mean losses are 0.605 (members) and 0.611 (non-members) for the final model.
- Verbatim (C235): 300 train-split discharge notes, 128-token prefixes, 64 greedy tokens; a match needs 50 consecutive aligned tokens.
  0 of 300 for the final causal export and 0 of 300 for the control. The truth NLL (an additional sanity figure) is 7.673 nats per
  token for the final causal export and 2.725 for the control. The control's value is in the range of a working language model on
  clinical text, so the final export's higher value reflects the export, not the probe.
- Causal export against the decision path (C236; additional): on 20 membership items (60 options), the argmax agrees on 16 and the largest
  probability difference is 0.768058. The decision path uses non-causal full attention (pplx's mask hook), which a plain causal LM does
  not reproduce.
- Reading: the membership result is the measured memorisation result for this loss. The verbatim result is uninformative for the final
  model, because its causal generation is poor (`SELF_AUDIT.md` §5.1; operator question 13). The release decision is the operator's.

## 8. Five-claim spot-check (re-run outputs, fresh processes)

Each row was re-run from its recorded recompute command in a new process. Figures are copied from the outputs.

| claim | recompute command (as recorded) | re-run output (figures) | recorded value | reproduced |
|---|---|---|---|---|
| C230 (Gate O4 point estimates) | `CUDA_VISIBLE_DEVICES= uv run --frozen python scripts/osler_v4/y9_rederive.py` | 5 checks, 0 mismatches; MR vs pplx +0.138710; panel vs pplx −0.042145; knowledge vs pplx −0.006691; MR vs MedDecider-4B +0.244914; knowledge vs MedDecider-4B +0.006048 | the same five figures | yes |
| C229 (pplx parity, CPU step) | `CUDA_VISIBLE_DEVICES= uv run --frozen python scripts/osler_v4/y9_parity.py` (the GPU step that produced the 200-item run is not repeated) | 200 items shared; argmax agrees on 198; correctness on 198; mean absolute probability difference 0.00529531; maximum 0.04502264 | 198 of 200; 0.00530; 0.04502 | yes |
| C237 (Y10 re-derivation) | `CUDA_VISIBLE_DEVICES= uv run --frozen python scripts/osler_v4/y10_rederive.py` | 2 AUC checks match; 0 gold-position disagreements; 2 verbatim recounts match; 0 mismatches | 0 mismatches | yes |
| C213 (Y8 step-log statistics) | `CUDA_VISIBLE_DEVICES= uv run --frozen python scripts/osler_v4/y8_steplog_stats.py` | logged rows 1,266 (first 50, last 63,300); missing 0; duplicates 0; non-finite 0; largest CE 14.5872 at step 22,450 (one item) | the same counts and CE | yes |
| C118 (Stage A choice) | the recorded `uv run python -c` command on `outputs/osler_v4/Y6/choice.json` | chosen A0; qualifying [A0, A4]; CG v2 A0 0.7879, A1 0.7856, A3 0.7843, A4 0.7836, A5 0.7791; A4 MR lower bound −0.0099, knowledge lower bound −0.0082 | the same | yes |

The re-runs rewrote their output files with the same figures (`rederive.json`, `parity.json`, `steplog_stats.json`, and the Y10
`rederive.json`); the recompute commands are the recorded ones.

## 9. Defects, deviations and open questions

- Defects and checks (§6 of STATE): the `mr_reported` lexical shortcut; the `synthea_indicator_evidence_v1` READOUT_FAIL; the DDXPlus
  two-letter collapse (question 10); the two flagged MR templates (question 11); the Y10 verbatim limitation (question 13).
- Earlier operator questions still open: question 4 (dev-proxy contamination in earlier loops), question 5 (the item id scheme), question 8
  (MedDecider's ten letters leave the panel NOT MEASURED) and question 12 (the account's disk quota; the gate passed once the operator
  freed space, C232).
- Deviations (STATE §7): the Y9 chain's exit records and torchvision flags (2026-10-25T01:44Z); the Y9 gate as two processes (11:28Z);
  the Y10 chain corrections and the probe's causal construction (12:30Z). Each is recorded with its reason; no check, gate or threshold was
  changed.

## 10. Done-when checks (GOAL.md)

- Every task is DONE or BLOCKED; no row is IN_PROGRESS (checked before Y11).
- The artifacts exist: `mimic.md`, `crafted.md`, `stage_a.md`, `blocks.md`, `gate_o4.md`, `results.md`, `audit.md`, and this file.
- Every number in this file has a CLAIMS row (C106 to C238 as cited).
- `NEXT.md` lists the proposals.
