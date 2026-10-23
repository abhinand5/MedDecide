# osler_v4 — Stage A (Y6): the recipe for Stage B

Aggregates only. Every number has a CLAIMS row (IDs in brackets). Written 2026-10-23 after the pre-registered choice ran.

## 1. Decision

**Chosen recipe: A0**, the control (osler_v3's A0 checkpoint: no KL to the base, the C1 mix). Stage B's B0 is A0 (ADVISORY Y7: no new run).

Why: A4 (KL 0.1 with osler_v3's broad replay) is the only other arm that qualifies against A0 (both lower bounds above −0.01), but its CG v2 accuracy is 0.0043 below A0's, so A0 is the best qualifying run. The tie rule (0.003) is not needed. A1, A3 and A5 do not qualify.

Two details the choice rests on:

- A4's MR lower bound is −0.0099 against the limit −0.01, a margin of 0.0001 [C118]. It qualifies, but the decision does not depend on it: A0 would win with or without A4.
- Under the batch-composition effect measured in this loop (§5), one changed decision moves a CG accuracy by about one item in 8,307. A4 trails A0 by 0.0043 (about 36 CG v2 items). To be chosen it would have to lead by the 0.003 tie margin, which is a shift of about 61 items. The measured effects are a few items, far below that [C119].

## 2. The runs

| run | recipe | source | steps | fast dev v4 accuracy | CG v2 accuracy | claims |
|---|---|---|---|---|---|---|
| A0 | control | osler_v3 checkpoint, re-scored | 13,992 | 0.7675 | 0.7879 | C106, C107 |
| A1 | KL 0.1 on the C1 replay | osler_v3 checkpoint, re-scored | 13,992 | 0.7645 | 0.7856 | C110, C111 |
| A3 | KL 0 on block_A3 with the broad replay | osler_v3 checkpoint, re-scored | 13,992 | 0.7746 | 0.7843 | C112, C113 |
| A4 | KL 0.1 with block_A3's broad replay | new run | 13,992 | 0.7805 | 0.7836 | C114, C115 |
| A5 | distillation 0.5 on the Y5 teacher distributions, KL 0 | new run | 13,992 | 0.7743 | 0.7791 | C108, C116, C117 |

A2 is not re-scored (ADVISORY Y6). The osler_v3 re-scores reproduce osler_v3's recorded decisions on every shared item except a few near ties and three CG items (§5) [C106, C110, C112].

## 3. Qualification against A0 (paired bootstrap within templates, 1,000 resamples, seed 0)

| arm | MR template-macro difference [95 % CI] | knowledge micro difference [95 % CI] | qualifies | CG v2 accuracy (13 templates) |
|---|---|---|---|---|
| A0 | reference | reference | (reference) | 0.7879 |
| A4 | +0.0003 [−0.0099, +0.0098] | +0.0005 [−0.0082, +0.0092] | yes (lower bounds −0.0099 and −0.0082) | 0.7836 |
| A1 | −0.0002 [−0.0083, +0.0081] | −0.0032 [−0.0104, +0.0032] | no (knowledge) | 0.7856 |
| A3 | +0.0022 [−0.0066, +0.0105] | −0.0027 [−0.0111, +0.0059] | no (knowledge) | 0.7843 |
| A5 | −0.0060 [−0.0148, +0.0031] | −0.0022 [−0.0111, +0.0054] | no (MR and knowledge) | 0.7791 |

The MR partition has 3,197 paired items and the knowledge partition 4,040 [C118]. The point estimates were re-derived in a fresh process from the raw predictions and match [C120].

## 4. The readout gate (D12, D21)

No readout cell fails for any arm, on fast dev v4 (60 templates) or on CG v2 (13 templates) [C107, C111, C113, C115, C117]. The restriction of the operator's answer to question 7 therefore changes nothing: each comparison uses the full sets.

## 5. Checks on the scores

- **Reproduction of osler_v3's records.** A0, A1 and A3 re-scored from osler_v3's checkpoints reproduce osler_v3's decisions on all shared CG v2 items except three in A3 (A1 and A3 also differ on a few dev items) [C106, C110, C112].
- **Batch dependence.** The three A3 CG items whose decisions changed were scored alone and in batches of eight on the GPU. Under bfloat16, one item moves by 0.22 in probability and changes its decision with batch composition, and a second changes its decision (0.029). Under float32 the largest change on the same items is 0.001 and no decision changes [C119]. The scores in this loop are bfloat16 with the batch composition of their scoring run. This is a measured limitation. It is not corrected here, because the choice is pre-registered on the scores as produced. Its consequence for the choice is in §1. Y8 and Y9 should decide the scoring precision for the final evaluations (NEXT.md).
- **Training validity.** Each new run's block result passed its checks: the step count as planned, the evaluation steps as planned, the logged steps checked with no mismatch, and the realised items equal to A3's replay (85,792) [C114, C116].

## 6. The new runs in detail

- **A4** (KL 0.1): in-training accuracy on fast dev v4 rose 0.7401, 0.7684, 0.7768, 0.7803 at steps 3,498, 6,996, 10,494 and 13,992. The full-dev pass of the selected checkpoint gives 0.7805 (template macro 0.8159, Brier 0.3109). The two differ by two items [C114, C115].
- **A5** (distillation): the distillation term is active on 125 of 279 logged batches (mean term 0.28, mean 1.36 teacher rows on those batches). In-training accuracy rose 0.7405, 0.7556, 0.7686, 0.7737 (the full-dev pass gives 0.7743, template macro 0.8067, Brier 0.3142). A5 is below A4 at every evaluation after step 3,498, and its MR and knowledge differences against A0 are negative [C116, C117, C118]. No instability: every loss is finite and the gradient norms match A4's.

## 7. What this means for Stage B

- The recipe for Stage B is the control, A0. The D48 distillation (A5) and the KL 0.1 recipe (A4) are reported as measured: neither beats the control on the choice's CG v2 criterion, and A5 is lower on MR and knowledge.
- The panel-gap question that motivated A4 and A5 (ADVISORY §2 question 3) is not answered by this stage: the external panel is never used for choices, and no panel number is in this document.
- The chosen recipe is 13,992 steps (the osler_v3 recipe), as ADVISORY Y7 requires.

## 8. Decisions recorded for this stage

- The operator's answer to STATE §6 question 7 (answered (b), 2026-10-22) is recorded in STATE §7; the restriction was implemented in `cmd_choose` and checked (C109), and it applied to nothing (§4).
- The A5 mix attaches the Y5 distributions by normalised text (not by item id, which is not unique) (C108).
- The A5 replay is written under outputs/osler_v4/Y6, not under osler_v3's outputs (STATE §7).
- The batch-dependence measurement uses the GPU after the chain; a CPU run is impossible in this environment (the model's forward pass needs a CUDA kernel) (C119).
