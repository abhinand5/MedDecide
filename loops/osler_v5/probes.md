# osler_v5 — Z1 probes on the osler_v4 final (additional; choose nothing)

Dev only. The checkpoint is osler_v4's final (Y8, 63,304 steps, `outputs/osler_v4/Y8/checkpoints/best`), scored in bf16 on the GPU by
`scripts/osler_v5/z1_probes.py` (job started 2026-10-26T19:59Z, finished 2026-10-26T22:54Z). The figures are recomputed from the saved
per-item rows by `scripts/osler_v5/z1_summarise.py` (`outputs/osler_v5/Z1/summary.json`); this file is written by
`scripts/osler_v5/z1_report.py`. Nothing here chooses an arm, a checkpoint or an alpha (ADVISORY Z1). No test set, panel, MIMIC
audit suite or held-out file was scored.

## 1. Consistency checks

- alpha 0 against the reference path (`MedDecideModel.reference_option_probs`, 16 CG items, bf16 on the GPU): max abs diff 1.19e-07.
- alpha 1 against osler_v4's Y8 predictions on the same items (the trained model, uncalibrated): fast dev v4 17,532 of 17,532 items with the same argmax (maximum absolute probability difference 0.0000); CG v2 8,307 of 8,307 (maximum 0.0000). Logged in `outputs/osler_v5/Z1/logs/consistency.log`.
- alpha 1 composite S 0.8386 against Z0's composite of the same predictions 0.8386 (C026).
- alpha 0 DDXPlus two-letter accuracy 0.6112 against zero-shot pplx-flash's 0.6112 in osler_v4 (C210).

## 2. Blend curve (alpha 0 is the base model through the reference path; all figures dev)

| alpha | S | MR macro | MIMIC macro | CG v2 micro | knowledge dev | knowledge minus alpha 0 [95 % CI] | DDX two-letter | DDX single | positional (orig / seed 1 / seed 2) |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.7820 | 0.8863 | 0.6718 | 0.7879 | 0.7725 | reference | 0.6112 | 0.7984 | pass / pass / pass |
| 0.3 | 0.8252 | 0.9215 | 0.7694 | 0.7849 | 0.7728 | +0.0003 [-0.0047, +0.0057] | 0.5150 | 0.7226 | pass / pass / pass |
| 0.5 | 0.8395 | 0.9475 | 0.7922 | 0.7787 | 0.7708 | -0.0018 [-0.0080, +0.0050] | 0.4148 | 0.7265 | pass / pass / pass |
| 0.7 | 0.8415 | 0.9613 | 0.8019 | 0.7613 | 0.7698 | -0.0027 [-0.0098, +0.0048] | 0.1844 | 0.7146 | FAIL / FAIL / FAIL |
| 0.85 | 0.8414 | 0.9681 | 0.8065 | 0.7497 | 0.7648 | -0.0077 [-0.0163, +0.0005] | 0.0762 | 0.6826 | FAIL / FAIL / FAIL |
| 1 | 0.8386 | 0.9672 | 0.8094 | 0.7393 | 0.7585 | -0.0140 [-0.0228, -0.0055] | 0.0040 | 0.6407 | FAIL / FAIL / FAIL |

MR and MIMIC are template macro accuracies, CG v2 is item micro, and S is their mean (D54). The knowledge difference is the paired bootstrap of item micro accuracy within templates (medqa, medmcqa), 1,000 resamples, seed 0, as in the keep rule.

## 3. Readout split (which part of the readout and the adapter carries the two-letter loss)

| condition | S | MR macro | MIMIC macro | CG v2 micro | knowledge dev | DDX two-letter | DDX single | positional (orig / seed 1 / seed 2) |
|---|---|---|---|---|---|---|---|---|
| alpha 0 (base adapter, base readout) | 0.7820 | 0.8863 | 0.6718 | 0.7879 | 0.7725 | 0.6112 | 0.7984 | pass / pass / pass |
| (a) trained adapter, base readout | 0.8439 | 0.9661 | 0.8105 | 0.7551 | 0.7560 | 0.1924 | 0.7445 | FAIL / FAIL / FAIL |
| (b) base adapter, trained readout | 0.7625 | 0.8859 | 0.6668 | 0.7348 | 0.7708 | 0.0381 | 0.4890 | FAIL / FAIL / FAIL |
| alpha 1 (trained adapter, trained readout) | 0.8386 | 0.9672 | 0.8094 | 0.7393 | 0.7585 | 0.0040 | 0.6407 | FAIL / FAIL / FAIL |

## 4. Readings (additional; labelled; none of them is a choice)

1. **Positional health is lost between alpha 0.5 and alpha 0.7** on this checkpoint. It passes at 0, 0.3 and 0.5 in all three orders and fails at 0.7, 0.85 and 1 in all three orders. The two-letter accuracy falls monotonically with alpha, from 0.6112 at 0 to 0.0040 at 1, while the single-letter accuracy stays at 0.7146 or above up to 0.7.
2. **Knowledge dev falls slightly with alpha, and significantly on MedMCQA at alpha 1.** Against alpha 0 the paired difference is -0.0140 [-0.0228, -0.0055] at alpha 1 (n = 4000). This is the GOAL's "knowledge below zero-shot" check on dev. By source at alpha 1: MedMCQA (3000 items) -0.0133 [-0.0233, -0.0027]; MedQA (1000 items) -0.0160 [-0.0340, +0.0030]. Gate O4's knowledge check is on the knowledge test set (osler_v4 measured -0.0067 there), a different set; the dev figure is additional. The log is `outputs/osler_v5/Z1/logs/knowledge_paired.log`.
3. **Both the readout and the adapter carry the two-letter loss.** The trained readout alone (split b) takes the two-letter accuracy from 0.6112 to 0.0381; the trained adapter alone with the base readout (split a) takes it to 0.1924; together (alpha 1) to 0.0040. The trained readout also lowers the single-letter accuracy to 0.4890 with the adapter off (alpha 0 gives 0.7984), so the readout rows for options A to Z changed too. (A hypothesis about the mechanism; the figures are measured.)
4. **A discrepancy with osler_v4's wording.** osler_v4's FINDINGS section 1 says single-letter gold is intact (C210, C212). C210's own figures give 0.6407 for the final against 0.7964 for zero-shot on the same 501 single-letter items, and this probe gives 0.6407 at alpha 1. The FINDINGS wording is not supported by its own numbers. It is noted here; osler_v4's files are not changed.
5. **What the Z4 rule would say about this checkpoint (what-if, not a choice).** The MR retention threshold is 0.9510 (80 % of the alpha 1 gain over alpha 0, which is 0.0809). Alpha 0.5 reaches 0.9475, below it; alpha 0.7 reaches 0.9613, above it, but fails positional health. Alpha 1 itself fails positional health. So no value measured here qualifies. Alpha 0.6 is not measured here and is on the Z4 grid. Z4 runs on the Z3 checkpoint selected by S among positional-health passes, not on this checkpoint.
6. **S is not a safe selector on its own.** S peaks at alpha 0.7 (0.8415) and alpha 0.85 (0.8414); both fail positional health. A selection by S alone would have taken a model that fails the two-letter check. This is why the Z3 rule restricts to positional passes.

## 5. Limits

- Dev only: fast dev v4 (17,532 items), CG v2 (8,307 items, of which 1,000 are DDXPlus and 499 have two-letter gold), knowledge dev v2 (4,000 items).
- Accuracy only; no calibration is applied, and accuracy does not depend on temperature.
- The probe job's own composite used item-micro MR and MIMIC accuracies. Its in-memory figures are superseded by `summary.json` (STATE §7.10).
- The alpha grid here (0, 0.3, 0.5, 0.7, 0.85, 1) is a probe grid. The Z4 grid (0.3 to 1.0 in steps of 0.1) runs on the Z3 checkpoint.

