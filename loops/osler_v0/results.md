# osler_v0 — results (O11)

Osler-4B (arm L) and Osler-9B on the O2 common set, Gate O1 (D24), the additional analyses, and the held-out generators. Every number comes from the **calibrated** outputs (deviation 59) unless a row says it is uncalibrated. Each number has a CLAIMS row (the O-id is in brackets). The verdict table with every paired interval is `gate_o1.md`; the held-out table is `heldout_generators.md`.

## 1. Gate O1 (D24) — verdicts

| model | verdict | what fails | knowledge guard (floor −0.02) |
|---|---|---|---|
| Osler-4B (arm L) | **FAIL** | external panel vs MedDecider-4B: accuracy −0.0526 [−0.0696, −0.0363]; Brier +0.0479 [+0.0386, +0.0568] | −0.0156 [−0.0291, −0.0018]: fails |
| Osler-9B | **FAIL** | external panel vs MedDecider-9B: accuracy −0.0169 [−0.0330, −0.0007]; Brier +0.0269 [+0.0184, +0.0357] | −0.0075 [−0.0196, +0.0055]: passes |

Both models pass the headline seen-template set against both of their baselines: Osler-4B +0.1102 [+0.0992, +0.1216] against MedDecider-4B and +0.1318 [+0.1203, +0.1434] against zero-shot Qwen3.5-4B; Osler-9B +0.0929 [+0.0838, +0.1030] against MedDecider-9B and +0.1035 [+0.0942, +0.1135] against JEV-9B; every headline Brier interval is below zero [O304, O305]. Osler-4B also passes the panel against zero-shot Qwen3.5-4B (+0.0208 [+0.0030, +0.0400]); Osler-9B passes the panel against JEV-9B (+0.0292 [+0.0142, +0.0452]) [O304, O305].

The rule and thresholds are those of D24, unchanged [O303]. The scope sizes: headline_seen 15,897; external_panel 4,353; heldout_templates 5,658; robustness 20,109; knowledge 5,456 [O303].

## 2. The common set, calibrated, both option orders

| model | accuracy, both orders | accuracy, single order | Brier, both orders | v0.2 | external panel | robustness | rows |
|---|---|---|---|---|---|---|---|
| Osler-4B L | 0.7720 | 0.7662 | 0.3074 | 0.8343 | 0.4964 | 0.7229 | O298, O299 |
| Osler-4B P | 0.7722 | 0.7676 | 0.3110 | 0.8342 | 0.4992 | 0.7231 | O308, O309 |
| Osler-4B N (non-causal) | 0.7794 | 0.7738 | 0.2932 | 0.8419 | 0.5109 | 0.7286 | O311, O312 |
| Osler-9B | 0.8000 | 0.7969 | 0.2682 | 0.8612 | 0.5428 | 0.7490 | O301, O302 |
| Osler-0.8B (reference) | 0.6973 | 0.6919 | 0.3678 | 0.7821 | 0.3221 | 0.6307 | O315, O316 |

Accuracy is the fraction of items whose top option is the gold option. Choice items are averaged over the two option orders; noul and score items are scored once. Uncalibrated comparisons: L (O292, O300) and P (O322). The uncalibrated outputs are kept and labelled and are not used by the gate.

## 3. The arms at 4B (additional; never in a verdict)

- **Non-causal N minus L:** accuracy +0.0050 [+0.0011, +0.0089] on the common set and +0.0152 [+0.0006, +0.0305] on the panel; Brier −0.0142 [−0.0161, −0.0123] and −0.0149 [−0.0231, −0.0066] [O313]. The gain is small, about half a point of accuracy.
- **Pointer P minus L:** accuracy +0.0013 [−0.0020, +0.0051] on the common set and +0.0046 [−0.0121, +0.0215] on the panel, so no difference; Brier +0.0036 [+0.0018, +0.0054] (worse) [O313].
- **Knowledge guard against zero-shot Qwen3.5-4B** (MedQA and MedMCQA, 5,456 items, micro accuracy): N +0.0051 [−0.0084, +0.0180] (passes); P −0.0220 [−0.0350, −0.0077] (fails); L −0.0156 [−0.0291, −0.0018] (fails; the same value as the gate's guard) [O314].
- **The head.** The pre-registered dev rule chose L: no challenger reached +0.010 on the dev template macro with a lower bound above zero. The dev margins were P +0.005145 and N +0.002749 [O241]. The test-set advantage of N is therefore a post-hoc finding. It does not reopen the head choice. Its serving cost (a custom attention mask, no stock GGUF path) is the question the ADVISORY §2 leaves to the operator.

## 4. Held-out generators (additional; baselines not measured)

| model | template macro accuracy, 95 % interval | Brier |
|---|---|---|
| Osler-4B L | 0.7267 [0.6917, 0.7600] | 0.3414 |
| Osler-4B P | 0.7767 [0.7433, 0.8067] | 0.3633 |
| Osler-4B N | 0.7683 [0.7350, 0.8000] | 0.3469 |
| Osler-9B | 0.8383 [0.8117, 0.8650] | 0.2250 |
| Osler-0.8B | 0.4583 [0.4200, 0.4983] | 0.7731 |

The majority baseline is 0.5000 for both held-out templates [O319, O320]. The 0.8B is below it, and its Brier is worse than a uniform guess over three options (0.6667) [O321]. Paired, additional: P − L +0.0500 [+0.0150, +0.0850]; N − L +0.0417 [+0.0033, +0.0800]; 9B − L +0.1117 [+0.0717, +0.1517] [O319]. The four D24 baselines were not scored on these items, so no held-out verdict is computed (deviation 58). The 9B's 0.99 on `note_lab_range_v1` is a numeric comparison task; the generator passed the O3 screen and the gold is computed from the parts [O320].

## 5. Checks run before any result was accepted

- **The calibration defect (deviation 59).** The arms' fitted temperatures never reached the saved checkpoints (all five have an empty calibration) [O293]. The fits reproduce their own NLL on the dev predictions to within 6e-4 [O294]; the calibrated scorer matches the transformed raw output to 2e-8 on the smoke rows [O295] and to 3.7e-8 on the first 25,088 rows of L [O297]. Calibration changes L's both-order accuracy from 0.7721 to 0.7720 (47 items change argmax) and Brier from 0.3161 to 0.3074; single-pass accuracy is unchanged [O300].
- **Saturation.** The uncalibrated P output is exactly one-hot on 7,256 of 43,894 choice items (L: 170) [O323]. The calibrated P output has 29 near-one-hot choice items, all in `ct_claim_set_choice_v1`, all correct, and 227 items above 1 − 1e-9 [O310].
- **Constant answers.** They concentrate on two held-out templates: `fda_boxed_warning_noul_v1` is answered `no` on 92–97 % of items for L, P and N (78 % for the 9B and 71 % for the 0.8B); `ct_arm_role_noul_v1` is answered `yes` on 89.6–96.8 % of items for all five models [O324]. The 0.8B also answers one option on 97 % of the unseen `pubmedqa_ynm_v1` items, and the dev predictions show the same collapse [O317]. These are transfer failures on templates the models did not train on. Calibration does not cause them, because it does not change any single-pass argmax [O317].
- **Dev–test gap on a seen template.** For the 0.8B, `nfcorpus_relevant_noul_v1` reaches 0.948 on dev and 0.600 on test, and the test answers are 84 % `yes` against a balanced gold [O317]. The student loop had recorded this template as a bag-of-words shortcut.
- **Leakage.** None: 573,611 training rows, with no match to a v0.2 test item or an external-panel item, by id or by question text [O307].
- **Chance.** The 0.8B's panel accuracy (0.3221) is above the uniform chance (0.1965) and the majority-of-gold baseline (0.2507) [O318].
- **Audits.** Every calibrated run has closed denominators, every choice item averaged over both orders, and identical noul and score rows in both files [O299, O302, O309, O312, O316].

## 6. Not measured (ADVISORY O11 items)

- **Baselines on the held-out generator items:** NOT MEASURED — the four D24 baselines were not scored on them (deviation 58) [O319].
- **Osler-0.8B against zero-shot 0.8B** (common set and knowledge guard): NOT MEASURED — the O2 common set has no zero-shot 0.8B predictions [O314].
- **The Osler-4B quantised export (≤ 4 GB, GGUF):** NOT MEASURED — the merged-adapter export failed the 1e-3 check at O5 (CLAIMS O094, O105), and no quantised export was built in this loop.
- **Selective accuracy at 50/80/90 % coverage, auto-accept at 95/99 % precision, latency per question and per record, the HLE supplementary set, and the long-record slice:** NOT MEASURED in this loop, because of the time budget (deviation 56). The per-template tables are in the audit files [O292, O299, O309, O312, O316] and are not summarised here.

## 7. Time

The calibrated scoring took about 7.8 hours of GPU time: L 5,951 s [O298], 9B 8,469 s [O301], P 5,949 s [O308], N 5,972 s [O311], 0.8B 1,771 s [O315]. The O11 stage ran from 01:54Z to about 13:42Z on 2026-10-13, about twelve hours of wall clock, against the ADVISORY's six-hour estimate. The difference comes from the slow long-record stretch of each run, the converter failure and the calibration fix. Deviation 56 leaves the time-versus-scope choice to the operator.

## 8. Deviations referenced

44 (both-order scoring), 52 (held-out generator scope), 53 (detached chains), 56 (time), 57 (converter failure and pre-flight), 58 (held-out scope), 59 (calibration defect).
