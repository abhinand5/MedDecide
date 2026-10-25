# osler_v4 — size-class scoreboard (Y9; additional rows, labelled; no verdict)

Generated 2026-10-25T06:35:08Z by `scripts/osler_v4/y9_gate.py` from `outputs/osler_v4/Y9/gate.json` (aggregates only). Each model's figures are its own, on the items it scored: the template macro accuracy, the item-mean Brier, and the readout status (D12). A model that cannot take an item (ten letters) has that item unscored here (D44(a)). Gate O4 is in gate_o4.md.

## mr_verdict

| model | scored items | template macro accuracy | Brier (item mean) | readout | skipped by runner |
|---|---|---|---|---|---|
| Osler-4B (final, Y8) | 13103 | 0.8431 | 0.1597 | PASS | 0 |
| zero-shot pplx-decider-flash-v1.2 | 13103 | 0.7044 | 0.3775 | PASS | 0 |
| MedDecider-4B | 12992 | 0.6051 | 0.4565 | PASS | 0 |
| osler_v2 Osler-4B | 13103 | 0.8621 | 0.1547 | PASS | 0 |
| MedDecider-9B | 12992 | 0.6223 | 0.4864 | PASS | 0 |
| JEV-9B | 13103 | 0.6583 | 0.4934 | PASS | 0 |
| pplx-decider-v1.2-27b (ceiling, additional) | 13103 | 0.7745 | 0.2775 | PASS | 0 |

## mr_reported

| model | scored items | template macro accuracy | Brier (item mean) | readout | skipped by runner |
|---|---|---|---|---|---|
| Osler-4B (final, Y8) | 6191 | 1.0000 | 0.0002 | PASS | 0 |
| zero-shot pplx-decider-flash-v1.2 | 6191 | 0.9960 | 0.0041 | PASS | 0 |
| MedDecider-4B | 0 | — | — | NOT MEASURED (no scored items) | 0 |
| osler_v2 Osler-4B | 6191 | 1.0000 | 0.0000 | PASS | 0 |
| MedDecider-9B | 0 | — | — | NOT MEASURED (no scored items) | 0 |
| JEV-9B | 6191 | 0.9974 | 0.0090 | PASS | 0 |
| pplx-decider-v1.2-27b (ceiling, additional) | 6191 | 1.0000 | 0.0009 | PASS | 0 |

## mr_heldout_test

| model | scored items | template macro accuracy | Brier (item mean) | readout | skipped by runner |
|---|---|---|---|---|---|
| Osler-4B (final, Y8) | 584 | 0.7679 | 0.6248 | READOUT_FAIL (D12) | 0 |
| zero-shot pplx-decider-flash-v1.2 | 584 | 0.6949 | 0.4632 | PASS | 0 |
| MedDecider-4B | 584 | 0.5937 | 0.4929 | PASS | 0 |
| osler_v2 Osler-4B | 584 | 0.7284 | 0.9843 | READOUT_FAIL (D12) | 0 |
| MedDecider-9B | 584 | 0.5424 | 0.6560 | READOUT_FAIL (D12) | 0 |
| JEV-9B | 584 | 0.5630 | 0.5959 | READOUT_FAIL (D12) | 0 |
| pplx-decider-v1.2-27b (ceiling, additional) | 584 | 0.8296 | 0.1404 | PASS | 0 |

## mimic_suite

| model | scored items | template macro accuracy | Brier (item mean) | readout | skipped by runner |
|---|---|---|---|---|---|
| Osler-4B (final, Y8) | 4253 | 0.7995 | 0.3156 | PASS | 0 |
| zero-shot pplx-decider-flash-v1.2 | 4253 | 0.6341 | 0.4961 | PASS | 0 |
| MedDecider-4B | 4253 | 0.6115 | 0.5179 | READOUT_FAIL (D12) | 0 |
| osler_v2 Osler-4B | 4253 | 0.5651 | 0.6905 | PASS | 0 |
| MedDecider-9B | 4253 | 0.6278 | 0.5034 | PASS | 0 |
| JEV-9B | 4253 | 0.5992 | 0.5401 | PASS | 0 |
| pplx-decider-v1.2-27b (ceiling, additional) | 4253 | 0.6819 | 0.4371 | PASS | 0 |

## mimic_heldout

| model | scored items | template macro accuracy | Brier (item mean) | readout | skipped by runner |
|---|---|---|---|---|---|
| Osler-4B (final, Y8) | 110 | 0.6384 | 0.6018 | PASS | 0 |
| zero-shot pplx-decider-flash-v1.2 | 110 | 0.7199 | 0.4319 | PASS | 0 |
| MedDecider-4B | 110 | 0.7176 | 0.5006 | PASS | 0 |
| osler_v2 Osler-4B | 110 | 0.6704 | 0.9268 | PASS | 0 |
| MedDecider-9B | 110 | 0.6713 | 0.5055 | PASS | 0 |
| JEV-9B | 110 | 0.7343 | 0.4221 | PASS | 0 |
| pplx-decider-v1.2-27b (ceiling, additional) | 110 | 0.7718 | 0.3431 | PASS | 0 |

## panel

| model | scored items | template macro accuracy | Brier (item mean) | readout | skipped by runner |
|---|---|---|---|---|---|
| Osler-4B (final, Y8) | 4353 | 0.5968 | 0.6201 | PASS | 0 |
| zero-shot pplx-decider-flash-v1.2 | 4353 | 0.6390 | 0.6416 | PASS | 0 |
| MedDecider-4B | 4141 | 0.6159 | 0.5763 | PASS | 0 |
| osler_v2 Osler-4B | 4353 | 0.6041 | 0.6370 | PASS | 0 |
| MedDecider-9B | 4141 | 0.6301 | 0.5484 | PASS | 0 |
| JEV-9B | 4141 | 0.5840 | 0.6207 | PASS | 212 |
| pplx-decider-v1.2-27b (ceiling, additional) | 4353 | 0.7203 | 0.4796 | PASS | 0 |

## knowledge

| model | scored items | template macro accuracy | Brier (item mean) | readout | skipped by runner |
|---|---|---|---|---|---|
| Osler-4B (final, Y8) | 5456 | 0.7024 | 0.4313 | PASS | 0 |
| zero-shot pplx-decider-flash-v1.2 | 5435 | 0.7097 | 0.4670 | PASS | 0 |
| MedDecider-4B | 5435 | 0.6969 | 0.4245 | PASS | 0 |
| osler_v2 Osler-4B | 5456 | 0.7005 | 0.4310 | PASS | 0 |
| MedDecider-9B | 5435 | 0.7265 | 0.3897 | PASS | 0 |
| JEV-9B | 5456 | 0.6692 | 0.4576 | PASS | 0 |
| pplx-decider-v1.2-27b (ceiling, additional) | 5435 | 0.8173 | 0.3038 | PASS | 0 |

## mednli_test

| model | scored items | template macro accuracy | Brier (item mean) | readout | skipped by runner |
|---|---|---|---|---|---|
| Osler-4B (final, Y8) | 1422 | 0.8762 | 0.1788 | PASS | 0 |
| zero-shot pplx-decider-flash-v1.2 | 1422 | 0.8720 | 0.1985 | PASS | 0 |
| MedDecider-4B | 1422 | 0.8488 | 0.2299 | PASS | 0 |
| osler_v2 Osler-4B | 1422 | 0.8692 | 0.1932 | PASS | 0 |
| MedDecider-9B | 1422 | 0.8425 | 0.2548 | PASS | 0 |
| JEV-9B | 1422 | 0.8221 | 0.2643 | PASS | 0 |
| pplx-decider-v1.2-27b (ceiling, additional) | 1422 | 0.8748 | 0.1891 | PASS | 0 |

## pairs_test

| model | scored items | template macro accuracy | Brier (item mean) | readout | skipped by runner |
|---|---|---|---|---|---|
| Osler-4B (final, Y8) | 500 | 0.9660 | 0.0568 | PASS | 0 |
| zero-shot pplx-decider-flash-v1.2 | 500 | 0.8740 | 0.1776 | PASS | 0 |
| MedDecider-4B | 500 | 0.6920 | 0.3639 | PASS | 0 |
| osler_v2 Osler-4B | 500 | 0.9140 | 0.1363 | PASS | 0 |
| MedDecider-9B | 500 | 0.7880 | 0.3261 | PASS | 0 |
| JEV-9B | 500 | 0.8920 | 0.1661 | PASS | 0 |
| pplx-decider-v1.2-27b (ceiling, additional) | 500 | 0.9400 | 0.0937 | PASS | 0 |

<!-- y8-code-split:begin -->
## Additional rows: CG v2 by option-code type, and per template (dev only)

Source: `outputs/osler_v4/Y8/code_split.json` (written by `scripts/osler_v4/y8_code_split.py`; aggregates only). CG v2 is a development set: every item has split `dev`. These are additional, labelled rows; no verdict or choice uses them.

### CG v2 DDXPlus accuracy by gold code, and the mean probability mass on two-letter codes

| model | gold A-Z: items | accuracy | gold two-letter (BA-BW): items | accuracy | mean mass on two-letter codes, all items | mean mass, A-Z gold | mean mass, two-letter gold |
|---|---|---|---|---|---|---|---|
| zero-shot pplx-decider-flash-v1.2 | 501 | 0.7964 | 499 | 0.6112 | 0.5017 | 0.1804 | 0.8243 |
| osler_v2 Osler-4B (Y4) | 501 | 0.7964 | 499 | 0.0641 | 0.3210 | 0.1943 | 0.4482 |
| A0 control (Stage A, 13,992 steps) | 501 | 0.8104 | 499 | 0.4790 | 0.4459 | 0.2425 | 0.6501 |
| Y8 final (63,304 steps) | 501 | 0.6407 | 499 | 0.0040 | 0.3105 | 0.2201 | 0.4013 |

Two-letter codes are option indexes 26 and up (keys BA to BW for the 49 options). The steer's label "AA+" is this split: no AA key exists, because the base-26 letters run A to Z. Share of the DDXPlus gold answers that are two-letter: 0.4990 (1,000 items).

### CG v2 accuracy per template (micro accuracy on the items each model scored)

| template | items | zero-shot pplx-decider-flash-v1.2 | osler_v2 Osler-4B (Y4) | A0 control (Stage A, 13,992 steps) | Y8 final (63,304 steps) |
|---|---|---|---|---|---|
| cg_v2_ddxplus_validate_v1 | 1,000 | 0.7040 | 0.4310 | 0.6450 | 0.3230 |
| cg_v2_headqa_dev_v1 | 1,342 | 0.8515 (n = 1,340) | 0.8323 | 0.8525 | 0.8271 |
| cg_v2_mednli_dev_v1 | 995 | 0.8714 | 0.8744 | 0.8764 | 0.8804 |
| cg_v2_pubmedqa_labelled_v1 | 1,000 | 0.7600 | 0.7790 | 0.7900 | 0.7880 |
| medmcqa_4opt_v1 | 2,753 | 0.7434 (n = 2,744) | 0.7356 | 0.7367 | 0.7276 |
| medqa_usmle4_v1 | 747 | 0.8782 | 0.8728 | 0.8876 | 0.8581 |
| medquad_routing_v1 | 87 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| mmlu_mc_v1 | 68 | 0.8971 | 0.8529 | 0.8529 | 0.8824 |
| nfcorpus_relevant_noul_v1 | 87 | READOUT_FAIL | 0.8506 | 0.7356 | 0.7586 |
| pubhealth_claim_true_noul_v1 | 86 | 0.8953 | 0.9419 | 0.9535 | 0.9419 |
| pubmedqa_ynm_v1 | 87 | 0.8046 | 0.8276 | 0.8276 | 0.7816 |
| trec_covid_graded_score_v1 | 33 | 0.3939 | 0.6970 | 0.7576 | 0.6970 |
| trec_covid_relevant_noul_v1 | 22 | 0.7273 | 0.6818 | 0.6818 | 0.6818 |

The items column is the count of the reference run, A0; a model with another count shows it in its cell.
zero-shot pplx-decider-flash-v1.2 skips 11 CG v2 items that the trainer-based runs keep (11 of them have repeated option labels; C086): 2 cg_v2_headqa_dev_v1, 9 medmcqa_4opt_v1.
<!-- y8-code-split:end -->
