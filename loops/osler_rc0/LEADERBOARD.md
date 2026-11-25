# RC0 leaderboard — 12 models, 9 frozen evaluation sets

Generated 2026-11-25T04:25:20Z from `rc0_board.json` (git `1ae8214014f0a2df22cfee0d3c23256d4d3fd277`) by `scripts/osler_rc0/rc0_leaderboard.py`. Every number is recomputed from per-item prediction files; accuracy is micro over the frozen set with a 95 % bootstrap CI (2,000 resamples, seed 0), macro is the unweighted mean over templates (CI resamples within template). Brier is the multi-class Brier (sum over offered options), ECE is 15-bin top-label. Coverage is scored / frozen universe; a set a model did not run is `NOT MEASURED — prediction file absent`.

## Headline sets

### `panel` (4,353 items)

| model | accuracy | 95% CI | template macro | Brier | ECE | coverage |
|---|---|---|---|---|---|---|
| pplx-27B | 0.6510 | [0.6363, 0.6653] | 0.7203 | 0.4796 | 0.1131 | 1.000 |
| Osler-9B-rc0 | 0.5927 | [0.5785, 0.6076] | 0.6650 | 0.5223 | 0.0229 | 1.000 |
| d3-flash | 0.5899 | [0.5745, 0.6046] | 0.6594 | 0.5387 | 0.1176 | 1.000 |
| Clef-Flash | 0.5711 | [0.5562, 0.5860] | 0.6502 | 0.5828 | 0.1333 | 1.000 |
| pplx-flash | 0.5615 | [0.5465, 0.5762] | 0.6390 | 0.6416 | 0.2285 | 1.000 |
| MedDecider-9B | 0.5603 | [0.5455, 0.5757] | 0.6301 | 0.5484 | 0.0461 | 0.951 |
| Osler-4B-rc0 | 0.5502 | [0.5353, 0.5642] | 0.6205 | 0.5643 | 0.0374 | 1.000 |
| MedDecider-4B | 0.5436 | [0.5279, 0.5586] | 0.6159 | 0.5763 | 0.0644 | 0.951 |
| Qwen3.5-9B | 0.5419 | [0.5265, 0.5557] | 0.6225 | 0.6546 | 0.2229 | 1.000 |
| JEV-9B | 0.5127 | [0.4977, 0.5274] | 0.5840 | 0.6207 | 0.1206 | 0.951 |
| Qwen3.5-4B | 0.4831 | [0.4689, 0.4980] | 0.5483 | 0.6985 | 0.2220 | 1.000 |
| d1-2B | 0.4259 | [0.4112, 0.4411] | 0.4810 | 0.6806 | 0.0584 | 1.000 |

### `knowledge` (5,456 items)

| model | accuracy | 95% CI | template macro | Brier | ECE | coverage |
|---|---|---|---|---|---|---|
| pplx-27B | 0.7823 | [0.7713, 0.7930] | 0.8173 | 0.3038 | 0.0507 | 0.996 |
| d3-flash | 0.7335 | [0.7220, 0.7458] | 0.7535 | 0.3519 | 0.0253 | 1.000 |
| Osler-9B-rc0 | 0.7322 | [0.7209, 0.7438] | 0.7535 | 0.3587 | 0.0441 | 1.000 |
| MedDecider-9B | 0.6997 | [0.6870, 0.7122] | 0.7265 | 0.3897 | 0.0385 | 0.996 |
| Clef-Flash | 0.6864 | [0.6738, 0.6983] | 0.7105 | 0.4260 | 0.0572 | 1.000 |
| pplx-flash | 0.6835 | [0.6712, 0.6962] | 0.7097 | 0.4670 | 0.1569 | 0.996 |
| Qwen3.5-9B | 0.6774 | [0.6651, 0.6901] | 0.7038 | 0.4485 | 0.1096 | 1.000 |
| Osler-4B-rc0 | 0.6728 | [0.6600, 0.6848] | 0.6995 | 0.4250 | 0.0194 | 1.000 |
| MedDecider-4B | 0.6703 | [0.6576, 0.6833] | 0.6969 | 0.4245 | 0.0234 | 0.996 |
| JEV-9B | 0.6448 | [0.6318, 0.6573] | 0.6692 | 0.4576 | 0.0125 | 1.000 |
| Qwen3.5-4B | 0.6162 | [0.6034, 0.6288] | 0.6451 | 0.5102 | 0.0842 | 1.000 |
| d1-2B | 0.4555 | [0.4423, 0.4687] | 0.4705 | 0.6635 | 0.0546 | 1.000 |

### `confirm_general` (5,988 items)

| model | accuracy | 95% CI | template macro | Brier | ECE | coverage |
|---|---|---|---|---|---|---|
| Osler-9B-rc0 | 0.9008 | [0.8933, 0.9083] | 0.8153 | 0.1673 | 0.0900 | 1.000 |
| Clef-Flash | 0.8933 | [0.8851, 0.9013] | 0.8086 | 0.1661 | 0.0298 | 1.000 |
| pplx-flash | 0.8889 | [0.8808, 0.8966] | 0.7907 | 0.1652 | 0.0234 | 1.000 |
| Osler-4B-rc0 | 0.8883 | [0.8806, 0.8963] | 0.8030 | 0.1792 | 0.0940 | 1.000 |
| pplx-27B | 0.8843 | [0.8761, 0.8923] | 0.8097 | 0.1659 | 0.0448 | 1.000 |
| MedDecider-9B | 0.8825 | [0.8738, 0.8905] | 0.7965 | 0.1735 | 0.0382 | 0.975 |
| MedDecider-4B | 0.8820 | [0.8738, 0.8900] | 0.7922 | 0.1782 | 0.0406 | 0.975 |
| d3-flash | 0.8774 | [0.8691, 0.8856] | 0.7881 | 0.1702 | 0.0363 | 1.000 |
| JEV-9B | NOT MEASURED | — | — | — | — | — |
| d1-2B | NOT MEASURED | — | — | — | — | — |
| Qwen3.5-9B | NOT MEASURED | — | — | — | — | — |
| Qwen3.5-4B | NOT MEASURED | — | — | — | — | — |

### `mr_verdict` (13,103 items)

| model | accuracy | 95% CI | template macro | Brier | ECE | coverage |
|---|---|---|---|---|---|---|
| Osler-9B-rc0 | 0.9016 | [0.8966, 0.9065] | 0.8654 | 0.1421 | 0.0220 | 1.000 |
| Osler-4B-rc0 | 0.8941 | [0.8890, 0.8997] | 0.8499 | 0.1514 | 0.0314 | 1.000 |
| pplx-27B | 0.8093 | [0.8025, 0.8156] | 0.7745 | 0.2775 | 0.0501 | 1.000 |
| pplx-flash | 0.7466 | [0.7390, 0.7537] | 0.7044 | 0.3775 | 0.1067 | 1.000 |
| d3-flash | 0.7351 | [0.7272, 0.7427] | 0.6876 | 0.3715 | 0.1042 | 1.000 |
| d1-2B | 0.7069 | [0.6991, 0.7146] | 0.6085 | 0.3939 | 0.0654 | 1.000 |
| Clef-Flash | 0.6831 | [0.6751, 0.6908] | 0.6830 | 0.4500 | 0.1290 | 1.000 |
| MedDecider-4B | 0.6681 | [0.6599, 0.6765] | 0.6051 | 0.4565 | 0.1546 | 0.992 |
| MedDecider-9B | 0.6653 | [0.6574, 0.6740] | 0.6223 | 0.4864 | 0.1792 | 0.992 |
| Qwen3.5-9B | 0.6315 | [0.6229, 0.6396] | 0.5953 | 0.5789 | 0.2366 | 1.000 |
| JEV-9B | 0.6303 | [0.6218, 0.6383] | 0.6583 | 0.4934 | 0.1741 | 1.000 |
| Qwen3.5-4B | 0.6176 | [0.6089, 0.6260] | 0.5574 | 0.5532 | 0.1981 | 1.000 |

### `mimic_suite` (4,253 items)

| model | accuracy | 95% CI | template macro | Brier | ECE | coverage |
|---|---|---|---|---|---|---|
| Osler-9B-rc0 | 0.7733 | [0.7606, 0.7863] | 0.7797 | 0.3373 | 0.0687 | 1.000 |
| Osler-4B-rc0 | 0.7597 | [0.7470, 0.7722] | 0.7660 | 0.3684 | 0.0818 | 1.000 |
| pplx-27B | 0.6729 | [0.6588, 0.6866] | 0.6819 | 0.4371 | 0.0592 | 1.000 |
| Clef-Flash | 0.6306 | [0.6167, 0.6450] | 0.6391 | 0.4958 | 0.0615 | 1.000 |
| d3-flash | 0.6301 | [0.6163, 0.6454] | 0.6389 | 0.4815 | 0.0611 | 1.000 |
| pplx-flash | 0.6240 | [0.6085, 0.6388] | 0.6341 | 0.4961 | 0.0811 | 1.000 |
| MedDecider-9B | 0.6189 | [0.6045, 0.6341] | 0.6278 | 0.5034 | 0.0715 | 1.000 |
| Qwen3.5-9B | 0.6029 | [0.5874, 0.6177] | 0.6116 | 0.5514 | 0.1462 | 1.000 |
| MedDecider-4B | 0.6019 | [0.5878, 0.6163] | 0.6115 | 0.5179 | 0.0752 | 1.000 |
| JEV-9B | 0.5885 | [0.5742, 0.6033] | 0.5992 | 0.5401 | 0.1046 | 1.000 |
| Qwen3.5-4B | 0.5603 | [0.5455, 0.5751] | 0.5697 | 0.5914 | 0.1592 | 1.000 |
| d1-2B | 0.5587 | [0.5439, 0.5735] | 0.5704 | 0.5602 | 0.0722 | 1.000 |

## Additional / held-out sets

### `mednli_test` (1,422 items)

| model | accuracy | 95% CI | coverage |
|---|---|---|---|
| Osler-9B-rc0 | 0.8917 | [0.8748, 0.9079] | 1.000 |
| Osler-4B-rc0 | 0.8910 | [0.8741, 0.9072] | 1.000 |
| d3-flash | 0.8629 | [0.8453, 0.8804] | 1.000 |
| pplx-flash | 0.8720 | [0.8544, 0.8896] | 1.000 |
| MedDecider-9B | 0.8425 | [0.8235, 0.8608] | 1.000 |
| MedDecider-4B | 0.8488 | [0.8305, 0.8671] | 1.000 |
| Clef-Flash | 0.8636 | [0.8460, 0.8812] | 1.000 |
| JEV-9B | 0.8221 | [0.8024, 0.8425] | 1.000 |
| d1-2B | 0.8080 | [0.7869, 0.8284] | 1.000 |
| Qwen3.5-9B | 0.8361 | [0.8165, 0.8559] | 1.000 |
| Qwen3.5-4B | 0.7862 | [0.7644, 0.8066] | 1.000 |
| pplx-27B | 0.8748 | [0.8579, 0.8924] | 1.000 |

### `mimic_heldout` (110 items)

| model | accuracy | 95% CI | coverage |
|---|---|---|---|
| Osler-9B-rc0 | 0.7000 | [0.6091, 0.7818] | 1.000 |
| Osler-4B-rc0 | 0.6727 | [0.5818, 0.7545] | 1.000 |
| d3-flash | 0.6818 | [0.5909, 0.7636] | 1.000 |
| pplx-flash | NOT MEASURED |
| MedDecider-9B | 0.6455 | [0.5545, 0.7273] | 1.000 |
| MedDecider-4B | 0.6636 | [0.5727, 0.7455] | 1.000 |
| Clef-Flash | 0.7182 | [0.6364, 0.8000] | 1.000 |
| JEV-9B | 0.7000 | [0.6091, 0.7818] | 1.000 |
| d1-2B | NOT MEASURED |
| Qwen3.5-9B | 0.6727 | [0.5818, 0.7545] | 1.000 |
| Qwen3.5-4B | 0.6364 | [0.5455, 0.7184] | 1.000 |
| pplx-27B | 0.7818 | [0.7000, 0.8545] | 1.000 |

### `mr_heldout_test` (584 items)

| model | accuracy | 95% CI | coverage |
|---|---|---|---|
| Osler-9B-rc0 | 0.4298 | [0.3904, 0.4709] | 1.000 |
| Osler-4B-rc0 | 0.3339 | [0.2962, 0.3733] | 1.000 |
| d3-flash | 0.4914 | [0.4521, 0.5308] | 1.000 |
| pplx-flash | 0.6130 | [0.5736, 0.6524] | 1.000 |
| MedDecider-9B | 0.4024 | [0.3630, 0.4418] | 1.000 |
| MedDecider-4B | 0.5205 | [0.4795, 0.5600] | 1.000 |
| Clef-Flash | NOT MEASURED |
| JEV-9B | 0.4041 | [0.3664, 0.4452] | 1.000 |
| d1-2B | NOT MEASURED |
| Qwen3.5-9B | NOT MEASURED |
| Qwen3.5-4B | NOT MEASURED |
| pplx-27B | 0.8990 | [0.8733, 0.9229] | 1.000 |

### `pairs_test` (500 items)

| model | accuracy | 95% CI | coverage |
|---|---|---|---|
| Osler-9B-rc0 | 0.9620 | [0.9460, 0.9780] | 1.000 |
| Osler-4B-rc0 | 0.9580 | [0.9400, 0.9740] | 1.000 |
| d3-flash | 0.9400 | [0.9180, 0.9600] | 1.000 |
| pplx-flash | 0.8740 | [0.8460, 0.9020] | 1.000 |
| MedDecider-9B | 0.7880 | [0.7500, 0.8220] | 1.000 |
| MedDecider-4B | 0.6920 | [0.6520, 0.7320] | 1.000 |
| Clef-Flash | NOT MEASURED |
| JEV-9B | 0.8920 | [0.8640, 0.9180] | 1.000 |
| d1-2B | NOT MEASURED |
| Qwen3.5-9B | NOT MEASURED |
| Qwen3.5-4B | NOT MEASURED |
| pplx-27B | 0.9400 | [0.9200, 0.9600] | 1.000 |

## Readout health (D12/D21)

D12 applies the label-mass gate to zero-shot letter readouts; for trained readouts that normalise over the offered options (D21) the label mass is a diagnostic and the constant-answer and below-chance checks are still applied. Cells that fail any applicable check read `READOUT_FAIL — <check>`.

| model | readout-health flags |
|---|---|
| Osler-9B-rc0 | `mr_heldout_test`: READOUT_FAIL — 1 of 3 templates (synthea_indicator_evidence_v1) |
| Osler-4B-rc0 | `mr_heldout_test`: READOUT_FAIL — 1 of 3 templates (synthea_indicator_evidence_v1) |
| d3-flash | — |
| pplx-flash | — |
| MedDecider-9B | `mr_heldout_test`: READOUT_FAIL — 1 of 3 templates (synthea_indicator_evidence_v1) |
| MedDecider-4B | `mimic_suite`: READOUT_FAIL — 1 of 9 templates (mimic_drg_cc_level_v1) |
| Clef-Flash | — |
| JEV-9B | `mr_heldout_test`: READOUT_FAIL — 1 of 3 templates (synthea_indicator_evidence_v1) |
| d1-2B | `mimic_suite`: READOUT_FAIL — 1 of 9 templates (mimic_drg_cc_level_v1) |
| Qwen3.5-9B | — |
| Qwen3.5-4B | `mr_verdict`: READOUT_FAIL — 1 of 7 templates (codiesp_icd_support_v1) |
| pplx-27B | — |

## Paired differences (accuracy, 95 % paired bootstrap CI)

### Osler-9B-rc0

| vs | set | Δ accuracy | 95% CI | paired n |
|---|---|---|---|---|
| d3-flash | panel | +0.0028 | [-0.0069, +0.0129] | 4,353 |
| d3-flash | knowledge | -0.0013 | [-0.0090, +0.0062] | 5,456 |
| d3-flash | confirm_general | +0.0234 | [+0.0180, +0.0291] | 5,988 |
| d3-flash | mr_verdict | +0.1665 | [+0.1590, +0.1738] | 13,103 |
| d3-flash | mimic_suite | +0.1432 | [+0.1281, +0.1580] | 4,253 |
| d3-flash | mednli_test | +0.0288 | [+0.0127, +0.0443] | 1,422 |
| d3-flash | mimic_heldout | +0.0182 | [-0.0727, +0.1091] | 110 |
| d3-flash | mr_heldout_test | -0.0616 | [-0.0959, -0.0274] | 584 |
| d3-flash | pairs_test | +0.0220 | [+0.0000, +0.0460] | 500 |
| MedDecider-9B | panel | +0.0275 | [+0.0152, +0.0403] | 4,141 |
| MedDecider-9B | knowledge | +0.0329 | [+0.0215, +0.0436] | 5,435 |
| MedDecider-9B | confirm_general | +0.0247 | [+0.0185, +0.0308] | 5,838 |
| MedDecider-9B | mr_verdict | +0.2371 | [+0.2290, +0.2446] | 12,992 |
| MedDecider-9B | mimic_suite | +0.1545 | [+0.1392, +0.1691] | 4,253 |
| MedDecider-9B | mednli_test | +0.0492 | [+0.0316, +0.0668] | 1,422 |
| MedDecider-9B | mimic_heldout | +0.0545 | [-0.0273, +0.1364] | 110 |
| MedDecider-9B | mr_heldout_test | +0.0274 | [-0.0051, +0.0634] | 584 |
| MedDecider-9B | pairs_test | +0.1740 | [+0.1379, +0.2120] | 500 |
| Clef-Flash | panel | +0.0216 | [+0.0085, +0.0354] | 4,353 |
| Clef-Flash | knowledge | +0.0458 | [+0.0352, +0.0577] | 5,456 |
| Clef-Flash | confirm_general | +0.0075 | [+0.0017, +0.0137] | 5,988 |
| Clef-Flash | mr_verdict | +0.2186 | [+0.2104, +0.2264] | 13,103 |
| Clef-Flash | mimic_suite | +0.1427 | [+0.1274, +0.1582] | 4,253 |
| Clef-Flash | mednli_test | +0.0281 | [+0.0113, +0.0443] | 1,422 |
| Clef-Flash | mimic_heldout | -0.0182 | [-0.1000, +0.0636] | 110 |
| Clef-Flash | mr_heldout_test | NOT MEASURED | — | — |
| Clef-Flash | pairs_test | NOT MEASURED | — | — |
| JEV-9B | panel | +0.0751 | [+0.0623, +0.0886] | 4,141 |
| JEV-9B | knowledge | +0.0874 | [+0.0750, +0.0999] | 5,456 |
| JEV-9B | confirm_general | NOT MEASURED | — | — |
| JEV-9B | mr_verdict | +0.2713 | [+0.2629, +0.2794] | 13,103 |
| JEV-9B | mimic_suite | +0.1848 | [+0.1691, +0.1996] | 4,253 |
| JEV-9B | mednli_test | +0.0696 | [+0.0506, +0.0893] | 1,422 |
| JEV-9B | mimic_heldout | +0.0000 | [-0.0818, +0.0818] | 110 |
| JEV-9B | mr_heldout_test | +0.0257 | [-0.0137, +0.0651] | 584 |
| JEV-9B | pairs_test | +0.0700 | [+0.0420, +0.1020] | 500 |
| pplx-27B | panel | -0.0584 | [-0.0712, -0.0450] | 4,353 |
| pplx-27B | knowledge | -0.0497 | [-0.0615, -0.0386] | 5,435 |
| pplx-27B | confirm_general | +0.0165 | [+0.0110, +0.0225] | 5,988 |
| pplx-27B | mr_verdict | +0.0923 | [+0.0864, +0.0986] | 13,103 |
| pplx-27B | mimic_suite | +0.1004 | [+0.0872, +0.1150] | 4,253 |
| pplx-27B | mednli_test | +0.0169 | [+0.0000, +0.0323] | 1,422 |
| pplx-27B | mimic_heldout | -0.0818 | [-0.1636, -0.0091] | 110 |
| pplx-27B | mr_heldout_test | -0.4692 | [-0.5155, -0.4229] | 584 |
| pplx-27B | pairs_test | +0.0220 | [+0.0020, +0.0420] | 500 |

### Osler-4B-rc0

| vs | set | Δ accuracy | 95% CI | paired n |
|---|---|---|---|---|
| pplx-flash | panel | -0.0113 | [-0.0218, -0.0007] | 4,353 |
| pplx-flash | knowledge | -0.0098 | [-0.0191, -0.0017] | 5,435 |
| pplx-flash | confirm_general | -0.0007 | [-0.0077, +0.0068] | 5,988 |
| pplx-flash | mr_verdict | +0.1474 | [+0.1400, +0.1552] | 13,103 |
| pplx-flash | mimic_suite | +0.1357 | [+0.1213, +0.1505] | 4,253 |
| pplx-flash | mednli_test | +0.0190 | [+0.0049, +0.0331] | 1,422 |
| pplx-flash | mimic_heldout | NOT MEASURED | — | — |
| pplx-flash | mr_heldout_test | -0.2791 | [-0.3219, -0.2346] | 584 |
| pplx-flash | pairs_test | +0.0840 | [+0.0560, +0.1140] | 500 |
| MedDecider-4B | panel | +0.0019 | [-0.0109, +0.0140] | 4,141 |
| MedDecider-4B | knowledge | +0.0035 | [-0.0083, +0.0142] | 5,435 |
| MedDecider-4B | confirm_general | +0.0116 | [+0.0043, +0.0194] | 5,838 |
| MedDecider-4B | mr_verdict | +0.2270 | [+0.2190, +0.2351] | 12,992 |
| MedDecider-4B | mimic_suite | +0.1578 | [+0.1425, +0.1723] | 4,253 |
| MedDecider-4B | mednli_test | +0.0422 | [+0.0253, +0.0577] | 1,422 |
| MedDecider-4B | mimic_heldout | +0.0091 | [-0.0818, +0.1091] | 110 |
| MedDecider-4B | mr_heldout_test | -0.1866 | [-0.2295, -0.1404] | 584 |
| MedDecider-4B | pairs_test | +0.2660 | [+0.2240, +0.3080] | 500 |
| Qwen3.5-4B | panel | +0.0671 | [+0.0535, +0.0811] | 4,353 |
| Qwen3.5-4B | knowledge | +0.0566 | [+0.0447, +0.0696] | 5,456 |
| Qwen3.5-4B | confirm_general | NOT MEASURED | — | — |
| Qwen3.5-4B | mr_verdict | +0.2765 | [+0.2683, +0.2851] | 13,103 |
| Qwen3.5-4B | mimic_suite | +0.1994 | [+0.1829, +0.2144] | 4,253 |
| Qwen3.5-4B | mednli_test | +0.1048 | [+0.0823, +0.1280] | 1,422 |
| Qwen3.5-4B | mimic_heldout | +0.0364 | [-0.0636, +0.1364] | 110 |
| Qwen3.5-4B | mr_heldout_test | NOT MEASURED | — | — |
| Qwen3.5-4B | pairs_test | NOT MEASURED | — | — |
| Osler-9B-rc0 | panel | -0.0425 | [-0.0551, -0.0299] | 4,353 |
| Osler-9B-rc0 | knowledge | -0.0594 | [-0.0711, -0.0476] | 5,456 |
| Osler-9B-rc0 | confirm_general | -0.0125 | [-0.0185, -0.0067] | 5,988 |
| Osler-9B-rc0 | mr_verdict | -0.0076 | [-0.0116, -0.0037] | 13,103 |
| Osler-9B-rc0 | mimic_suite | -0.0136 | [-0.0233, -0.0038] | 4,253 |
| Osler-9B-rc0 | mednli_test | -0.0007 | [-0.0141, +0.0120] | 1,422 |
| Osler-9B-rc0 | mimic_heldout | -0.0273 | [-0.1002, +0.0545] | 110 |
| Osler-9B-rc0 | mr_heldout_test | -0.0959 | [-0.1233, -0.0668] | 584 |
| Osler-9B-rc0 | pairs_test | -0.0040 | [-0.0200, +0.0100] | 500 |

## Coverage and NOT MEASURED

| model | sets measured | not measured |
|---|---|---|
| Osler-9B-rc0 | 9/9 | — |
| Osler-4B-rc0 | 9/9 | — |
| d3-flash | 9/9 | — |
| pplx-flash | 8/9 | mimic_heldout |
| MedDecider-9B | 9/9 | — |
| MedDecider-4B | 9/9 | — |
| Clef-Flash | 7/9 | mr_heldout_test, pairs_test |
| JEV-9B | 8/9 | confirm_general |
| d1-2B | 5/9 | confirm_general, mimic_heldout, mr_heldout_test, pairs_test |
| Qwen3.5-9B | 6/9 | confirm_general, mr_heldout_test, pairs_test |
| Qwen3.5-4B | 6/9 | confirm_general, mr_heldout_test, pairs_test |
| pplx-27B | 9/9 | — |

Reasons for `NOT MEASURED`: competitor sets the model never ran (see §2 of the ADVISORY) and, where relevant, runner timeboxes. A missing cell is never imputed.

