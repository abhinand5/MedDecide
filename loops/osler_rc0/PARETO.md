# RC0 Pareto frontier

Generated 2026-11-25T04:25:20Z from `rc0_board.json` by `scripts/osler_rc0/rc0_leaderboard.py`. Both axes are accuracies (higher is better). A model is *comparable* only when it has a measured number on every axis; incomparable models are listed, never imputed.

## ≤10B quality frontier (5 headline sets)

Comparable models: Clef-Flash, MedDecider-4B, MedDecider-9B, Osler-4B-rc0, Osler-9B-rc0, d3-flash, pplx-flash.

Not comparable (missing a headline set): JEV-9B, Qwen3.5-4B, Qwen3.5-9B, d1-2B.

| dominator | dominated | panel | knowledge | confirm_general | mr_verdict | mimic_suite |
|---|---|---|---|---|---|---|
| Osler-9B-rc0 | Osler-4B-rc0 | +0.0425 | +0.0594 | +0.0125 | +0.0076 | +0.0136 |
| Osler-9B-rc0 | pplx-flash | +0.0312 | +0.0487 | +0.0119 | +0.1550 | +0.1493 |
| Osler-9B-rc0 | MedDecider-9B | +0.0324 | +0.0325 | +0.0183 | +0.2363 | +0.1545 |
| Osler-9B-rc0 | MedDecider-4B | +0.0491 | +0.0619 | +0.0188 | +0.2335 | +0.1714 |
| Osler-9B-rc0 | Clef-Flash | +0.0216 | +0.0458 | +0.0075 | +0.2186 | +0.1427 |
| Osler-4B-rc0 | MedDecider-4B | +0.0066 | +0.0026 | +0.0063 | +0.2260 | +0.1578 |
| pplx-flash | MedDecider-4B | +0.0179 | +0.0132 | +0.0070 | +0.0785 | +0.0221 |
| Clef-Flash | MedDecider-4B | +0.0275 | +0.0161 | +0.0113 | +0.0149 | +0.0287 |

Undominated on all five: Osler-9B-rc0, d3-flash.

## 2-axis Pareto (OOD General Avg vs Clinical Audit Avg)

OOD General Avg = mean(panel, knowledge, confirm_general); Clinical Audit Avg = mean(mr_verdict, mimic_suite).

| model | OOD General Avg | Clinical Audit Avg | Pareto |
|---|---|---|---|
| pplx-27B | 0.7725 | 0.7411 | front |
| Osler-9B-rc0 | 0.7419 | 0.8375 | front |
| d3-flash | 0.7336 | 0.6826 |  |
| Clef-Flash | 0.7169 | 0.6568 |  |
| MedDecider-9B | 0.7142 | 0.6421 |  |
| pplx-flash | 0.7113 | 0.6853 |  |
| Osler-4B-rc0 | 0.7038 | 0.8269 |  |
| MedDecider-4B | 0.6986 | 0.6350 |  |
| Qwen3.5-9B | 0.6097 | 0.6172 |  |
| JEV-9B | 0.5787 | 0.6094 |  |
| Qwen3.5-4B | 0.5497 | 0.5889 |  |
| d1-2B | 0.4407 | 0.6328 |  |

Frontier: Osler-9B-rc0, pplx-27B.

Not comparable: none.

## ≤4B tier frontier

Comparable: MedDecider-4B, Osler-4B-rc0, pplx-flash; not comparable: Qwen3.5-4B, d1-2B.

| dominator | dominated | panel | knowledge | confirm_general | mr_verdict | mimic_suite |
|---|---|---|---|---|---|---|
| Osler-4B-rc0 | MedDecider-4B | +0.0066 | +0.0026 | +0.0063 | +0.2260 | +0.1578 |
| pplx-flash | MedDecider-4B | +0.0179 | +0.0132 | +0.0070 | +0.0785 | +0.0221 |

Undominated: Osler-4B-rc0, pplx-flash.

## Efficiency frontier

params (B) and measured batch-1 p50 latency vs the three accuracy averages; pplx-27B included as the out-of-tier cost/accuracy reference

| model | params (B) | p50 ms/item (batch 1) | peak VRAM (GB) | 5-set macro | OOD avg | Audit avg |
|---|---|---|---|---|---|---|
| d1-2B | 2.00 | NOT MEASURED | NOT MEASURED | 0.5367 | 0.4407 | 0.6328 |
| Osler-4B-rc0 | 4.00 | 26.5430 | 9.6000 | 0.7530 | 0.7038 | 0.8269 |
| pplx-flash | 4.00 | 25.5100 | 9.5790 | 0.7009 | 0.7113 | 0.6853 |
| MedDecider-4B | 4.00 | NOT MEASURED | NOT MEASURED | 0.6732 | 0.6986 | 0.6350 |
| Qwen3.5-4B | 4.00 | NOT MEASURED | NOT MEASURED | 0.5693 | 0.5497 | 0.5889 |
| Osler-9B-rc0 | 8.39 | 33.7140 | 17.3230 | 0.7801 | 0.7419 | 0.8375 |
| d3-flash | 8.39 | 33.7990 | 17.3230 | 0.7132 | 0.7336 | 0.6826 |
| MedDecider-9B | 9.00 | NOT MEASURED | NOT MEASURED | 0.6853 | 0.7142 | 0.6421 |
| Clef-Flash | 9.00 | NOT MEASURED | NOT MEASURED | 0.6929 | 0.7169 | 0.6568 |
| JEV-9B | 9.00 | NOT MEASURED | NOT MEASURED | 0.5941 | 0.5787 | 0.6094 |
| Qwen3.5-9B | 9.00 | NOT MEASURED | NOT MEASURED | 0.6134 | 0.6097 | 0.6172 |
| pplx-27B | 27.00 | 90.9570 | 52.9220 | 0.7600 | 0.7725 | 0.7411 |

Latency values come from `outputs/osler_rc0/R5/latency.json`; `NOT MEASURED` means the R5 timebox for that model was not met or its runner cannot batch.

## Paired comparison tables

### Osler-9B-rc0 vs d3-flash

| set | Δ accuracy | 95% CI | Δ Brier [95% CI] (lower better) |
|---|---|---|---|
| panel | +0.0028 | [-0.0069, +0.0129] | -0.0164 [-0.0239, -0.0091] |
| knowledge | -0.0013 | [-0.0090, +0.0062] | +0.0068 [+0.0020, +0.0116] |
| confirm_general | +0.0234 | [+0.0180, +0.0291] | -0.0029 [-0.0082, +0.0021] |
| mr_verdict | +0.1665 | [+0.1590, +0.1738] | -0.2294 [-0.2378, -0.2212] |
| mimic_suite | +0.1432 | [+0.1281, +0.1580] | -0.1442 [-0.1596, -0.1285] |
| mednli_test | +0.0288 | [+0.0127, +0.0443] | -0.0584 [-0.0738, -0.0420] |
| mimic_heldout | +0.0182 | [-0.0727, +0.1091] | -0.0256 [-0.1421, +0.0975] |
| mr_heldout_test | -0.0616 | [-0.0959, -0.0274] | +0.1765 [+0.1357, +0.2189] |
| pairs_test | +0.0220 | [+0.0000, +0.0460] | -0.0550 [-0.0823, -0.0291] |

### Osler-9B-rc0 vs MedDecider-9B

| set | Δ accuracy | 95% CI | Δ Brier [95% CI] (lower better) |
|---|---|---|---|
| panel | +0.0275 | [+0.0152, +0.0403] | -0.0231 [-0.0309, -0.0152] |
| knowledge | +0.0329 | [+0.0215, +0.0436] | -0.0314 [-0.0389, -0.0239] |
| confirm_general | +0.0247 | [+0.0185, +0.0308] | -0.0136 [-0.0208, -0.0072] |
| mr_verdict | +0.2371 | [+0.2290, +0.2446] | -0.3455 [-0.3560, -0.3350] |
| mimic_suite | +0.1545 | [+0.1392, +0.1691] | -0.1661 [-0.1829, -0.1489] |
| mednli_test | +0.0492 | [+0.0316, +0.0668] | -0.0983 [-0.1156, -0.0800] |
| mimic_heldout | +0.0545 | [-0.0273, +0.1364] | -0.0261 [-0.1338, +0.0806] |
| mr_heldout_test | +0.0274 | [-0.0051, +0.0634] | +0.1890 [+0.1446, +0.2348] |
| pairs_test | +0.1740 | [+0.1379, +0.2120] | -0.2630 [-0.3206, -0.2087] |

### Osler-9B-rc0 vs Clef-Flash

| set | Δ accuracy | 95% CI | Δ Brier [95% CI] (lower better) |
|---|---|---|---|
| panel | +0.0216 | [+0.0085, +0.0354] | -0.0605 [-0.0724, -0.0483] |
| knowledge | +0.0458 | [+0.0352, +0.0577] | -0.0673 [-0.0776, -0.0571] |
| confirm_general | +0.0075 | [+0.0017, +0.0137] | +0.0012 [-0.0067, +0.0087] |
| mr_verdict | +0.2186 | [+0.2104, +0.2264] | -0.3079 [-0.3177, -0.2982] |
| mimic_suite | +0.1427 | [+0.1274, +0.1582] | -0.1585 [-0.1758, -0.1414] |
| mednli_test | +0.0281 | [+0.0113, +0.0443] | -0.0572 [-0.0751, -0.0373] |
| mimic_heldout | -0.0182 | [-0.1000, +0.0636] | +0.0509 [-0.0415, +0.1421] |
| mr_heldout_test | NOT MEASURED | — | — |
| pairs_test | NOT MEASURED | — | — |

### Osler-9B-rc0 vs JEV-9B

| set | Δ accuracy | 95% CI | Δ Brier [95% CI] (lower better) |
|---|---|---|---|
| panel | +0.0751 | [+0.0623, +0.0886] | -0.0954 [-0.1057, -0.0853] |
| knowledge | +0.0874 | [+0.0750, +0.0999] | -0.0988 [-0.1078, -0.0900] |
| confirm_general | NOT MEASURED | — | — |
| mr_verdict | +0.2713 | [+0.2629, +0.2794] | -0.3513 [-0.3609, -0.3417] |
| mimic_suite | +0.1848 | [+0.1691, +0.1996] | -0.2028 [-0.2199, -0.1854] |
| mednli_test | +0.0696 | [+0.0506, +0.0893] | -0.1079 [-0.1268, -0.0898] |
| mimic_heldout | +0.0000 | [-0.0818, +0.0818] | +0.0574 [-0.0324, +0.1491] |
| mr_heldout_test | +0.0257 | [-0.0137, +0.0651] | +0.2492 [+0.1974, +0.2982] |
| pairs_test | +0.0700 | [+0.0420, +0.1020] | -0.1030 [-0.1387, -0.0702] |

### Osler-9B-rc0 vs pplx-27B

| set | Δ accuracy | 95% CI | Δ Brier [95% CI] (lower better) |
|---|---|---|---|
| panel | -0.0584 | [-0.0712, -0.0450] | +0.0427 [+0.0311, +0.0544] |
| knowledge | -0.0497 | [-0.0615, -0.0386] | +0.0545 [+0.0444, +0.0657] |
| confirm_general | +0.0165 | [+0.0110, +0.0225] | +0.0014 [-0.0048, +0.0076] |
| mr_verdict | +0.0923 | [+0.0864, +0.0986] | -0.1354 [-0.1426, -0.1286] |
| mimic_suite | +0.1004 | [+0.0872, +0.1150] | -0.0998 [-0.1159, -0.0856] |
| mednli_test | +0.0169 | [+0.0000, +0.0323] | -0.0326 [-0.0492, -0.0152] |
| mimic_heldout | -0.0818 | [-0.1636, -0.0091] | +0.1363 [+0.0441, +0.2255] |
| mr_heldout_test | -0.4692 | [-0.5155, -0.4229] | +0.7047 [+0.6399, +0.7671] |
| pairs_test | +0.0220 | [+0.0020, +0.0420] | -0.0306 [-0.0559, -0.0055] |

### Osler-4B-rc0 vs pplx-flash

| set | Δ accuracy | 95% CI | Δ Brier [95% CI] (lower better) |
|---|---|---|---|
| panel | -0.0113 | [-0.0218, -0.0007] | -0.0773 [-0.0896, -0.0647] |
| knowledge | -0.0098 | [-0.0191, -0.0017] | -0.0427 [-0.0521, -0.0326] |
| confirm_general | -0.0007 | [-0.0077, +0.0068] | +0.0140 [+0.0060, +0.0215] |
| mr_verdict | +0.1474 | [+0.1400, +0.1552] | -0.2260 [-0.2357, -0.2162] |
| mimic_suite | +0.1357 | [+0.1213, +0.1505] | -0.1278 [-0.1427, -0.1131] |
| mednli_test | +0.0190 | [+0.0049, +0.0331] | -0.0396 [-0.0521, -0.0274] |
| mimic_heldout | NOT MEASURED | — | — |
| mr_heldout_test | -0.2791 | [-0.3219, -0.2346] | +0.3004 [+0.2583, +0.3392] |
| pairs_test | +0.0840 | [+0.0560, +0.1140] | -0.1072 [-0.1401, -0.0741] |

### Osler-4B-rc0 vs MedDecider-4B

| set | Δ accuracy | 95% CI | Δ Brier [95% CI] (lower better) |
|---|---|---|---|
| panel | +0.0019 | [-0.0109, +0.0140] | -0.0085 [-0.0167, +0.0000] |
| knowledge | +0.0035 | [-0.0083, +0.0142] | -0.0001 [-0.0077, +0.0077] |
| confirm_general | +0.0116 | [+0.0043, +0.0194] | -0.0058 [-0.0152, +0.0030] |
| mr_verdict | +0.2270 | [+0.2190, +0.2351] | -0.3065 [-0.3169, -0.2965] |
| mimic_suite | +0.1578 | [+0.1425, +0.1723] | -0.1496 [-0.1665, -0.1325] |
| mednli_test | +0.0422 | [+0.0253, +0.0577] | -0.0711 [-0.0853, -0.0563] |
| mimic_heldout | +0.0091 | [-0.0818, +0.1091] | +0.0333 [-0.0937, +0.1623] |
| mr_heldout_test | -0.1866 | [-0.2295, -0.1404] | +0.2706 [+0.2295, +0.3102] |
| pairs_test | +0.2660 | [+0.2240, +0.3080] | -0.2935 [-0.3412, -0.2445] |

### Osler-4B-rc0 vs Qwen3.5-4B

| set | Δ accuracy | 95% CI | Δ Brier [95% CI] (lower better) |
|---|---|---|---|
| panel | +0.0671 | [+0.0535, +0.0811] | -0.1341 [-0.1476, -0.1209] |
| knowledge | +0.0566 | [+0.0447, +0.0696] | -0.0852 [-0.0963, -0.0741] |
| confirm_general | NOT MEASURED | — | — |
| mr_verdict | +0.2765 | [+0.2683, +0.2851] | -0.4018 [-0.4128, -0.3911] |
| mimic_suite | +0.1994 | [+0.1829, +0.2144] | -0.2230 [-0.2424, -0.2042] |
| mednli_test | +0.1048 | [+0.0823, +0.1280] | -0.1636 [-0.1902, -0.1395] |
| mimic_heldout | +0.0364 | [-0.0636, +0.1364] | -0.0672 [-0.2167, +0.0806] |
| mr_heldout_test | NOT MEASURED | — | — |
| pairs_test | NOT MEASURED | — | — |

### Osler-4B-rc0 vs Osler-9B-rc0

| set | Δ accuracy | 95% CI | Δ Brier [95% CI] (lower better) |
|---|---|---|---|
| panel | -0.0425 | [-0.0551, -0.0299] | +0.0420 [+0.0331, +0.0506] |
| knowledge | -0.0594 | [-0.0711, -0.0476] | +0.0662 [+0.0569, +0.0752] |
| confirm_general | -0.0125 | [-0.0185, -0.0067] | +0.0119 [+0.0078, +0.0162] |
| mr_verdict | -0.0076 | [-0.0116, -0.0037] | +0.0094 [+0.0058, +0.0132] |
| mimic_suite | -0.0136 | [-0.0233, -0.0038] | +0.0311 [+0.0202, +0.0418] |
| mednli_test | -0.0007 | [-0.0141, +0.0120] | +0.0024 [-0.0095, +0.0138] |
| mimic_heldout | -0.0273 | [-0.1002, +0.0545] | +0.0545 [-0.0382, +0.1523] |
| mr_heldout_test | -0.0959 | [-0.1233, -0.0668] | -0.0815 [-0.1176, -0.0459] |
| pairs_test | -0.0040 | [-0.0200, +0.0100] | +0.0073 [-0.0146, +0.0319] |

