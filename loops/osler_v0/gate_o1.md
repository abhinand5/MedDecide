# O11 — Gate O1 (PROGRAM D24)

Osler rows: both-order averages for choice items; noul and score items are scored once (deviation 44).
Accuracy is the template macro (G1's definition), bootstrapped within templates (1,000 resamples, seed 0);
Brier is the item mean, bootstrapped the same way. An interval is shown as point [lower, upper].

| scope | items in scope |
|---|---|
| headline_seen | 15897 |
| external_panel | 4353 |
| heldout_templates | 5658 |
| robustness | 20109 |
| knowledge | 5456 |

## Osler-4B

| scope | baseline | items (paired) | accuracy diff [lower, upper] | Brier diff [lower, upper] | pass | skipped / missing (baseline, Osler) |
|---|---|---|---|---|---|---|
| headline_seen | MedDecider-4B | 15897 | +0.1102 [+0.0992, +0.1216] | -0.1328 [-0.1392, -0.1267] | yes | 0 / 0, 0 |
| headline_seen | zero-shot Qwen3.5-4B | 15897 | +0.1318 [+0.1203, +0.1434] | -0.1815 [-0.1893, -0.1745] | yes | 0 / 0, 0 |

| external_panel | MedDecider-4B | 4141 | -0.0526 [-0.0696, -0.0363] | +0.0479 [+0.0386, +0.0568] | no | 212 / 0, 0 |
| external_panel | zero-shot Qwen3.5-4B | 4353 | +0.0208 [+0.0030, +0.0400] | -0.0746 [-0.0881, -0.0611] | yes | 0 / 0, 0 |

| heldout_templates | MedDecider-4B | 5658 | -0.0992 [-0.1110, -0.0881] | +0.2042 [+0.1897, +0.2180] | no | 0 / 0, 0 |
| heldout_templates | zero-shot Qwen3.5-4B | 5658 | -0.1271 [-0.1399, -0.1153] | +0.2441 [+0.2288, +0.2594] | no | 0 / 0, 0 |

| robustness | MedDecider-4B | 18751 | +0.0682 [+0.0582, +0.0777] | -0.1410 [-0.1474, -0.1340] | yes | 1358 / 0, 0 |
| robustness | zero-shot Qwen3.5-4B | 20109 | +0.1754 [+0.1649, +0.1857] | -0.3152 [-0.3241, -0.3065] | yes | 0 / 0, 0 |

Knowledge guard (MedQA + MedMCQA, micro accuracy, vs zero-shot Qwen3.5-4B): 5456 paired items, -0.0156 [-0.0291, -0.0018]; guard fails (lower bound must exceed -0.02).

**Osler-4B verdict: FAIL.** a primary comparison or the knowledge guard does not pass.

## Osler-9B

| scope | baseline | items (paired) | accuracy diff [lower, upper] | Brier diff [lower, upper] | pass | skipped / missing (baseline, Osler) |
|---|---|---|---|---|---|---|
| headline_seen | MedDecider-9B | 15897 | +0.0929 [+0.0838, +0.1030] | -0.1252 [-0.1312, -0.1197] | yes | 0 / 0, 0 |
| headline_seen | JEV-9B | 15897 | +0.1035 [+0.0942, +0.1135] | -0.1427 [-0.1484, -0.1375] | yes | 0 / 0, 0 |

| external_panel | MedDecider-9B | 4141 | -0.0169 [-0.0330, -0.0007] | +0.0269 [+0.0184, +0.0357] | no | 212 / 0, 0 |
| external_panel | JEV-9B | 4141 | +0.0292 [+0.0142, +0.0452] | -0.0454 [-0.0555, -0.0356] | yes | 212 / 0, 0 |

| heldout_templates | MedDecider-9B | 5658 | -0.0004 [-0.0107, +0.0104] | +0.0316 [+0.0210, +0.0419] | no | 0 / 0, 0 |
| heldout_templates | JEV-9B | 5658 | -0.0769 [-0.0881, -0.0650] | +0.1633 [+0.1511, +0.1760] | no | 0 / 0, 0 |

| robustness | MedDecider-9B | 18751 | +0.1118 [+0.1019, +0.1208] | -0.1783 [-0.1853, -0.1710] | yes | 1358 / 0, 0 |
| robustness | JEV-9B | 19461 | +0.2289 [+0.2176, +0.2392] | -0.3235 [-0.3317, -0.3149] | yes | 648 / 0, 0 |

Knowledge guard (MedQA + MedMCQA, micro accuracy, vs zero-shot Qwen3.5-9B): 5456 paired items, -0.0075 [-0.0196, +0.0055]; guard passes (lower bound must exceed -0.02).

**Osler-9B verdict: FAIL.** a primary comparison or the knowledge guard does not pass.

