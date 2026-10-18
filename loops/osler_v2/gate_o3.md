# Gate O3 (osler_v2): Osler-4B - verdict

Generated from `outputs/osler_v2/W8/gate_o3.json` by `scripts/osler_v2/w8_report.py` (2026-10-18T17:36:09Z).
Aggregates only; every number has a row in `loops/osler_v2/CLAIMS.md` (W053–W062).

**Verdict (as the gate code states it): NOT MEASURED** - not measured: knowledge guard

**Measured failures:** external panel vs zero-shot pplx-decider-flash-v1.2 (lower bound -0.0344 must exceed -0.02).
**Not measured:** knowledge guard (zero-shot pplx).
**Gate passes:** False. The gate passes only when every required comparison passes; a measured failure stands whatever the unmeasured checks show. The 9B verdict is NOT MEASURED (deferred by the operator, D39).

## Scopes (items)

| scope | items |
|---|---|
| mr_verdict | 13103 |
| mr_reported_only | 6191 |
| external_panel | 4353 |
| knowledge | 5456 |
| mr_heldout_test | 584 |

## Primary checks (decide the verdict)

Accuracy and Brier are Osler-4B minus the baseline, paired by item; intervals are bootstrapped within templates (1000 resamples, seed 0).

| check | paired items | accuracy diff [95 % CI] | Brier diff [95 % CI] | accuracy lower > 0 | panel floor passed |
|---|---|---|---|---|---|
| MR suite v1 (verdict templates) vs MedDecider-4B | 12992 | +0.2634 [+0.2511, +0.2756] | -0.3080 [-0.3178, -0.2973] | yes | yes |
| MR suite v1 (verdict templates) vs zero-shot pplx-decider-flash-v1.2 | 13103 | +0.1577 [+0.1422, +0.1733] | -0.2428 [-0.2525, -0.2324] | yes | yes |
| external panel vs MedDecider-4B | 4141 | -0.0038 [-0.0199, +0.0112] | +0.0404 [+0.0282, +0.0510] | no | yes |
| external panel vs zero-shot pplx-decider-flash-v1.2 | 4353 | -0.0198 [-0.0344, -0.0060] | -0.0945 [-0.1070, -0.0810] | no | no |
| knowledge guard (test) vs zero-shot pplx-decider-flash-v1.2 | NOT MEASURED - 21 knowledge items are not paired |  |  |  |

## Additional rows (labelled; never in the verdict)

| row | paired items | accuracy diff [95 % CI] | Brier diff [95 % CI] | accuracy lower > 0 | panel floor passed |
|---|---|---|---|---|---|
| MR verdict, generated part vs MedDecider-4B | 1748 | +0.2395 [+0.2194, +0.2595] | -0.3482 [-0.3744, -0.3213] | yes | yes |
| MR verdict, generated part vs zero-shot pplx-decider-flash-v1.2 | 1748 | +0.1533 [+0.1359, +0.1698] | -0.2568 [-0.2852, -0.2295] | yes | yes |
| MR verdict, real-text part vs MedDecider-4B | 11244 | +0.2873 [+0.2744, +0.3003] | -0.3017 [-0.3128, -0.2905] | yes | yes |
| MR verdict, real-text part vs zero-shot pplx-decider-flash-v1.2 | 11355 | +0.1609 [+0.1363, +0.1851] | -0.2407 [-0.2523, -0.2289] | yes | yes |
| MR verdict without PMC vs MedDecider-4B | 4892 | +0.2799 [+0.2652, +0.2941] | -0.4210 [-0.4403, -0.4024] | yes | yes |
| MR verdict without PMC vs zero-shot pplx-decider-flash-v1.2 | 5003 | +0.1643 [+0.1467, +0.1817] | -0.3311 [-0.3523, -0.3100] | yes | yes |
| MR reported-only templates vs MedDecider-4B | NOT MEASURED - mr_reported_only vs MedDecider-4B: no paired items | | | | |
| MR reported-only templates vs zero-shot pplx-decider-flash-v1.2 | 6191 | +0.0040 [+0.0005, +0.0091] | -0.0026 [-0.0038, -0.0016] | yes | yes |
| held-out MR test vs MedDecider-4B | 70 | +0.3098 [+0.2047, +0.4257] | -0.3908 [-0.5071, -0.2864] | yes | yes |
| held-out MR test vs zero-shot pplx-decider-flash-v1.2 | 70 | +0.2038 [+0.1087, +0.3071] | -0.3956 [-0.5497, -0.2496] | yes | yes |
| MR verdict vs zero-shot Qwen3.5-4B (not a verdict baseline under D39) | 13103 | +0.3054 [+0.2874, +0.3234] | -0.4037 [-0.4136, -0.3935] | yes | yes |
| external panel vs zero-shot Qwen3.5-4B (not a verdict baseline under D39) | 4353 | +0.0700 [+0.0526, +0.0880] | -0.0849 [-0.1005, -0.0693] | yes | yes |
| knowledge test vs zero-shot Qwen3.5-4B (full set) | 5456 | +0.0687 [+0.0550, +0.0812] | | yes | |
| knowledge test vs zero-shot pplx, items pplx can score (21 skipped) | 5435 | +0.0018 [-0.0063, +0.0110] | | yes | |
| MR verdict, osler_v1's Osler-4B (V10) vs MedDecider-4B | 12992 | +0.0805 [+0.0700, +0.0916] | +0.0351 [+0.0287, +0.0414] | yes | yes |
| MR verdict, osler_v1's Osler-4B (V10) vs zero-shot pplx-decider-flash-v1.2 | 13103 | -0.0274 [-0.0413, -0.0147] | +0.0996 [+0.0915, +0.1083] | no | no |

d1-3B: NOT MEASURED - not run in this loop (ADVISORY W8 lists it as an additional row).

