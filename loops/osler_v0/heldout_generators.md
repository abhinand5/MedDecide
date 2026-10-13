# osler_v0 — held-out generators (O11; additional, never in a verdict)

The two locked held-out generators (`loops/osler_v0/heldout.md`): the 600 test items of `note_lab_range_v1` (choice, 300 items) and `policy_triage_v1` (score, 300 items). Held-out results are reported here and never pooled with the seen generators (heldout.md rule 3). Choice items use both option orders averaged; score items have one order. Intervals: 95 % percentile bootstrap over the items, resampled within each generator (2,000 resamples, seed 0).

**Baselines: NOT MEASURED.** The four D24 baselines (MedDecider-4B, zero-shot Qwen3.5-4B, MedDecider-9B, JEV-9B) were not scored on these items, so the held-out verdicts are not computed (deviation 58).

| model | template macro accuracy (95 % CI) | note_lab_range_v1 | policy_triage_v1 | Brier (mean) | majority baseline (template macro) | single-order template macro |
|---|---|---|---|---|---|---|
| osler_4b_L | 0.7267 (0.6917 to 0.7600) | 0.8567 (n 300) | 0.5967 (n 300) | 0.3414 | 0.5000 | 0.7233 |
| osler_4b_P | 0.7767 (0.7433 to 0.8067) | 0.8767 (n 300) | 0.6767 (n 300) | 0.3633 | 0.5000 | 0.7767 |
| osler_4b_N | 0.7683 (0.7350 to 0.8000) | 0.9033 (n 300) | 0.6333 (n 300) | 0.3469 | 0.5000 | 0.7667 |
| osler_9b | 0.8383 (0.8117 to 0.8650) | 0.9900 (n 300) | 0.6867 (n 300) | 0.2250 | 0.5000 | 0.8383 |
| osler_0p8b | 0.4583 (0.4200 to 0.4983) | 0.5333 (n 300) | 0.3833 (n 300) | 0.7731 | 0.5000 | 0.4650 |

Paired differences (additional; template macro, items resampled within generator):

| comparison | difference | 95 % CI | items |
|---|---|---|---|
| osler_4b_P - osler_4b_L | +0.0500 | +0.0150 to +0.0850 | 600 |
| osler_4b_N - osler_4b_L | +0.0417 | +0.0033 to +0.0800 | 600 |
| osler_9b - osler_4b_L | +0.1117 | +0.0717 to +0.1517 | 600 |
