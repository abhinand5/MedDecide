# Baselines (osler_v2 W5) — aggregates only

Written from `outputs/osler_v2/W5/baselines_summary.json` (recompute: `uv run --frozen python scripts/osler_v2/w5_baselines_summary.py`).
No item text. Every number is a row in `loops/osler_v2/CLAIMS.md` (W037–W041).

## Knowledge dev v2 (zero-shot; the C0 knowledge checks use these)

Items: 4,000 knowledge dev v2 items minus 12 with a repeated option label (pplx's reference criteria are keyed by label) = 3988 scored, on both sides.

| model | accuracy | readout (D12) |
|---|---|---|
| zero-shot Qwen3.5-4B | 0.6642 | PASS (MedQA, MedMCQA) |
| zero-shot pplx-decider-flash-v1.2 | 0.7728 | reference readout (not a letter readout; D12 not applicable) |

## EHR-Rel test (new template `ehr_rel_relatedness_v1`, 373 items, four ordered levels)

Chance (uniform over four levels) is 0.25. The ordinal error is the mean absolute difference between the expected level and the gold level (levels 1–4).

| model | accuracy | mean abs. error (levels) |
|---|---|---|
| zero-shot Qwen3.5-4B | 0.3995 | 1.5243 |
| zero-shot Qwen3.5-9B | 0.5067 | 1.4323 |
| MedDecider-4B | 0.5442 | 1.1765 |
| MedDecider-9B | 0.4906 | 1.3926 |
| JEV-9B | 0.2386 | 1.6273 |

## Not yet measured in W5

- zero-shot pplx-decider-flash-v1.2 on the MR suite v1 (test and held-out), the external panel and the knowledge test items: queued after the C0 stage (`scripts/osler_v2/w5_pplx_test.sh`); needed only at W8 if D39 chooses pplx.
- d1-3B (an additional row, not in the chain): NOT MEASURED in this session.
- Qwen3.5-9B and JEV-9B on the MR suite v1 and the knowledge test items: from osler_v0/osler_v1 outputs at W8 (not re-run here).
