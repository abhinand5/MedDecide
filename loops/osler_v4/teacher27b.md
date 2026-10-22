# osler_v4 Y5 — pplx-decider-v1.2-27b soft labels (aggregates only)

Teacher: `perplexity-ai/pplx-decider-v1.2-27b@cce0f3f80033e9d89d80831d5281fd72124aacce`; licence: Apache-2.0 (pinned snapshot). Temperature (saved): 1.3823494677922652.

## Parity (40 CG v2 items, dev)

| comparison | argmax agreement | max abs diff | mean abs diff |
|---|---|---|---|
| reference_batched vs reference_single | 40/40 | 0.03532 | 0.001922 |
| fast_batched vs reference_single | 40/40 | 0.02407 | 0.001372 |

Scoring path: `reference_batched`. Seconds: reference_single 3.9, reference_batched 6.4, fast_batched 35.7.

## Anchor pool

Eligible unique anchor rows after exclusions: 870161; written: 400000 (cap 400000). Counts:

- excluded: overlaps a benchmark row (record): 854
- excluded: overlaps a benchmark row (text and record): 4748
- excluded: overlaps a benchmark row (text): 1685
- kept: item id shared with an earlier pool row (different text): 22104
- not an anchor (domain template prefix): 101725
- rows: block_C1.jsonl: 830076
- rows: replay_broad_train.jsonl: 162797
- skipped: cannot be sent exactly (repeated_label): 7564
- skipped: same text as an earlier pool row: 6136

## Teacher accuracy on dev diagnostics (not used for any choice)

| set | qtype | items | accuracy |
|---|---|---|---|
| union | all | 9122 | 0.8516 |
| union | choice | 8894 | 0.8554 |
| union | noul | 195 | 0.7487 |
| union | score | 33 | 0.4242 |
| cg_v2 | all | 8296 | 0.8498 |
| cg_v2 | choice | 8068 | 0.8540 |
| cg_v2 | noul | 195 | 0.7487 |
| cg_v2 | score | 33 | 0.4242 |
| knowledge_dev_v2 | all | 3988 | 0.8601 |
| knowledge_dev_v2 | choice | 3988 | 0.8601 |

## Soft labels joined

Overlap with benchmark rows: 0.

- pool row not scored (time budget): 287872
- soft label written: 112128
