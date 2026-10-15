# osler_v1 — V8 blocks, run results and keep decisions

Written 2026-10-15T08:11:10Z by `scripts/osler_v1/v8_blocks_report.py` from the V7 manifest (`outputs/osler_v1/V7/blocks.json`), the V8 block results and the keep records. Aggregates only. A run that has not finished reads `pending`.

## 1. Block contents (V7)

| block | adds | rows | replay share | largest template | tabular page share | leakage hits (text, bare, qualified) | teacher prose final |
|---|---|---|---|---|---|---|---|
| B0 | mix v2 train plus the V3 replay (HellaSwag, WinoGrande, CommonsenseQA, QASC) | 653,874 | 15.31 % | 6.48 % | 0.00 % | 0, 0 | not applicable |
| B1 | + V1 generators and V5a Synthea items (tabular page category capped at 2 %) | 696,532 | 14.38 % | 6.08 % | 2.00 % | 0, 0 | not applicable |
| B2 | + V4 teacher prose: indicator (train pairs) and narrative pages | 734,514 | 13.63 % | 5.77 % | 2.00 % | 0, 0 | yes |
| B3 | + V5b real text: MTSamples, CodiEsp train, PMC-Patients pairs | 792,260 | 12.64 % | 6.63 % | 2.00 % | 0, 0 | yes |

The replay is the same 100,132 rows in every block (HellaSwag 39,905, WinoGrande 40,398, CommonsenseQA 10,765, QASC 9,064); its share falls as teacher rows are added. The 2 % tabular cap (V7) and the 8 % template cap (D17) hold in every block; the narrative pages (V4-D6) are counted under their own source, not under the tabular cap.

## 2. Runs: steps, realised items, tokens and the selected checkpoint

| run | steps | evaluations | realised items | real tokens | padded tokens | selected step | status |
|---|---|---|---|---|---|---|---|
| B0 | 13,992 | 4 | 80,008 | 69,413,742 | 76,590,400 | 13,992 | done |
| B0b | 13,992 | 4 | 80,008 | 69,413,742 | 76,590,400 | 13,992 | done |
| B1 | 13,992 | 4 | 80,851 | 68,687,777 | 78,513,024 | 13,992 | done |
| B2 | pending | | | | | | not run |
| B3 | pending | | | | | | not run |

Every run uses the same step count (13,992 optimiser steps, the 80,000-item replay of B0; V8-D4) and the same recipe (Qwen3.5-4B, the option-code head, the batching constant of V0). B0b changes only the LoRA targets (attention projections).

## 3. Dev results of each run (the full fast dev set, 5,723 items)

| run | accuracy | template macro | Brier | mean NLL | temperatures fitted on dev |
|---|---|---|---|---|---|
| B0 | 0.8548 | 0.8582 | 0.2165 | 0.4080 | choice, noul, score |
| B0b | 0.8406 | 0.8573 | 0.2490 | 0.5351 | choice, noul, score |
| B1 | 0.9037 | 0.8781 | 0.1409 | 0.2696 | choice, noul, score |

The keep rule compares partitions (MR, general, knowledge), not these full-dev figures; the partition statistics are in section 4.

## 4. Keep rule (ADVISORY V8; paired bootstrap within templates, 1,000 resamples, seed 0)

A block X is kept against the current C iff the lower bound of X minus C is above 0 on MR, above -0.01 on general and above -0.02 on knowledge.

| candidate vs C | partition | items | difference [lower, upper] | margin | passes |
|---|---|---|---|---|---|
| B1 vs B0 | mr | 2,397 | +0.0947 [+0.0807, +0.1098] | +0.00 | yes |
| B1 vs B0 | general | 3,326 | +0.0062 [-0.0048, +0.0177] | -0.01 | yes |
| B1 vs B0 | knowledge | 500 | +0.0480 [+0.0080, +0.0900] | -0.02 | yes |
| B2 vs B1 | | pending | | | |
| B3 vs B2 | | pending | | | |

## 5. Knowledge checks (B0 and B0b against zero-shot Qwen3.5-4B, 500 fast-dev items; D12 gate)

| block | items | difference [lower, upper] | margin | status | passes |
|---|---|---|---|---|---|
| B0 | 500 | -0.0600 [-0.1020, -0.0200] | -0.02 | MEASURED | no |
| B0b | 500 | -0.0080 [-0.0420, +0.0260] | -0.02 | MEASURED | no |

## 6. Decisions and the current C

- Knowledge checks: B0: fails; B0b: fails. C stays B0 when B0 and B0b both fail (ADVISORY V8).
- B1 against B0: KEPT; B2 against B1: pending; B3 against B2: pending.

