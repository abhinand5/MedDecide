# Blocks v2 (osler_v2 W6): base choice, C0, keep decisions

Aggregates only (no item text). Recompute: `scripts/osler_v2/w6_keep.py` (`--base`, `--candidate ... --current ...`,
`--knowledge-check ... --zs ...`) on `outputs/osler_v2/W6/<block>/dev_predictions.jsonl`. Every number has a row in
`loops/osler_v2/CLAIMS.md` (W043 onwards). Written 2026-10-17 before C1 started.

## 1. Base choice (D39), fixed before any C1 run

Rule (D39): C0p (pplx-decider-flash-v1.2) minus C0q (Qwen3.5-4B) on fast dev v2, paired bootstrap within templates, 1,000
resamples, seed 0. pplx is chosen iff the lower bounds are > −0.01 on MR, > −0.01 on general and > −0.02 on knowledge, **and** the
MR or general point difference is > 0.

| partition (items) | point difference | 95 % bootstrap interval | check |
|---|---|---|---|
| MR (3,597) | +0.0378 | [+0.0283, +0.0471] | lower > −0.01: pass |
| general (7,726) | +0.0045 | [−0.0063, +0.0150] | lower > −0.01: pass |
| knowledge (4,500) | +0.0918 | [+0.0793, +0.1038] | lower > −0.02: pass |

Point difference positive on MR and general: yes. **Chosen base: pplx-decider-flash-v1.2.** C1 and C2 run on pplx. Gate O3
(W8) uses zero-shot pplx-decider-flash-v1.2 in place of zero-shot Qwen3.5-4B (D39); MedDecider-4B stays.

Source: `outputs/osler_v2/W6/keep/base_choice.json`.

Partition note (D-W5-7): the keep rule's MR partition is osler_v1's `MR_SOURCES`. MR (3,597) = 3,997 MR-prefixed fast-dev items
minus the 400 EHR-Rel items, which the code puts in general. General (7,726) = 2,826 general + 400 EHR-Rel + 4,500 knowledge (the
knowledge items are also general, as in the code).

## 2. C0 (the first block; both backbones; same data, adapter and steps)

Training file: `data/train/osler_v2/block_C0.jsonl` = osler_v1 B1 (771,009 − 74,477 = 696,532 rows of B1) + the fixed
share-alike replay (74,477 rows). Replay share of C0: 22.65 %. Recipe: attention-only LoRA (r 32, alpha 32), batch 8, 8,192
tokens, gradient checkpointing above 8,192 padded tokens, 13,992 steps, one dev evaluation per quarter on the whole fast dev v2.

| run | selected checkpoint | fast dev v2 accuracy (11,323) | template macro | Brier | validation |
|---|---|---|---|---|---|
| C0q (Qwen3.5-4B) | step 13,992 (final) | 0.7849 | 0.8589 | 0.3136 | 13,992 steps as planned; 279 logged steps, 0 mismatches |
| C0p (pplx-flash) | step 10,494 | 0.8355 | 0.8702 | 0.2556 | 13,992 steps as planned; 279 logged steps, 0 mismatches |

## 3. C0 knowledge checks (knowledge dev v2; D12-gated; margin −0.02 on the lower bound)

Knowledge dev v2 = 3,988 items (4,000 minus 12 with a repeated option label; excluded on both sides).

| block | zero-shot base | block − zero-shot (point) | 95 % interval | passes (lower > −0.02) |
|---|---|---|---|---|
| C0q | zero-shot Qwen3.5-4B (D12 PASS) | +0.0203 | [+0.0078, +0.0321] | yes |
| C0p | zero-shot pplx-decider-flash-v1.2 (reference readout; D12 not applicable) | −0.0013 | [−0.0095, +0.0070] | yes |

Sources: `outputs/osler_v2/W6/keep/knowledge_C0_qwen_vs_qwen.json`, `outputs/osler_v2/W6/keep/knowledge_C0_pplx_vs_pplx.json`.

## 4. Keep rule for C1 and C2 (D34; decided after each run)

The candidate is kept iff the lower bounds of (candidate − last kept block) are > 0 on MR, > −0.01 on general and > −0.02 on
knowledge (paired bootstrap, 1,000, seed 0). The last kept block after C0 is C0p (the base). C1 = C0p-base + real text; C2 =
last kept + teacher prose (indicator and narrative; 37,222 train rows). See the sections below as the runs finish.

## 4a. C1 (pplx backbone; decided after the run)

C1 = C0 + real text (MTSamples 668, CodiEsp 2,908 after the carve-out, PMC 52,549, EHR-Rel train 2,942); 13,992 steps; validation
as planned (0 log mismatches). Fast dev v2 full: accuracy 0.8602, template macro 0.8880, Brier 0.2025 (C0p: 0.8355, 0.8702, 0.2556).

Keep rule, C1 against C0p (the last kept block):

| partition | point | 95 % interval | lower bound vs margin | check |
|---|---|---|---|---|
| MR (3,597) | +0.0750 | [+0.0645, +0.0855] | > 0 | pass |
| general (7,726) | +0.0032 | [−0.0043, +0.0105] | > −0.01 | pass |
| knowledge (4,500) | +0.0027 | [−0.0056, +0.0111] | > −0.02 | pass |

**C1 is kept.** Source: `outputs/osler_v2/W6/keep/C1_pplx_vs_C0_pplx.json`. C2 (C1 + teacher prose) started after this decision.

## 4b. C2 (C1 + teacher prose; decided after the run)

C2 = C1 + teacher prose (indicator 23,974 and narrative 13,248 train rows); 13,992 steps; validation as planned. Fast dev v2 full:
accuracy 0.8560, template macro 0.8783, Brier 0.2122 (C1: 0.8602, 0.8880, 0.2025).

Keep rule, C2 against C1 (the last kept block):

| partition | point | 95 % interval | lower bound vs margin | check |
|---|---|---|---|---|
| MR (3,597) | −0.0042 | [−0.0121, +0.0037] | > 0 | fail |
| general (7,726) | −0.0112 | [−0.0197, −0.0027] | > −0.01 | fail |
| knowledge (4,500) | −0.0051 | [−0.0131, +0.0029] | > −0.02 | pass |

**C2 is not kept.** The teacher prose lowers MR and general on fast dev v2 against C1 (the osler_v1 B2 finding repeats). The kept
mix is therefore **C1** (the final mix, W7). Source: `outputs/osler_v2/W6/keep/C2_pplx_vs_C1_pplx.json`.

## 5. Replay share per block (the W6 guard for the fixed replay set)

| block | rows | replay share (v1 + v2 share-alike rows) | note |
|---|---|---|---|
| C0 | 771,009 | 22.65 % | fixed set (74,477 v2 rows) |
| C1 | 830,076 | 21.04 % | + MTSamples 668, CodiEsp 2,908 (carve-out removed), PMC 52,549, EHR-Rel 2,942 |
| C2 | 867,298 | 20.13 % | + teacher prose 37,222 |

No chart rows exist (W3b BLOCKED), so C3 is not built. The largest block, C2, keeps the 20 % target (not kept, so not the final mix).
