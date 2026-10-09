# Tier-1 contamination probe (F5 / T4)

Generated: `2026-10-06T09:04:43Z`  
Probe: Min-K% Prob, K=0.2, over the item's state text; control = deterministic reorder of the same text (sentences, else clauses, else lines)  
Sample: up to 500 tier-1 **test** items per source, seed 0.

> **Probes, not proofs.** A score gap between the original text and a sentence-reordered
> control is *consistent with* verbatim memorisation, but the original ordering can simply
> be more predictable prose. A small gap does not prove a model never saw the data. No
> tier-1 accuracy in this loop is adjusted by these numbers.

Min-K% Prob is the mean log-probability of the least likely 20 % of the text's tokens under
the model, so **less negative = easier to predict**. The control reorders the *same* text at
the finest available granularity - sentences, else clauses, else lines - keeping every
token; the granularity used per source is in `control_kinds`. Items with only one sentence,
clause and line have no control and are excluded from the gap (counted in the JSON).

## Mean Min-K% score, control, and gap by model and source

| model | source | n scored | score | control | gap |
|---|---|---|---|---|---|
| `LiquidAI/LFM2.5-350M` | medmcqa | 500 | -20.3213 | -19.6501 | -0.6711 |
| `LiquidAI/LFM2.5-350M` | medqa | 500 | -16.9134 | -16.5361 | -0.3774 |
| `LiquidAI/LFM2.5-350M` | medquad | 500 | -25.8188 | -20.4144 | -5.4044 |
| `LiquidAI/LFM2.5-350M` | mmlu_medical | 500 | -18.6811 | -17.3625 | -1.3186 |
| `LiquidAI/LFM2.5-350M` | nfcorpus | 500 | -18.6937 | -17.2714 | -1.4223 |
| `LiquidAI/LFM2.5-350M` | pubmedqa | 477 | -16.6650 | -16.6908 | +0.0258 |
| `LiquidAI/LFM2.5-350M` | scifact | 500 | -19.6254 | -17.6486 | -1.9768 |
| `LiquidAI/LFM2.5-350M` | trec_covid | 195 | -18.4077 | -17.4167 | -0.9910 |
| `Qwen/Qwen3.5-0.8B-Base` | medmcqa | 500 | -9.2550 | -9.3815 | +0.1264 |
| `Qwen/Qwen3.5-0.8B-Base` | medqa | 500 | -5.6480 | -6.3262 | +0.6782 |
| `Qwen/Qwen3.5-0.8B-Base` | medquad | 500 | -8.9207 | -12.1129 | +3.1922 |
| `Qwen/Qwen3.5-0.8B-Base` | mmlu_medical | 500 | -7.1136 | -7.1266 | +0.0131 |
| `Qwen/Qwen3.5-0.8B-Base` | nfcorpus | 500 | -6.7726 | -7.4879 | +0.7152 |
| `Qwen/Qwen3.5-0.8B-Base` | pubmedqa | 477 | -6.1131 | -6.5879 | +0.4748 |
| `Qwen/Qwen3.5-0.8B-Base` | scifact | 500 | -6.8902 | -7.5365 | +0.6462 |
| `Qwen/Qwen3.5-0.8B-Base` | trec_covid | 195 | -7.2040 | -7.8853 | +0.6812 |
| `Qwen/Qwen3.5-0.8B` | medmcqa | 500 | -9.4843 | -9.8311 | +0.3468 |
| `Qwen/Qwen3.5-0.8B` | medqa | 500 | -6.0519 | -6.7529 | +0.7010 |
| `Qwen/Qwen3.5-0.8B` | medquad | 500 | -9.5318 | -12.4857 | +2.9540 |
| `Qwen/Qwen3.5-0.8B` | mmlu_medical | 500 | -7.4232 | -7.5646 | +0.1414 |
| `Qwen/Qwen3.5-0.8B` | nfcorpus | 500 | -7.4916 | -8.0216 | +0.5301 |
| `Qwen/Qwen3.5-0.8B` | pubmedqa | 477 | -6.6045 | -7.0514 | +0.4469 |
| `Qwen/Qwen3.5-0.8B` | scifact | 500 | -7.6711 | -8.0972 | +0.4261 |
| `Qwen/Qwen3.5-0.8B` | trec_covid | 195 | -7.9288 | -8.4520 | +0.5232 |
| `google/medgemma-1.5-4b-it` | medmcqa | 500 | -17.9771 | -16.1468 | -1.8303 |
| `google/medgemma-1.5-4b-it` | medqa | 500 | -10.0724 | -10.8516 | +0.7792 |
| `google/medgemma-1.5-4b-it` | medquad | 500 | -19.0360 | -19.0949 | +0.0589 |
| `google/medgemma-1.5-4b-it` | mmlu_medical | 500 | -15.1534 | -12.5182 | -2.6351 |
| `google/medgemma-1.5-4b-it` | nfcorpus | 500 | -10.9566 | -11.7007 | +0.7440 |
| `google/medgemma-1.5-4b-it` | pubmedqa | 477 | -10.4669 | -10.9151 | +0.4482 |
| `google/medgemma-1.5-4b-it` | scifact | 500 | -11.7121 | -12.1312 | +0.4191 |
| `google/medgemma-1.5-4b-it` | trec_covid | 195 | -11.4212 | -12.1858 | +0.7647 |
| `Qwen/Qwen3.5-4B` | medmcqa | 500 | -8.9191 | -9.1133 | +0.1942 |
| `Qwen/Qwen3.5-4B` | medqa | 500 | -5.3387 | -6.0839 | +0.7452 |
| `Qwen/Qwen3.5-4B` | medquad | 500 | -8.9334 | -12.4144 | +3.4810 |
| `Qwen/Qwen3.5-4B` | mmlu_medical | 500 | -6.7907 | -6.8826 | +0.0919 |
| `Qwen/Qwen3.5-4B` | nfcorpus | 500 | -6.6457 | -7.1893 | +0.5436 |
| `Qwen/Qwen3.5-4B` | pubmedqa | 477 | -5.5956 | -6.1430 | +0.5474 |
| `Qwen/Qwen3.5-4B` | scifact | 500 | -6.7823 | -7.2158 | +0.4334 |
| `Qwen/Qwen3.5-4B` | trec_covid | 195 | -7.4043 | -7.7352 | +0.3309 |
| `Qwen/Qwen3.5-9B` | medmcqa | 500 | -8.7068 | -8.8871 | +0.1803 |
| `Qwen/Qwen3.5-9B` | medqa | 500 | -5.0392 | -5.7732 | +0.7341 |
| `Qwen/Qwen3.5-9B` | medquad | 500 | -8.5211 | -12.0426 | +3.5215 |
| `Qwen/Qwen3.5-9B` | mmlu_medical | 500 | -6.5733 | -6.6078 | +0.0345 |
| `Qwen/Qwen3.5-9B` | nfcorpus | 500 | -6.5861 | -6.9555 | +0.3695 |
| `Qwen/Qwen3.5-9B` | pubmedqa | 477 | -5.3286 | -5.8789 | +0.5504 |
| `Qwen/Qwen3.5-9B` | scifact | 500 | -6.6098 | -6.9888 | +0.3790 |
| `Qwen/Qwen3.5-9B` | trec_covid | 195 | -7.1928 | -7.5365 | +0.3437 |

## Summary

| model | mean gap across sources |
|---|---|
| `LiquidAI/LFM2.5-350M` | -1.5170 |
| `Qwen/Qwen3.5-0.8B-Base` | +0.8159 |
| `Qwen/Qwen3.5-0.8B` | +0.7587 |
| `google/medgemma-1.5-4b-it` | -0.1564 |
| `Qwen/Qwen3.5-4B` | +0.7959 |
| `Qwen/Qwen3.5-9B` | +0.7641 |

## How to read this

* A **negative** gap means the reordered control was *easier* to predict than the original
  ordering; that is the expected direction for prose whose original order is informative.
* The probe is informative only in comparison: a model whose gap is much larger than the
  others' on a given source is the one a reviewer should suspect of having seen that source.
* Tier-1 numbers in `FINDINGS.md` are reported as they are measured, with this caveat beside
  them; the contamination-resistant claim rests on **tier 2**, not on this probe.

