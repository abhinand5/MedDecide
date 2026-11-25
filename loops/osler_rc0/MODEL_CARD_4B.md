# Osler-4B-rc0 — model card (RC0)

**Status:** release candidate staged locally on 2026-11-25T05:29:32Z; not uploaded. Staged tree: `/workspace/MedDecide/outputs/osler_rc0/R8/release_staging/osler-4b-rc0` (weights, calibration, provenance manifest; no training data, no MIMIC text).

## What it is

`Osler-4B-rc0` is a calibrated medical decision model: given a state (a note, a trial record, a question, or a query + passage) and a list of typed questions, it returns a probability for every offered option in one prefill pass — no text generation. Question types: `noul` (yes/no), `choice` (2-255 named options) and `score` (2-10 ordered levels).

- **Base model:** `perplexity-ai/pplx-decider-flash-v1.2` at revision `9ff7cf9838e59df461c276b7699d56b244d2af58`
- **Architecture:** pplx Qwen3_5Model backbone with LoRA (r=32) and a separate option-code readout; noncausal full attention
- **Calibration (dev-only, fastdev_v4):** choice 1.522233, noul 2.374354, score 1.538844
- **Licence:** Apache-2.0. The base's LICENSE and NOTICE travel with the project; no MIMIC text or training rows are in the release tree.
- **Serving:** one prefill pass, greedy readout over the offered option codes; no sampling, no generation.

```python
# Osler-4B-rc0: one prefill pass over the offered option codes
from meddecide.model.pplx import PplxDecisionModel
model = PplxDecisionModel.load('staged/weights', device='cuda')
model.eval_mode()
decisions = model.predict(items, batch_size=16)  # temperature from model.json is applied
p_yes = decisions[0].probs  # aligned with the item's option keys
```

## Headline measurements (12-model comparison, 9 frozen sets)

| set | accuracy | 95% CI | template macro | Brier | ECE | coverage |
|---|---|---|---|---|---|---|
| `panel` | 0.5502 | [0.5353, 0.5642] | 0.6205 | 0.5643 | 0.0374 | 1.000 |
| `knowledge` | 0.6728 | [0.6600, 0.6848] | 0.6995 | 0.4250 | 0.0194 | 1.000 |
| `confirm_general` | 0.8883 | [0.8806, 0.8963] | 0.8030 | 0.1792 | 0.0940 | 1.000 |
| `mr_verdict` | 0.8941 | [0.8890, 0.8997] | 0.8499 | 0.1514 | 0.0314 | 1.000 |
| `mimic_suite` | 0.7597 | [0.7470, 0.7722] | 0.7660 | 0.3684 | 0.0818 | 1.000 |

## Paired differences against its own zero-shot base

| set | Δ accuracy | 95% CI |
|---|---|---|
| `panel` | -0.0113 | [-0.0218, -0.0007] |
| `knowledge` | -0.0098 | [-0.0191, -0.0017] |
| `confirm_general` | -0.0007 | [-0.0077, +0.0068] |
| `mr_verdict` | +0.1474 | [+0.1400, +0.1552] |
| `mimic_suite` | +0.1357 | [+0.1213, +0.1505] |

## 12-model Pareto summary

This model is on the 2-axis Pareto front (OOD General Avg vs Clinical Audit Avg): **no**; undominated on all five headline sets among <=10B models: **no**. Full frontier tables: `loops/osler_rc0/PARETO.md`.

| model | params (B) | 5-set macro | panel | knowledge | confirm_general | mr_verdict | mimic_suite |
|---|---|---|---|---|---|---|---|
| Osler-9B-rc0 | 8.39 | 0.7801 | 0.5927 | 0.7322 | 0.9008 | 0.9016 | 0.7733 |
| pplx-27B | 27.00 | 0.7600 | 0.6510 | 0.7823 | 0.8843 | 0.8093 | 0.6729 |
| Osler-4B-rc0 | 4.00 | 0.7530 | 0.5502 | 0.6728 | 0.8883 | 0.8941 | 0.7597 |
| d3-flash | 8.39 | 0.7132 | 0.5899 | 0.7335 | 0.8774 | 0.7351 | 0.6301 |
| pplx-flash | 4.00 | 0.7009 | 0.5615 | 0.6835 | 0.8889 | 0.7466 | 0.6240 |
| Clef-Flash | 9.00 | 0.6929 | 0.5711 | 0.6864 | 0.8933 | 0.6831 | 0.6306 |
| MedDecider-9B | 9.00 | 0.6853 | 0.5603 | 0.6997 | 0.8825 | 0.6653 | 0.6189 |
| MedDecider-4B | 4.00 | 0.6732 | 0.5436 | 0.6703 | 0.8820 | 0.6681 | 0.6019 |

## Efficiency

- Parameters: 4.00 B
- Batch-1 p50 latency: 26.5430 ms/item (p95 63.5540 ms/item)
- Batch-16 throughput: 21.3450 items/s
- Peak allocated VRAM (batch 1): 9.6000 GB

## MIMIC PHI / memorisation audit (D28; local only)

- Visited-member membership AUC (members visited by both release candidates vs late-anchor `mimic_suite` non-members, n=2000+2000): **0.5644** [0.5466, 0.5813]; base control 0.5396; difference 0.0248 [0.0097, 0.0397].
- Y10 historical compatibility set AUC: 0.5033 (osler_v4 reference 0.5042).
- 2,000 members sampled from 12,993 MIMIC rows visited by both runs; the training mix has 539,807 MIMIC-derived rows (35.4 % of 1,526,405).
- Verbatim-extraction probe on 300 MIMIC discharge notes: 0 matches of ≥50 consecutive tokens for the adapter-on causal export; the zero-shot `pplx-flash` control on the same notes: 0.
- Adapter-disabled generation is token-for-token identical to the untouched `pplx-flash` base on 20 fixed prompts (adapter-on control differs).

## Honest limitations

- General OOD QA versus its own zero-shot base: `panel` -0.0113 [-0.0218, -0.0007], `knowledge` -0.0098 [-0.0191, -0.0017] — **below the base by about one point, both intervals excluding zero**; the model trades general-OOD accuracy for clinical-audit gains and better Brier on four of five headline sets.
- **Held-out failure, reported:** on `mr_heldout_test` / `synthea_indicator_evidence_v1` (n=514, a template excluded from training) both release candidates read below chance (9B accuracy 0.356, CI upper 0.3988; 4B 0.2451, upper 0.2821; chance 0.5) while their bases pass the same cell (d3-flash 0.4591, pplx-flash 0.5973). The cell is flagged `READOUT_FAIL` and is not quoted as an accuracy anywhere in this card.
- Dev-only calibration does not transfer to the fresh post-cutoff set: mean confidence on `confirm_general` is 0.8137 (9B) / 0.8005 (4B) against accuracies 0.9008 / 0.8883, so the ECE there is 0.0900 / 0.0940 versus 0.0192 on the dev fit set (the models are under-confident on the fresh distribution). No test-set temperature was fitted.
- The 27B comparison model is stronger on hard multi-step medical QA (for example MedXpertQA); the release candidate is not claimed to beat it.
- Held-out transfer (`mimic_heldout`, `mr_heldout_test`, `mednli_test`, `pairs_test`) is reported as measured, with wide intervals on the small sets.
- Not for unsupervised clinical decisions. Outputs are probabilities from a research model; a clinician remains responsible for any decision.
- MIMIC: trained on credentialed data locally; no MIMIC text was sent to any hosted service, and the release tree contains no MIMIC text.

## Provenance

- Staged and verified 2026-11-25T05:29:32Z at git `3d82a407c72ed779ef7a6340684b6259d5c00353`; probability parity max |Δp| 0 with 20/20 argmax agreement on 20 fixed dev items.
- Full manifest with SHA-256 of every staged file: `provenance_manifest.json` in the staged tree.

