# Osler-9B-rc0 — model card (RC0)

**Status:** release candidate staged locally on 2026-11-25T05:29:32Z; not uploaded. Staged tree: `/workspace/MedDecide/outputs/osler_rc0/R8/release_staging/osler-9b-rc0` (weights, calibration, provenance manifest; no training data, no MIMIC text).

## What it is

`Osler-9B-rc0` is a calibrated medical decision model: given a state (a note, a trial record, a question, or a query + passage) and a list of typed questions, it returns a probability for every offered option in one prefill pass — no text generation. Question types: `noul` (yes/no), `choice` (2-255 named options) and `score` (2-10 ordered levels).

- **Base model:** `vllm-sr/d3-flash` at revision `67e03e1bc157961c582ee7c6a35d9b15fada3dec`
- **Architecture:** Qwen3_5Model with attention-only LoRA (r=32) and a [255, 4096] option-code readout; non-generative (no lm_head)
- **Calibration (dev-only, fastdev_v4):** choice 1.527412, noul 2.318485, score 1.703540
- **Licence:** Apache-2.0. The base's LICENSE and NOTICE travel with the project; no MIMIC text or training rows are in the release tree.
- **Serving:** one prefill pass, greedy readout over the offered option codes; no sampling, no generation.

```python
# Osler-9B-rc0: one prefill pass over the offered option codes
import torch
from transformers import AutoModel
from meddecide.distill.d3_train import D3Student, prepare_rows
snapshot = '/path/to/vllm-sr--d3-flash@67e03e1b'
runtime = AutoModel.from_pretrained(snapshot, trust_remote_code=True, dtype=torch.bfloat16,
                                   device='cuda:0').runtime
student = D3Student(runtime, lora_r=32, lora_alpha=32).to(runtime.device)
state = torch.load('staged/weights/osler_9b_rc0.pt', map_location=student.device)['model']
student.load_checkpoint_state(state)  # LoRA + [255, 4096] option-code readout
student.eval()
prepared = prepare_rows(runtime, rows)  # rows: state / question / options / gold
probs = student.probabilities([list(r.sequence) for r in prepared],
                              [r.count for r in prepared], batch_size=16)
# apply the per-qtype temperatures in calibration.json to the option logits before softmax
```

## Headline measurements (12-model comparison, 9 frozen sets)

| set | accuracy | 95% CI | template macro | Brier | ECE | coverage |
|---|---|---|---|---|---|---|
| `panel` | 0.5927 | [0.5785, 0.6076] | 0.6650 | 0.5223 | 0.0229 | 1.000 |
| `knowledge` | 0.7322 | [0.7209, 0.7438] | 0.7535 | 0.3587 | 0.0441 | 1.000 |
| `confirm_general` | 0.9008 | [0.8933, 0.9083] | 0.8153 | 0.1673 | 0.0900 | 1.000 |
| `mr_verdict` | 0.9016 | [0.8966, 0.9065] | 0.8654 | 0.1421 | 0.0220 | 1.000 |
| `mimic_suite` | 0.7733 | [0.7606, 0.7863] | 0.7797 | 0.3373 | 0.0687 | 1.000 |

## Paired differences against its own zero-shot base

| set | Δ accuracy | 95% CI |
|---|---|---|
| `panel` | +0.0028 | [-0.0069, +0.0129] |
| `knowledge` | -0.0013 | [-0.0090, +0.0062] |
| `confirm_general` | +0.0234 | [+0.0180, +0.0291] |
| `mr_verdict` | +0.1665 | [+0.1590, +0.1738] |
| `mimic_suite` | +0.1432 | [+0.1281, +0.1580] |

## 12-model Pareto summary

This model is on the 2-axis Pareto front (OOD General Avg vs Clinical Audit Avg): **yes**; undominated on all five headline sets among <=10B models: **yes**. Full frontier tables: `loops/osler_rc0/PARETO.md`.

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

- Parameters: 8.39 B
- Batch-1 p50 latency: 33.7140 ms/item (p95 93.1080 ms/item)
- Batch-16 throughput: 16.7970 items/s
- Peak allocated VRAM (batch 1): 17.3230 GB

## MIMIC PHI / memorisation audit (D28; local only)

- Visited-member membership AUC (members visited by both release candidates vs late-anchor `mimic_suite` non-members, n=2000+2000): **0.5876** [0.5703, 0.6051]; base control 0.5408; difference 0.0468 [0.0316, 0.0626].
- Y10 historical compatibility set AUC: 0.5018 (osler_v4 reference 0.5042).
- 2,000 members sampled from 12,993 MIMIC rows visited by both runs; the training mix has 539,807 MIMIC-derived rows (35.4 % of 1,526,405).
- The verbatim-extraction probe is **not applicable**: the staged weights carry no `lm_head` and no language-model head supports generation (verified on the staged weights).

## Honest limitations

- General OOD QA versus its own zero-shot base: `panel` +0.0028 [-0.0069, +0.0129], `knowledge` -0.0013 [-0.0090, +0.0062] — parity (both intervals include zero), not a win.
- **Held-out failure, reported:** on `mr_heldout_test` / `synthea_indicator_evidence_v1` (n=514, a template excluded from training) both release candidates read below chance (9B accuracy 0.356, CI upper 0.3988; 4B 0.2451, upper 0.2821; chance 0.5) while their bases pass the same cell (d3-flash 0.4591, pplx-flash 0.5973). The cell is flagged `READOUT_FAIL` and is not quoted as an accuracy anywhere in this card.
- Dev-only calibration does not transfer to the fresh post-cutoff set: mean confidence on `confirm_general` is 0.8137 (9B) / 0.8005 (4B) against accuracies 0.9008 / 0.8883, so the ECE there is 0.0900 / 0.0940 versus 0.0192 on the dev fit set (the models are under-confident on the fresh distribution). No test-set temperature was fitted.
- The 27B comparison model is stronger on hard multi-step medical QA (for example MedXpertQA); the release candidate is not claimed to beat it.
- Held-out transfer (`mimic_heldout`, `mr_heldout_test`, `mednli_test`, `pairs_test`) is reported as measured, with wide intervals on the small sets.
- Not for unsupervised clinical decisions. Outputs are probabilities from a research model; a clinician remains responsible for any decision.
- MIMIC: trained on credentialed data locally; no MIMIC text was sent to any hosted service, and the release tree contains no MIMIC text.

## Provenance

- Staged and verified 2026-11-25T05:29:32Z at git `3d82a407c72ed779ef7a6340684b6259d5c00353`; probability parity max |Δp| 0 with 20/20 argmax agreement on 20 fixed dev items.
- Full manifest with SHA-256 of every staged file: `provenance_manifest.json` in the staged tree.

