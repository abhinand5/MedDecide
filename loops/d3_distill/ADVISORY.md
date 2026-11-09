# d3_distill — ADVISORY (step 2–3 pre-registration)

Written `2026-11-09T16:15Z`, **before any student forward or training step**. The PLAN.md says "same recipe as
distill_v0", but no `distill_v0` artifacts exist in this repo's history; this file freezes the recipe that will be run.
Every choice below is either from the d3_distill PLAN, from the osler_v6 pre-registration (D51/D57, the recipe it shares),
or a conservative reading flagged in STATE.md "Questions for the operator". None may change after an arm starts.

## 1. Port parity gate (per student, before any training)

- Load the student with the card's `AutoModel.from_pretrained(trust_remote_code=True)`; the students are
  `noncausal_full_attention`, temperature 1.0, no `prompt_layout`, so the runtime's reference path is `D3.probabilities`
  over left-padded full-attention batches — the same math our trainable forward uses.
- Our trainable forward reuses the student's own `D3.prepare` (render/tokenize), `PromptEncoder` and readout:
  `embeds = backbone.get_input_embeddings()(ids)`; `hidden = runtime.encoder(embeds, mask)[:, -1]`;
  `logits = hidden.float() @ readout.T`; softmax over the offered codes. It re-implements no model step.
- Gate: on 40 pool items (one one-question request per item, batch 1), the trainable forward reproduces the runtime's
  `D3.probabilities` within **1e-4** (max absolute difference). If a student fails, that student is BLOCKED; the teacher
  labels still stand as a dataset (PLAN.md).
- Additional diagnostic, not the gate: agreement with `system_one` and the forward at batch 8.

## 2. Recipe (both students, both arms, the final)

- Frozen base + LoRA r=32, alpha=32, dropout 0, **attention only** (D33): `q_proj`, `k_proj`, `v_proj`, `o_proj` of the
  full-attention layers. The decision readout (`[255, hidden]`, FP32) is trained too; the rest of the backbone is frozen.
- Loss: `CE(gold)` on every item + `λ · KL(teacher ‖ student)` on every labelled item, both over the offered option codes
  after the 255-way logits are masked to the row's options; sum over items divided by `batch_size` (`loss_weighting:
  per_item`, D51). At 8 items per batch, D51's `n_items / batch_size` factor is 1 for every batch, so D51 is satisfied by
  construction and recorded as such.
- Temperature: 1.0, the students' `decision_config` value (no `temperature_by_questions`); the teacher's labels are
  distributions over options, so the same soft targets apply to the students' own prompt format.
- Optimizer AdamW, betas 0.9/0.95, weight_decay 0; LoRA lr 1e-4, readout lr 1e-3; cosine schedule, 3 % warmup, floor 0.1;
  grad clip 1.0. Gradient checkpointing on batches above 8,192 padded tokens (the D40 speed rule).
- Batch 8, `max_batch_tokens` 8,192; sampler: each epoch's order by a seeded stable hash, split into windows of 64,
  each window length-sorted and cut into batches of 8 (less padding, no item order effect on coverage).

## 3. Dosing (D57) and λ calibration

- The 200,000 labels are one label per pool row, so every training batch holds 8 labelled rows: the D57 minimum (3) is
  met on every step by construction. The per-step labelled count is logged; a step below 3 stops the arm.
- λ is calibrated in a separate 200-step probe at λ = 1 on the arm's data order (no weights kept). Per step, the pre-clip
  gradient norms of the CE term and of the KL term are measured separately (two backward passes on the same batch), and
  `s0 = mean(‖g_KL‖ / (‖g_CE‖ + ‖g_KL‖))`. `λ = (0.4/0.6) · (1 − s0) / s0`, frozen before the arm. The arm's own first
  200 steps must show the same pre-clip share in **0.30 – 0.50**; a miss blocks that student's arms (not re-tuned).

## 4. Arms, final, selection rule

- **Arms = seeds.** Per student: arm s0 and arm s1, same recipe and data order seed 0/1, 13,992 steps each. Evaluations on
  the train-side dev carve (below) at steps 3,500, 7,000, 10,500 and 13,992.
- **Dev carve:** the 5 % of pool rows assigned `split=dev` by `stable_hash(row_id) % 100 < 5` (10,005 rows: 9,006 MedMCQA,
  999 synthetic). Never trained on. Selection uses **gold accuracy only**; teacher KL on dev is a diagnostic and is used to
  nothing.
- **Zero-shot reference:** each student's dev accuracy before training (same forward).
- **Arm qualification:** no NaN/inf, the λ band holds, and the arm's dev accuracy at 13,992 steps is not below the
  zero-shot student by more than 2 points (a collapse guard, not a positional test — the students' attention is noncausal
  and no positional check exists for them in this loop; recorded as NOT MEASURED).
- **Selection:** the qualifying arm with the higher dev gold accuracy; a tie goes to seed 0. If neither arm qualifies, no
  final is trained and the outcome is reported.
- **Final:** 63,304 steps from the frozen student base with the chosen seed's data order, ten evaluations (every 6,330
  steps), checkpoint by dev gold accuracy among evaluations that are not below the zero-shot student by more than 2 points.

## 5. Gate O7 (PLAN.md step 3) — what can be measured here

Each check is reported against zero-shot pplx-flash or the osler_v4 final on the **same frozen subsample**:

| check | PLAN | data on this machine |
|---|---|---|
| panel | lower bound of (student − zero-shot pplx-flash) > 0 | 500-item seeded panel subsample; pplx-flash row in `outputs/d3_bench/aggregate.json` — measurable |
| knowledge | lower bound > −0.02 | 1,000-item knowledge subsample; pplx-flash 0.6774 — measurable |
| MR | lower bound within −0.02 of the osler_v4 final | 500-item mr_verdict subsample exists, but the osler_v4 final has no row here and its weights are not on this machine — **NOT MEASURED** |
| MIMIC | within −0.02 of the osler_v4 final | no MIMIC data on this machine (operator instruction) — **NOT MEASURED** |

Lower bounds are 95 % paired bootstrap intervals (1,000 resamples, seed 0) over items scored by both sides; a check whose
baseline row is missing is NOT MEASURED, never re-labelled. The teacher (d3-ultra) is reported as an additional row from
the existing d3_bench aggregate, not scored again.

## 6. Guards

- No test-side item is ever trained on: the dev carve is train-side; the panel/knowledge/mr_verdict subsamples are used
  only for the gate, after the final is frozen.
- No MIMIC text leaves the local GPU (and none is read at all on this machine).
- The teacher's labels are not used to select on any test set; arm selection is dev gold accuracy only.
- Every run records git commit, command, seed, data sha256, step count, wall clock and peak GPU memory.
