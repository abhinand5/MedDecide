# d3_distill — FINDINGS

Status: **in flight** (2026-11-09T19:05Z). Steps 0 and 1 are complete; step 2 is training and step 3 will be computed on
the full frozen sets once both students' finals exist. This file is updated at closure; numbers are only those with a
CLAIMS row and a gitignored artifact.

## Summary (to be completed at closure)

- Step 0: the teacher's batched path passes the throughput gate at **29.056 items/s** on 1,000 MedMCQA-train items with
  the probability check passing for every item (the single-request path measured 6.734 items/s; the operator accepted the
  batched path, and the batched-vs-single difference of up to 0.035 / mean 0.003 / argmax 40/40 as a teacher fact).
- Step 1: **200,000 labels, 0 invalid, 0 constant**, from 182,777 MedMCQA-train rows and 17,223 code-generated rows
  (7 clinical-decision families + 3 MR audit generators), with 0 text overlap against the 22,912 frozen Y9 items.
- Step 2: port parity PASS for both students (max difference 0.0, argmax 40/40); λ(4B) probe s0 = 0.4317 → λ = 0.8774
  (arms in flight).
- Step 3: the Gate O7 baseline (zero-shot pplx-decider-flash-v1.2, authors' code) was scored locally on the frozen sets:
  knowledge 0.7105, panel 0.6362, mr_verdict 0.7050 (template macro; D12 PASS; within 0.003 of the published osler_v4
  row). The students' gate rows are pending their finals.

## 1. What was asked, and what was possible here

The loop distils `vllm-sr/d3-ultra` (local, Apache-2.0) into `vllm-sr/d3-mini` (4.54B) and `vllm-sr/d3-flash` (8.39B)
under the plan's steps 0–3. The machine had no `data/` tree at the start; the three frozen Y9 evaluation sets were added
by the operator on 2026-11-09 (SHA-256 verified). MIMIC and its derived files stay off the machine by instruction, so
every MIMIC-dependent check is BLOCKED/NOT MEASURED rather than substituted.

## 2. Step 0 — teacher throughput (DONE, PASS)

The plan's gate needs batched inference, not the single-request API. The batched path stays inside the teacher's own code:
`D3.prepare` renders and tokenizes one one-question request per item exactly as `system_one` does, and
`D3.probabilities` runs the model's own `prefix_bidir` mask, forward, FP32 readout and softmax over length-sorted,
left-padded batches of 8 (`meddecide.distill.d3_batch`).

| path | items/s | probabilities valid |
|---|---:|---|
| single request per call (the step-0 fact) | 6.734 | 1,000/1,000 |
| batched, end to end (the gate) | **29.056** | 1,000/1,000 |

The gate passed with 0 invalid and 0 constant distributions; dev accuracy on the same 1,000 items was 0.945 on both paths.
Batched vs single on 40 items: argmax 40/40, mean max-abs difference 0.0028, max 0.0354. The operator accepted the batched
teacher on 2026-11-09; the 1e-4 bar applies to the students' port only.

## 3. Step 1 — the label set (DONE)

Pool: 200,000 rows = MedMCQA official train 182,777 + `meddecide.gen` clinical families 12,900 + `meddecide.mr` audit
generators 4,323; 189,995 train / 10,005 train-side dev (5 %, never trained on). Labels: 200,000/200,000 valid, 0
invalid, 0 constant, every answer a distribution over exactly the row's options (sums within 2e-7). The teacher's top-1
gold accuracy on the labels is 0.9330 overall (in-distribution diagnostic only). Exclusions enforced: MedMCQA validation
texts (the knowledge test side), held-out generator families (laboratory-range, triage, AKI, heart-failure) and non-train
generator patients; 0 text matches against all 22,912 frozen Y9 items. MIMIC, MedNLI test and the CG v2 dev proxy remain
NOT MEASURED (files absent).

## 4. Step 2 — students (IN FLIGHT)

Port parity: the trainable harness (the student's own `D3.prepare`/`D3.encoder` + readout, attention-only LoRA r=32 at
zero) reproduces the student's own `D3.probabilities` exactly (0.0 max abs difference, argmax 40/40) for both students at
the reference's one-row shape. Recipe: CE(gold) + λ·KL(teacher ‖ student), batch 8, AdamW (LoRA 1e-4, readout 1e-3),
cosine 3 % warmup, λ calibrated by a 200-step probe (4B: s0 = 0.4317, λ = 0.8774; the arm's first 200 steps must be in the
0.30–0.50 share band). Two arms = seeds 0/1 at 13,992 steps, selection on the train-side dev carve, final at 63,304 steps.

## 5. Step 3 — Gate O7 (baseline ready, students pending)

The PLAN's gate: panel lower bound > 0 against zero-shot pplx-decider-flash-v1.2; knowledge lower bound > −0.02; MR and
MIMIC within −0.02 of the osler_v4 final. The baseline was scored on this machine with the authors' code on the frozen
sets (readout gate PASS everywhere): knowledge 0.7105, panel 0.6362, mr_verdict 0.7050 (template macro). The osler_v4
final's predictions/weights are not on this machine, so the MR-vs-osler and MIMIC checks will be NOT MEASURED; the
mr_verdict row against pplx is reported as additional.

## 6. Claims index

C001–C006 (step 0), C007–C010 and C014–C020 (step 1), C021–C027 (step 2/3 infrastructure and baseline). See
`loops/d3_distill/CLAIMS.md`; every value there carries its artifact and recompute command.
