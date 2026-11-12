# d3_distill — FINDINGS

Status: **plan steps 0–3 complete, gate reported, awaiting operator review** (2026-11-10T05:20Z). Numbers are only those
with a CLAIMS row and a gitignored artifact.

## Summary

- Step 0: the teacher's batched path passes the throughput gate at **29.056 items/s** on 1,000 MedMCQA-train items with
  the probability check passing for every item (the single-request path measured 6.734 items/s; the operator accepted the
  batched path, and the batched-vs-single difference of up to 0.035 / mean 0.003 / argmax 40/40 as a teacher fact).
- Step 1: **200,000 labels, 0 invalid, 0 constant**, from 182,777 MedMCQA-train rows and 17,223 code-generated rows
  (7 clinical-decision families + 3 MR audit generators), with 0 text overlap against the 22,912 frozen Y9 items.
- Step 2: port parity PASS for both students (max difference 0.0, argmax 40/40). λ probes: 4B s0 = 0.4317 → λ = 0.8774;
  9B s0 = 0.4173 → λ = 0.9310. Four arms trained, all bands PASS; finals at 63,304 steps: d3-mini dev 0.875862
  (zero-shot 0.8141), d3-flash dev 0.893253 (zero-shot 0.8417).
- Step 3: the Gate O7 baseline (zero-shot pplx-decider-flash-v1.2, authors' code) scored locally on the frozen sets:
  knowledge 0.7105, panel 0.6362, mr_verdict 0.7050 (template macro; D12 PASS; within 0.003 of the published osler_v4 row).
  **Gate O7: FAIL for both students on the panel check** (−4.62 points [−6.52, −2.81] for mini; −1.51 [−3.21, +0.15] for
  flash); knowledge PASS (+1.54 [0.10, 2.91]; +5.15 [3.77, 6.49]); MR-vs-osler_v4-final and MIMIC NOT MEASURED.

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
cosine 3 % warmup, λ calibrated by a 200-step probe (4B: s0 = 0.4317, λ = 0.8774; 9B: s0 = 0.4173, λ = 0.9310; the arm's
first 200 steps must be in the 0.30–0.50 share band). Two arms = seeds 0/1 at 13,992 steps, selection on the train-side
dev carve, final at 63,304 steps.

Measured so far (dev = the train-side 10,005-row carve; the final's Gate O7 rows are pending):

| student | zero-shot dev | arm s0 (band) | arm s1 (band) | final dev (best step) | final band | final time |
|---|---:|---|---|---|---|---:|
| d3-mini (4.54B) | 0.8141 | 0.8432 (0.4015) | 0.8424 (0.4066) | **0.8759** (63,300) | 0.3977 PASS | 9,364.5 s |
| d3-flash (8.39B) | 0.8417 | 0.8687 (0.4009) | 0.8727 (0.4058) | **0.8933** (44,310) | 0.3951 PASS | 12,783.0 s |

Seeds: mini chose seed 0 (0.843178 vs 0.842379); flash chose seed 1 (0.872664 vs 0.868666). The finals' dev curves rise
monotonically to the plateau (mini 0.8231 → 0.8759; flash 0.8475 → 0.8933) and every run's fresh-process audit passed
8/8 (steps, losses, band, evals, checkpoint, step count).

The mini final's public-split diagnostics: MedMCQA validation 0.694, MedQA test 0.7541, knowledge subsample 0.701 micro /
0.7167 template macro (the locally scored pplx baseline is 0.7105 macro on the same 1,000 items). The MR/panel rows need
the Y9 scoring that follows the chain (the chain's own scoring block ran the pre-update set list; the scores are produced
separately and the gate report re-run).

## 5. Step 3 — Gate O7 (baseline ready, students pending)

The PLAN's gate: panel lower bound > 0 against zero-shot pplx-decider-flash-v1.2; knowledge lower bound > −0.02; MR and
MIMIC within −0.02 of the osler_v4 final. The baseline was scored on this machine with the authors' code on the frozen
sets (readout gate PASS everywhere): knowledge 0.7105, panel 0.6362, mr_verdict 0.7050 (template macro), reproducing the
published osler_v4 Y9 row within 0.003. The osler_v4 final's predictions/weights are not on this machine, so the MR-vs-
osler and MIMIC checks are NOT MEASURED, and no other number is substituted for them.

| student | panel (limit > 0) | knowledge (limit > −0.02) | mr_verdict vs pplx (additional) | MR vs osler_v4 final | MIMIC |
|---|---|---|---|---|---|
| d3-mini | **FAIL** −0.04622 [−0.06522, −0.02808] | **PASS** +0.01543 [+0.00096, +0.02907] | +0.02314 [+0.01072, +0.03676] | NOT MEASURED | NOT MEASURED |
| d3-flash | **FAIL** −0.01513 [−0.03212, +0.00149] | **PASS** +0.05153 [+0.03769, +0.06486] | +0.00339 [−0.01063, +0.01781] | NOT MEASURED | NOT MEASURED |

Own template-macro accuracies on the full sets (paired items; knowledge 5,435 of 5,456, 21 repeated-label items unpaired
at 0.38 %): mini panel 0.5900 / knowledge 0.7260 / mr_verdict 0.7281; flash panel 0.6210 / knowledge 0.7620 / mr_verdict
0.7084; pplx 0.6362 / 0.7105 / 0.7050.

Reading. The distillation moved knowledge clearly (the 9B by 5.2 points with a CI clear of zero) and left the MR verdict
flat-to-slightly-positive, but it did not close the external-panel gap: both students lose the panel to zero-shot pplx,
and the 4B by about as much as the osler_v4 final did. The 9B's panel interval just crosses zero (−3.2 to +0.15) but the
pre-registered rule is the lower bound, so it is a FAIL, not a near-pass to relabel (R4). The teacher itself scores 0.638
on the 500-item d3_bench panel subsample, so this is not a teacher-quality artefact alone; the panel is a
general-medical-reasoning set that the MedMCQA/synthetic mixture does not target.

Training effect on the frozen sets (additional; same paired rule, frozen student vs pplx):

| student | panel, zero-shot → final | knowledge, zero-shot → final | mr_verdict, zero-shot → final |
|---|---|---|---|
| d3-mini | −0.01083 [−0.02486, +0.00288] → **−0.04622** [−0.06522, −0.02808] | +0.00173 → **+0.01543** | READOUT_FAIL → **+0.02314** |
| d3-flash | **+0.02316** [+0.00870, +0.03802] → **−0.01513** [−0.03212, +0.00149] | +0.04247 → **+0.05153** | −0.01755 → **+0.00339** |

The 9B's frozen base would have satisfied the panel check (+2.3 points with the CI clear of zero); the 63,304-step
distillation cost it 3.8 panel points and flipped the check to FAIL. On the 4B the panel was already below pplx before
training and the recipe cost another 3.5 points. Knowledge and the MR verdict improved on both sizes. So the binding
constraint is the recipe's objective/mixture, not the base or the teacher: the CL/CE mixture is MedMCQA-heavy (91 % of
the pool) and teaches the knowledge and audit templates while moving the model away from the external panel's
general-reasoning calibration (panel Brier also rises sharply after training, e.g. 0.54 → 0.65 for the 9B).

## 6. Claims index

C001–C006 (step 0), C007–C010 and C014–C020 (step 1), C021–C027 (step 2/3 infrastructure and baseline). See
`loops/d3_distill/CLAIMS.md`; every value there carries its artifact and recompute command.
