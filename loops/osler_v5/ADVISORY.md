# ADVISORY — osler_v5: fix and max the Osler-4B

Planned 2026-10-25 by the advisor (Claude Opus) with the operator, after the osler_v4 review. Read in full at the start of
every iteration. New decisions: D51–D55 in `docs/plans/PROGRAM.md` §3 (D26–D50 still hold).

## 1. Where things stand

osler_v4's final Osler-4B (63,304 steps, pplx-flash base, mix composition `9d25e30f…`: osler_v2's C1 + MIMIC B1–B3, 1,456,593
rows) is the best open model we measured on real-record audit, and it fails Gate O4 on one check:

| Gate O4 check | vs zero-shot pplx-flash | vs MedDecider-4B |
|---|---|---|
| (i) MR suite v1 | +0.139 [+0.124, +0.153] PASS | +0.245 PASS |
| (ii) MIMIC audit suite v1 | +0.165 [+0.150, +0.180] PASS | +0.165 PASS (8 templates) |
| (iii) external panel | **−0.042 [−0.058, −0.027] FAIL** | NOT MEASURED (4.87 % unpaired, MedDecider's 10-letter prompt) |
| (iv) knowledge | −0.007 [−0.017, +0.005] PASS | +0.006 PASS |

MIMIC suite scoreboard (template macro): osler_v4 0.800, pplx-27B 0.682, zero-shot pplx 0.634, MedDecider-9B 0.628.
osler_v2 scores 0.862 on the MR verdict, osler_v4 0.843. Membership AUC 0.504 (no signal).

**Diagnosis (advisor, after the loop; dev data and training logs only, never the panel):**
1. **Readout collapse on many-option items.** Training items average 2–6 options, so readout rows past F rarely see a gold.
   DDXPlus dev (49 options) on items whose gold has a two-letter code: zero-shot pplx 0.611, A0 at 14k 0.479, osler_v2 0.064,
   osler_v4 final 0.004 (C210). With single-letter gold, osler_v2 equals pplx (0.796), so the knowledge is there. DDXPlus alone
   accounts for osler_v2's whole CG v2 deficit, and for 0.039 of osler_v4's 0.049 deficit against A0 (C212).
2. **Per-step loss averaging over-weights long items.** Batches are filled to 8 items or 8,192 tokens after a length sort, so a
   long MIMIC note often trains alone and weighs as much as eight replay items. In Y8, MIMIC was 36 % of trained items and about
   60 % of the gradient weight (from `outputs/osler_v4/Y8/logs/train.jsonl`). The replay floor of 20 % was nominal.
3. **General skill erodes beyond the readout.** The panel also drops on yes/no and 4-option datasets, and the 110 MIMIC held-out
   items fall below zero-shot (0.638 vs 0.720). Weight-space distance from the base is the lever left: KL 0.1 and broad replay
   each moved osler_v3's panel gap by about +0.013 (single runs, additional rows), and a blend toward the base (D53) needs no
   retraining.
4. **Selection ignored CG.** Fast dev v4's pooled macro rose to 0.850 while CG v2 stayed at 0.73 (D54 fixes this).

## 2. The questions this loop answers

1. Does per-item loss weighting (D51) and option-count augmentation (D52) fix the CG v2 loss **without** losing the audit lead?
2. Does KL 0.1 on general anchors add to them?
3. How much of the panel-proxy gap does a blend toward the base (D53) recover, and at what audit cost?
4. Does the fixed, blended Osler-4B pass **Gate O5** (D55), and beat osler_v4's final on every scope?

Outcomes (write the ones that apply into FINDINGS):
- **(1) Gate O5 passes:** this recipe (arm, length, α rule) goes to the 9B next loop unchanged.
- **(2) Fixes restore CG v2 but the panel still fails:** the panel loss is not visible on dev; report per dataset (additional) and
  propose nothing that selects on the panel.
- **(3) The blend trades too much audit:** report the α curve on dev; the rule's choice stands.
- **(4) Positional health fails everywhere:** the augmentation did not reach the readout; report and stop the recipe question.

**Release targets** (reported, not gates): MIMIC suite ≥ +0.10 over zero-shot pplx (point estimate); MR verdict template macro ≥
osler_v2's − 0.01; panel non-inferior to zero-shot pplx (the gate check).

## 3. Ground rules specific to this loop

1. **Everything from osler_v4 §3 still holds** (base pplx-flash, D33 attention-only LoRA, D40 speed recipe, D37, labels by code,
   dev-only selection, held-out lists unchanged, MIMIC local only, nothing MIMIC-derived committed). The KL anchor prefixes are
   osler_v4's list unchanged.
2. **The mix is osler_v4's final mix, recomposed in memory** with `scripts/osler_v4/y8_train.py`'s composer: its composition
   SHA-256 must equal `9d25e30f53d7823d051b9c71334c248660318c86ce904c26f855b9a9dbc618b8` before any run. No new data sources.
   **Never train on DDXPlus** (any split): CG v2's DDXPlus must stay an unbiased proxy.
3. **Frozen sets reused, never rebuilt:** fast dev v4, CG v2, knowledge dev v2, the MR suite v1, the MIMIC audit suite v1, the
   held-out files, the panel, the knowledge test. Check every set's hash against osler_v4's manifests before use.
4. **Baselines are not re-scored.** Gate O5 and the scoreboard reuse osler_v4's prediction files (Y4, Y9) for zero-shot pplx,
   MedDecider-4B/-9B, JEV-9B, pplx-27B, osler_v2 and the osler_v4 final, after checking that their item ids match the frozen sets.
   Only osler_v5 models are scored.
5. **Disk:** keep at least 30 GB free (`df` is misleading on this volume; test with a 2 GB write). Full checkpoints keep the newest
   two. Delete each finished arm's `full/` resume states once its result is recorded. The quota stalled osler_v4 for 5 hours.
6. **Commit cadence (AGENTS.md, amended for this loop):** ruff and the full suite run before a commit that changes code
   (`src/`, `scripts/`, `tests/`, `configs/`); a commit that changes only `loops/` files skips the suite. Run the suite in the
   osler_v4 chunked, CPU-hidden form (C217).
7. **Nemotron is not available** (its weights were removed on 2026-10-25 to free space). No teacher work in this loop.

## 4. Tasks

### Z0 — Orientation and the recipe code (CPU, ~5 h)

1. Read this ADVISORY, PROGRAM D49–D55, osler_v4's FINDINGS and NEXT. Ruff and the suite (osler_v4 deselects); record counts.
2. **D51:** `loss_weighting: per_step | per_item` in the trainer config (default `per_step`). Under `per_item`, multiply the step's
   total loss (CE + Brier + KL + distillation) by `n_items / batch_size`. Unit test: a 1-item and an 8-item batch of identical
   items give per-item gradients in the ratio 1 : 1; `per_step` reproduces the old loss to the bit.
3. **D52(a):** `option_count_augment: {p: 0.3, min_options: 8, max_options: 64}` (default off), applied in `iter_batches` after
   option shuffling, seeded by (seed, epoch, item).
   - **The closed-set list:** for each `choice` template in the mix, decide by reading its builder whether the gold is a
     single-valued field of the record (exactly one label of the set is true: e.g. an admission's MS-DRG, a radiology exam
     name, a page category, an ED disposition, a principal diagnosis). Write the list with one reason per template, and the
     label-set size from train, to STATE §2 **before any run**. Templates whose other labels could also be true (multi-label
     coding, free-text MCQ options) are excluded. Never change the list after a run.
   - Distractors come from the template's **train** label set, never from dev or test items.
   - Unit tests on synthetic items: gold kept, no duplicate labels, count within range, gold position uniform over 10,000 draws.
   - Record the share of trained items expected to have > 26 options.
4. **D52(b):** the positional health check (`meddecide.eval.code_split` extended): for every dev template with > 26 options,
   two-letter-gold accuracy ≥ 0.5 × single-letter-gold accuracy. Test on synthetic predictions.
5. **D53:** `blend(checkpoint, alpha) -> checkpoint`: scale the LoRA update by α (merge with scale α, or set the adapter scaling)
   and interpolate the readout with its step-0 values (`R_init` = the readout as loaded before training). α = 1 must reproduce
   the trained model's probabilities (test, max abs diff < 1e-5 in fp32); α = 0 must reproduce zero-shot pplx (same test).
6. **D54:** the composite S (dev) as a pure function over prediction files, with the three partitions taken exactly as osler_v4
   Y6 took the MR partition, the fast dev v4 MIMIC partition, and CG v2. Compute S, its parts and positional health for zero-shot
   pplx, A0 and the osler_v4 final from existing predictions (additional rows; they calibrate expectations, choose nothing).
7. Write the Z2 and Z3 configs (below). Commit.

### Z1 — Probes on the osler_v4 final (GPU, ~3 h; additional, choose nothing)

On the osler_v4 final checkpoint, dev only (fast dev v4 + CG v2):
1. **Blend curve:** α ∈ {0.3, 0.5, 0.7, 0.85, 1.0}: S and its three parts, knowledge dev, positional health, CG v2 per template.
2. **Readout reset:** the final with readout rows ≥ 26 restored to `R_init`: DDXPlus code split and CG v2.

Report in `loops/osler_v5/probes.md`. These rows inform the reader; the pre-registered rules below do not change because of them.

### Z2 — Recipe arms (GPU, 4 runs, ~4.5 h each)

Each arm: osler_v4's final mix (§3.2), 13,992 steps, the osler_v4 Y8 recipe otherwise (configs/osler_v3/block_arm.yaml, D40,
seed 0), evaluations at 6,996 and 13,992 on fast dev v4 + CG v2.

| arm | recipe |
|---|---|
| R0 | control: osler_v4's final recipe (`per_step`, no augmentation, KL 0) |
| R1 | `loss_weighting: per_item` |
| R2 | R1 + `option_count_augment` (D52a) |
| R3 | R2 + `kl_to_base: 0.1` on anchors (osler_v4 prefixes; augmented anchors included) |

**Choice rule** (pre-registered; paired bootstrap within templates, 1,000 resamples, seed 0; values at step 13,992):
- An arm **qualifies** if positional health passes and its knowledge dev lower bound against R0 > −0.01.
- **Choose the qualifying arm with the highest S.** If a simpler arm (lower number) is within 0.003 of it, the simpler arm wins.
- If no arm qualifies, choose the highest-S arm, record it as a deviation and continue.

Delete each arm's `full/` after its result is recorded. Write `loops/osler_v5/arms.md`.

### Z3 — Final Osler-4B (GPU, ~21 h)

- The chosen arm's recipe on the same mix, **63,304 steps** (osler_v4's length, so the final compares one-to-one with osler_v4's),
  10 evaluations on fast dev v4 + CG v2, resumable full checkpoints every 5,000 steps (newest two kept).
- **Checkpoint selection by S** among evaluations that pass positional health; if none passes, the highest S, flagged.
- Record S, its parts, knowledge dev and positional health at every evaluation.
- Temperatures fitted on dev and saved with the checkpoint. Refuse to run twice (the `arm_result.json` guard).

### Z4 — Blend (GPU, ~3 h)

α ∈ {0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0} on the selected checkpoint; temperatures refitted per α on fast dev v4.
**Rule** (pre-registered):
- An α qualifies if positional health passes, knowledge dev ≥ the α = 1 value − 0.01 (point estimate), and CG v2 template macro
  ≥ zero-shot pplx's CG v2 template macro − 0.01.
- Choose the qualifying α with the highest S; within 0.002 of it, the larger α wins.
- If no α meets the CG floor, choose the α with the highest CG v2 template macro among those whose S is within 0.01 of the
  highest S.

Export the blended model as one checkpoint (`outputs/osler_v5/Z4/final_blended/`). Write `loops/osler_v5/blend.md`.

### Z5 — Evaluation and Gate O5 (GPU, ~3 h)

Score the blended final (calibrated) on the frozen sets: the MR suite v1 (verdict and reported-only), the MIMIC audit suite v1,
the held-out files, the external panel, the knowledge test, MedNLI test. Also score the unblended (α = 1) checkpoint as an
additional row.

**Apply Gate O5 (D55) exactly**, with the pre-registration written to STATE §7 before any test-set scoring. Scoreboard rows
(additional, from existing predictions): zero-shot pplx, MedDecider-4B/-9B, JEV-9B, pplx-27B (ceiling), osler_v2, osler_v4 final.
Report the panel per dataset, the release targets of §2, and the DDXPlus code split. Write `gate_o5.md` and `results.md`.

### Z6 — Membership audit (small GPU, ~1 h)

osler_v4 Y10's membership test (2,000 MIMIC train vs 2,000 test items, same templates) on the blended final; AUC with interval.
The verbatim probe is not repeated (osler_v4 C236: uninformative for the causal export). Write `loops/osler_v5/audit.md`.

### Z7 — Findings and closure — HARD STOP

FINDINGS (Summary first; §2 outcomes; arms, the final's S trend, the blend curve, the gate, the scoreboard; the five-claim
spot-check), NEXT.md, `Loop status: STOPPED`, commit, push, stop.

## 5. Task table (seed STATE.md with this)

| id | task | GPU | deps |
|---|---|---|---|
| Z0 | Orientation and recipe code (D51–D54) | no | — |
| Z1 | Probes on the osler_v4 final (blend curve, readout reset) | yes | Z0 |
| Z2 | Recipe arms R0–R3 and the choice | yes | Z0 |
| Z3 | Final Osler-4B (63,304 steps) | yes | Z2 |
| Z4 | Blend α on dev and export | yes | Z3 |
| Z5 | Evaluation, Gate O5, scoreboard | yes | Z4 |
| Z6 | Membership audit | small | Z4 |
| Z7 | Findings and closure — HARD STOP | no | all |

GPU sequence: Z1 → Z2 → Z3 → Z4 → Z5 → Z6. GPU hours (planning guesses): Z1 3, Z2 19, Z3 21, Z4 3, Z5 3, Z6 1 — **about 50
GPU-hours; 3.5–4 days with overhead.** While Z3 trains, pre-write Z4–Z5 code (CPU only).

## 6. Out of scope

- Osler-9B (next loop, on this loop's recipe) and Osler-0.8B; base, attention, head or adapter changes.
- New data sources, teachers, Nemotron work, 27B distillation (osler_v4 A5 trailed A4; the anchors are under-scored).
- DDXPlus training data; any panel, test, MIMIC-suite or held-out influence on a choice.
- Changing a frozen set, Gate O5, the choice rules or the closed-set list after any run.
- Releasing or uploading anything; the HF backup.

## 7. Failure modes to avoid

1. **The augmentation leaks a label.** A distractor equal to another true label makes the gold wrong. Only single-answer
   closed-set templates, with reasons; when unsure, exclude.
2. **Per-item weighting silently changes the learning rate.** Long-item steps now move the weights less; report the realised
   gradient-weight share by family (MIMIC vs other) for R0 and R1 from the step logs.
3. **The blend breaks calibration.** Temperatures are refitted per α; report Brier at each α.
4. **Re-using a baseline file that does not match the set.** Check item ids and counts before Gate O5; a mismatch is BLOCKED,
   not re-scored silently.
5. **Disk quota.** Check headroom before every GPU job; clean arm resume states as you go.
