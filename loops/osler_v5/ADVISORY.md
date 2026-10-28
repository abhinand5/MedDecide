# ADVISORY — osler_v5: fix and max the Osler-4B

Planned 2026-10-25 by the advisor (Claude Opus) with the operator; revised 2026-10-26 after an independent review (Opus 5.5,
max effort) that checked the plan against osler_v4's logs. New decisions: D51–D55 in `docs/plans/PROGRAM.md` §3 (D26–D50
still hold). Read in full at the start of every iteration.

## 1. Where things stand

osler_v4's final Osler-4B (63,304 steps; osler_v2's C1 mix plus MIMIC B1–B3, 1,456,593 rows, composition SHA-256
`9d25e30f53d7823d051b9c71334c248660318c86ce904c26f855b9a9dbc618b8`) is the best open model we measured on real-record audit,
and it fails Gate O4 on one check:

| Gate O4 check | vs zero-shot pplx-flash | vs MedDecider-4B |
|---|---|---|
| (i) MR suite v1 | +0.139 [+0.124, +0.153] PASS | +0.245 PASS |
| (ii) MIMIC audit suite v1 | +0.165 [+0.150, +0.180] PASS | +0.165 PASS (8 templates) |
| (iii) external panel | **−0.042 [−0.058, −0.027] FAIL** | NOT MEASURED (4.87 % unpaired, MedDecider's 10-letter prompt) |
| (iv) knowledge | −0.007 [−0.017, +0.005] PASS | +0.006 PASS |

MIMIC suite scoreboard (template macro): osler_v4 0.800, pplx-27B 0.682, zero-shot pplx 0.634, MedDecider-9B 0.628.
Membership AUC 0.504 (no signal).

**Diagnosis (dev data and training logs only; never the panel):**
1. **Readout collapse on many-option items.** In the final, readout rows 26–254 are bit-identical to the base; rows A–Z moved by
   1.8× their base norm. No training item offers more than 26 options, so those rows never get a gradient. DDXPlus (49 options)
   two-letter gold: zero-shot pplx 0.611, osler_v4 0.004 (2 of 499); single-letter gold is intact (C210, C212).
2. **Per-step loss averaging over-weights long items.** 69 % of logged steps clip at norm 1.0 (median pre-clip norm 9.7). Batches
   are filled after a length sort, so MIMIC was 36 % of trained items and about 60 % of the applied update.
3. **General skill erodes beyond the readout.** The panel also drops on yes/no and 4-option datasets. Weight-space distance from
   the base is the lever left; KL 0.1 and broad replay moved osler_v3's panel gap by about +0.013 each (single runs).
4. **Selection ignored CG.** Fast dev v4's pooled macro rose to 0.850 while CG v2 micro stayed at 0.74 (D54 fixes this).

## 2. The questions this loop answers

1. Do per-item weighting (D51) and option-count augmentation (D52) fix the CG v2 readout loss without losing the audit lead?
2. Does KL 0.1 on general anchors add to them?
3. How much of the panel-proxy gap does a blend toward the base (D53) recover, and at what audit cost?
4. Does the blended Osler-4B pass **Gate O5** (D55: Gate O4 unchanged)?

Outcomes (write the ones that apply into FINDINGS):
- **(1) Every measured check passes against both baselines:** the MedDecider-4B panel reads NOT MEASURED (ten letters). FINDINGS
  states "PASS vs zero-shot pplx-flash; NOT MEASURED vs MedDecider-4B"; the recipe goes to Osler-9B.
- **(2) Fixes restore CG v2 but the panel still fails:** the panel loss is not visible on dev; report per dataset (additional)
  and propose nothing that selects on the panel.
- **(3) The blend trades too much audit:** report the α curve on dev; the Z4 rule's choice stands.
- **(4) Positional health fails everywhere, or the CG-first arm rule selects an arm that fails it:** the augmentation did not reach
  the readout. Report it and stop the recipe question; no arm is chosen on S alone in that case.

**Release targets** (reported, not gates): MIMIC suite ≥ +0.10 over zero-shot pplx (point estimate); MR verdict template macro ≥
osler_v2's − 0.01; panel non-inferior to zero-shot pplx (the gate check).

## 3. Ground rules specific to this loop

1. **Everything from osler_v4 §3 still holds** (base pplx-flash, D33 attention-only LoRA, D40 speed recipe, D37, labels by code,
   dev-only selection, held-out lists unchanged, MIMIC local only, nothing MIMIC-derived committed). The KL anchor prefixes are
   osler_v4's list unchanged.
2. **The mix is osler_v4's final mix, recomposed in memory** with `scripts/osler_v4/y8_train.py`'s composer. Its composition SHA-256
   must equal `9d25e30f…` before any run. No new data sources. **Never train on DDXPlus** (any split).
3. **Frozen sets reused, never rebuilt:** fast dev v4, CG v2, knowledge dev v2, the MR suite v1, the MIMIC audit suite v1, the
   held-out files, the panel, the knowledge test. Check every set's hash against osler_v4's manifests before use.
4. **Baselines are not re-scored.** Gate O5 and the scoreboard reuse osler_v4's prediction files (Y4, Y9) for zero-shot pplx,
   MedDecider-4B/-9B, JEV-9B, pplx-27B, osler_v2 and the osler_v4 final, after checking item ids. Only osler_v5 models are scored.
   The zero-shot CG v2 reference is the α = 0 model on identical items (D54), not a baseline file's macro.
5. **Disk:** keep at least 30 GB free (`df` is misleading on this volume; test with a 2 GB write). Full checkpoints keep the newest
   two. Delete each finished arm's `full/` resume states once its result is recorded.
6. **No suite and no large writes during a GPU job.** osler_v4 lost arms to a silent death during a CPU build and to a quota filled
   by a suite run. Code commits that change `src/`, `scripts/`, `tests/` or `configs/` run ruff and the suite only between GPU jobs
   (osler_v4's chunked, CPU-hidden form, C217). `loops/`-only commits skip the suite.
7. **Watchdog:** a GPU job whose log has not advanced for 15 minutes is treated as dead: it is resumed from its newest full
   checkpoint, and the resume is recorded. The `arm_result.json` refuse-twice guard allows a resume and refuses a second
   completed run.
8. **Nemotron is not available** (weights removed 2026-10-25). No teacher work in this loop.

## 4. Tasks

### Z0 — Orientation and the recipe code (CPU, ~8–10 h)

1. Read this ADVISORY, PROGRAM D49–D55, osler_v4's FINDINGS and NEXT. Ruff and the suite (osler_v4 deselects); record counts.
2. **D51:** `loss_weighting: per_step | per_item` (default `per_step`). Under `per_item` the clip is applied first, then the
   gradients are multiplied by `n_items / batch_size`; KL and distillation are summed over masked items and divided by
   `batch_size` (not averaged over anchors). Tests use mixed batches with clipping on: a lone MIMIC item and eight replay items
   must give the item-weighted update ratio. Replay Y8's logged batches on the CPU and record the MIMIC share of the applied update
   under `per_step` (expected near 0.59) and under `per_item` (expected near 0.34). If `per_item` does not move the share to about
   0.34–0.40, stop and record it before any run.
3. **D52(a):** the closed list is `mimic_principal_dx_v1` and `mimic_msdrg_v1` (only 5.9 % of items together). Write the list with
   reasons to STATE §2 before any run, together with the expected number of two-letter golds per row; raise p if that number is
   below 20 per row. Also flag the defect: `fda_route_choice_v1` sets gold = routes[0] while offering the drug's other routes;
   it is not augmented and is reported to the operator as a finding. Unit tests on synthetic items: gold kept, no duplicate codes or
   titles, count within range, gold position uniform, frequency-weighted distractors, the fallback below max(original, 8).
4. **D52(b):** the positional check on DDXPlus dev under the original order and two seeded permutations.
5. **D53:** `blend(checkpoint, alpha)` by adapter scaling with R_base the readout at step 0. Tests: α = 0 reproduces the base
   model's probabilities against `MedDecideModel.reference_option_probs` (not an export file), max abs diff < 1e-5 in fp32; α = 1
   reproduces the trained model. Re-score the exported blend on dev before Z5.
6. **D54:** the composite S (dev) as a pure function over prediction files: MR partition (osler_v3's), fast dev v4's MIMIC partition,
   CG v2 micro. Compute S, its parts and positional health for zero-shot pplx, A0 and the osler_v4 final from existing predictions
   (additional rows; they calibrate expectations and choose nothing). **Check:** the osler_v4 final must fail the revised CG
   qualifier (ADVISORY Z4) on these numbers, or the check has no discriminating power and the plan is BLOCKED for review.
7. Write the configs. Commit.

### Z1 — Probes on the osler_v4 final (GPU, ~3 h; additional, choose nothing)

On the osler_v4 final checkpoint, dev only (fast dev v4 + CG v2):
1. **Blend curve:** α ∈ {0.3, 0.5, 0.7, 0.85, 1.0}: S and its parts, knowledge dev, positional health, CG v2 micro per template.
2. **Readout split:** (a) the trained adapter with the base readout; (b) the base adapter (adapter off) with the trained readout.
   Report DDXPlus two-letter accuracy and CG v2 micro for each. (The earlier plan's "restore rows ≥ 26" is a no-op: those rows are
   already at base values.)

Report in `loops/osler_v5/probes.md`. These rows inform the reader; the rules below do not change because of them.

### Z2 — Recipe arms (GPU, 4 runs, ~4.5 h each)

Each arm: osler_v4's final mix (§3.2), 13,992 steps, the osler_v4 Y8 recipe otherwise (configs/osler_v3/block_arm.yaml, D40),
evaluations at 6,996 and 13,992 on fast dev v4 + CG v2.

| arm | recipe |
|---|---|
| R0 | control: osler_v4's final recipe (`per_step`, no augmentation, KL 0), seed 0 |
| R0' | R0 at seed 1 (measures the seed noise that the tie band must exceed) |
| R2 | `loss_weighting: per_item` + option-count augmentation (D52a), seed 0 |
| R3 | R2 + `kl_to_base: 0.1` on anchors (osler_v4 prefixes; augmented anchors included), seed 0 |

**Choice rule** (pre-registered; values at step 13,992, paired bootstrap within templates, 1,000 resamples, seed 0):
- An arm **qualifies** if positional health passes, its knowledge-dev lower bound against R0 > −0.01, and its MR and MIMIC lower
  bounds against R0 > −0.03.
- **Choose the qualifying arm with the highest CG v2 micro.** The fuller-fix arm (the higher-numbered) wins ties within
  max(0.005, |R0 − R0'|).
- If no arm qualifies, outcome (4) applies. Choose nothing and report.

Delete each arm's `full/` after its result is recorded. Write `loops/osler_v5/arms.md`.

### Z3 — Final Osler-4B (GPU, ~24 h)

- The chosen arm's recipe on the same mix, **63,304 steps** (osler_v4's length, so the final compares one-to-one with osler_v4's;
  Y8 took 20.7 h without KL, and this run takes about 24.5 h with it).
- Ten evaluations on fast dev v4 + CG v2 at Y8's steps (6,330 … 63,304), so the step-63,304 R0 is free for comparison.
- Full checkpoints every 2,500 steps (newest two kept), watchdog per ADVISORY §3.7.
- **Checkpoint selection by S** among evaluations that pass positional health; if none passes, outcome (4).
- Record S, its parts, knowledge dev and positional health at every evaluation.
- Temperatures fitted on dev and saved with the checkpoint. Refuse to run twice (the `arm_result.json` guard, resume allowed).

### Z4 — Blend (GPU, ~3 h)

α ∈ {0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0} on the selected checkpoint; temperatures refitted per α on fast dev v4. Report Brier
and the D12 readout gate at each α.

**Rule** (pre-registered):
- Retention: for each of the MR and MIMIC dev macros, the gain over α = 0 at α must be ≥ 80 % of the gain at α = 1.
- An α qualifies if it passes the retention rule, positional health, and CG v2 micro ≥ its α = 1 value.
- **Choose the smallest qualifying α.** If none qualifies, α = 1 is kept and the blend is reported as not used (outcome 3).

Export the chosen blend (adapter scaling, calibration saved) to `outputs/osler_v5/Z4/final_blended/`. Write `loops/osler_v5/blend.md`.

### Z5 — Evaluation and Gate O5 (GPU, ~3 h)

Score the blended final (calibrated) on the frozen sets: the MR suite v1 (verdict and reported-only), the MIMIC audit suite v1, the
held-out files, the external panel, the knowledge test, MedNLI test. Also score the unblended (α = 1) checkpoint as an additional row.

**Apply Gate O5 (D55 = Gate O4 unchanged) exactly.** Write the pre-registration into STATE §7 before any test-set scoring. Scoreboard
rows (additional, from existing predictions): zero-shot pplx, MedDecider-4B/-9B, JEV-9B, pplx-27B (ceiling), osler_v2, osler_v4 final.
Report the panel per dataset, the MedDecider-4B panel comparison on the 4,141 pairable items as an additional row, the release
targets of §2, and the DDXPlus code split. Write `gate_o5.md` and `results.md`.

### Z6 — Membership audit (small GPU, ~1 h)

osler_v4 Y10's membership test (2,000 MIMIC train vs 2,000 test items, same templates) on the blended final; AUC with interval.
The verbatim probe is not repeated (osler_v4 C236). Write `loops/osler_v5/audit.md`.

### Z7 — Findings and closure — HARD STOP

FINDINGS (Summary first; §2 outcomes; arms, the final's S trend, the blend curve, the gate, the scoreboard; the five-claim
spot-check), NEXT.md, `Loop status: STOPPED`, commit, push, stop.

## 5. Task table (seed STATE.md with this)

| id | task | GPU | deps |
|---|---|---|---|
| Z0 | Orientation, recipe code (D51–D54), CPU replays and checks | no | — |
| Z1 | Probes on the osler_v4 final (blend curve, readout split) | yes | Z0 |
| Z2 | Recipe arms R0, R0', R2, R3 and the choice | yes | Z0 |
| Z3 | Final Osler-4B (63,304 steps) | yes | Z2 |
| Z4 | Blend α on dev and export | yes | Z3 |
| Z5 | Evaluation, Gate O5, scoreboard | yes | Z4 |
| Z6 | Membership audit | small | Z4 |
| Z7 | Findings and closure — HARD STOP | no | all |

GPU sequence: Z1 → Z2 → Z3 → Z4 → Z5 → Z6. GPU hours (planning estimates): Z1 3, Z2 18, Z3 25, Z4 3, Z5 3, Z6 1 — **about 53
GPU-hours; budget 4.5 days with overhead.** While Z3 trains, pre-write Z4–Z5 code (CPU only; no suite run during the job).

## 6. Out of scope

- Osler-9B (next loop, on this loop's recipe) and Osler-0.8B; base, attention, head or adapter changes.
- New data sources, teachers, Nemotron work, 27B distillation.
- DDXPlus training data; any panel, test, MIMIC-suite or held-out influence on a choice.
- Changing a frozen set, Gate O5, the choice rules, the blend rule or the closed list after any run.
- Releasing or uploading anything; the HF backup.

## 7. Failure modes to avoid

1. **Augmentation mislabels.** A distractor that is also a true label makes the gold wrong. Only the closed list; deduplicate by code
   and title; when unsure, exclude.
2. **Per-item weighting does not reach MIMIC's share.** Z0.2's replay check is a gate for the arms.
3. **The blend breaks calibration.** Temperatures are refitted per α; Brier at each α is reported.
4. **A baseline file does not match its set.** Check item ids and counts before Gate O5; a mismatch is BLOCKED, never re-scored silently.
5. **Silent process death.** Watchdog and no large writes during GPU jobs (ADVISORY §3.6–3.7).

## 8. Corrections of 2026-10-27 (D56; supersedes §4 Z2–Z3 where they differ)

Reason: the Z2 seed repeat (R0 prime) fails positional health and drops CG micro by 0.042 against R0 (C052–C053); the mix holds
155 `fda_route_choice_v1` rows whose gold is `routes[0]` while other routes are also true (C048, STATE §6.1). The corrections
were decided by the operator in chat on 2026-10-27 and are pre-registered here before any corrected-mix arm runs.

1. **Stop R3 now** (the running R3 job, on the defective mix). Record it as stopped in STATE §7 with its last step. Its partial
   results are kept as additional rows, not a candidate.
2. **Corrected mix (D56a).** Drop every `fda_route_choice_v1` row from the training mix. Do not try to keep the rows whose defect
   condition holds: the template is dropped whole, because the condition cannot be verified per row. Report the counts (rows
   dropped, rows remaining), the new composition hash, and the reason. Check the dropped set against C048 (155 rows with 17 or 18
   options); if the number differs, report the difference. Nothing else in the mix changes. Its hash is recorded in STATE §2.
3. **Corrected arms (D56b).** Each runs 13,992 steps on the corrected mix, evaluations at 6,996 and 13,992:
   - F0: osler_v4's recipe, per-step weighting, no augmentation, seed 0;
   - F2: per-item weighting and option augmentation, seed 0;
   - F2s1: F2 at seed 1.
   Order: F2, then F0, then F2s1. The chain runs under the watchdog (§3.7). Delete each `full/` after its result.
4. **Choice rule (D56c; pre-registered; fixed now).** A qualifier passes positional health in all three orders at 13,992 steps
   (for F2s1, the seed-1 pass is reported whatever it is), has a knowledge-dev lower bound against F0 > −0.01, and has MR and MIMIC
   lower bounds against F0 > −0.03 (paired bootstrap, 1,000 resamples, seed 0). The choice is the qualifier with the highest CG v2
   micro. If the choice is F2 and F2s1 is not a positional pass, the recipe is reported as seed-unstable and still goes to the final.
   An arm within 0.010 of the top CG micro that is the higher-numbered (fuller-fix) arm wins the tie.
5. **Final (Z3, corrected).** The chosen recipe on the corrected mix, seed 0, **63,304 steps**, ten evaluations at Y8's steps,
   positional health at every evaluation, checkpoint selection by S among passing evaluations (§4 Z3 rule, unchanged).
6. **Readout-scale check (D56d; CPU only, may run while the GPU job runs).** Load the osler_v4 final, the F2 checkpoint (corrected mix)
   and pplx's base readout; for rows A–Z and rows 26–254 report the mean row norm, and the mean distance from the base row. Report
   the paired difference between F2 and the osler_v4 final. Additional row, labelled; no choice uses it. Do not load any MIMIC text.
7. **Z4–Z6 and the stop** unchanged. Expected timing (planning estimate): steps 1–3 about 14 GPU-hours, the final about 24, Z4–Z6
   about 7. Closure before 2026-11-01 is the target.
8. **What this loop cannot claim:** whether the corrected mix alone fixes the collapse is measured by F2 and F2s1, and by the
   corrected final's ten evaluations. A result that does not fix it is reported as such.

## 9. Addendum Z8 — the corrected recipe at full length (additional; operator-approved 2026-10-28)

The operator approved, in chat on 2026-10-28, one additional run after the closure of Z7. It is labelled additional and chooses
nothing.

1. **Run:** the F2 recipe (per-item weighting, option augmentation, seed 0) on the **corrected mix** (D56a), **63,304 steps**, with
   the Z3 driver (ten evaluations at Y8's steps, positional health at every evaluation, full checkpoints every 2,500 steps, the
   watchdog of §3.7). The recipe and the mix are those of F2 (C062), so this run is F2 at full length.
2. **Refuse to run a second copy** (the `arm_result.json` guard). Resume is allowed.
3. **Reporting:** the ten-evaluation curve of S, its parts, CG micro, knowledge dev and positional health (fraction of passing
   evaluations and the two-letter values), and the final checkpoint's dev figures. Test scoring after training is allowed as an
   **additional, labelled** row (MR suite v1, MIMIC audit suite v1, panel, knowledge test), with no verdict and no effect on any choice.
4. **Gate O5 is not applied.** Its rule is for a chosen recipe, and this run is not chosen (outcome 4 stands; C070).
5. **Z8 does not reopen Z2.** The choice rule of D56c is unchanged, and F2's qualifier status at 13,992 steps stays as C070 records.
6. **Time:** about 24 GPU-hours. Write `loops/osler_v5/z8_full_length.md` and the CLAIMS rows. Then update `docs/paper/EVIDENCE.md` §5
   with the new rows (aggregates only), commit and push, and stop.
