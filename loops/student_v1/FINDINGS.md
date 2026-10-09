# student_v1 — findings (loop 1b: correct G1, padding fix, data diversity, three decision heads)

Generated from the loop's claims (`loops/student_v1/CLAIMS.md`). Every number below has a claim row; the
artifact and recompute command are in that row. Scope: MedDecide-Bench v0.2 test items (37,289: fresh 23,768,
established 13,521), held-out templates per D14, the D12 readout-health gate, and the G1 rule of D16.

## Summary

1. **Arm A (pointer head) and arm C (Unsloth decision head) are close overall; neither clearly dominates.**
   Overall test accuracy: A 0.7591, C 0.7545, B 0.7528 (V113). Arm A passes G1 on the seen fresh templates against
   zero-shot Qwen3.5-0.8B and against JEV-9B (V079, V080), and fails on the held-out templates (V081, V082).
   Arm C passes G1 on the seen fresh templates (V105) and fails on the held-out templates against JEV-9B (V105).
2. **The arm-vs-arm sign depends on the scope.** A minus C on the seen templates is -2.78 pts on the 7 fresh seen
   templates (V109), but +1.14 pts on all 15 seen templates including the established tier (V110). On the held-out
   templates, C is ahead in both scopes: -5.90 pts on the 3 fresh held-out templates with a verdict (V109) and -4.48
   pts on all 4 held-out templates (V110). The decision between heads therefore rests on which scope the program
   reads, which is operator question 6 (§5).
3. **Arm B (letter readout) does not pass the readout-health gate on most templates.** D12 fails on 8 of the 11 fresh
   templates and on all four held-out cells (V089): the median full-vocabulary label mass on the offered letters is
   0.05 to 0.32 there. A probe shows this is how the readout behaves, not a measurement error: the top token is an
   offered letter in all 152 probe items (V100), but in the failing templates a fifth to a third of the probability sits on
   letters that are not offered (V101). Arm B's G1 therefore rests on three fresh templates (seen PASS vs zero-shot +0.3056 and vs JEV-9B
   +0.0953, V090, V091) and its held-out G1 is NOT MEASURED (V092).
4. **No arm transfers to the held-out templates.** On `ct_arm_role_noul_v1` (balanced, held out) arms A and B answer
   the same letter for 1,999 and 1,973 of 2,000 items, so both sit at chance (V104); arm C is excluded there by the
   constant-answer check (V105). The held-out G1 verdicts for A and C are FAIL (V081, V082, V105).
5. **The training data's option order carried a strong position prior, which the arms must not learn.** The train
   file's gold option is last in 90.8 % of 6-option choice items and first in 64.4 % of 3-option items (V085). Arms A
   and B permute option order each epoch; arm C initially did not, so arm C would have learned the prior. With the
   same permutation applied to arm C, every choice group is uniform (V086). This is the main design lesson of this loop
   for any future head comparison.
6. **Measurement defects found during this loop were corrected, and none changed a number already reported.** G1
   misclassified the Unsloth head's rows as letter readouts, which failed all 11 fresh cells for arm C (V106); the fix
   is a classification correction with thresholds unchanged, and arms A and B reproduce exactly. Path bugs in three
   evaluation scripts were fixed and their outputs re-run (V117). A duplicate training run started by a stale shell
   was stopped before it wrote anything (V8 self-audit).
7. **Not measured (R3):** HLE (supplementary; not scored by any arm), the option-shuffle flip rate, the long-record
   slice (the membership is not stored in the items and the counts file predates the screen's 495-item drop),
   per-record latency, and arm B's held-out G1. Reasons are in `arms.md` §5.

## §1 Setup, in brief

- Base: `Qwen/Qwen3.5-0.8B`. Training mix: V4 (482,889 train rows, 24 templates; dev 5,619 rows, 28 templates).
- Arm A: pointer head, LoRA r=8, 15,000 steps, batch 8 (`configs/student_v1_arm_a.yaml`). Arm B: letter readout,
  same LoRA, restricted softmax over the offered letters + Brier. Arm C: Unsloth `FastDecisionModel` with the
  decision head, LoRA r=16, 15,000 steps, batch 8 with the arms A/B batch order (D15) and option order (D16).
- Dev selection on the V4 dev set: pooled-class macro accuracy, Brier tie-break (ADVISORY S9 rule).
- Evaluation: every arm in file order (no option shuffle), no padding (batch 1). Baselines: zero-shot
  Qwen3.5-0.8B and JEV-9B (`autotrust/JEV-9B`, scored on 35,097 of 37,289 items; the rest exceed the 8,192 cap).
- Expectation references (V116): uniform guess 0.3345, per-template majority 0.3506 (test items).

## §2 Per-arm outcome

### Arm A — pointer head (selected step 12,500)

- Training: 15,000 steps, wall 8,609 s, dev pooled macro 0.7603 at the selected step (V067–V071).
- Overall test accuracy 0.7591; fresh 0.8313; established 0.6323 (V113).
- G1 (D16, fresh tier): seen PASS against zero-shot (macro +0.1626, CI [+0.1500, +0.1755]) and JEV-9B on the
  intersection (+0.0426, CI [+0.0260, +0.0598]) (V079, V080). Held-out FAIL against zero-shot (-0.0302, CI
  [-0.0488, -0.0115]) and JEV-9B (-0.2250, CI [-0.2445, -0.2054]) (V081, V082). Strict-slice verdicts in
  `g1_arm_a.md`.
- Constant answer on `ct_arm_role_noul_v1` (V104): accuracy 0.5005.

### Arm B — letter readout (selected step 5,000 by the tripwire rule)

- Training: 15,000 steps, wall 8,688 s. The pre-clip grad norm exceeded 5,000 once, at step 13,254 (6,022.8); the
  ADVISORY rule records the run as diverged and evaluates it at its best checkpoint at or before the trip, step
  5,000 (dev pooled macro 0.6997; the unrestricted best, step 14,000, is reported as additional, operator question 3).
- Temperatures refitted at step 5,000 on the dev set (V088).
- Overall test accuracy 0.7528 (V113).
- D12 (fresh tier, V089): PASS on `ct_claim_set_choice_v1`, `fda_class_choice_v2`, `fda_route_claim_noul_v1`; READOUT_FAIL
  on the other eight, on median label mass 0.05 to 0.32; two also fail the constant-answer check.
- Readout probe (additional, V100 and V101): the top token is an offered letter in 152 of 152 items; the restricted
  argmax equals the greedy letter in 151 of 152; the mass on non-offered letters is 0.005 to 0.337 by template.
  Corrected range in V103.
- Prompt cap (V102): 2,213 of 37,289 test items (5.9 %) reach the 8,192-token cap, all in three openFDA templates;
  the scoring path cuts their leading tokens, the same for every arm.
- G1 (fresh, D12 applied): seen PASS vs zero-shot (macro +0.3056, CI [+0.2986, +0.3132]) and JEV-9B
  (+0.0953, CI [+0.0879, +0.1026]), both on the three PASS templates only (V090, V091). Held-out NOT MEASURED (V092).

### Arm C — Unsloth decision head (selected step 5,000, unrestricted; no trip)

- Training: 15,000 steps, 6,025 s after a 1,042 s dataset build (V096, V118). Mean training loss 0.5056. Pre-clip grad
  norm p50 3.485, p95 14.623, max 384.776: the tripwire (5,000) did not trip (V096). 120,000 items used, all
  distinct (V097); option order changed for 285,077 items (V098).
- Dev selection (S9 rule): step 5,000, pooled macro 0.6851, accuracy 0.7325, Brier 0.3696 (V099). The runner-up,
  step 7,500, is 0.6842, a margin of 0.0009. The lowest dev macro over the 30 checkpoints is 0.512, above the 0.50
  divergence rule.
- Calibration on seen dev (Unsloth `calibrate`): accuracy 0.7322, ECE 0.0256 (V117).
- Overall test accuracy 0.7545; fresh 0.8392 (the best fresh figure of the three); established 0.6057 (V113).
- G1 (fresh, D12 applied; only `ct_arm_role_noul_v1` excluded by the constant-answer check): seen PASS against zero-shot
  and JEV-9B; held-out FAIL against JEV-9B; strict seen FAIL against JEV-9B (CI lower bound -0.0001); strict held-out
  FAIL (V105).
- Latency (V115) is not like-for-like: arm C ran on the reference convolution kernel because `causal_conv1d` is
  not installed in its environment.

## §3 Arm-vs-arm

Difference = left minus right, paired bootstrap, 95 % percentile interval, macro accuracy (V107–V112; full tables in
`arms.md` §1).

| pair | scope | seen | held-out |
|---|---|---|---|
| A − B | verdict (fresh; B's D12 cells dropped) | +1.13 [+0.78, +1.48], 3 templates (V107) | not measured: no common item |
| A − B | additional (all tiers, no D12) | -1.24 [-2.08, -0.44], 15 templates (V108) | -0.06 [-1.39, +1.28], 4 templates (V108) |
| A − C | verdict (fresh; ct_arm_role dropped for both) | -2.78 [-4.15, -1.41], 7 templates (V109) | -5.90 [-7.68, -4.24], 3 templates (V109) |
| A − C | additional (all tiers, no D12) | +1.14 [+0.21, +2.08], 15 templates (V110) | -4.48 [-5.79, -3.18], 4 templates (V110) |
| B − C | verdict (fresh; B's D12 cells dropped) | -0.49 [-0.87, -0.13], 3 templates (V111) | not measured: no common item |
| B − C | additional (all tiers, no D12) | +2.38 [+1.53, +3.22], 15 templates (V112) | -4.41 [-5.90, -2.91], 4 templates (V112) |

Reading: arm C is ahead of arm A on the held-out templates under both scopes. On the seen templates the sign flips
with the scope. Arm B's comparisons rest on the three templates where its readout passes D12.

Dev selection and test accuracy disagree in part: on the dev metric arm A (0.7603) is ahead of arm B (0.6997) and
arm C (0.6851), but on the test set arm A and arm C are within 0.5 pts overall (V113, V099, V067).

Selective accuracy (additional, V114): at 50 % coverage C 0.9902, A 0.9620, B 0.9464; at 80 % and 90 % the three arms
are within 0.9 pts.

## §4 Measurement and process notes

- **Path bugs (fixed, outputs re-run).** `v7_letter_eval.py` (final print; its outputs were complete), `v7_temperatures.py`
  (the checkpoint path), and `v8_finalize.py` (the final `relative_to`; its first run wrote all test predictions but no
  `finalize.json`). The finalize rerun is identical to the first run (V117).
- **G1 classification (fixed, V106).** Arm C's rows were classed as letter readouts until the decision head was declared
  non-letter. A and B were rerun with their saved commands and are identical on every field.
- **Duplicate training run (stopped, no output).** A stale chain shell started a second arm C training run at 15:47 after
  its script had been overwritten. It was stopped before writing anything. The lesson is in the memory notes.
- **The 14:50 arm C run** reported "done" after failing, because its chain did not gate on exit codes. It is superseded
  by the gated run.
- **Pair scope.** The first A−B pair (V093–V095) excluded only arm B's cells. The rerun (V107–V108) also applies arm A's
  exclusions, which leave the seen verdict unchanged and affect only the held-out set, which is NOT MEASURED for that pair.

## §5 Decisions for the operator

Open items in `STATE.md` §6 (summarised here; the loop continued with the default shown in brackets):

1. V2 acceptance item 1 (the padded-batch tolerance) [the no-padding batch-1 path is the evaluation path].
2. V4 record–claim consistency designs and catalog sources not built [deferred to loop 2].
3. Arm B's tripwire rule: step 5,000 versus 14,000 for its verdict [5,000, by the rule as written].
4. Arm C's batch order and option order: the matched design versus the first design [the matched design].
5. D12 on the trained letter readout: report as the protocol states (arm B's G1 on three templates; held-out NOT
   MEASURED) versus redefine the readout or amend D12 [report as the protocol states].
6. Which scope the head comparison reads: the fresh verdict scope (seen templates favour arm C by -2.78 pts; held-out
   favour arm C by -5.90 pts) or the all-tier additional scope (seen favour arm A by +1.14 pts; held-out favour arm C by
   -4.48 pts) [no default: this is a program decision, and the loop does not pick a head].

Also open from earlier loops: the JEV-9B coverage gap (2,192 items over the 8,192 cap), and the seen-template
score dev count (33 against a target of 200, D9).

## §6 Program gate

The evidence does not clearly favour one head. Arm C is ahead on the held-out templates in both scopes and on the
fresh seen verdict scope; arm A is ahead on the all-tier seen scope and has the highest overall test accuracy, by
0.46 pts over arm C (V113). Arm B fails the readout-health gate on most templates, so its comparison is not a full result.
Under the ADVISORY rule the program keeps the head the evidence favours; that decision belongs to the operator
(question 6) and is not made in this loop.

## Spot-check (five claims, recomputed in this closure; outputs written to the scratchpad, saved artifacts untouched)

| claim | recompute (re-run) | recorded | re-run |
|---|---|---|---|
| V099 arm C selection | `v8_select.py` (dev_raw, dev items) | step 5,000, macro 0.6851 | step 5,000, macro 0.6851 |
| V086 option-order uniformity | `v8_option_positions.py --augment` | largest choice abs z 2.62; 285,077 changed | 2.62; 285,077 |
| V089 D12 on arm B | count of fresh arm B cells with no failing check, `g1_arm_b.json` | 3 PASS of 11 | 3 PASS of 11 (ct_claim_set_choice_v1, fda_class_choice_v2, fda_route_claim_noul_v1) |
| V109 arm A minus arm C, verdict scope | `v9_arms.py` with the per-arm exclusions | seen -0.0278, n = 17,442 | seen -0.0278, n = 17,442 |
| V105 arm C G1 verdicts | `g1.py` for arm C with the tier-1 option | fresh_seen PASS, fresh_heldout FAIL, strict_seen FAIL, strict_heldout FAIL | the same four verdicts; ct_arm_role_noul_v1 the only excluded fresh cell |
