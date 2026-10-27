# osler_v5 — RUN STATE

> **This file is the loop's memory.** The live copy is `loops/osler_v5/STATE.md`
> (committed, so the operator can review it remotely). Read it at the start of every
> iteration, before the plan. If it disagrees with your recollection, **the file
> wins** — your context may have been compacted since the last iteration.
>
> **How to update:** set the task `IN_PROGRESS` with a UTC start time *before* working;
> on completion set `DONE` or `BLOCKED — <reason>`, fill the finished time, and append a
> ≤5-line entry to the iteration log. Never delete a log entry; append only. Timestamps
> are `date -u +%FT%TZ`. Never paste item text, predictions, or secrets into this file.

Loop status: `RUNNING`  <!-- set to STOPPED at the hard stop (Z7), or when no PENDING task can proceed without the operator -->
Run started (UTC): `2026-10-26T19:24:27Z`
Last updated (UTC): `2026-10-27T00:00Z`
Iterations so far: `1`

---

## 1. Task board

Statuses: `PENDING` → `IN_PROGRESS` → `DONE` | `BLOCKED — reason`

| id | task | GPU | deps | status | started (UTC) | finished (UTC) |
|---|---|---|---|---|---|---|
| Z0 | Orientation and recipe code (D51–D54) | no | — | DONE | 2026-10-26T19:24:27Z | 2026-10-26T19:58Z |
| Z1 | Probes on the osler_v4 final (blend curve, readout reset) | yes | Z0 | DONE | 2026-10-26T19:59Z | 2026-10-26T23:00Z |
| Z2 | Recipe arms R0–R3 and the choice | yes | Z0 | PENDING | | |
| Z3 | Final Osler-4B (63,304 steps) | yes | Z2 | PENDING | | |
| Z4 | Blend α on dev and export | yes | Z3 | PENDING | | |
| Z5 | Evaluation, Gate O5, scoreboard | yes | Z4 | PENDING | | |
| Z6 | Membership audit | small | Z4 | PENDING | | |
| Z7 | Findings and closure — HARD STOP | no | all | PENDING | | |

Rules: take the **first** `PENDING` task whose deps are all `DONE` (order exceptions and the GPU sequence in ADVISORY §5).
Never run two GPU jobs at once. A `BLOCKED` task does not block unrelated work. No row may be
left `IN_PROGRESS` at closure.

---

## 2. Key values discovered during the run

**Operator confirmations** (filled by the operator before kickoff; the loop never edits this line):
- D51–D55 (per-item weighting, option-count augmentation and positional health, blend, dev composite, Gate O5): `yes` — recorded 2026-10-26 by the advisor on the operator's standing instruction in chat ("I agree with whatever you go with"), after the independent review (commit a9ca386).

| key | value | source | task |
|---|---|---|---|
| suite counts (passed / skipped / deselected); disk headroom test | baseline 1042 passed, 1 skipped, 2 deselected, exit 0 (174 s, 2026-10-26T19:24:52Z), with the two osler_v4 deselects; ruff clean. Disk: 2 GB write in 2.0 s (1.1 GB/s), test file removed (2026-10-26T19:24:49Z) | `outputs/osler_v5/Z0/logs/pytest_baseline.log` | Z0 |
| mix composition SHA-256 (must equal 9d25e30f…) | `9d25e30f53d7823d051b9c71334c248660318c86ce904c26f855b9a9dbc618b8`, equal to osler_v4's. 1,456,593 rows (base 830,076; family kept 509,807; replay 291,319 = 20.00 %; top-up 116,710); 74 templates; largest template 4.80 % | `outputs/osler_v5/Z0/logs/compose_check.log`; `outputs/osler_v5/Z0/mix_template_stats.json` | Z0 |
| corrected mix (ADVISORY §8.2, D56a): composition SHA-256 after dropping every fda_route_choice_v1 row | `78d7e5478060fc828784cb3365569e177c6961270d38a76a9cff144684e9f3be` (recorded 2026-10-27T14:32Z). Rows before 1,456,593; dropped 155 (fda_route_choice_v1 only; difference from C048's 155 rows: 0); remaining 1,456,438; templates 74 → 73; no other template's count changed. Replay rows 291,319 (share 20.002 %, minimum kept). Family rows 509,807 unchanged: family share 35.0037 %, above the 35 % cap by 54 rows (reported, not re-sampled, per ADVISORY §8.2: nothing else changes) | `outputs/osler_v5/Z0/corrected_mix.json`; C059 | Z2 |
| closed-set list (template, label-set size, reason); expected share of trained items with > 26 options | **On the list** (reasons in `CLOSED_LIST`, `src/meddecide/train/option_augment.py`): `mimic_principal_dx_v1` (label set 4,457; 37,843 rows; one principal diagnosis per admission, distractors deduplicated by code and title) and `mimic_msdrg_v1` (672; 37,844 rows; one MS-DRG per admission). **Excluded by name:** `mimic_rad_exam_v1` (ADVISORY Z0.3), `mimic_icd_support_v1` (not exclusive by construction; conservative), `fda_route_choice_v1` (defect, §6). **Every other template is excluded** (not a closed set by construction, or unverified). Expected share of trained items above 26 options: 3.51 % at 13,992 steps, 3.46 % at 63,304 | `outputs/osler_v5/Z0/augment_expectation.json` | Z0 |
| expected two-letter golds per row (mean over rows 26–63 at p = 1; the minimum row) | arm (13,992 steps): 2,818 eligible of 53,572 realised items (5.26 %); mean 19.35, minimum 0.77 (row 63); the p that would give a mean of 20 is 1.03, so unreachable. Final (63,304 steps): 12,584 eligible of 242,255 (5.19 %); mean 86.41, minimum 3.45; p needed 0.23. **Chosen p = 1.0** (§7, deviation 1) | `outputs/osler_v5/Z0/augment_expectation.json` | Z0 |
| per_item check: MIMIC share of the applied update, replayed on Y8's batches | per_step 0.5880 (clip-aware on the 1,266 logged steps; full run unclipped 0.5773). per_item 0.3365 (clip-aware; full run unclipped 0.3417; 82,789 MIMIC of 242,255 realised items). Replay against Y8's record: 0 mismatches in items, real tokens and padded tokens over 63,304 steps; the 1,266 logged steps match on counts. Stop rule (about 0.34–0.40): met, as 0.3365 rounds to 0.34 (§7, deviation 4) | `outputs/osler_v5/Z0/replay/summary.json`; `outputs/osler_v5/Z0/logs/replay.log` | Z0 |
| S, its parts and positional health for zero-shot pplx, A0, osler_v4 final (additional) | **osler_v4 final:** S 0.8386 (MR template macro 0.9672 on 3,197; MIMIC template macro 0.8094 on 5,562; CG micro 0.7393 on 8,307); positional (original order) two-letter 0.0040 (2 of 499) against single-letter 0.6407: **FAILS**, below the 0.0204 chance. **A0:** S 0.7869 (MR 0.9603; MIMIC 0.6126; CG micro 0.7879); positional 0.4790 against 0.8104: passes. **Zero-shot pplx-flash:** S NOT MEASURED (no MR-partition file for fast dev v4; MIMIC template macro 0.6659 on 5,562; CG micro 0.7888 on 8,296 items); positional 0.6112 against 0.7964: passes. The MR and MIMIC parts are template macro (D54's default; CG micro is explicit); the earlier micro values in C016–C018 are superseded (§7.10) | `outputs/osler_v5/Z0/composite.json`; `outputs/osler_v5/Z0/logs/composite.log` | Z0 |
| CG check: the osler_v4 final against the α = 0 reference on identical items (D54; ADVISORY Z0.6) | on the 8,296 identical CG v2 items: final 0.7395, zero-shot pplx-flash 0.7888, gap −0.0493. The final **fails** the CG qualifier, so the check discriminates. The α = 0 reference itself is re-scored on GPU in Z1 with the reference path (§7, deviation 6) | `outputs/osler_v5/Z0/composite.json` (`cg_micro_identical_items`) | Z0 |
| D52(b) positional check under two seeded permutations (DDXPlus dev) | NOT MEASURED in Z0: scoring under the permutations needs the model on the GPU. The code is in place (`permutation_for`, `positional_health`, `run_arm.py`); the osler_v4 final, A0 and zero-shot pplx are scored in Z1 | — | Z0 → Z1 |
| probes: blend curve on the osler_v4 final; readout-reset DDXPlus split | alpha 0 to 1: S (MR and MIMIC template macro; CG micro) 0.7820, 0.8252, 0.8395, 0.8415, 0.8414, 0.8386. Positional health passes at alpha 0, 0.3, 0.5 and fails from 0.7 in all orders; DDXPlus two-letter 0.6112 → 0.0040. Knowledge dev -0.0140 [-0.0228, -0.0055] at alpha 1 (MedMCQA significant, C036). Readout split: the trained readout alone takes two-letter to 0.0381; the trained adapter alone to 0.1924 (C032–C039) | `loops/osler_v5/probes.md`; `outputs/osler_v5/Z1/summary.json` | Z1 |
| arms R0–R3: S, parts, knowledge, positional health, MIMIC gradient share; the chosen arm | | `loops/osler_v5/arms.md` | Z2 |
| final: selected step; S trend; positional health per evaluation | | | Z3 |
| blend: S, parts, CG floor per α; the chosen α | | `loops/osler_v5/blend.md` | Z4 |
| Gate O5 verdict; release targets | | `loops/osler_v5/gate_o5.md` | Z5 |
| membership AUC | | `loops/osler_v5/audit.md` | Z6 |

---

## 3. Current task checklist

**Fill this in before starting any task**, by copying that task's concrete steps out of
the plan as unticked boxes. Tick each one the moment it is finished, not at the end.
This is the only record of progress *within* a task — without it, a compaction mid-task
leaves you unable to tell what you already did.

Clear this section and write the new task's checklist when you start the next task; the
completed checklist goes into the iteration-log entry.

```
Task in flight: Z2 — recipe arms, started 2026-10-26T23:01Z (GPU; one job at a time)
Working dir:    outputs/osler_v5/Z2/ (gitignored; the committed record is the claims, arms.md and this file)
Running job:    Z2 chain (scripts/osler_v5/z2_chain.sh: R0, R0p, R2, R3 in turn); chain log outputs/osler_v5/Z2/chain.log. R0 finished (exit 0 at 03:41:17Z; S 0.8288 at 13,992; positional passes in all orders; C041–C044). R0 prime finished (exit 0 at 08:13:55Z; S 0.8159; CG micro 0.7346; positional FAILS in all orders, two-letter 0.0000 / 0.0041 / 0.0021 below the 0.0204 chance; C052–C053; tie band 0.0416). R2 finished (exit 0 at 12:50:49Z; S 0.8311; CG micro 0.7986; positional passes in all orders; MR −0.0008, MIMIC −0.0146, knowledge +0.0135 against R0 with the lower bounds −0.0088, −0.0234, +0.0045, so R2 meets the qualifying bounds; C056–C058). R3 started 2026-10-27T12:50:52Z: python PID 3292313 (uv 3292307); log outputs/osler_v5/Z2/R3/logs/run.log. Memory: allocated peaks 57 GB; the allocator's reserved cache reached about 95 GB in R0 prime.

Z2 checklist (ADVISORY Z2):
- [ ] 1. disk headroom: 2 GB write before each job
- [x] 2. smoke run of scripts/osler_v5/run_arm.py with --smoke (R2 config, 8 steps, one evaluation on cut dev sets): passed 2026-10-26T23:09Z; its outputs deleted
- [ ] 3. scripts/osler_v5/z2_chain.sh: R0, R0p, R2, R3 at 13,992 steps (evaluations at 6,996 and 13,992); each arm's full/ deleted after its arm_result.json
- [ ] 4. scripts/osler_v5/z2_choose.py → outputs/osler_v5/Z2/choice.json (qualifiers, the choice, the tie band); loops/osler_v5/arms.md written from it
- [ ] 5. CLAIMS rows for the arms and the choice; STATE; commit; push
```

---

## 4. Iteration log (append only, newest last)

<!-- Template for each entry:
### <task-id> — <DONE|BLOCKED> — <UTC timestamp>
- What ran: <command or script>
- Output: <path>
- Headline: <one number or one sentence, with its CLAIMS id>
- Surprises: <anything unexpected, or "none">
- Next: <what this unblocks>
-->

<!-- The Headline line matters beyond this file: these headlines are the raw material
     FINDINGS.md's Summary section is written from at closure. A headline that only
     makes sense with full context ("done, see report") starves the summary. Write
     each one so a reader who has not seen the task can repeat it: what was measured,
     what came out, with what denominator. -->

### Z0 — DONE — 2026-10-26T19:58Z
- What ran: ruff and the suite before and after the code (C001–C002); composer check; `scripts/osler_v5/z0_{mix_stats,replay,augment_expectation,composite,blend_check,rederive}.py` (CPU); 38 new unit tests; outputs in `outputs/osler_v5/Z0/` (gitignored), claims C001–C025, `SELF_AUDIT.md`.
- Headline: the replay of Y8's 63,304 batches matches Y8's record exactly (0 mismatches, C007). The MIMIC share of the applied update is 0.588 under per_step and 0.337 under per_item (clip-aware, C010–C011). The α = 0 blend matches the reference path to 2e-10 and α = 1 matches the trained model exactly (C021, C023). The osler_v4 final fails the positional check (two-letter 0.0040 against single-letter 0.6407, C019) and trails zero-shot on identical CG items (0.7395 against 0.7888, C020).
- Surprises: p = 1.0 gives 19.35 expected two-letter golds per row at arm length (target 20, C013) and 86 at final length (C014). The prediction ids are re-derived, so the item-id question (operator 5) stays open. The zero-shot CG file lacks 11 items and has no MR partition on fast dev v4.
- Next: Z1 (GPU probes): blend curve, readout split, α = 0 reference and permutations on dev, written to `loops/osler_v5/probes.md`.

### Z1 — DONE — 2026-10-26T23:00Z
- What ran: `scripts/osler_v5/z1_probes.py` on the GPU (eight settings and the reference check, dev only, 2 h 55 min; α = 0 against the reference path 1.19e-07); `z1_summarise.py` and `z1_report.py` (CPU; `loops/osler_v5/probes.md`); claims C030–C039; ruff and the suite (1097 passed, 1 skipped, 2 deselected, exit 0).
- Headline: on the osler_v4 final, positional health passes at α 0 to 0.5 and fails from 0.7 (DDXPlus two-letter 0.6112 → 0.0040); S (template macro) peaks near 0.7; knowledge dev falls by 0.0140 at α = 1 (MedMCQA significant, C036). The trained readout alone takes two-letter to 0.0381, the trained adapter alone to 0.1924 (C035).
- Surprises: the probe's composite was item micro for MR and MIMIC (superseded, §7.10); the final's single-letter DDXPlus accuracy 0.6407 contradicts osler_v4's "intact" wording (C039); the Monitor's log-age stall alarm fired while the GPU was at 100 % (no resume).
- Next: Z2 — a smoke run, then the four-arm chain (R0, R0', R2, R3) and the choice rule.

## 5. Blocked items

| id | what is blocked | exact reason | what would unblock it |
|---|---|---|---|

---

## 6. Questions for the operator

Anything you could not resolve without a human. Be specific enough to answer without
re-reading the run: state the ambiguity, the options, and which you would pick.

1. **`fda_route_choice_v1` is defective (ADVISORY Z0.3; the finding asked for).** The item builder
   (`src/meddecide/bench/fresh/openfda.py`) sets gold to `routes[0]` and offers the whole route vocabulary, so a label that lists
   several routes has other true answers. The template has 155 rows in the mix. Options: (a) report it and leave the mix unchanged,
   which this loop must do because the mix hash is fixed; (b) rebuild the template with multi-route handling in a new loop. **I would
   pick (a) now and list (b) in NEXT.md.** It is excluded from augmentation (`EXCLUDED_NAMED`).
2. **Item-id scheme (operator question 5, still open).** The prediction files carry the id that `mix_row_to_item` derives, not the id
   on the frozen item row. Every join in this loop uses the derived id and checks the id set against the frozen file. **I would
   pick leaving it as it is**; a change would touch every frozen prediction file.
3. **The mix trains readout rows 0 to 17, not 0 to 12 (correction to my Z0 reasoning).** The composed mix has 155 rows with 17 or 18 options, all from `fda_route_choice_v1`, the defective template of §6.1 (C048). Their gold is `routes[0]`, so the readout rows 10 and 15 (K and P) receive gradient from label-questionable rows. R0 prime (seed 1) at step 6,996 answers P on 37 % and K on 29 % of the DDXPlus items and gets 0 of 499 two-letter items right (C046; below the 0.0204 chance); seed 0 does not (C047). The training dynamics look normal (CE and grad norms like R0's, C049 context), so this is a seed-dependent readout outcome, not a crash. The hypothesis that the defective template drives the K and P collapse is untested (measured facts only in C046–C049). The mix is frozen, so nothing here changes the arms; it goes to the operator with §6.1.

---

## 7. Deviations from the plan

Any place you departed from GOAL/ADVISORY, with the reason. An empty section is the
expected outcome. Editing code or a check to make it pass is never an acceptable
deviation — that is a `BLOCKED`.

1. **p = 1.0 for the closed list (D52a; ADVISORY Z0.3).** The rule says to raise p while the expected two-letter golds per row stay below
   20. At arm length the mean over rows 26–63 is 19.35 at p = 1, the maximum, so the target is unreachable there; at final length p of
   0.23 or more suffices. p = 1.0 goes into R2, R3 and the final, so one recipe is compared throughout. The arm-length shortfall is
   reported with the arms; the positional check (D52b) is the measured test of whether rows 26–63 get trained.
2. **Per-item KL and distillation (D51).** The text divides the KL and distillation sums by batch size and multiplies the whole
   post-clip gradient by n/B. I implemented that literally, so an anchor's KL weight is (n/B)·λ/B after clipping and equals λ/B only in
   full batches. R3 runs on this reading; its consequence is that KL weighs less in short batches.
3. **The per_item check (ADVISORY Z0.2).** The primary value is the clip-aware share on the 1,266 logged steps, as the replay script's
   docstring fixed it before the run. It is 0.3365, which rounds to 0.34, the lower edge of "about 0.34–0.40". I judge the rule met; the
   unclipped full-run value 0.3417 agrees. The move from 0.588 to 0.337 is the effect the rule was for.
4. **Advisor figures not reproduced (notes, not changes).** "MIMIC 36 % of the items": the rows with the `mimic_` prefix are 34.27 % of the
   mix and 34.17 % of the realised items; the MIMIC family including MedNLI is 35.0 % (the family cap). "Closed-list items 5.9 %": the two
   templates are 5.20 % of mix rows (75,687) and 5.26 % of the arm's realised items. The replay reproduces the advisor's shares 0.59 and 0.34.
5. **The CG qualifier (ADVISORY Z0.6).** The check names "the revised CG qualifier (ADVISORY Z4)". Z4 as written compares an α with
   α = 1, and the osler_v4 final is α = 1, so it cannot fail that comparison. I read the check as the Z4 qualifier against the α = 0
   reference on identical items (D54's reference). On existing predictions the final fails: 0.7395 against 0.7888 on 8,296 items. Z1
   re-scores the α = 0 reference on the reference path and repeats the check there.
6. **Arm choice (GOAL against ADVISORY Z2).** GOAL says the arms are "chosen by a dev composite of MR, MIMIC and CG (D54)". ADVISORY Z2
   gives the rule: qualify on positional health, knowledge and the MR and MIMIC bounds, then take the highest CG v2 micro. I follow Z2, the
   more specific text. S is the checkpoint-selection statistic (ADVISORY Z3).
7. **Zero-shot composite incomplete.** Its fast-dev-v4 MR partition has no prediction file, and 11 of 8,307 CG v2 items are absent from its
   CG file (9 MedMCQA, 2 HeadQA; the cause is not recorded in osler_v4). The D54 value of 0.791 for its CG micro is not reproduced on
   those items (0.7888). S is NOT MEASURED for it until Z1 scores the α = 0 reference on all items of fast dev v4 and CG v2.
8. **Full fast dev in the trainer's evaluation.** The arm driver keeps the trainer's own full-dev evaluation (its dev-macro tripwire, as
   in Y8) and scores the same set again in its hook. This doubles the dev scoring at each evaluation and changes no number.
10. **Composite definition corrected before Z2 (D54).** Z0 first computed MR and MIMIC as item-micro accuracies. D54 names CG v2 micro
   explicitly as "not template macro", and the keep rule and Gate O4 use template macro for the MR and MIMIC suites, so the MR and
   MIMIC parts of S are now template macro (`src/meddecide/eval/composite.py`). The micro values in C016–C018 are superseded by
   C026–C028. The probe job of Z1 (started before the correction) prints its S with micro MR and MIMIC; its S is recomputed from
   the saved rows with the corrected definition before it is reported.
9. **No augmented row is a KL anchor in R3.** Both closed templates carry the `mimic_` prefix that the osler_v4 anchor list excludes, so
   "augmented anchors included" holds with an empty set.

---


11. **R3 stopped by the operator's steer (2026-10-27T14:27Z; ADVISORY §8.1, D56).** R3 (the defective mix) was terminated at PID 3292313 (exit 143). Its last logged step is 5,850 of 13,992 (117 logged steps). No evaluation was reached (the first is at 6,996), so R3 has no evaluation rows. Its training log is kept as the partial result; it is an additional record and not a candidate.
## 8. Closure summary feed

<!-- Filled at closure, feeding FINDINGS.md's Summary section. One row per major
     outcome, in the order a human should hear them. Each claim cites its CLAIMS id;
     failures and blocked tasks get rows too. -->

| # | outcome (one sentence, plain language) | claims | artifact |
|---|---|---|---|
| | | | |
