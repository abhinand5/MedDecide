# osler_v1 — FINDINGS (closure)

Written 2026-10-16 after V10. Loop status: STOPPED — a hard stop for operator review. Every number has a CLAIMS row in
`loops/osler_v1/CLAIMS.md` (V001 to V189). Gate O2 detail is in `gate_o2.md` and `results.md`; the next steps are in `NEXT.md`.

## Summary

- **Gate O2 fails for Osler-4B** (CLAIMS V165). The question (ADVISORY §2) was whether Osler-4B learns medical-record audit
  decisions well enough to beat MedDecider-4B and its own base, without losing the external panel or the base's knowledge. On
  Gate O2 as written, the answer is no.
  - MR suite against MedDecider-4B: accuracy +0.0805 [+0.0700, +0.0916] passes; Brier +0.0351 [+0.0287, +0.0414] fails, because
    the upper bound must be below zero (V167).
  - External panel against MedDecider-4B: accuracy −0.0223 [−0.0365, −0.0085]. Non-inferiority needs the lower bound above
    −0.02, so it fails (V169).
  - Against zero-shot Qwen3.5-4B the MR suite passes (accuracy +0.0943; Brier −0.0449; V166), the panel passes (+0.0568; Brier
    −0.0924; V168), and the knowledge guard passes (+0.0440 [+0.0324, +0.0548]; V170).
- **Osler-9B: NOT MEASURED.** The operator deferred it (V9-D1).
- **ADVISORY §2 outcomes that apply: 3 and 4.** Outcome 3: the MR suite fails against MedDecider-4B. Its loss sits on the
  real-text ICD templates: CodiEsp accuracy −0.0992 and Brier +0.4738; mkurman accuracy −0.1624 and Brier +0.3972 (V185).
  Outcome 4: B2 and B3 were not kept against B1 (V140, V148). Outcome 5 is reported separately and is mixed (see §2).
  Outcomes 1 and 2 do not apply.
- **The audit data helps on generated items and not on real text.** On the generated part of the MR suite (7,939 items),
  Osler-4B is +0.2351 [+0.2148, +0.2563] in accuracy against MedDecider-4B on the 1,748 items MedDecider-4B can answer, and
  +0.1588 against zero-shot (V184). On the real-text part (11,355 items), it is −0.0742 [−0.0837, −0.0658] against
  MedDecider-4B, with a Brier of +0.0934, and +0.0136 [−0.0049, +0.0333] against zero-shot (V184). The kept block holds no real
  clinical text (V071).
- **Which block helped on dev.** B1, the kept block, beat B0 on the MR dev macro (+0.0947; lower bound +0.0807), and its
  general and knowledge checks passed (V112). B3, which adds real text, raised the MR dev macro by +0.0539 over B1 (lower bound
  +0.0382), but failed two margins: general −0.0035 (lower bound −0.0141 against −0.01) and knowledge −0.0400 (lower bound
  −0.0860 against −0.02) (V148). B2, with teacher prose, did not help MR (−0.0046) and was not kept (V140).
- **Defects.** No below-chance, constant-answer or divergence flag on the final model (V161) or on the V10 rows (V180).
  Too-good flags on three MR templates at 1.0000 on V10: synthea_page_category_v1 (5,891 items) and
  mr_page_category_tabular_v1 (300) are the lexical failures accepted at build; mr_icd_support_template_v2 (554 items) is not
  explained by the lexical screen and is open (V180, V183). Removing the three templates after the flag (post hoc) leaves the
  direction of every MR comparison unchanged (V181).
- **Corrections made in the loop; no primary number changed.** The gate's note called the MR suite "generated"; it holds
  real-text items too (V186, V10-D2). The first held-out rows against MedDecider-4B included its readout-failed cell and are
  withdrawn in part (V188, V10-D4).

## 1. What ran

| task | status | what | CLAIMS |
|---|---|---|---|
| V0 | DONE | environment, MIMIC and teacher checks; batching recipe (batch 8, 8,192 max tokens, gradient checkpointing; 4.118 items/s) | V003, V004, V017, V018 |
| V1 | DONE | MR generators: indicator, ICD v2 (v1 design replaced, V1-D1), page category tabular (accepted lexical failure, V1-D2); O3 screens; held-out isolation | V022 to V031 |
| V2 | DONE | ICD-10-CM codes: the 48 curated codes are in the FY2026 order file and billable; FY2027 checked for Q4 | V035, V101 |
| V3 | DONE | replay: the 20 % target was not met; B0 replay 15.31 % (the operator accepted 15.3 %, Q2) | V070 |
| V4 | DONE (closed) | teacher prose: 14,903 indicator pairs and 16,450 narrative pages verified; ICD prose BLOCKED (pilot 59.8 % against the 60 % threshold); the OpenCode Go weekly guard closed it at $36.01; the DeepSeek continuation was not run (Q10) | V126 to V128 |
| V5a | DONE | Synthea: 40,000 encounters; indicator screen PASS (Naive Bayes 0.576); page category FAIL, accepted (V5a-D1) | V044 to V048 |
| V5b | DONE | real text: MTSamples, CodiEsp (machine-translated), PMC-Patients pairs (52,549 train), mkurman (evaluation only) | V052 to V057 |
| V5c | BLOCKED | MIMIC: the PhysioNet files are absent (re-checked 2026-10-14T00:15Z) | V097 |
| V6 | DONE | MR evaluation suite (19,294 test; 1,518 held-out) and the baselines; readout gates | V065, V082 to V089 |
| V7 | DONE | blocks B0 to B3 with replay shares and leakage checks; blocks.md | V070 to V074 |
| V8 | DONE | block runs; the keep rule; B1 kept; B2 and B3 not kept; the too-good investigation | V104 to V152 |
| V9 | DONE | final Osler-4B on B1: 43,136 steps, attention-only LoRA (V9-D2); Osler-9B deferred (V9-D1) | V153 to V164 |
| V10 | DONE | scoring of 30,621 items; Gate O2 for Osler-4B: FAIL; additional rows | V165 to V189 |
| V11 | BLOCKED (condition not met) | the memorisation audit runs only if MIMIC is in the final mix; it is not (V11-D1) | — |
| V12 | DONE | this file, NEXT.md, the five-claim spot-check, `Loop status: STOPPED` | — |

## 2. ADVISORY §2 outcomes

**Outcome 3 applies: the MR suite fails against MedDecider-4B.** The implication in the ADVISORY is "our data adds too little",
and it asks which block helped on dev and on which template the loss sits.

- Which block helped on dev: B1 over B0 (V112); B3 over B1 on the MR dev macro, which the keep rule did not accept because two
  margins failed (V148); B2 did not help (V140).
- Where the loss sits: against MedDecider-4B on the 12,992 paired MR items, the two real-text ICD templates lose
  (codiesp_icd_support_v1: −0.0992 accuracy, +0.4738 Brier; mkurman_icd_support_v1: −0.1624, +0.3972). The synthetic and PMC
  templates favour Osler-4B, most strongly the synthetic indicator and ICD templates (+0.17 to +0.28 accuracy; V185). The
  template-macro difference is +0.0805, the gate's figure.
- The kept block contains no real clinical text (V071). B3 contained it and helped the MR dev macro, but the keep rule did not
  accept it. That is the trade-off the data loop has to resolve (NEXT §2).

**Outcome 4 applies: a block is not kept.** B2 (V140) and B3 (V148) were not kept against B1, and their dev numbers are reported
in §4. A block that did not help on dev was dropped, not tuned.

**Outcome 5 is reported separately and is mixed (never in a verdict).**

- Held-out indicator (aki_creatinine_rise): 538 test items. Against zero-shot Qwen3.5-4B the accuracy gain is +0.0344
  [−0.0554, +0.1356], which includes zero, and the Brier is much worse (+0.2499; V173). MedDecider-4B's readout fails on the
  synthetic indicator template in the held-out set (V089), so only the 24 MR-indicator items are comparable against it: accuracy
  +0.2917 [+0.1250, +0.4583] (V188).
- Held-out ICD family (heart_failure): 46 test items. Against MedDecider-4B accuracy +0.2391 [+0.0652, +0.4130] and Brier
  −0.3001; against zero-shot accuracy +0.3478 [+0.1951, +0.5000] and Brier −0.4282 (V174). The intervals are wide.
- Reading: the family result is positive against both baselines, but on 46 items. The indicator result does not show that the
  model learned a rule that transfers in calibration to zero-shot, and it is not measurable against MedDecider-4B on the
  synthetic template.

**Outcomes 1 and 2 do not apply:** the gate does not pass for Osler-4B, and the MR suite does not pass against MedDecider-4B.

## 3. Gate O2 (D31) verdicts

| check | baseline | items (paired) | accuracy diff [lower, upper] | Brier diff [lower, upper] | pass | CLAIMS |
|---|---|---|---|---|---|---|
| (i) MR suite | MedDecider-4B | 12,992 (6,302 skipped) | +0.0805 [+0.0700, +0.0916] | +0.0351 [+0.0287, +0.0414] | no (Brier) | V167 |
| (i) MR suite | zero-shot Qwen3.5-4B | 19,294 | +0.0943 [+0.0829, +0.1054] | −0.0449 [−0.0496, −0.0404] | yes | V166 |
| (ii) external panel | MedDecider-4B | 4,141 (212 skipped) | −0.0223 [−0.0365, −0.0085] | +0.0332 [+0.0261, +0.0409] | no (accuracy) | V169 |
| (ii) external panel | zero-shot Qwen3.5-4B | 4,353 | +0.0568 [+0.0410, +0.0741] | −0.0924 [−0.1047, −0.0806] | yes | V168 |
| (iii) knowledge guard (MedQA and MedMCQA test) | zero-shot Qwen3.5-4B | 5,456 | +0.0440 [+0.0324, +0.0548] (micro) | — | yes | V170 |

**Osler-4B: FAIL. Osler-9B: NOT MEASURED (V9-D1).**

The 6,302 MR items and 212 panel items that MedDecider-4B skips have more than 10 options; its letter set is A to J (V084,
V172). Those items are outside the MedDecider comparisons, as the O1 and O2 protocol requires. The panel's superiority bound is
reported separately: against zero-shot it holds (lower bound +0.0410); against MedDecider-4B it does not.

Additional, never in the verdict (V179): against the osler_v0 Osler-4B (arm L, the option-code head that osler_v0 kept), the
osler_v1 Osler-4B is higher on the same panel items (+0.0358 [+0.0186, +0.0520] in accuracy) and on the same knowledge items
(+0.0764 [+0.0619, +0.0913]); its Brier is also lower on both. The osler_v0 MR suite and the osler_v0 Osler-9B are NOT MEASURED.

## 4. Data blocks and the keep rule (dev only; ADVISORY V8)

The keep rule compares the MR dev macro, the general dev and the knowledge items against margins of 0.0, −0.01 and −0.02, each
on its lower bound (paired bootstrap within templates; 1,000 resamples; seed 0).

| block | rows | replay share | decision | the three checks against the comparison block | CLAIMS |
|---|---|---|---|---|---|
| B0 | 653,874 | 15.31 % | control; its knowledge check against zero-shot FAILED (−0.060; lower −0.102); B0b FAILED too (−0.008; lower −0.042) | — | V070, V104, V111 |
| B1 | 696,532 | 14.38 % | **KEPT** against B0 | MR +0.0947 (lower +0.0807); general +0.0062 (lower −0.0048); knowledge +0.0480 (lower +0.0080) | V071, V112 |
| B2 | 734,514 | 13.63 % | NOT KEPT against B1 | MR −0.0046 (lower −0.0181); general −0.0038 (lower −0.0149); knowledge −0.0220 (lower −0.0640) | V132, V140 |
| B3 | 792,260 | 12.64 % | NOT KEPT against B1 | MR +0.0539 (lower +0.0382, passes); general −0.0035 (lower −0.0141, fails); knowledge −0.0400 (lower −0.0860, fails) | V133, V148 |

Dev numbers for each block (fast dev, 5,723 items): B1 accuracy 0.9037, template macro 0.8781 (V108); B2 0.9015 (V138, V139);
B3 0.9123 (V146, V147). The replay share never reached the 20 % target: the largest was B0's 15.31 %, which the operator accepted
(operator answer Q2).

## 5. The kept block and the final Osler-4B

- Training (V9): 43,136 steps; 250,024 realised items; 211,304,913 real tokens; attention-only LoRA targets (V9-D2); training
  wall clock 41,386.2 s (V156); the process exited with code 0 at 2026-10-16T08:38Z (STATE).
- Completeness: 862 logged steps with 0 mismatches; eight evaluations on the planned steps; 0 non-finite losses; the calibration
  equals the dev fits (V157, V160).
- Final dev (5,723 items, scored after the run): accuracy 0.9084 (5,199 correct), template macro 0.8834, Brier 0.1396, mean NLL
  0.2762 (V158). The majority baseline is 0.3365 (V158).
- In-run figures: the template macro rose at every evaluation (0.8471 at step 5,392 to 0.8818 at step 43,136). The in-run final
  evaluation was 6 correct items below the post-run figure (V159); the post-run figure is reported.
- Temperatures: choice 1.4307 (3,089 items), noul 1.6592 (2,515), score 1.3400 (119); the checkpoint's calibration equals the fits
  (V160).
- Defect screen on dev: no below-chance, constant-answer or divergence flag; the too-good flag names the same four templates as
  B1 and B2 (V161, V144).

## 6. Investigations

- **Too-good flags on dev (V161, V162).** The same four templates as B1 and B2 (V144). No MR dev item shares text or a
  dataset-qualified record with the B1 training file (V143). Two of them are the lexical page-category failures accepted at
  build (Naive Bayes 1.000; V025, V045). The other two pass the lexical screen (ICD v2 0.804, V023; synthetic indicator 0.5757,
  V046). Their near-perfect dev accuracy is not explained by overlap or by the bag-of-words shortcut. Open (NEXT §4).
- **Too-good flags on V10 (V180, V183).** Three templates at 1.0000, the same kind of split. The ICD template's 100 % on 554 test
  items is open. No overlap check on the V10 test items was run in this loop.
- **The in-run versus post-run gap (V159).** Six correct items, from weights that are the same at step 43,136. Cause not
  established. The hypotheses (batch composition and padding; bfloat16 checkpoint rounding; evaluation batch order) are untested.
- **pmc_similar_patient_v1 (V150, V151, V163, V177, V181).** B1 has no training rows for this template, and its dev accuracy is
  0.8525 (V163). B3 trained on it and reached 0.99 on dev, while a word-overlap score alone reaches AUC 0.9734 (V151). It is
  8,100 of 19,294 MR test items (41.98 %). Without it, the MR Brier against zero-shot is −0.0020 (V177), so the MR Brier advantage
  over zero-shot rests on this template. The Naive Bayes O3 screen on it has not been run (ADVISORY §8.2; NEXT §5).
- **The Brier loss against MedDecider-4B (V177, V185).** It is largest on the non-pmc items (+0.1597) and on the real-text ICD
  templates. Hypothesis, untested: the temperatures fitted on the fast dev do not transfer to the MR test distribution.
- **The gate's wording and the readout rule (V186, V188).** The gate's note described the MR suite as generated; it holds
  real-text items too. The held-out rows against MedDecider-4B included its readout-failed cell. Both were corrected, with the
  verdict and every primary number unchanged; the pre-fix gate output was checked in session, not kept in the repo.

## 7. Limitations

- One model size was scored: Osler-4B. Osler-9B is deferred (V9-D1).
- One training seed and one run per block. The intervals are bootstrap intervals over items; no run-to-run variance was measured.
- The MR suite is 59 % real text (11,355 of 19,294 items; V187). Its generated part is 7,939 items. MIMIC is absent (NOT MEASURED),
  and MDACE is absent (NOT MEASURED).
- The MedDecider comparisons cover subsets: 12,992 of 19,294 MR items and 4,141 of 4,353 panel items, because MedDecider-4B
  answers at most 10 options (V084, V172).
- The held-out family has 46 test items, and the MR indicator 24. Their intervals are wide.
- The replay share in the kept block is 14.38 %, below the 20 % target (V071).
- Post hoc analyses are labelled additional (V10-D1). None enters a verdict.
- The osler_v0 comparisons use the osler_v0 arm L outputs for the panel and knowledge items only. The osler_v0 MR suite and the
  osler_v0 Osler-9B are NOT MEASURED (V179).
- The teacher (DeepSeek) continuation was not run (Q10); the ICD teacher prose was not built (BLOCKED).

## 8. Five-claim spot-check (re-run at closure, 2026-10-16)

Each claim was re-derived in a fresh process with its recorded recompute command (the gate and the checks print their values; the
additional script rewrites its JSON from the same inputs). The outputs match the CLAIMS values.

**V165 (the Gate O2 verdict).** `uv run --frozen python scripts/osler_v1/v10_gate.py`

```
Osler-4B: FAIL. failed: MR suite vs MedDecider-4B; external panel vs MedDecider-4B (accuracy lower bound must exceed -0.02)
wrote /workspace/MedDecide/loops/osler_v1/gate_o2.md and /workspace/MedDecide/outputs/osler_v1/V10/gate_o2.json
```

**V167 (the MR suite against MedDecider-4B).** `uv run --frozen python -I scripts/osler_v1/v10_check.py` (lines re-derived mr_med, matches)

```
re-derived mr_med: accuracy diff +0.0805; Brier diff +0.0351; paired 12992
re-derivation matches the gate's point estimates (abs tol 0.0005): True
```

**V158 (the final checkpoint's dev metrics).** `uv run --frozen python -I scripts/osler_v1/v9_check.py`

```
dev items 5723; accuracy 0.908440 (5199 correct)
template macro 0.883377 over 45 templates
majority baseline 0.336537
brier 0.139618; mean nll 0.276238
```

**V171 (the V10 scoring is complete).** `uv run --frozen python -I scripts/osler_v1/v10_check.py` (lines run, both_orders, single_order)

```
run: rows in items file 30621; scored now 30621; written before 0; calibration equals dev fits: True; device cuda:0; dtype bfloat16; wall clock 1537.4 s; git commit 9f94e93
both_orders.jsonl: rows 30621; distinct ids 30621; status {'scored': 30621}; rows with non-finite or unnormalised probabilities 0
single_order.jsonl: rows 30621; distinct ids 30621; status {'scored': 30621}; rows with non-finite or unnormalised probabilities 0
```

**V185 (the per-template breakdown against MedDecider-4B).** `uv run --frozen python -I scripts/osler_v1/v10_additional.py`, then the per-template rows from `outputs/osler_v1/V10/additional.json`

```
templates 6; template-macro accuracy difference +0.0805
codiesp_icd_support_v1: paired 1654; accuracy diff -0.0992; brier diff +0.4738
mkurman_icd_support_v1: paired 1490; accuracy diff -0.1624; brier diff +0.3972
pmc_similar_patient_v1: paired 8100; accuracy diff +0.0390; brier diff -0.0402
synthea_indicator_evidence_v1: paired 618; accuracy diff +0.1650; brier diff -0.2336
mr_indicator_evidence_v1: paired 576; accuracy diff +0.2569; brier diff -0.3725
mr_icd_support_template_v2: paired 554; accuracy diff +0.2834; brier diff -0.4246
```

The five claims reproduce. The re-runs wrote the same values into the gate outputs (gitignored JSON) and into the gate report
(`gate_o2.md`, committed with this loop).
