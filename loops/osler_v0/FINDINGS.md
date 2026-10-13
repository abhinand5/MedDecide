# osler_v0 — findings

## Summary

**The question (ADVISORY §2):** can a 4B decision model trained on gold-by-construction clinical decisions plus human-labelled data beat the best open medical decision model of its size without losing the base model's knowledge, and which readout should the Osler series use?

**Answer under the pre-registered gate: no.** Gate O1 (D24), applied to the calibrated outputs, gives **FAIL for Osler-4B and FAIL for Osler-9B** [O304, O305].

- **Osler-4B (arm L, option-code head).** It beats MedDecider-4B and zero-shot Qwen3.5-4B on the headline seen-template set: accuracy +0.1102 [+0.0992, +0.1216] and +0.1318 [+0.1203, +0.1434]; every Brier interval is below zero [O304]. It loses the external panel to MedDecider-4B: accuracy −0.0526 [−0.0696, −0.0363] [O304]. It fails the knowledge guard against zero-shot Qwen3.5-4B: −0.0156 [−0.0291, −0.0018], floor −0.02 [O304].
- **Osler-9B.** It beats MedDecider-9B and JEV-9B on the headline set: accuracy +0.0929 [+0.0838, +0.1030] and +0.1035 [+0.0942, +0.1135] [O305]. It passes the knowledge guard: −0.0075 [−0.0196, +0.0055] [O305]. It loses the external panel to MedDecider-9B: accuracy −0.0169 [−0.0330, −0.0007] [O305].

**Which ADVISORY §2 outcomes apply.**

- **Outcome 3 (the knowledge guard fails): applies to Osler-4B.** §2 makes replay ratio, LoRA scope or early stopping the next loop's first job, before any scaling.
- **Outcome 2 (pass on the fresh tier, fail on the external panel): applies to Osler-9B** (and to the 4B's panel loss). §2 reads this as "we learned our benchmark, not medicine; the next loop is data breadth, not scale."
- **Outcome 4 (arms within noise at 4B): applies to the pointer arm.** P does not differ from L on accuracy (+0.0013 [−0.0020, +0.0051]) [O313]. The head chosen by the pre-registered dev rule stays L [O241], so the readout is the option-code head.
- **Outcome 5 (a non-causal arm wins clearly): not met cleanly.** N is ahead of L on the common set by +0.0050 [+0.0011, +0.0089] and on the panel by +0.0152 [+0.0006, +0.0305] [O313]. That is a small advantage found after the head was chosen on dev. Its serving cost (a custom attention mask, no stock GGUF path) is recorded for the operator. This loop does not call it a clear win.
- **Outcome 1 (both pass): not met.**

**Cautions to read first.**

1. **The O11 probabilities were uncalibrated until deviation 59.** The arms' fitted temperatures never reached the saved checkpoints [O293]. All O11 Brier numbers from before that fix are uncalibrated, and so are the dev Brier values in the O6–O8 and O10 closures. The gate uses calibrated outputs only [O300, O304]. Single-pass accuracy is unaffected, because temperature is monotone within a question type; the two-order average can move a few items [O300].
2. **The external panel is the weak spot.** Every Osler model loses the panel to the competitor of its size, except the 4B against zero-shot Qwen3.5-4B and the 9B against JEV-9B [O304, O305].
3. **The knowledge guard is the independent check against memorisation.** The 4B fails it (L −0.0156; P −0.0220), N passes (+0.0051), and the 9B passes (−0.0075) [O304, O305, O314]. The 0.8B cannot be checked on the common set [O314].
4. **Held-out templates and held-out generators are reported separately and never in a verdict** [O319, O324]. The constant-answer behaviour concentrates on held-out templates [O324].
5. **Time.** The O11 stage ran about twelve hours of wall clock against the ADVISORY's six-hour estimate. The calibrated scoring alone took about 7.8 hours of GPU time [O298, O301, O308, O311, O315]. Deviation 56 leaves the time-versus-scope choice to the operator.

## What the loop measured, by task

The lines below summarise each task from its STATE board row. Numbers marked with an O-id come from CLAIMS.md.

- **O0 (orientation, snapshot, environments, smoke, throughput):** DONE, 2026-10-09. No numbers beyond the board.
- **O1 (external clinical panel and robustness pack):** DONE. The panel has 4,353 items and the robustness pack 20,109 items in the gate's scopes [O303].
- **O2 (competitor scoreboard):** DONE. All rows for eleven models on the common set (59,538 items); D12 applied to the zero-shot cells; the pplx lead was investigated (CLAIMS O123–O125); Question 16 (near-ceiling templates shared by all models) stays open. The baselines are in `scoreboard.md` (CLAIMS O123–O160).
- **O3 (clinical generators, gold by construction; held-out list):** DONE. Nine generators, two held out (`note_lab_range_v1`, choice; `policy_triage_v1`, score), 300 test items each (CLAIMS O048–O050; `heldout.md`). The O3 screen passed [O320].
- **O4 (training mix v2):** DONE. 573,611 training rows and 10,843 dev rows; the audit is at CLAIMS O085–O088. The leakage check against the v0.2 test items and the panel is clean [O307].
- **O5 (readouts):** DONE with a question open. The option-code head uses 255 codes (CLAIMS O093). The merged-adapter export failed the 1e-3 check at 1.04e-02 (CLAIMS O094, O105); the separate-adapter export passed at 2.15e-06 (CLAIMS O104). Question 14 (the precision of the O5 "fp32" verdict) stays open.
- **O6–O8 (three matched 4B arms, 25,000 steps each, no tripwire):** DONE. Dev, full 10,843 items, dev only, uncalibrated: L accuracy 0.8932, template macro 0.8636 (CLAIMS O190–O193); P 0.8939, 0.8687 (CLAIMS O211–O216); N 0.8940, 0.8663 (CLAIMS O235–O240).
- **O9 (head choice on dev, the pre-registered rule):** DONE. L is chosen. No challenger reaches +0.010 on the dev template macro with a lower bound above zero: P +0.005145, N +0.002749 [O241].
- **O10 (Osler-9B and Osler-0.8B reference):** DONE. Osler-9B dev, uncalibrated: accuracy 0.9021, template macro 0.8691 (CLAIMS O261–O266). Osler-0.8B: 0.8285, 0.8143 (CLAIMS O279–O283). Knowledge flag on both: the trained models' tier-1 MedQA is below the zero-shot base on the same dev items, −0.4 for the 9B and −6.4 for the 0.8B; the 4B arms are below their base on both sources (CLAIMS O246–O248, O280).
- **O11 (evaluation and Gate O1):** DONE. The calibrated common-set results, the verdicts, the arms, the 0.8B and the held-out items are in `results.md`. Key rows: the verdicts [O303–O306]; the common set [O298–O302, O308–O312, O315–O316]; the arms [O313–O314]; the held-out generators [O319–O321]; the self-audit at `outputs/osler_v0/O11/SELF_AUDIT.md`.

## Defects, checks and what they showed

- **Calibration defect (deviation 59).** The fitted temperatures were recorded but not saved with the checkpoints, so every uncalibrated O11 probability was off. The fits reproduce their own NLL [O294]; the calibrated scorer matches the transformed raw output to 2e-8 on the smoke rows [O295] and to 3.7e-8 on the first 25,088 rows of L [O297]. Fixed by scoring every pass with the recorded fits. This is not fixed in `arm_run.py` itself: the next loop should save the fit with the checkpoint.
- **Saturation.** The uncalibrated P output is exactly one-hot on 7,256 of 43,894 choice items (L: 170) [O323]. In the calibrated P output, 29 near-one-hot choice items are all in one robustness template and all correct [O310]. This is the pointer head's confidence, not a scoring error.
- **Constant answers.** They sit on two held-out templates: `fda_boxed_warning_noul_v1` is answered `no` on 92–97 % of items for L, P and N; `ct_arm_role_noul_v1` is answered `yes` on 89.6–96.8 % for all five models [O324]. The 0.8B also answers one option on 97 % of the unseen `pubmedqa_ynm_v1` items, and its dev predictions collapse the same way [O317]. These are transfer failures, not readout failures: calibration leaves the argmax of each pass unchanged [O317].
- **A dev–test gap on a seen template.** The 0.8B reaches 0.948 on dev and 0.600 on test for `nfcorpus_relevant_noul_v1`, and the test answers are 84 % `yes` against a balanced gold [O317]. The student loop had recorded this template as a bag-of-words shortcut.
- **Leakage.** None: 573,611 training rows; no match to a v0.2 test item or an external-panel item, by id or by question text [O307].
- **Chance.** The 0.8B's panel accuracy (0.3221) is above the uniform chance (0.1965) and the majority-of-gold baseline (0.2507) [O318]. The 0.8B is above chance everywhere it is in a verdict or an additional row.
- **The result that looked too good.** The 9B scores 0.99 on the held-out `note_lab_range_v1`. The generator passed the O3 screen, the gold is computed from the parts, and the items require a numeric comparison of a measured value against a reference range [O320]. No leak was found.
- **The 0.8B on the held-out generators** is below the 0.5 majority baseline and worse than a uniform guess on Brier [O321].
- **Errors I made and corrected** (listed in `SELF_AUDIT.md`): a 0.8B guard that used the wrong base, discarded before recording [O314]; a row that counted a growing file, replaced by a snapshot [O297]; an unqualified claim that calibration leaves accuracy unchanged, corrected for two-order averages [O300]; a miscount of the 9B's `ct_arm_role` share, corrected [O324]; an uncalibrated audit quoted before it had a row, now recorded [O322, O323].

## Deviations (summary; the full text is in STATE §7)

- **44** — both-order scoring for choice items; noul and score scored once.
- **52–53** — held-out generator scope (the Osler side is resolved by 58) and the detached chains.
- **56** — O11 time: the stage ran about twelve hours against six estimated. Open for the operator.
- **57** — the converter failure on reversed noul robustness rows, fixed and pre-flighted (CLAIMS O284–O287).
- **58** — held-out generator scope added for the Osler models; the baselines on those items are NOT MEASURED.
- **59** — the calibration defect; the calibrated outputs are the ones used by the gate (CLAIMS O293–O296).

## Not measured (ADVISORY O11 items)

- The four D24 baselines on the held-out generator items (deviation 58) [O319].
- The 0.8B against a zero-shot 0.8B base, on the common set and in the knowledge guard [O314].
- The Osler-4B quantised export (≤ 4 GB, GGUF): the merged export failed the O5 check (CLAIMS O094, O105), and no quantised export was built in this loop.
- Selective accuracy at coverage, auto-accept at precision, per-item latency, the HLE supplementary set and the long-record slice (time, deviation 56). The per-template tables are in the audit files.

## Open for the operator

1. **Time versus scope (deviation 56).** O11 took about twelve hours. The next loop's budget should reflect this.
2. **The next loop's first job.** Outcome 3 makes the knowledge fix the first job, before any scaling. The operator should confirm the next loop's scope.
3. **The non-causal serving cost** (arm N, +0.0050 on the common set [O313]) against its serving cost. This is a decision for the operator, not for this loop.
4. **Held-out baselines** (deviation 58): whether to wire the four baselines to the held-out items.
5. **Questions carried from earlier loops:** Question 14 (the O5 precision), Question 16 (near-ceiling templates), Question 21 (the example budget; option (a) applied).

## Five-claim spot-check (re-run at closure, 2026-10-13T13:45Z)

Each claim was re-derived in this session with its recorded recompute command. The outputs match the CLAIMS values.

**Claim O304 — Osler-4B gate verdict.** Command: `uv run --frozen python -c "import json; g=json.load(open('outputs/osler_v0/O11/calibrated/gate.json')); r=g['models']['osler_4B']; print(r['verdict']); [print(c['scope'], c['baseline'], c['n_paired'], c['n_baseline_skipped'], round(c['accuracy']['point'],4), round(c['accuracy']['lo'],4), round(c['accuracy']['hi'],4), round(c['brier']['point'],4), round(c['brier']['lo'],4), round(c['brier']['hi'],4), c['passes']) for c in r['comparisons']]; print(r['knowledge_guard'])"`

```
FAIL
headline_seen MedDecider-4B 15897 0 0.1102 0.0992 0.1216 -0.1328 -0.1392 -0.1267 True
headline_seen zero-shot Qwen3.5-4B 15897 0 0.1318 0.1203 0.1434 -0.1815 -0.1893 -0.1745 True
external_panel MedDecider-4B 4141 212 -0.0526 -0.0696 -0.0363 0.0479 0.0386 0.0568 False
external_panel zero-shot Qwen3.5-4B 4353 0 0.0208 0.003 0.04 -0.0746 -0.0881 -0.0611 True
heldout_templates MedDecider-4B 5658 0 -0.0992 -0.111 -0.0881 0.2042 0.1897 0.218 False
heldout_templates zero-shot Qwen3.5-4B 5658 0 -0.1271 -0.1399 -0.1153 0.2441 0.2288 0.2594 False
robustness MedDecider-4B 18751 1358 0.0682 0.0582 0.0777 -0.141 -0.1474 -0.134 True
robustness zero-shot Qwen3.5-4B 20109 0 0.1754 0.1649 0.1857 -0.3152 -0.3241 -0.3065 True
{'baseline': 'zero-shot Qwen3.5-4B', 'n_paired': 5456, 'accuracy': {'n': 5456, 'point': -0.015579178885630499, 'lo': -0.02914681085043988, 'hi': -0.0018236803519061668}, 'passes': False}
```

**Claim O305 — Osler-9B gate verdict.** Command: the same as O304 with `osler_9B`.

```
FAIL
headline_seen MedDecider-9B 15897 0 0.0929 0.0838 0.103 -0.1252 -0.1312 -0.1197 True
headline_seen JEV-9B 15897 0 0.1035 0.0942 0.1135 -0.1427 -0.1484 -0.1375 True
external_panel MedDecider-9B 4141 212 -0.0169 -0.033 -0.0007 0.0269 0.0184 0.0357 False
external_panel JEV-9B 4141 212 0.0292 0.0142 0.0452 -0.0454 -0.0555 -0.0356 True
heldout_templates MedDecider-9B 5658 0 -0.0004 -0.0107 0.0104 0.0316 0.021 0.0419 False
heldout_templates JEV-9B 5658 0 -0.0769 -0.0881 -0.065 0.1633 0.1511 0.176 False
robustness MedDecider-9B 18751 1358 0.1118 0.1019 0.1208 -0.1783 -0.1853 -0.171 True
robustness JEV-9B 19461 648 0.2289 0.2176 0.2392 -0.3235 -0.3317 -0.3149 True
{'baseline': 'zero-shot Qwen3.5-9B', 'n_paired': 5456, 'accuracy': {'n': 5456, 'point': -0.007514662756598241, 'lo': -0.019611436950146627, 'hi': 0.005498533724340176}, 'passes': True}
```

**Claim O299 — the calibrated L audit.** Command: `uv run --frozen python scripts/osler/o11_audit.py --name osler_4b_L --out-root outputs/osler_v0/O11/calibrated`

```
osler_4b_L: 59538 rows in both_orders, 59538 in single_order; choice rows averaged 43894 of 43894; fixed-order rows identical 15644 of 15644
accuracy both orders 0.7720, single order 0.7662, Brier 0.3074; worst per-template top predicted share 0.958 (ct_arm_role_noul_v1, 26 templates)
```

**Claim O319 — the held-out report.** Command: `uv run --frozen python scripts/osler/o11_heldout_report.py`, then the template macro and Brier read from `report.json`.

```
report.json and loops/osler_v0/heldout_generators.md written
{'osler_4b_L': (0.7267, 0.3414), 'osler_4b_P': (0.7767, 0.3633), 'osler_4b_N': (0.7683, 0.3469), 'osler_9b': (0.8383, 0.225), 'osler_0p8b': (0.4583, 0.7731)}
```

**Claim O313 — the arm-vs-arm comparisons.** Command: `uv run --frozen python scripts/osler/o11_additional.py`

```
common_set osler_4b_N minus osler_4b_L 59538 0.005 0.0011 0.0089 | brier -0.0142 -0.0161 -0.0123
common_set osler_4b_P minus osler_4b_L 59538 0.0013 -0.002 0.0051 | brier 0.0036 0.0018 0.0054
external_panel osler_4b_N minus osler_4b_L 4353 0.0152 0.0006 0.0305 | brier -0.0149 -0.0231 -0.0066
external_panel osler_4b_P minus osler_4b_L 4353 0.0046 -0.0121 0.0215 | brier 0.0146 0.0057 0.0235
knowledge guard osler_4b_N {'point': 0.005131964809384164, 'lo': -0.00843108504398827, 'hi': 0.017961876832844576}
knowledge guard osler_4b_P {'point': -0.021994134897360705, 'lo': -0.03500733137829912, 'hi': -0.0076979472140762464}
knowledge guard osler_4b_L {'point': -0.015579178885630499, 'lo': -0.02914681085043988, 'hi': -0.0018236803519061668}
knowledge guard osler_0p8b no zero-shot base predictions
```

The five claims reproduce. The working tree was clean after the re-runs, so the committed artifacts are unchanged.
