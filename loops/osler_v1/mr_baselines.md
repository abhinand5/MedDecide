# osler_v1 V6 — MR evaluation suite and baselines

Aggregates only (no item text, no predictions). Source: `outputs/osler_v1/V6/baseline_summary.json`, written by
`scripts/osler_v1/v6_analyse.py`; re-derived and checked in a fresh process by `scripts/osler_v1/v6_self_audit.py`
(107 checks, 0 failed; `outputs/osler_v1/V6/SELF_AUDIT.md`). The runners are the O2 runners with the O2 protocol (authors'
code and both orders for choice items). Predictions: `outputs/osler_v1/V6/baselines/<model>/predictions.jsonl`.

No Gate O2 verdict is given here: that is V10. The zero-shot cells pass the D12 readout gate on all twelve cells; the
competitor cells use the D21 check, and one held-out template fails it for all three competitors (below).

## 1. The suite

| set | items | templates | where it comes from |
|---|---|---|---|
| `mr_test` | 19,294 | 9 | test splits of V1 (1,430), V5a Synthea (6,509) and V5b real text (11,355); `data/bench/osler_mr_v0/manifest.json` (counts.test) |
| `mr_heldout` | 1,518 | 3 | the held-out indicator (aki_creatinine_rise) and the held-out ICD family (heart_failure), reported separately and never in a verdict (V066) |

The teacher prose (V4) is not in the suite yet. It joins when V4 is complete, by re-running the V6 assembly and the baseline
scoring for the new templates (ADVISORY §6, order exception). Test and held-out items share no item id (CLAIMS V068).

## 2. Headline results (readout-gated cells)

Accuracy is taken over the cells that pass the readout gate (D12 for zero-shot, D21 for competitors). A cell that reads
READOUT_FAIL reports no accuracy and is excluded from the set's figures, and it is named. The 95 % interval is the
percentile bootstrap of the summary (`summarise`, O2 code). ECE is the O2 binning.

| model | `mr_test`: scored of in scope | accuracy [95 % CI] | template macro | Brier | ECE | `mr_heldout`: scored in pass cells | accuracy [95 % CI] | template macro | Brier | ECE | readout-failed templates |
|---|---|---|---|---|---|---|---|---|---|---|---|
| zero-shot Qwen3.5-4B | 19,294 of 19,294 (100.00 %) | 0.7383 [0.7320, 0.7445] | 0.6550 | 0.3791 | 0.1269 | 1,518 | 0.8478 [0.8307, 0.8656] | 0.6944 | 0.2260 | 0.0539 | none |
| zero-shot Qwen3.5-9B | 19,294 of 19,294 (100.00 %) | 0.7480 [0.7420, 0.7543] | 0.6842 | 0.3958 | 0.1580 | 1,518 | 0.8314 [0.8129, 0.8511] | 0.7051 | 0.2879 | 0.1025 | none |
| MedDecider-4B | 12,992 of 19,294 (67.34 %) | 0.6681 [0.6603, 0.6760] | 0.6051 | 0.4565 | 0.1546 | 1,518 (176 in pass cells) | 0.6477 [0.5795, 0.7102] | 0.6608 | 0.4446 | 0.1049 | `synthea_indicator_evidence_v1` |
| MedDecider-9B | 12,992 of 19,294 (67.34 %) | 0.6653 [0.6581, 0.6734] | 0.6223 | 0.4864 | 0.1792 | 1,518 (176 in pass cells) | 0.6080 [0.5341, 0.6761] | 0.6191 | 0.5052 | 0.1406 | `synthea_indicator_evidence_v1` |
| JEV-9B | 19,294 of 19,294 (100.00 %) | 0.7473 [0.7409, 0.7537] | 0.7336 | 0.3380 | 0.1143 | 1,518 (176 in pass cells) | 0.5966 [0.5283, 0.6648] | 0.6204 | 0.6066 | 0.2382 | `synthea_indicator_evidence_v1` |

Reading the table:
- MedDecider cannot score 6,302 test items: its letter set is A to J, and those items have more options (the page-category
  templates, 12 options). Its test figures are on the 67.34 % it can score. Section 3 gives the like-for-like comparison.
- On `mr_heldout`, the MedDecider and JEV-9B pass-cell figures cover 176 items, because `synthea_indicator_evidence_v1`
  (1,342 items) fails their readout gate. The zero-shot figures are on all 1,518 items. Section 3 puts all five on the
  same 176 items.

## 3. Matched comparison (additional, labelled)

Common support: the items that both MedDecider models scored in each set, with the readout-failed templates of any model
excluded for every model (one item set per set). Additional analysis; not a gate.

| model | `mr_test` (12,992 items) accuracy [95 % CI] | template macro | `mr_heldout` (176 items) accuracy [95 % CI] | template macro |
|---|---|---|---|---|
| zero-shot Qwen3.5-4B | 0.6177 [0.6091, 0.6260] | 0.5497 | 0.5795 [0.5057, 0.6477] | 0.6002 |
| zero-shot Qwen3.5-9B | 0.6312 [0.6230, 0.6393] | 0.5834 | 0.6136 [0.5453, 0.6875] | 0.6277 |
| MedDecider-4B | 0.6681 [0.6603, 0.6760] | 0.6051 | 0.6477 [0.5795, 0.7102] | 0.6608 |
| MedDecider-9B | 0.6653 [0.6581, 0.6734] | 0.6223 | 0.6080 [0.5341, 0.6761] | 0.6191 |
| JEV-9B | 0.6301 [0.6218, 0.6377] | 0.6584 | 0.5966 [0.5283, 0.6648] | 0.6204 |

On the items MedDecider can score, its two models are ahead of both zero-shot models on accuracy, and JEV-9B is behind them.
On the 176 held-out items the order changes, and the intervals overlap widely. The unmatched zero-shot figures are higher on
`mr_test` because they include the 12-option items, where zero-shot does well.

## 4. Per-template accuracy (readout-gated cells)

`NM` = not measured (no scored item: MedDecider's letter set cannot take the 12-option items). `F` = READOUT_FAIL (no accuracy).

`mr_test`

| template | items | zero-shot 4B | zero-shot 9B | MedDecider-4B | MedDecider-9B | JEV-9B |
|---|---|---|---|---|---|---|
| codiesp_icd_support_v1 | 1,654 | 0.111 | 0.130 | 0.186 | 0.229 | 0.312 |
| mkurman_icd_support_v1 | 1,490 | 0.256 | 0.273 | 0.397 | 0.417 | 0.538 |
| mr_icd_support_template_v2 | 554 | 0.708 | 0.709 | 0.717 | 0.724 | 0.738 |
| mr_indicator_evidence_v1 | 576 | 0.658 | 0.689 | 0.708 | 0.688 | 0.748 |
| mr_page_category_tabular_v1 | 300 | 1.000 | 0.997 | NM | NM | 1.000 |
| mtsamples_page_category_v1 | 111 | 0.604 | 0.667 | NM | NM | 0.658 |
| pmc_similar_patient_v1 | 8,100 | 0.765 | 0.767 | 0.798 | 0.777 | 0.672 |
| synthea_indicator_evidence_v1 | 618 | 0.801 | 0.932 | 0.824 | 0.900 | 0.942 |
| synthea_page_category_v1 | 5,891 | 0.994 | 0.994 | NM | NM | 0.995 |

`mr_heldout`

| template | items | zero-shot 4B | zero-shot 9B | MedDecider-4B | MedDecider-9B | JEV-9B |
|---|---|---|---|---|---|---|
| mr_icd_support_template_v2 | 102 | 0.471 | 0.539 | 0.578 | 0.549 | 0.471 |
| mr_indicator_evidence_v1 | 74 | 0.730 | 0.716 | 0.743 | 0.689 | 0.770 |
| synthea_indicator_evidence_v1 | 1,342 | 0.883 | 0.860 | F | F | F |

## 5. Readout health

- Zero-shot (D12, label mass, greedy agreement, accuracy interval above chance, constant answer): all 9 test cells and all
  3 held-out cells PASS.
- Competitors (D21: constant answer and accuracy interval not below chance): `synthea_indicator_evidence_v1` reads
  READOUT_FAIL for MedDecider-4B, MedDecider-9B and JEV-9B on the held-out set. The accuracy interval's
  upper bound is below chance (0.5000) in each case: 0.4963 (MedDecider-4B), 0.3458 (MedDecider-9B) and 0.3525 (JEV-9B). Its accuracy is therefore not reported; the zero-shot models pass there. The
  cell is excluded from every figure of the competitors, and the matched block excludes it for all five models.
- `mr_test`: no competitor cell fails. The three MedDecider page-category cells are NM, not failures.

## 6. Caveats

- Page-category templates are lexical by construction (V1-D2, V5a-D1): the section headers name the class, which is why the
  zero-shot accuracies there are near 1.0. This is accepted; it is not evidence of clinical skill.
- The `d1-3B` additional row was measured after this section was first written, in the GPU idle window before V4 closed
  (section 8). It is an additional row only (D31); no Gate O2 verdict uses it. `envs/d1/` was not needed: the main
  environment has transformers 5.18, which meets the card's requirement of 5.14.
- The zero-shot Qwen3.5-9B has the highest held-out template macro of the zero-shot models; the held-out sets have 74 to
  1,342 items per template, so per-template differences of a few points are within the intervals.
- These baselines are on the MR suite only. The knowledge baselines of Gate O2 come from the O2 common set (V10).

## 7. Provenance

- Predictions: `outputs/osler_v1/V6/baselines/{qwen35-4b, qwen35-9b, meddecider-4b, meddecider-9b, jev-9b}/predictions.jsonl`
  (20,812 rows each). Model revisions: as in `scripts/osler_v1/v6_baselines_run.sh` (MedDecider-4B
  `570b37090c379ad722186a79a3432a31be463fc9`, MedDecider-9B `ec8a69da2cd96b2ddd2ff4e314ba9a40b71936c3`).
- The first baseline attempt is in STATE (V6-D1); its zero-shot 4B predictions were removed before the rerun.
- Recompute: `uv run --frozen python scripts/osler_v1/v6_analyse.py` then `uv run --frozen python -I scripts/osler_v1/v6_self_audit.py`.

## 8. Addendum: d1-3B (additional row, measured 2026-10-14)

LiquidAI/d1-3B (revision `051bcc464b01b9f92942b364d9586b0ef5912432`) through its own code (`system_one`), with both option orders
for choice items averaged per option, as the O2 protocol does for the competitors. Additional only (D31): no Gate O2 verdict
uses it. Predictions: `outputs/osler_v1/V6/baselines/d1-3b/predictions.jsonl` (20,812 rows, 0 skipped; CLAIMS V114–V117).

| set | scored | pass-cell items | accuracy [95 % CI] | template macro | Brier | ECE | readout-failed templates |
|---|---|---|---|---|---|---|---|
| `mr_test` | 19,294 of 19,294 | 9,540 | 0.8062 [0.7984, 0.8142] | 0.6921 | 0.2494 | 0.0554 | codiesp_icd_support_v1, pmc_similar_patient_v1 |
| `mr_heldout` | 1,518 of 1,518 | 176 | 0.5227 [0.4489, 0.5966] | 0.5344 | 0.5441 | 0.1113 | synthea_indicator_evidence_v1 |

On the matched block (the common support of the Gate O2 comparators, unchanged from section 3), d1-3B scores 0.4756 on the
3,238 `mr_test` items that are outside its own two failing templates (CI [0.4586, 0.4914]), and 0.5227 on the 176 held-out
items. Its readout failures are on CodiEsp and PMC-Patients, where the D21 check fails; the comparators pass there.
