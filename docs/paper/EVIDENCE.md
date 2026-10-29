# Osler — evidence reference (for the paper)

Status: 2026-10-28. A reference for writing, not the paper. Every number names its claim ID (`loops/<loop>/CLAIMS.md`), where
the artifact is, and what was pre-registered. Aggregates only: no item, note, prediction or MIMIC-derived text is here. Hypotheses
are marked as hypotheses (R10). Anything not listed is not claimed.

---

## 1. The question and the answer

**Question.** Can a small open decision model (≤ 10B) be the best open model for medical-record audit (payer-style coding and
documentation checks), and be non-inferior to its base on general clinical decisions?

**Answer as measured.**
- **Record audit: yes, on our benchmarks.** osler_v4's final Osler-4B is the strongest open model ≤ 10B we measured on the MIMIC-IV
  audit suite, and it beats pplx-27B on that suite (C220, scoreboard in `loops/osler_v4/results.md`).
- **General clinical: no.** On the external clinical panel it is −0.042 against its base (limit −0.02), so Gate O4 **FAILS** (C220).
- **Corrected recipes (osler_v5): no recipe qualified under the pre-registered rule** (C070). No new final was trained.

So the model is "the best open audit model we measured under 10B, with a general-skill gap not yet closed." It is not "world class."

---

## 2. Timeline of the loops

| loop | what it tested | result | where |
|---|---|---|---|
| bench_v0 / bench_v0_fix0 | benchmark, readout-health gate D12 | repaired; D12 added | `loops/bench_v0*/FINDINGS.md` |
| student_v0 / student_v1 | 0.8B heads | first MedDecide-0.8B; held-out transfer failed | `loops/student_v*/FINDINGS.md` |
| osler_v0 | four clinical generators; 4B and 9B | Gate O1 FAIL (panel vs MedDecider) | `loops/osler_v0/FINDINGS.md` |
| osler_v1 | synthetic-first audit data; attention-only LoRA | Gate O2 FAIL (Brier; real-text ICD) | `loops/osler_v1/FINDINGS.md` |
| osler_v2 | base choice pplx-flash (D39); real text | Gate O3 FAIL: panel −0.0198 [−0.0344, −0.0060] vs pplx | `loops/osler_v2/FINDINGS.md`, C-rows W055–W066 |
| osler_v3 | Stage A (KL, replay); Stage B blocks | control chosen; no block kept | `loops/osler_v3/FINDINGS.md` |
| osler_v4 | MIMIC-IV audit templates (D46); Nemotron minimal pairs (D45); 27B distillation (A5); 63,304-step final; Gate O4 | Gate O4 FAIL on panel; best audit model; closure 1a188ba | `loops/osler_v4/FINDINGS.md`, `gate_o4.md`, `results.md` |
| osler_v5 | per-item loss weighting, option augmentation, blend, corrected mix | no recipe qualified (outcome 4); no final | `loops/osler_v5/FINDINGS.md`, `arms.md`, `NEXT.md` |

---

## 3. The model

- **Base:** `perplexity-ai/pplx-decider-flash-v1.2` (Apache-2.0), a Qwen3.5-4B decision model distilled from pplx's 27B (D39).
  Its architecture is pplx's: noncausal full attention on full-attention layers, and a 255-row option-code readout
  (`src/meddecide/model/pplx.py`).
- **Our training:** attention-only LoRA on the base (D33). The readout is pplx's, and it is trained (rows initialised from pplx's
  own readout, not `lm_head`; osler_v5 Z1 probes).
- **Final (osler_v4):** 63,304 steps (D50); mix = osler_v2's C1 plus MIMIC blocks B1–B3 (osler_v4 STATE §7; composition hash
  `9d25e30f…`). Checkpoint selected by the pre-registered rule on fast dev v4 (C200).
- **Calibration:** per-question-type temperatures fitted on dev and saved (C203).

---

## 4. Benchmarks and frozen sets

| set | what | size | role | where |
|---|---|---|---|---|
| MR suite v1 | synthetic medical-record audit templates (osler_v2 lineage) | 13,103 verdict items, 7 templates (reported-only set separate) | Gate O4 check (i) | osler_v4 Y9 `gate.json` |
| MIMIC audit suite v1 | late-anchor test items of 9 MIMIC templates, frozen before training (D49) | 4,253 items | Gate O4 check (ii); scoreboard | `data/bench/osler_mimic_v1/` (manifest), C-rows from osler_v4 Y4 |
| External clinical panel | 7 public datasets (never trained on) | 4,353 items | Gate O4 check (iii) | osler_v4 Y9; per-dataset C224 |
| Knowledge | MedQA / MedMCQA test | 5,435 paired (Gate); 4,000 dev (osler_v5) | Gate O4 check (iv) | osler_v4 Y9; osler_v5 Z1 |
| CG v2 (dev proxy) | general clinical dev items from non-panel sources, 13-gram-checked | 8,307 items | dev only | osler_v4 Y4 (C051) |
| Fast dev v4 | dev items for selection | 17,532 items | dev only | osler_v4 Y4 |
| DDXPlus (in CG v2) | 49-option symptom-to-diagnosis dev set | 1,000 items; 501 single-letter, 499 two-letter gold | positional-health check | osler_v4 Y8 `code_split.json` (C210) |

**Contamination history (disclose).** Earlier loops found about 12 % dev-proxy contamination (osler_v4 STATE, C-rows near the Y4
section); osler_v4's proxies were cleaned, and verdict sets were checked by text (C-rows in osler_v4 Y4). Test sets were never used
for choices.

---

## 5. Results

### 5.1 Gate O4 (osler_v4 final, calibrated; baselines: zero-shot pplx-flash, MedDecider-4B) — C220, C221

| check | vs zero-shot pplx-flash | vs MedDecider-4B |
|---|---|---|
| (i) MR verdict | +0.1387 [+0.1243, +0.1530]: PASS | +0.2449 [+0.2328, +0.2568]: PASS |
| (ii) MIMIC audit suite | +0.1654 [+0.1502, +0.1795]: PASS | +0.1651 [+0.1490, +0.1804] (8 templates): PASS |
| (iii) external panel | **−0.0421 [−0.0584, −0.0269]: FAIL** (limit −0.02) | NOT MEASURED: 212 of 4,353 items (4.87 %) have more than ten options (MedDecider's prompt) |
| (iv) knowledge | −0.0067 [−0.0172, +0.0048]: PASS | +0.0060 [−0.0071, +0.0194]: PASS |

Limit: the pre-registered rule (D49) requires a pass on all four checks against both baselines. The panel check against
MedDecider-4B is therefore NOT MEASURED; the rule does not allow a verdict without it.

### 5.2 Size-class scoreboard (additional; no verdict) — `loops/osler_v4/results.md`

Template macro accuracy, osler_v4 final (C228 and results.md):

| set | osler_v4 final | zero-shot pplx-4B | pplx-27B (ceiling) | MedDecider-9B | MedDecider-4B | osler_v2 |
|---|---|---|---|---|---|---|
| MIMIC audit suite | **0.7995** | 0.6341 | 0.6819 | 0.6278 | 0.6115 (READOUT_FAIL) | 0.5651 |
| MR verdict | 0.8431 | 0.7044 | 0.7745 | 0.6223 | 0.6051 | **0.8621** |
| MR heldout (584) | 0.7679 (READOUT_FAIL) | 0.6949 | 0.8296 | 0.5424 | 0.5937 | 0.7284 |

On MR verdict the final is below osler_v2 (0.8431 against 0.8621). The difference is not tested for significance (osler_v4 NEXT item 4).
On knowledge the final scores 0.7024, against 0.7265 for MedDecider-9B and 0.7097 for zero-shot pplx (osler_v4 FINDINGS item 5).

### 5.3 Per-dataset panel (additional, C224)

Against zero-shot pplx (C224): symptom_to_diagnosis −0.0991 (212 items); medical_question_pairs_noul −0.0847 (295); medconceptsqa −0.0550
(1,000); mmlu_pro_health −0.0294 (781). Their intervals exclude zero. The other three datasets (medexpqa_en, medexqa, medxpertqa) are within
−0.023 to +0.008, with intervals that include zero.

### 5.4 Membership and verbatim audit (Y10) — C233, C235, C236

- Membership AUC for the final: 0.5042 [0.4853, 0.5221]; zero-shot control 0.5090. No measurable membership signal.
- Verbatim probe: 0 of 300 matches for both models. **Uninformative:** the causal export of the final generates poorly (true-continuation
  NLL 7.673 against 2.725 for the control; C235), and its argmax agrees with the decision path on 16 of 20 items (C236). A zero count
  does not show absence of memorisation.

### 5.5 osler_v5 corrected-mix arms (13,992 steps; dev only) — C041–C075, `loops/osler_v5/arms.md`

| arm | S | MR macro | MIMIC macro | CG micro | knowledge dev | positional, all three orders |
|---|---|---|---|---|---|---|
| F0 (osler_v4 recipe, seed 0) | 0.8296 | 0.9325 | 0.7822 | 0.7740 | 0.7658 | fails (original order) |
| F2 (per-item + augmentation, seed 0) | 0.8259 | 0.9375 | 0.7558 | 0.7844 | 0.7595 | passes |
| F2s1 (F2, seed 1) | 0.8243 | 0.9442 | 0.7496 | 0.7791 | 0.7633 | passes |

Paired against F0 (C066, C069; the D56c bounds): F2 knowledge −0.0063 [−0.0150, +0.0030] (bound −0.01: fails); MIMIC −0.0264
[−0.0343, −0.0186] (bound −0.03: fails). F2s1 knowledge −0.0025; MIMIC −0.0327 [−0.0420, −0.0239] (fails).
**Choice: none** (C070; outcome 4 of the osler_v5 ADVISORY).

### 5.6 Blend curve on the osler_v4 final (dev; additional) — C032–C034

| α | 0 | 0.3 | 0.5 | 0.7 | 0.85 | 1 |
|---|---|---|---|---|---|---|
| S | 0.7820 | 0.8252 | 0.8395 | 0.8415 | 0.8414 | 0.8386 |
| positional health | pass | pass | pass | fail | fail | fail |
| DDXPlus two-letter | 0.6112 | 0.5150 | 0.4148 | 0.1844 | 0.0762 | 0.0040 |

The blend improves S while positional health fails from α = 0.7. The rule (D53, Z4) was not run, because no final exists.

### 5.7 Readout split and readout scale — C035, C063

- Trained adapter with base readout: DDXPlus two-letter 0.1924; single-letter 0.7445. Base adapter with trained readout: two-letter
  0.0381; single-letter 0.4890 (C035). So the trained readout damages single-letter answers too.
- Readout rows 26–254 of the osler_v4 final sit 0.0057 from pplx's base; F2's sit 0.330 (C063, additional).

### 5.8 Readout probe on the osler_v4 final (dev; calibrated; additional) — C078–C087, `loops/osler_v5/z9_z10.md`

| setting | S | MR macro | MIMIC macro | CG v2 micro | knowledge dev | knowledge minus zero-shot pplx [95 % interval] | DDXPlus two-letter | DDXPlus single-letter | positional, three orders |
|---|---|---|---|---|---|---|---|---|---|
| base readout (pplx's step-0 readout) | 0.8439 | 0.9661 | 0.8105 | 0.7551 | 0.7560 | −0.0165 [−0.0258, −0.0065] | 0.1924 | 0.7445 | fails in all three |
| trained readout (the final as it is) | 0.8386 | 0.9672 | 0.8094 | 0.7393 | 0.7585 | −0.0140 [−0.0228, −0.0055] | 0.0040 | 0.6407 | fails in all three |

Both settings are calibrated per question type on fast dev v4. The trained readout's temperatures match osler_v4's saved fit to 3 × 10⁻⁴.
Both settings reproduce the Z1 figures exactly (C035, C032). The base readout alone does not fix positional health, so the adapter is
the next suspect (ADVISORY §10 Z9.4). These are additional rows; no choice uses them.

### 5.9 Comparator scoring on the test-side sets (additional; no choice) — C088–C103, `loops/osler_v5/z9_z10.md`

| model | MR verdict macro / micro | MR reported macro / micro | MIMIC suite macro / micro | MIMIC held-out (110) macro / micro | MedNLI test | panel macro / micro | knowledge micro | knowledge minus pplx, paired [95 % interval] |
|---|---|---|---|---|---|---|---|---|
| Qwen3.5-4B zero-shot | 0.5574 / 0.6176 | 0.9968 / 0.9939 | 0.5697 / 0.5603 | 0.6958 / 0.6364 | 0.7862 | 0.5483 / 0.4831 | 0.6162 | −0.0670 [−0.0791, −0.0539] |
| Qwen3.5-9B zero-shot | 0.5953 / 0.6315 | 0.9955 / 0.9945 | 0.6116 / 0.6029 | 0.7218 / 0.6727 | 0.8361 | 0.6225 / 0.5419 | 0.6774 | −0.0055 [−0.0186, +0.0070] |
| Clef-Flash (authors' code) | 0.6830 / 0.6830 | 0.9928 / 0.9953 | 0.6391 / 0.6306 | 0.6954 / 0.7182 | 0.8636 | 0.6502 / 0.5711 | 0.6864 | +0.0029 [−0.0092, +0.0155] |

All 93 cells pass the readout-health gate (D12 for the zero-shot models, D21 for Clef-Flash; C089), so no item is excluded. Three cells
sit at or near chance and pass the gate; they are reported as they are (C097). On the MIMIC ED-acuity cell all three models score below its majority-gold baseline (0.5567; C103). Clef-Flash's MIMIC cells are measured after its local-only
check passed (C088). The knowledge pairing is joined on osler_v4's item id (C102); the Qwen pairs compare two readouts and are labelled so.
These are additional rows; no choice uses them.

---

## 6. Defects and corrections (disclose)

1. **Single-letter DDXPlus was not intact** (my earlier statement). The osler_v4 final scores 0.6407 on single-letter gold against
   0.7964 for zero-shot (C039, corrected from C210's "intact" reading).
2. **Label defect, `fda_route_choice_v1`:** gold is `routes[0]` while other routes are also true; 155 rows with 17–18 options in the
   mix (C048). It was in every osler_v4 and osler_v5 run before the correction. The corrected mix drops the template (D56a, C059); the
   template is not repaired. Screen covered option counts only (SELF_AUDIT §4.6 of osler_v5); other templates are unchecked.
3. **Family-cap excess:** after the drop the corrected mix's family share is 0.350037, 54 rows above the 35 % cap (C059). The cap was
   not re-applied.
4. **Seed instability:** the osler_v4 recipe passes positional health at seed 0 and collapses at seed 1 (C052); F0 passes at 6,996 steps
   and fails at 13,992 (C064, C065). A single evaluation is weak evidence.
5. **osler_v4 Gate O4 pre-registration:** the rule text was fixed before the run (STATE §7, 2026-10-24T06:03:11Z; C220).
6. **Earlier dev-proxy contamination** (about 12 % in earlier loops; §4).
7. **Post hoc rule attempt:** D55 (a pairing amendment written after reading a gate result) was drafted and then dropped before any run
   (osler_v5 review; commit a9ca386). It is not in force.

---

## 7. Diagnosis (hypotheses — not measured as causes)

- **H1 — Readout coverage:** training rarely offers more than 26 options, so rows past Z drift or go unused. Partly supported: the
  two-letter collapse exists (C210), and augmentation helps it at the same step (C062, C065: two-letter 0.5391 for F2 against 0.3647 for F0 at 13,992 steps).
  Not isolated: augmentation also changes the loss.
- **H2 — Per-step loss averaging over-weights long MIMIC items.** Supported in the arithmetic: MIMIC share of the applied update 0.588
  under per-step against 0.3365 under per-item (C010–C011). The MIMIC macro fell under per-item (C066, C069). The causal link is not tested.
- **H3 — Readout scale drift.** The trained rows move far from base (C063). Not tested as a cause; the osler_v5 corrected arms did not
  constrain it.
- **H4 — General-skill loss beyond the readout.** The panel drop is visible on yes/no and 4-option datasets (C224), not only on
  many-option items. Not tested.

---

## 8. What is not measured (say so in the paper)

- The osler_v5 final, its blend, Gate O5 and the membership audit for the corrected recipe (Z3–Z6: BLOCKED; C070).
- The MedDecider-4B panel comparison (NOT MEASURED: 10-letter prompt, C221).
- Osler-9B and Osler-0.8B on any recipe.
- Significance of the MR gap between the final and osler_v2 (osler_v4 NEXT item 4).
- Independent validation of any claim: all benchmarks are ours, except the public panel and knowledge sets.

---

## 9. How to reproduce a number

Each claim ID has a recompute command in its CLAIMS row. Examples:
- Gate O4 numbers: `uv run --frozen python scripts/osler_v4/y9_gate.py` (writes `outputs/osler_v4/Y9/gate.json`).
- osler_v5 arm summaries: `uv run --frozen python scripts/osler_v5/z2_arm_summary.py <ARM>` (CPU).
- osler_v5 paired differences: `uv run --frozen python scripts/osler_v5/z2_pairs.py --control F0 F2 F2s1` (CPU).
- Choice under D56c: `scripts/osler_v5/z2_choose_d56c.py` (C070).

Raw outputs are gitignored (`outputs/`); the committed record is the CLAIMS rows, FINDINGS, gate documents and this file.

---

## 10. Pending (addendum, operator-approved 2026-10-28)

- **Z8 (additional, labelled):** the F2 recipe on the corrected mix at 63,304 steps. It is not a candidate and no rule chooses it.
  It shows whether the corrected recipe holds at full length. Its result appears in §5 as a new row once it runs. See
  `loops/osler_v5/ADVISORY.md` §9.
