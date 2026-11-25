# Osler Paper Drafting Guide (`osler_rc0`)

> **Purpose:** A completely self-contained blueprint and numerical reference for drafting the arXiv paper on **Osler-9B**, **Osler-4B**, and **MedDecide-Bench**. Every figure, table, statistical test, ablation, privacy audit, and limitation required for publication is either embedded directly in this document or queryable via SQL from the companion reporting database [`docs/paper/osler_rc0_reporting.duckdb`](osler_rc0_reporting.duckdb) (21 aggregate tables, 11.76 MB, zero raw benchmark text, zero MIMIC text, zero patient IDs, zero secrets).
>
> **Pointer Policy:** Every internal reference in this guide uses **within-repo relative paths** pointing strictly to committed files. Public model weights are linked by their pinned Hugging Face repositories.

---

## 0. Artifacts & Quick Start

| Artifact | Location (relative to repo root) | Role for Paper Drafting |
|---|---|---|
| **Reporting DuckDB** | [`docs/paper/osler_rc0_reporting.duckdb`](osler_rc0_reporting.duckdb) | Single source of truth for all plots, curves, CIs, per-source/template breakdowns, calibration bins, selective prediction, latency, training trajectories, and privacy audits. |
| **DuckDB Builder** | [`scripts/osler_rc0/build_paper_duckdb.py`](../../scripts/osler_rc0/build_paper_duckdb.py) | Deterministic script that builds the reporting database from frozen run artifacts. |
| **RC0 Synthesis** | [`loops/osler_rc0/FINDINGS.md`](../../loops/osler_rc0/FINDINGS.md) | Executive summary, 5-claim fresh-process spot-check, deviations, and sign-off checklist. |
| **Unified Leaderboard** | [`loops/osler_rc0/LEADERBOARD.md`](../../loops/osler_rc0/LEADERBOARD.md) | 12-model × 9-set accuracy, macro, Brier, ECE, coverage, readout health, and paired deltas. |
| **Pareto Analysis** | [`loops/osler_rc0/PARETO.md`](../../loops/osler_rc0/PARETO.md) | Global 2-axis frontier, ≤10B & ≤4B strict 5-set dominance, efficiency frontier, paired tables. |
| **Per-Category Tables** | [`loops/osler_rc0/PER_CATEGORY.md`](../../loops/osler_rc0/PER_CATEGORY.md) | Full source-level and template-level accuracies across all 9 frozen sets. |
| **PHI / Memorisation Audit** | [`loops/osler_rc0/AUDIT.md`](../../loops/osler_rc0/AUDIT.md) | Mandatory local D28 MIMIC membership-inference, Y10, verbatim-probe, and architecture audit. |
| **Claims Index** | [`loops/osler_rc0/CLAIMS.md`](../../loops/osler_rc0/CLAIMS.md) | All 29 verifiable claims (`C001`–`C029`) with exact provenance and recompute commands. |
| **Model Cards** | [`loops/osler_rc0/MODEL_CARD_9B.md`](../../loops/osler_rc0/MODEL_CARD_9B.md), [`loops/osler_rc0/MODEL_CARD_4B.md`](../../loops/osler_rc0/MODEL_CARD_4B.md) | Release specifications, serving snippets, and measured limitations. |
| **Program & Decision Log** | [`docs/plans/PROGRAM.md`](../plans/PROGRAM.md) | Complete settled decisions (`D1`–`D61`) and multi-loop research chronology. |
| **Benchmark Schema** | [`docs/benchmark/schema.md`](../benchmark/schema.md) | Executable item schema (`noul`, `choice`, `score`), deterministic IDs, and split invariants. |
| **Public Leaderboard** | [`docs/leaderboard.md`](../leaderboard.md) | Top-level repository leaderboard mirror. |
| **Published Weights (HF)** | [`abhinand/Osler-9B`](https://huggingface.co/abhinand/Osler-9B) (`1fba04ea...`), [`abhinand/Osler-4B`](https://huggingface.co/abhinand/Osler-4B) (`607b283f...`) | Standalone merged Apache-2.0 safetensors + readout + bundled `AutoModel` / `vllm-sr` runtime + dev calibration (no external library required). |

### Querying the Reporting Database
```python
import duckdb
con = duckdb.connect("docs/paper/osler_rc0_reporting.duckdb", read_only=True)
print(con.execute("SELECT table_name, row_count, paper_use FROM _catalog ORDER BY table_name").df())
```
Or from CLI using the repo environment:
```bash
uv run --frozen --with duckdb python -c "
import duckdb
con = duckdb.connect('docs/paper/osler_rc0_reporting.duckdb', read_only=True)
print(con.execute('SELECT * FROM pareto_two_axis ORDER BY five_set_macro DESC').fetchdf())
"
```

---

## 1. Paper Identity, Core Thesis & Claim Discipline

### Working Title Options
1. **Osler: Calibrated Single-Pass Medical Decision Models on the Efficiency–Accuracy Pareto Frontier** *(recommended — crisp, accurate, highlights both calibration and Pareto frontier)*
2. **System-One Clinical Decisions: Calibrated Single-Pass Models for Medical Record Audit and General Clinical Judgment**

### Core Thesis (3 Sentences)
Clinical record auditing, coding validation, and structured medical triage require bounded, well-calibrated probability distributions over arbitrary option sets at tens of milliseconds per decision—requirements poorly served by slow, uncalibrated autoregressive generation. We introduce **Osler-9B** (8.39B) and **Osler-4B** (4.55B), open Apache-2.0 "System-One" medical decision models trained with a data-engineered, format-matched recipe (clinical audit + structured real EHR tasks + format-matched distillation + KL-to-base + option-count augmentation + dev-only per-type temperature scaling) over a single noncausal prefill pass. Evaluated across a unified **12-model, 9-frozen-set (35,769-item) benchmark**, **Osler-9B** is the only ≤10B model on the global two-axis Pareto frontier alongside a 27B ceiling (`pplx-27B`), achieves the highest 5-set macro accuracy of all 12 models (**0.7801** vs. **0.7600** for `pplx-27B`) at **2.7× lower latency** (33.7 ms vs. 91.0 ms) and **3.1× lower VRAM** (17.3 GB vs. 52.9 GB), strictly dominates every comparable ≤10B medical and general decision model on all five headline sets, and clears a local MIMIC-IV PHI/memorisation audit.

### Strict Claim Discipline (What We DO vs. DO NOT Claim)
Follow these boundaries without exception when writing the manuscript:

**What we DO claim (all verified in `CLAIMS.md` and DuckDB):**
1. **Global 2-axis Pareto front:** On `(OOD General Avg, Clinical Audit Avg)`, the global front across all 12 models consists solely of **`Osler-9B` (0.7419, 0.8375)** and **`pplx-27B` (0.7725, 0.7411)**. `Osler-9B` is the only ≤10B model on the front.
2. **Strict ≤10B 5-set dominance:** `Osler-9B` strictly dominates `Osler-4B`, `pplx-flash`, `MedDecider-9B`, `MedDecider-4B`, and `Clef-Flash` on **all five** headline sets (`panel`, `knowledge`, `confirm_general`, `mr_verdict`, `mimic_suite`) with positive margins on every cell. Only its own zero-shot base `d3-flash` is undominated alongside it (a statistical tie on `knowledge`: `0.7322` vs. `0.7335`, Δ `-0.0013 [-0.0090, +0.0062]`).
3. **#1 on fresh post-cutoff tier-2 (`confirm_general`, 5,988 items):** `Osler-9B` scores **0.9008** (`[0.8933, 0.9083]`), beating its own base (`+0.0234 [+0.0180, +0.0291]`), `pplx-27B` (`+0.0165 [+0.0110, +0.0225]`), `Clef-Flash` (`+0.0075 [+0.0017, +0.0137]`), and `MedDecider-9B` (`+0.0247 [+0.0185, +0.0308]` on the 5,838-item intersection).
4. **Substantial clinical-audit & EHR gains:** `Osler-9B` beats `pplx-27B` by **+9.23 pts** on `mr_verdict` (`0.9016` vs. `0.8093`) and **+10.04 pts** on `mimic_suite` (`0.7733` vs. `0.6729`), and beats its base by **+16.65 pts** / **+14.32 pts**, while achieving **#1** on `mednli_test` (`0.8917`) and `pairs_test` (`0.9620`).
5. **Calibration & selective triage:** Dev-only per-`qtype` temperature scaling cuts `Osler-9B` panel ECE from `0.1085` → **`0.0229`** (lowest of all 12 models on `panel`) and `mr_verdict` ECE to **`0.0220`** without flipping a single argmax decision (`0/35,769`). At 80% selective coverage, `Osler-9B` reaches **98.22%** accuracy on `mr_verdict` (`1.78%` error vs. `11.04%` for `pplx-27B`) and **82.81%** on `mimic_suite` (`90.31%` at 50% coverage).
6. **≤4B specialist frontier:** `Osler-4B` (`0.7530` 5-set macro, `26.54 ms`, `9.60 GB`) strictly dominates `MedDecider-4B` on all five headline sets (`+22.60 pts` on `mr_verdict`, `+15.78 pts` on `mimic_suite`, `+0.66 pts` on `panel`, `+0.26 pts` on `knowledge`, `+0.63 pts` on `confirm_general`) and beats its own `pplx-flash` base by `+14.74 pts` (`mr_verdict`) and `+13.57 pts` (`mimic_suite`) with substantially better Brier on four of five headline sets.
7. **Privacy / PHI audit clearance:** Under our local D28 audit, visited-member MIMIC membership-inference AUC minus zero-shot base control is **`+0.0468`** (9B) and **`+0.0248`** (4B) — both clearing the pre-registered `≤ 0.05` rule; 4B verbatim extraction yields **`0/300`** matches at ≥50, ≥25, and ≥10 tokens; and `Osler-9B` carries no `lm_head` (architecturally non-generative).

**What we DO NOT claim (must be stated transparently as limitations):**
1. **Do NOT claim `Osler-9B` beats `d3-flash` on `panel` or `knowledge`:** It achieves **statistical parity** (`panel +0.0028 [-0.0069, +0.0129]`, `knowledge -0.0013 [-0.0090, +0.0062]`).
2. **Do NOT claim `Osler-9B` beats `pplx-27B` on hard multi-step medical QA:** `pplx-27B` leads `panel` (`0.6510` vs. `0.5927`) and `knowledge` (`0.7823` vs. `0.7322`), especially `medxpertqa_text` (`0.375` vs. `0.258`), `mmlu_pro_health` (`0.776` vs. `0.698`), and `medqa` (`0.883` vs. `0.793`).
3. **Do NOT hide `Osler-4B`'s OOD regression:** `Osler-4B` sits ~1 pt below its own `pplx-flash` base on `panel` (`-0.0113 [-0.0218, -0.0007]`) and `knowledge` (`-0.0098 [-0.0191, -0.0017]`). Report it explicitly as a capacity trade-off in 4B models.
4. **Do NOT quote `mr_heldout_test / synthea_indicator_evidence_v1` as an ordinary accuracy:** Both RCs fail the D21 below-chance check (`READOUT_FAIL`: 9B `0.356`, 4B `0.245`, chance `0.50`) while their bases pass (`0.459`, `0.597`). Present this as a key diagnostic finding on held-out template shift.
5. **Do NOT claim universal calibration transfer:** Dev-only temperatures transfer well to `panel`, `knowledge`, `mr_verdict`, and `mimic_suite`, but leave both RCs **under-confident** on fresh `confirm_general` (mean confidence `0.814` / `0.801` vs. accuracy `0.901` / `0.888`, ECE `0.090` / `0.094`).
6. **Do NOT present the PHI audit as a mathematical proof of zero memorisation:** Only ~15.5% of the 1.53M-row mix was visited in 63,304 steps, and the audit is a bounded empirical release check, not a differential-privacy guarantee.
7. **Do NOT impute `NOT MEASURED` cells:** `JEV-9B` on `confirm_general` (offline head loader error), `MedDecider-4B/9B` and `Clef-Flash` standardized latency, and `Qwen3.5`/`d1-2B` on unrun additional sets remain explicitly `NOT MEASURED`.

---

## 2. Section-by-Section Paper Blueprint

### Section 1 — Introduction
- **Motivation:** High-volume medical workflows—utilization management, ICD-10-CM/MS-DRG coding validation, clinical indicator verification (sepsis, AKI, acute respiratory failure), trial/label triage—are fundamentally **typed decision problems** (`noul` yes/no, `choice` over 2–255 named options, `score` over 2–10 ordered levels). Autoregressive chain-of-thought LLMs are slow (seconds/item), produce uncalibrated or unparsed outputs, and expose free-text hallucination/PHI-generation risk.
- **The System-One Paradigm:** A decision model performs **one prefill pass** over `(state, question, options)` and returns a calibrated probability distribution over the offered options in 26–34 ms.
- **The Core Research Challenge:** Across 11 prior iterative training loops (`osler_v0`–`v6`, `distill_v0`, `d3_distill`), naive domain fine-tuning and even large-scale teacher distillation from 27B–120B teachers systematically caused **catastrophic forgetting on out-of-domain clinical QA** (`-2.0` to `-4.2 pts` on an external 7-dataset panel) and **two-letter option-code collapse** (accuracy dropping to `0.4%` on >26-option tasks like DDXPlus).
- **Contributions (bulleted in Intro):**
  1. **Osler-9B & Osler-4B:** Open Apache-2.0 calibrated medical decision models (`abhinand/Osler-9B`, `abhinand/Osler-4B`) with verified offline parity and a non-generative 9B readout architecture.
  2. **Data-Engineered Anti-Forgetting Recipe (`v7`):** Diagnosis and resolution of two-letter option-code collapse (`D51`/`D52`), empirical demonstration of why pure teacher distillation fails to preserve OOD panel performance, and a format-matched C1–C4 + KL-to-base + token-capped training recipe that achieves full OOD parity at 9B alongside `+14`–`+17 pt` clinical audit gains.
  3. **Unified 12-Model, 9-Set Evaluation & MedDecide-Bench:** 35,769 frozen items spanning established exams, a 7-dataset OOD external panel, a post-cutoff structured-gold tier (`confirm_general`, ≥2026-03-01), real/synthetic medical record audits, MIMIC-IV payer-audit tasks, minimal pairs, and held-out templates—enforcing D12/D21 readout-health checks and paired bootstrap CIs.
  4. **Local MIMIC PHI & Memorisation Audit Protocol:** Visited-member membership inference calibrated against zero-shot base controls, 300-note verbatim extraction probing, and architectural non-generativity verification.

### Section 2 — Related Work & The Decision-Model Landscape
- Contrast four System-One readout families (see [`docs/plans/PROGRAM.md`](../plans/PROGRAM.md) §2):
  1. **Letter verbalizer:** single-token logits over option letters (`Qwen3.5`, `MedDecider-4B/9B` capped at 10 options, `JEV-9B` decision head capped at 16 options).
  2. **Pointer head:** small learned scoring head over option spans (`MedJev`, tested in our `student_v1` / `osler_v0` head ablations).
  3. **Joint schema head:** cross-attention over options and state (`Clef-Flash`).
  4. **Separate 255-row option-code readout:** rows initialized from base `lm_head` option-code tokens (`A`–`Z`, `AA`–`IV`), softmax normalized strictly over offered options (`pplx-decider`, `d3-flash`, **Osler-4B/9B**).
- Note our early controlled architecture findings (`student_v1` at 0.8B and `osler_v0` at 4B): pointer, joint schema, and option-code heads finished within **0.5–0.6 pts** of one another; all major performance gaps traced to **data composition, loss weighting, option cardinality coverage, and regularization** (`D26`).

### Section 3 — Architecture & Method
- **Typed Decision Schema** ([`docs/benchmark/schema.md`](../benchmark/schema.md), [`src/meddecide/bench/schema.py`](../../src/meddecide/bench/schema.py)):
  - `noul` (`yes`/`no`, returns `p(yes)`), `choice` (2–255 named options), `score` (2–10 ascending ordinal levels).
  - Deterministic 16-hex `item_id` over canonical JSON; split-by-record-hash prevents cross-template record leakage.
- **Model Architectures:**
  - **Osler-9B** (8.39B exact `8,401,603,824` params): Base `vllm-sr/d3-flash` (`67e03e1bc157961c582ee7c6a35d9b15fada3dec`, `Qwen3_5Model` noncausal full attention) + attention-only LoRA (`r=32, α=32`, [`D33`]) + separate `[255, 4096]` option-code readout evaluated at the final prompt position (`src/meddecide/distill/d3_train.py`). **Carries no `lm_head`** — state dict contains strictly `{lora, readout}` (39.8 MB weights), making text generation architecturally impossible.
  - **Osler-4B** (4.55B exact `4,545,556,992` params): Base `perplexity-ai/pplx-decider-flash-v1.2` (`9ff7cf9838e59df461c276b244d2af58`, `Qwen3_5Model` noncausal full attention) + attention-only LoRA (`r=32, α=32`) + separate `[255, d]` option-code readout evaluated at per-question markers (`src/meddecide/model/pplx.py`). Base generation is **20/20 token-identical** when the adapter is disabled (verified offline in R8).
  - **Why attention-only LoRA (`D33`, `D40`):** In `osler_v1`, all-layer LoRA degraded medical knowledge dev accuracy by `-6.0 pts` while attention-only LoRA preserved it (`-0.8 pts`, `+4.4 pts` on 5,456 test items) and accelerated training by **1.8×** (`6,406–6,838` vs. `3,744` real tokens/s).
- **Training Objective & Matched Option-A Recipe (63,304 steps for both 9B and 4B):**
  - **Loss:** Per-item weighted cross-entropy (`D51`, scaling post-clip gradients by `n_items / batch_size` so long MIMIC notes no longer dominate ~60% of updates for 36% of items) + multi-class Brier loss + **D57-dosed teacher KL** (`λ = 3.243598`, active on `32.64%` of 4B steps, stratified sampler over `438,188` teacher-labelled anchors) + **KL-to-base** (`0.1 × KL(base ‖ model)` on the 69,967 format-matched `C1–C4` cells, active on `3.63%` of 4B steps / `3.36–3.5%` of 9B steps under token-capped batching).
  - **Token-Capped Batching:** Cap of `8` items and `8,192` padded tokens per step (`407,250` plan batches for 4B; `408,236` for 9B; seed `0`). Across 63,304 steps, each run visits ~15.5% of the 1,526,405-row mix (`235,968` rows for 4B; `234,251` rows for 9B; Wall-clock for 4B: `14.07 h` on 1× RTX PRO 6000, peak VRAM `57.1 GB`, `0` non-finite losses, `0` tripwires).
  - **Option-Count Augmentation (`D52a`) & Positional Health (`D52b`):** Dynamically expands eligible `choice` templates (`mimic_principal_dx_v1`, `mimic_msdrg_v1`) to `8–64` shuffled options so readout rows `26–254` (`AA`+) receive gradient updates, eliminating `osler_v4`'s two-letter collapse (`0.4%` → `48.9%` 4B / `44.5%` 9B on DDXPlus dev, passing `≥ 0.5 × single-letter` across original + 2 permuted orders).
  - **Dev-Only Per-Type Temperature Calibration:** Fitted strictly on `fastdev_v4` (`14,597` rows: `9,284 choice`, `3,680 noul`, `1,633 score`) by minimizing NLL:
    - **Osler-9B:** `T_choice = 1.5274`, `T_noul = 2.3185`, `T_score = 1.7035` (dev NLL `0.6109 → 0.5342`, Brier `0.2822 → 0.2706`, ECE `0.0697 → 0.0192`, `0` argmax changes).
    - **Osler-4B:** `T_choice = 1.5222`, `T_noul = 2.3744`, `T_score = 1.5388` (dev NLL `choice 0.6455 → 0.5884`, `noul 0.4443 → 0.3104`, `score 0.9364 → 0.8692`, `0` argmax changes).
  - **WiSE-FT Interpolation Check (`osler_v7` T7/T9):** Weight-space interpolation toward the base (`s ∈ {0.0, 0.25, 0.5, 0.75, 1.0}`) did not beat the trained endpoint (`s = 1.00` won both 4B and 9B on dev `S`, and 9B blends `0.25`/`0.50` failed all-orders positional health).

### Section 4 — Training Mix & Multi-Loop Empirical Evolution
- **Training Mix (`mix_v7`, 1,526,405 rows total; private, aggregate stats only in `training_mix_composition`):**
  - By cell: `C0` base mix `1,456,438` (`95.42%`), format-matched `C1–C4` KL/distillation pool `69,967` (`4.58%`: `C1` `20,000`, `C2` `10,000`, `C3` `20,000`, `C4` `19,967`, labelled locally by `pplx-27B` in 3.72 h with 0 errors).
  - By `qtype`: `choice` `934,743` (`61.24%`), `noul` `434,283` (`28.45%`), `score` `157,379` (`10.31%`).
  - By key provenance: MIMIC-derived (local-only, patient-split, pre-2014 train) `539,807` (`35.36%`); teacher-labelled general anchors `438,188` (`28.71%`). Top sources: `mimic_iv` (`34.11%`), `clinicaltrials` (`16.02%`), `pubmed` (`8.46%`), `aqua_rat` (`4.58%`), `snli` (`4.04%`), `openfda` (`4.03%`), `mr_pmc` (`3.44%`), `medmcqa` (`3.34%`), `winogrande` (`2.65%`), `hellaswag` (`2.61%`), `medquad` (`2.57%`).
  - **Contamination Audit (`C003`):** Against `medconceptsqa_all` (1,000 external panel items), `mix_v7` has **`0/1,000`** exact normalized `(state + question)` matches and **`0/1,000`** items with any 13-gram hit (`max containment = 0.0` across 248,029,866 distinct mix 13-grams).
- **Why Pure Teacher Distillation Alone Failed (Table / Narrative from `loop_lineage`):**
  - `osler_v2` → `osler_v4`: Adding real clinical text and MIMIC payer-audit templates lifted `mimic_suite` to `0.7947`, but drove `panel` down `-4.21 pts` (`0.5254`) and caused complete two-letter readout collapse (`0.4%` on DDXPlus).
  - `osler_v6` (385k `pplx-27B` anchors, D57 dosing): Restored `knowledge` to parity (`0.6820`) while keeping audit gains, yet `panel` remained `-2.45 pts` below base (`0.5360`).
  - `distill_v0` (`d1-ultra` hosted teacher, panel `0.683`, 495k pool) & `d3_distill` (`d3-ultra` local teacher, 200k pool, 91% MedMCQA): Teacher exam strength boosted `knowledge` (`+5.2 pts` in `d3_distill`) but **did not transfer to OOD panel** (`-1.15 pts` in `distill_v0`, `-3.83 pts` vs. base in `d3_distill`).
  - **Resolution (`osler_v7 / rc0`):** Combining format-matched `C1–C4` cells, `KL-to-base`, `D51` per-item weighting, `D52` option-count augmentation, and a stronger 9B backbone (`d3-flash`) eliminated the OOD panel deficit completely at 9B (`+0.0028`, parity on `panel` and `knowledge`, `+2.34 pts` on fresh `confirm_general`) while delivering `+16.65` / `+14.32 pts` on clinical audit.

### Section 5 — MedDecide-Bench & Unified 12-Model Evaluation Protocol
- **9 Frozen Evaluation Sets (35,769 total items; zero test-set tuning):**
  - **5 Headline Sets (33,153 items):**
    1. `panel` (`4,353` items, 7 OOD public datasets: `medconceptsqa_all`, `medexpqa_en`, `medexqa_test`, `medical_question_pairs`, `medxpertqa_text`, `mmlu_pro_health`, `symptom_to_diagnosis`) — **OOD General**.
    2. `knowledge` (`5,456` items, 2 established medical exams: `medmcqa` 4,183 + `medqa` 1,273) — **OOD General**.
    3. `confirm_general` (`5,988` items, 12 fresh post-cutoff templates ≥2026-03-01 from `clinicaltrials` 2,486, `openfda` 1,304, `pubmed` 2,198; gold strictly from structured source fields) — **OOD Fresh Tier-2**.
    4. `mr_verdict` (`13,103` items, 7 clinical-record audit templates spanning `pmc_patients`, `codiesp`, `mkurman`, `synthea`, `mtsamples`, generated ICD/indicator records) — **Clinical Audit**.
    5. `mimic_suite` (`4,253` items, 9 MIMIC-IV late-anchor `≥2014` payer-audit templates: DRG CC/MCC level, MS-DRG, ED acuity/disposition, ICD specificity/support, lab validation, eMAR med-given, principal dx) — **Clinical Audit**.
  - **4 Additional / Held-Out Sets (2,616 items):** `mednli_test` (`1,422`), `pairs_test` (`500` Nemotron-edited MIMIC minimal pairs), `mr_heldout_test` (`584`, 3 held-out templates), `mimic_heldout` (`110`, held-out heart-failure ICD family / AKI indicator).
- **12-Model Roster:** 2 Release Candidates (`Osler-9B-rc0`, `Osler-4B-rc0`), 2 Direct Zero-Shot Bases (`d3-flash`, `pplx-flash`), 2 Open Medical Decision Competitors (`MedDecider-9B`, `MedDecider-4B`), 3 General System-One Competitors (`Clef-Flash`, `JEV-9B`, `d1-2B`), 2 Raw Zero-Shot LLM Backbones (`Qwen3.5-9B`, `Qwen3.5-4B`), and 1 Out-of-Tier 27B Ceiling (`pplx-27B`).
- **Evaluation Hygiene:** Every cell recomputed from per-item probability files with zero unmapped/duplicate IDs; 2,000-resample item-paired bootstrap 95% CIs (seed 0); D12/D21 readout-health checks (label mass for zero-shot, constant-answer + below-chance checks for all); explicit coverage reporting + exact ≤10-option paired intersections (`panel` 4,141; `confirm_general` 5,838) for fair comparison against `MedDecider-4B/9B`.

### Section 6 — Main Results & Pareto Frontier

#### Table 1 (Headline Accuracy & 2-Axis Pareto Summary — exact values from `leaderboard` & `pareto_two_axis`)

| Model | Params (B) | `panel` (4,353) | `knowledge` (5,456) | `confirm_general` (5,988) | `mr_verdict` (13,103) | `mimic_suite` (4,253) | OOD Avg | Audit Avg | 5-Set Macro | 2-Axis Front |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| **Osler-9B-rc0** | 8.39 | 0.5927 | 0.7322 | **0.9008** | **0.9016** | **0.7733** | 0.7419 | **0.8375** | **0.7801** | **YES** |
| `pplx-27B` *(ceiling)* | 26.09 | **0.6510** | **0.7823** | 0.8843 | 0.8093 | 0.6729 | **0.7725** | 0.7411 | 0.7600 | **YES** |
| **Osler-4B-rc0** | 4.55 | 0.5502 | 0.6728 | 0.8883 | 0.8941 | 0.7597 | 0.7038 | 0.8269 | 0.7530 | no |
| `d3-flash` *(9B base)* | 8.39 | 0.5899 | 0.7335 | 0.8774 | 0.7351 | 0.6301 | 0.7336 | 0.6826 | 0.7132 | no |
| `pplx-flash` *(4B base)* | 4.54 | 0.5615 | 0.6835 | 0.8889 | 0.7466 | 0.6240 | 0.7113 | 0.6853 | 0.7009 | no |
| `Clef-Flash` | 9.00 | 0.5711 | 0.6864 | 0.8933 | 0.6831 | 0.6306 | 0.7169 | 0.6568 | 0.6929 | no |
| `MedDecider-9B` | 9.00 | 0.5603* | 0.6997* | 0.8825* | 0.6653* | 0.6189 | 0.7142 | 0.6421 | 0.6853 | no |
| `MedDecider-4B` | 4.00 | 0.5436* | 0.6703* | 0.8820* | 0.6681* | 0.6019 | 0.6986 | 0.6350 | 0.6732 | no |
| `Qwen3.5-9B` | 9.00 | 0.5419 | 0.6774 | NM | 0.6315 | 0.6029 | 0.6097† | 0.6172 | — | no |
| `JEV-9B` | 9.00 | 0.5127* | 0.6448 | NM | 0.6303 | 0.5885 | 0.5787† | 0.6094 | — | no |
| `Qwen3.5-4B` | 4.00 | 0.4831 | 0.6162 | NM | 0.6176 | 0.5603 | 0.5497† | 0.5889 | — | no |
| `d1-2B` | 2.00 | 0.4259 | 0.4555 | NM | 0.7069 | 0.5587 | 0.4407† | 0.6328 | — | no |

*\*Partial coverage due to 10-option (`MedDecider`) or 16-option (`JEV`) prompt caps: `panel` 0.951 (4,141/4,353), `confirm_general` 0.975 (5,838/5,988), `knowledge` 0.996 (5,435/5,456), `mr_verdict` 0.992 (12,992/13,103). †Two-set OOD average (`panel`, `knowledge`); `confirm_general` is `NM` (NOT MEASURED).*

#### Key Sub-Result Highlights (Additional Sets & Per-Category)
- **Additional Sets:** `Osler-9B` / `Osler-4B` rank #1 and #2 on **`mednli_test`** (`0.8917` / `0.8910` vs. `pplx-27B` `0.8748`, `d3-flash` `0.8629`, `MedDecider-9B` `0.8425`) and **`pairs_test`** (`0.9620` / `0.9580` vs. `pplx-27B` `0.9400`, `d3-flash` `0.9400`, `MedDecider-9B` `0.7880`, `MedDecider-4B` `0.6920`). On **`mimic_heldout`** (n=110), `Osler-9B` scores `0.7000` (`+0.0182` vs. base, `-0.0818` vs. `pplx-27B` `0.7818`).
- **Paired ≤10-Option Intersections:** On `panel` (4,141 items): `Osler-9B` `0.5878` > `MedDecider-9B` `0.5603` (`+0.0275 [+0.0152, +0.0403]`) > `Osler-4B` `0.5455` > `MedDecider-4B` `0.5436` (`+0.0019 [-0.0109, +0.0140]`). On `confirm_general` (5,838 items): `Osler-9B` `0.9072` > `Osler-4B` `0.8936` > `MedDecider-9B` `0.8825` > `MedDecider-4B` `0.8820`.
- **Where the Audit Gains Come From (Per-Template):**
  - `codiesp_icd_support_v1` (real Spanish/English clinical case ICD coding, n=1,654): `Osler-4B` `0.7696`, `Osler-9B` `0.7310` vs. `pplx-27B` `0.4921`, `d3-flash` `0.2836`, `MedDecider-9B` `0.2291`.
  - `mimic_ed_acuity_v1` (n=494): `Osler-4B` `0.6862`, `Osler-9B` `0.6802` vs. `pplx-27B` `0.4899`, `d3-flash` `0.4170`.
  - `mimic_drg_cc_level_v1` (n=498): `Osler-9B` `0.7169`, `Osler-4B` `0.7149` vs. `pplx-27B` `0.5743`, `d3-flash` `0.5040`.
  - `mimic_msdrg_v1` (n=500): `Osler-9B` `0.6640`, `Osler-4B` `0.6000` vs. `pplx-27B` `0.5100`, `d3-flash` `0.4680`.
  - `fda_boxed_warning_noul_v1` (fresh post-cutoff openFDA labels, n=768): `Osler-4B` `0.9023`, `Osler-9B` `0.8880` vs. `pplx-27B` `0.7617`, `d3-flash` `0.7539`.

### Section 7 — Calibration, Selective Prediction, Cardinality Scaling & Efficiency
- **Calibration (Brier & ECE):**
  - On `panel`: `Osler-9B` raw ECE `0.1085` → calibrated **`0.0229`** (Brier `0.5409 → 0.5223`, #1 calibration of all 12 models, beating `d3-flash` ECE `0.1176` and `pplx-27B` ECE `0.1131`); `Osler-4B` raw ECE `0.1450` → calibrated **`0.0374`** (Brier `0.5942 → 0.5643`, `-0.0773` better Brier than `pplx-flash` `0.6416`).
  - On `mr_verdict`: `Osler-9B` ECE **`0.0220`** (Brier `0.1421` vs. `pplx-27B` `0.2775`, `d3-flash` `0.3715`); `Osler-4B` ECE **`0.0314`** (Brier `0.1514`).
  - On `knowledge`: `Osler-4B` ECE **`0.0194`**, `Osler-9B` ECE `0.0441`.
- **Selective Prediction (Confidence-Thresholded Abstention — Table / Figure 3 from `selective_prediction`):**
  - **`mr_verdict` (13,103 items):** At 80% coverage, `Osler-9B` achieves **98.22%** accuracy (`1.78%` error, min conf `0.819`) and `Osler-4B` **97.93%** (`2.07%` error) vs. `pplx-27B` `88.96%` (`11.04%` error) and `MedDecider-9B` `73.77%`. At 50% coverage, both Osler models reach **99.76%–99.77%** accuracy (`~0.23%` error).
  - **`mimic_suite` (4,253 items):** At 80% coverage, `Osler-9B` reaches **82.81%** (`+9.08 pts` over `pplx-27B` `73.73%`; `+14.25 pts` over `d3-flash` `68.56%`). At 50% coverage, `Osler-9B` reaches **90.31%** (`+6.77 pts` over `pplx-27B` `83.54%`).
  - **`panel` (4,353 items):** At 50% coverage, `Osler-9B` reaches **82.18%** at a well-calibrated confidence threshold (`0.567` vs. overconfident `0.736` for `d3-flash` at `81.99%` and `0.872` for `pplx-flash` at `76.21%`).
- **Option-Count Cardinality Scaling (`option_count_scaling`):**
  - Unlike `MedDecider-4B/9B` (hard 10-option cap, dropping `symptom_to_diagnosis` [212 items], `mtsamples` [111 items], and 150 fresh `confirm_general` items), Osler scores 2–255 options natively: on `11–26` options, `Osler-9B` reaches **`0.8108`** on `mr_verdict` (`mtsamples`) vs. `pplx-27B` `0.7297` and `d3-flash` `0.6757`, and **`0.6887`** on `panel` (`symptom_to_diagnosis`, matching `pplx-27B` `0.6934`).
  - On `mimic_suite`, `Osler-9B` gains across every bucket: `2` options `0.8724` (vs. `pplx-27B` `0.7500`), `3–4` options `0.7477` (vs. `0.6590`), `5–10` options `0.7429` (vs. `0.6460`).
- **Standardized Latency & Footprint (500-item stratified sample: 250 `panel` + 250 `mr_verdict`, seed 0, 10 warm-up excluded, 1× RTX PRO 6000):**
  - `Osler-4B-rc0`: **`26.54 ms`** b=1 p50 (`63.55 ms` p95, `29.37 items/s`, `9.60 GB` VRAM, `4.55B` params) — negligible `+1.03 ms` LoRA overhead over `pplx-flash` (`25.51 ms`, `9.58 GB`).
  - `Osler-9B-rc0`: **`33.71 ms`** b=1 p50 (`93.11 ms` p95, `23.56 items/s`, `17.32 GB` VRAM, `8.40B` params) — zero measurable overhead vs. `d3-flash` (`33.80 ms`, `17.32 GB`), **2.70× faster** and **3.06× smaller VRAM** than `pplx-27B` (`90.96 ms` p50, `301.16 ms` p95, `52.92 GB` VRAM, `26.09B` params).
  - **Architectural finding on batching:** Batch-16 throughput (`21.35 / 16.80 / 4.77 items/s` for 4B / 9B / 27B) is **lower** than batch-1 (`29.37 / 23.56 / 7.95 items/s`) across all 5 measured models because noncausal full attention pads every sequence in a batch to the longest item (heavy-tailed clinical note lengths). Serving in production should use length-bucketed micro-batching or batch-1 continuous scheduling.

### Section 8 — Privacy, Local MIMIC Processing & Memorisation Audit (D28)
- **Strict Data Governance:** MIMIC-IV 3.1 / MIMIC-IV-Note 2.2 / MIMIC-IV-ED were processed strictly on operator-controlled local hardware (never sent to any hosted API); train/test splits are patient-disjoint (`anchor_year_group ≥ 2014` reserved for test); zero MIMIC text or training rows exist in the released weights or repo.
- **Visited-Member Audit Design (`AUDIT.md`):** Because 63,304 steps visit ~15.5% of the mix (`235,968` rows for 4B, `234,251` for 9B, intersection `36,439`, Jaccard `0.084`), auditing random mix rows would dilute membership signal. We therefore sampled **2,000 MIMIC rows visited by BOTH runs** (stratified across all 9 `mimic_suite` templates from the `12,993` shared visited MIMIC rows) against **2,000 late-anchor `mimic_suite` non-members** verified to have **`0/2,000`** content-hash overlap with the training mix.
- **Results:**
  - **Rule (a) Membership Inference AUC (`-log p(gold)`):**
    - `Osler-9B-rc0`: **`0.5876`** `[0.5703, 0.6051]` vs. zero-shot `d3-flash` control **`0.5408`** `[0.5234, 0.5583]` → **Δ `+0.0468`** `[+0.0316, +0.0626]` (**CLEAR**, `≤ 0.05`).
    - `Osler-4B-rc0`: **`0.5644`** `[0.5466, 0.5813]` vs. zero-shot `pplx-flash` control **`0.5396`** `[0.5224, 0.5569]` → **Δ `+0.0248`** `[+0.0097, +0.0397]` (**CLEAR**, `≤ 0.05`).
    - *(Why controls are ~0.54 rather than 0.50: temporal admission-year shift between pre-2014 train and ≥2014 test notes; subtracting the zero-shot base isolates training-induced signal.)*
    - Y10 historical compatibility set (`2,000 + 2,000`, only 35 visited by both): `Osler-9B` `0.5018`, `Osler-4B` `0.5033`, `d3-flash` `0.4975`, `pplx-flash` `0.5090`.
  - **Rule (b) Verbatim Extraction Probe (`Osler-4B-rc0`):** 300 MIMIC discharge notes, 128-token prefix, 64 greedy tokens from adapter-on causal export → **`0/300`** matches at ≥50, ≥25, and ≥10 consecutive tokens (max aligned run `1`, median `0`; truth continuation NLL `8.56` vs. `2.72` for untouched `pplx-flash` control, which also had `0/300` ≥50-token matches).
  - **Architectural Non-Generativity (`Osler-9B-rc0`):** Verified `has_lm_head = false` (only `{lora, readout}` keys, `[255, 4096]` readout); text extraction is structurally impossible.

### Section 9 — Honest Limitations & Failure Modes
Dedicate a full, candid Limitations section to all six measured failure modes:
1. **OOD QA Ceiling vs. 27B & Base Parity:** `Osler-9B` ties rather than beats `d3-flash` on `panel` (`+0.0028`) and `knowledge` (`-0.0013`), and trails `pplx-27B` on multi-step clinical reasoning (`medxpertqa_text` `0.258` vs. `0.375`, `mmlu_pro_health` `0.698` vs. `0.776`, `medqa` `0.793` vs. `0.883`). Single-pass System-One models excel at grounded verification/audit and calibrated triage, not deep multi-step diagnostic deduction.
2. **4B Capacity Trade-Off:** `Osler-4B` loses ~1 pt vs. `pplx-flash` on `panel` (`-0.0113`) and `knowledge` (`-0.0098`) in exchange for `+14.7` / `+13.6 pts` on clinical audit and large Brier gains.
3. **Held-Out Template Failure (`READOUT_FAIL`):** On `mr_heldout_test / synthea_indicator_evidence_v1` (n=514, 94.4% majority class, excluded from training), both RCs fall below chance (`9B 0.356`, `4B 0.245`, `READOUT_FAIL`) while their bases pass (`0.459` / `0.597`). Fine-tuning sharpened structured-evidence decision boundaries in ways that inverted on this unseen synthetic indicator template.
4. **Fresh-Distribution Calibration Transfer Gap:** Dev-fitted temperatures (`T_noul ≈ 2.32–2.37`, `T_choice ≈ 1.52–1.53`) calibrate `panel` (`ECE 0.023/0.037`) and `mr_verdict` (`0.022/0.031`), but make both RCs **under-confident** on the higher-accuracy fresh `confirm_general` set (`mean conf 0.814 / 0.801` vs. `acc 0.901 / 0.888`, `ECE 0.090 / 0.094`). Domain-shift recalibration on a small target-domain dev split remains important in deployment.
5. **Noncausal Padding Latency Penalty:** Standard batch-16 inference is slower than batch-1 for all measured noncausal models due to long-note padding overhead.
6. **Audit & Coverage Boundaries:** Only ~15.5% of the mix was visited in 63,304 steps; the PHI audit is an empirical release check, not a formal DP guarantee. `JEV-9B` on `confirm_general` and latency for `MedDecider`/`Clef-Flash` are `NOT MEASURED`. Neither model is approved for unsupervised clinical decision-making.

### Section 10 — Reproducibility, Release Artifacts & Verification
- **Public Standalone Merged Releases (Apache-2.0, zero external library dependency, base LICENSE + NOTICE included):**
  - **`abhinand/Osler-9B`**: `https://huggingface.co/abhinand/Osler-9B` (HF commit `1fba04eaa8fdcaa151373fd901b2d15aa5449907`, 29 files: 4 merged BF16 safetensors shards + FP32 `[255, 4096]` readout + bundled `d3_runtime` / `d3_server.py` + regenerated `MODEL_MANIFEST.json` + per-`qtype` dev calibration; base `vllm-sr/d3-flash@67e03e1bc157961c582ee7c6a35d9b15fada3dec`). Loads directly via `AutoModel.from_pretrained("abhinand/Osler-9B", trust_remote_code=True)` or `vllm-sr serve -e abhinand/Osler-9B`.
  - **`abhinand/Osler-4B`**: `https://huggingface.co/abhinand/Osler-4B` (HF commit `607b283f092e79e416eb718157c36ea269dbcd59`, 25 files: 2 merged BF16 safetensors shards + `[255, 2560]` readout + bundled `modeling_osler.py` + `pplx_decider` runtime with text-only fallback + per-`qtype` dev calibration; base `perplexity-ai/pplx-decider-flash-v1.2@9ff7cf9838e59df461c276b7699d56b244d2af58`). Loads directly via `AutoModel.from_pretrained("abhinand/Osler-4B", trust_remote_code=True)`.
- **Offline & Merged Verification (`C028` + standalone merge check):** Reloading staged adapters reproduces original probabilities to `max |Δp| = 0.0` (`20/20` argmax) for both models; merged standalone BF16 safetensors packages reproduce calibrated R2/R3 predictions on 5 panel items with `5/5` argmax agreement (`max |Δp| = 0.0109` 9B, `0.0111` 4B); `Osler-4B` adapter-disabled generation is `20/20` token-identical to untouched `pplx-flash`; `Osler-9B` carries `has_lm_head = false`.
- **Fresh-Process Spot-Check (`FINDINGS.md`):** 5 independent headline claims re-derived from raw artifacts all **PASS**; `ruff check .` clean; `pytest` passes all runnable tests (1 documented pre-existing strict-xfail `tests/test_padding_trained.py::test_trained_checkpoint_logits_are_padding_invariant`).

---

## 3. DuckDB Database Catalog & Exact Plot/Table SQL Recipes

Database path: [`docs/paper/osler_rc0_reporting.duckdb`](osler_rc0_reporting.duckdb) (DuckDB 2.0.1, 21 data tables + `_catalog` + `_metadata`).

### Complete Table Catalog

| Table | Rows | Primary Paper Role | Key Columns |
|---|---:|---|---|
| `models` | 12 | Model Setup Table | `model, params_b, size_tier, role, base_repo, base_revision, hf_release_repo, readout_type, non_generative, license` |
| `eval_sets` | 9 | Benchmark Overview Table | `set_name, role, is_headline, is_ood, n_items, n_templates, sha256, description` |
| `leaderboard` | 98 | Table 1 (Main Results) | `model, set_name, status, universe, scored, coverage, accuracy, accuracy_ci_lo, accuracy_ci_hi, macro_accuracy, brier, ece, readout_health` |
| `per_source` | 257 | Panel & Appendix Tables | `model, set_name, source, n_items, accuracy, brier, ece` |
| `per_template` | 461 | Per-Template & Failure Analysis | `model, set_name, template_id, n_items, n_options_mode, chance, majority_share, modal_share, accuracy, brier, ece, readout_status` |
| `paired_comparisons` | 81 | Significance / Delta Tables | `candidate, baseline, set_name, status, n_paired, acc_diff, acc_ci_lo, acc_ci_hi, brier_diff, brier_ci_lo, brier_ci_hi` |
| `paired_intersections_10opt` | 20 | Fair MedDecider Table | `set_name, model, n_items, accuracy, brier, ece` |
| `pareto_two_axis` | 12 | Figure 1 (Pareto Frontier) | `model, params_b, p50_ms_b1, peak_vram_gb, ood_general_avg, clinical_audit_avg, five_set_macro, on_two_axis_front, undominated_le10b_5set, undominated_le4b_5set` |
| `pareto_dominance` | 10 | Dominance Proofs Table | `scope, dominator, dominated, d_panel, d_knowledge, d_confirm_general, d_mr_verdict, d_mimic_suite` |
| `reliability_bins` | 1,560 | Figure 2 (Reliability Diagrams) | `model, set_name, calibration_variant, bin_idx, bin_lo, bin_hi, n_items, mean_confidence, mean_accuracy, abs_gap` |
| `selective_prediction` | 1,034 | Figure 3 (Risk-Coverage) | `model, set_name, coverage_target, actual_coverage, n_accepted, n_total, selective_accuracy, selective_error, min_confidence` |
| `option_count_scaling` | 225 | Figure 4 (Cardinality Scaling) | `model, set_name, option_bucket, n_items, mean_chance, accuracy, brier, ece` |
| `calibration_fits` | 7 | Calibration Table | `model, qtype, n_dev, temperature, nll_before, nll_after, brier_before, brier_after, ece_before, ece_after, argmax_changes` |
| `latency_benchmarks` | 13 | Figure 5 (Efficiency) | `model, batch_size, status, params_exact, p50_ms, p95_ms, mean_ms, items_per_s, peak_vram_gb, note` |
| `training_eval_trajectories` | 10 | Figure 6 (Dev Trajectory) | `model, step, dev_s, mr_macro, mimic_macro, cg_micro, knowledge_dev, positional_original_pass, positional_all_orders_pass, two_letter_acc, single_letter_acc` |
| `training_step_curves` | 2,533 | Appendix Loss Dynamics | `model, step, loss, ce, brier, kl, distill, accuracy, n_items, real_tokens, padded_tokens, elapsed_s, grad_norm, gpu_peak_gb, lr_head, lr_lora` |
| `phi_audit_membership` | 8 | Section 8 Audit Table | `audit_set, model, n_members, n_nonmembers, auc, auc_ci_lo, auc_ci_hi, base_control_model, base_control_auc, delta_auc, delta_ci_lo, delta_ci_hi, rule_a_clear` |
| `phi_audit_verbatim_arch` | 2 | Section 8 Verbatim/Arch | `model, generative_capable, has_lm_head, n_notes, matches_ge_50, matches_ge_25, matches_ge_10, longest_run_max, truth_nll_mean, control_matches_ge_50, control_truth_nll_mean, visited_rows_4b, visited_rows_9b, visited_intersection, nonmember_hash_overlap, verdict` |
| `training_mix_composition` | 53 | Section 4 Data Table | `dimension, key, row_count, share` |
| `loop_lineage` | 12 | Section 4 Evolution Table | `seq, loop_id, target_model, base_model, key_intervention, panel_acc, panel_delta_vs_base, knowledge_acc, mr_verdict_acc, mimic_suite_acc, outcome` |
| `claims` | 29 | Reproducibility Audit | `claim_id, claim, value, artifact, recompute_command, task` |

### Ready-to-Run Figure Queries

#### Figure 1 — Global 2-Axis Pareto Frontier (OOD General Avg vs. Clinical Audit Avg, marker size = params / latency)
```sql
SELECT model, params_b, p50_ms_b1, peak_vram_gb,
       ood_general_avg, clinical_audit_avg, five_set_macro,
       on_two_axis_front, undominated_le10b_5set
FROM pareto_two_axis
ORDER BY clinical_audit_avg DESC;
```
*Plot design:* Scatter `ood_general_avg` (x) vs. `clinical_audit_avg` (y), highlight Pareto frontier connecting `Osler-9B-rc0` `(0.7419, 0.8375)` and `pplx-27B` `(0.7725, 0.7411)`, draw arrows from `d3-flash → Osler-9B-rc0` and `pplx-flash → Osler-4B-rc0`, annotate parameter size and b=1 latency.

#### Figure 2 — Calibration Reliability Diagrams (Raw vs. Dev-Calibrated on `panel`, `mr_verdict`, and Under-Confidence on `confirm_general`)
```sql
SELECT model, set_name, calibration_variant, bin_idx,
       (bin_lo + bin_hi) / 2.0 AS bin_mid, n_items,
       mean_confidence, mean_accuracy, abs_gap
FROM reliability_bins
WHERE set_name IN ('panel', 'mr_verdict', 'confirm_general')
  AND model IN ('Osler-9B-rc0', 'Osler-4B-rc0', 'd3-flash', 'pplx-27B')
ORDER BY set_name, model, calibration_variant, bin_idx;
```
*Plot design:* 3-panel reliability diagram (`panel`, `mr_verdict`, `confirm_general`) comparing `Osler-9B-rc0` `raw` vs. `calibrated` against `d3-flash` and `pplx-27B` with diagonal `y = x` reference and ECE callouts. Panel 3 visually demonstrates the honest fresh-set under-confidence gap (`ECE 0.090`).

#### Figure 3 — Selective Prediction Risk–Coverage Curves (`mr_verdict`, `mimic_suite`, `panel`)
```sql
SELECT set_name, model, actual_coverage, selective_accuracy, selective_error, min_confidence
FROM selective_prediction
WHERE set_name IN ('mr_verdict', 'mimic_suite', 'panel')
  AND model IN ('Osler-9B-rc0', 'Osler-4B-rc0', 'd3-flash', 'pplx-flash', 'MedDecider-9B', 'pplx-27B')
ORDER BY set_name, model, actual_coverage;
```
*Plot design:* 3 subplots showing selective accuracy (or risk `1 - acc`) vs. coverage `[0.10, 1.00]`, highlighting the 80%-coverage triage operating point (`98.2%` `mr_verdict`, `82.8%` `mimic_suite` for `Osler-9B-rc0`).

#### Figure 4 — Option-Count Cardinality Scaling (`2`, `3-4`, `5-10`, `11-26` options)
```sql
SELECT set_name, option_bucket, model, n_items, mean_chance, accuracy, brier
FROM option_count_scaling
WHERE set_name IN ('panel', 'mr_verdict', 'mimic_suite')
  AND model IN ('Osler-9B-rc0', 'Osler-4B-rc0', 'd3-flash', 'pplx-flash', 'MedDecider-9B', 'pplx-27B')
ORDER BY set_name, option_bucket, model;
```
*Plot design:* Grouped bar or line chart across option buckets showing Osler maintaining strong accuracy and Brier into `11–26` options where 10-option verbalizers (`MedDecider`) drop out completely.

#### Figure 5 — Efficiency Frontier (Batch-1 p50 Latency & Peak VRAM vs. 5-Set Macro / Clinical Audit Avg, plus Batch-1 vs. Batch-16 Comparison)
```sql
SELECT l.model, l.batch_size, l.params_exact / 1e9 AS params_b,
       l.p50_ms, l.p95_ms, l.items_per_s, l.peak_vram_gb,
       p.five_set_macro, p.clinical_audit_avg, p.ood_general_avg
FROM latency_benchmarks l
JOIN pareto_two_axis p USING (model)
WHERE l.status = 'MEASURED'
ORDER BY l.batch_size, l.p50_ms;
```

#### Figure 6 — 63,304-Step Training Trajectory & Two-Letter Positional Health (`Osler-9B-rc0` vs. `Osler-4B-rc0`)
```sql
SELECT model, step, dev_s, mr_macro, mimic_macro, cg_micro, knowledge_dev,
       two_letter_acc, single_letter_acc, positional_original_pass, positional_all_orders_pass
FROM training_eval_trajectories
ORDER BY model, step;
```
*Appendix optimization dynamics:* Query `training_step_curves` (every 50 steps) for `ce`, `kl`, `distill`, `grad_norm`, `elapsed_s`.

#### Appendix Tables (Per-Source Panel Breakdown & Loop Evolution)
```sql
-- Panel per-source breakdown (Figure / Appendix Table)
SELECT source, n_items, model, accuracy, brier, ece
FROM per_source
WHERE set_name = 'panel'
ORDER BY source, accuracy DESC;

-- Research lineage showing why pure distillation failed on panel
SELECT seq, loop_id, target_model, key_intervention,
       panel_acc, panel_delta_vs_base, knowledge_acc, mr_verdict_acc, mimic_suite_acc, outcome
FROM loop_lineage
ORDER BY seq;
```

---

## 4. Within-Repo Code & Evidence Reading Map

- **Core model implementations:**
  - 9B readout & LoRA wrapper: [`src/meddecide/distill/d3_train.py`](../../src/meddecide/distill/d3_train.py)
  - 4B `pplx` noncausal decision model: [`src/meddecide/model/pplx.py`](../../src/meddecide/model/pplx.py)
  - Core trainer, per-item weighting (`D51`), option augmentation (`D52`), KL-to-base (`D41`), D57 distillation: [`src/meddecide/train/trainer.py`](../../src/meddecide/train/trainer.py)
  - Calibration (`apply_temperature`, NLL fitting, ECE): [`src/meddecide/eval/calibration.py`](../../src/meddecide/eval/calibration.py)
  - Unified RC0 leaderboard & Pareto engine: [`src/meddecide/eval/rc0_board.py`](../../src/meddecide/eval/rc0_board.py)
- **RC0 execution & release scripts:**
  - Leaderboard & Pareto generator: [`scripts/osler_rc0/rc0_leaderboard.py`](../../scripts/osler_rc0/rc0_leaderboard.py)
  - Local MIMIC PHI / memorisation audit: [`scripts/osler_rc0/r6_phi_audit.py`](../../scripts/osler_rc0/r6_phi_audit.py)
  - Standardized latency benchmark: [`scripts/osler_rc0/r5_latency.py`](../../scripts/osler_rc0/r5_latency.py)
  - Offline release staging & verification: [`scripts/osler_rc0/r8_stage_release.py`](../../scripts/osler_rc0/r8_stage_release.py)
  - Standalone merge, GPU parity check & HF publisher: [`scripts/osler_rc0/merge_and_push_standalone.py`](../../scripts/osler_rc0/merge_and_push_standalone.py)
  - Reporting DuckDB builder: [`scripts/osler_rc0/build_paper_duckdb.py`](../../scripts/osler_rc0/build_paper_duckdb.py)
- **Prior loop findings (for Section 4 narrative):**
  - [`loops/bench_v0_fix0/FINDINGS.md`](../../loops/bench_v0_fix0/FINDINGS.md), [`loops/student_v1/FINDINGS.md`](../../loops/student_v1/FINDINGS.md), [`loops/osler_v0/FINDINGS.md`](../../loops/osler_v0/FINDINGS.md), [`loops/osler_v1/FINDINGS.md`](../../loops/osler_v1/FINDINGS.md), [`loops/osler_v2/FINDINGS.md`](../../loops/osler_v2/FINDINGS.md), [`loops/osler_v4/FINDINGS.md`](../../loops/osler_v4/FINDINGS.md), [`loops/osler_v5/FINDINGS.md`](../../loops/osler_v5/FINDINGS.md), [`loops/osler_v6/FINDINGS.md`](../../loops/osler_v6/FINDINGS.md), [`loops/osler_v7/FINDINGS.md`](../../loops/osler_v7/FINDINGS.md), [`loops/osler_rc0/FINDINGS.md`](../../loops/osler_rc0/FINDINGS.md).
