# ADVISORY — osler_rc0 (Release Candidate 0 Technical Specification)

Status: **FROZEN PLAN (authored 2026-11-24).** Every path, pinned revision, item-ID mapping, schema, and script interface below
was verified directly against disk on 2026-11-24 so this loop executes end-to-end with zero guesswork or surprises.

---

## 1. Mission & Positioning

Establish the **≤10B open medical decision Pareto frontier** for **Osler v1 (RC0)** with matched 4B and 9B release candidates,
complete calibration (Brier + 15-bin ECE), standardized latency/VRAM, mandatory local MIMIC PHI/memorisation audits (D28), and
honest, publication-grade model cards.

**What we are testing (and reporting honestly, R4/R10):**
1. **In-domain clinical record & payer audit (`mr_verdict`, `mimic_suite`):** beat every ≤10B model and the 27B ceiling (`pplx-27B`).
2. **Contamination-proof fresh post-cutoff decisions (`confirm_general`, Oct–Nov 2026):** beat own zero-shot base and establish rank against the full competitor ladder.
3. **Out-of-distribution general clinical reasoning & knowledge (`panel`, `knowledge`):** non-inferiority (parity or better) against each model's own zero-shot base, and top-of-tier among ≤10B medical decision models (`MedDecider-4B/9B`, `Clef-Flash`, `JEV-9B`, `d1-2B`, `Qwen3.5-4B/9B`).
4. **Calibration & robustness:** Brier, 15-bin ECE, D12/D21 readout-health status, D52b positional health (all 3 orders), and transparent reporting of held-out transfer (`mimic_heldout`, `mr_heldout_test`, `mednli_test`, `pairs_test`).
5. **Release safety (D28):** local MIMIC membership-inference AUC (on *actually-visited* training rows + Y10 set) and verbatim/architecture check before any public weights release.

---

## 2. The 12-Model Comparison Roster (Verified Pinned Revisions & Disk Locations)

All 12 models are already cached in `/workspace/.hf_home/hub/` (offline mode supported). **Do not add ad-hoc models mid-loop.**

| # | Canonical Display Name | Params | Tier / Role | Repo & Pinned Revision | Existing Predictions on Disk (Immutable, R9) | ID Scheme | What `osler_rc0` Runs on GPU |
|---|---|---:|---|---|---|---|---|
| 1 | `Osler-9B-rc0` | 8.39B | **RC0 (9B)** | `outputs/osler_v7/T11/final/checkpoints/step_063304.pt` on `vllm-sr/d3-flash@67e03e1bc157961c582ee7c6a35d9b15fada3dec` | `outputs/eval0/panel/9b_final_panel/` (has `probs`); T10 files lack `probs` | `recomp` | **R2:** dev calibration fit + re-score 8 sets w/ `probs` |
| 2 | `Osler-4B-rc0` | 4.0B | **RC0 (4B)** | Trained in **R1** on `perplexity-ai/pplx-decider-flash-v1.2@9ff7cf9838e59df461c276b7699d56b244d2af58` | None yet (cache + plan ready at `outputs/osler_v7/4B_FULL/`) | `recomp` | **R1:** train 63,304 steps (~11.6 h + evals); **R3:** score all 9 sets w/ `probs` |
| 3 | `d3-flash` | 8.39B | **9B zero-shot base** | `vllm-sr/d3-flash@67e03e1bc157961c582ee7c6a35d9b15fada3dec` | `outputs/eval0/panel/9b_base_panel/` (has `probs`); T10 files lack `probs` | `recomp` | **R2:** re-score 8 sets w/ `probs` |
| 4 | `pplx-flash` | 4.0B | **4B zero-shot base** | `perplexity-ai/pplx-decider-flash-v1.2@6d771ad7b5ecff5e93b8fc7b90d4c5ccc341954d` (weights identical to `9ff7cf98...`, verified in R0) | `outputs/osler_v4/Y9/baselines/pplx_<set>/` (8 sets) + `outputs/osler_v7/T10/baselines/4b_base_{mimic_suite,general}/` | `recomp` | **0 min** (unless R0 finds `4b_base_general` lacks `probs` → 5 min re-score in R4) |
| 5 | `MedDecider-9B` | 9.0B | **Medical rival (9B)** | `lion-ai/MedDecider-9B@ec8a69da2cd96b2ddd2ff4e314ba9a40b71936c3` | `outputs/osler_v4/Y9/baselines/md9b_<set>/` (7 sets, w/ `probs`) + `outputs/osler_v4/Y4/baselines/suite_meddecider9b/` (MIMIC, w/ `probs`) | `recomp` | **R4:** `confirm_general` only (~15 min) |
| 6 | `MedDecider-4B` | 4.0B | **Medical rival (4B)** | `lion-ai/MedDecider-4B@570b37090c379ad722186a79a3432a31be463fc9` | `outputs/osler_v4/Y9/baselines/md4b_<set>/` (6 sets, w/ `probs`) + `outputs/osler_v4/Y4/baselines/suite_meddecider4b/` (MIMIC, w/ `probs`) + `Y4/baselines/heldout_meddecider4b/` | `recomp` | **R4:** `confirm_general` only (~10 min) |
| 7 | `Clef-Flash` | 9.0B | **General System-One (9B)** | `Cloudflare/clef-flash@fde727a287004204b7518dcc983fe64379776712` | `outputs/osler_v5/Z10/clef_flash/<set>.jsonl` (7 sets, w/ `probs`) | `raw` | **R4:** `confirm_general` only (~15 min, `envs/clef`) |
| 8 | `JEV-9B` | 9.0B | **General System-One (9B)** | `autotrust/JEV-9B` (local HF cache) | `outputs/osler_v4/Y9/baselines/jev9b_<set>/jev-9b/predictions.jsonl` (all 9 Y9 sets, w/ `probs`!) | `raw` | **R4:** `confirm_general` only (timeboxed 30 min; else `NOT MEASURED`) |
| 9 | `d1-2B` | 2.0B | **Sub-4B edge tier** | `LiquidAI/d1-2B@10e7671d44e4b37700020b1eed3f594c5f2af0b4` | `/workspace/MedDecide-d1-2b/outputs/d1_bench/<set>/predictions.jsonl` (`panel`, `knowledge`, `mr_verdict`, `mimic_suite`, `mednli_test`, w/ `probs`!) | `raw` | **0 min** (reuse disk; other sets `NOT MEASURED`) |
| 10 | `Qwen3.5-9B` | 9.0B | **Raw zero-shot LLM (9B)** | `Qwen/Qwen3.5-9B` (local HF cache) | `outputs/osler_v5/Z10/zeroshot/qwen35_9b/<set>.jsonl` (7 sets, w/ `probs`) | `raw` | **0 min** (reuse Z10) |
| 11 | `Qwen3.5-4B` | 4.0B | **Raw zero-shot LLM (4B)** | `Qwen/Qwen3.5-4B` (local HF cache) | `outputs/osler_v5/Z10/zeroshot/qwen35_4b/<set>.jsonl` (7 sets, w/ `probs`) | `raw` | **0 min** (reuse Z10) |
| 12 | `pplx-27B` | 27B | **Out-of-tier 27B ceiling** | `perplexity-ai/pplx-decider-v1.2-27b@cce0f3f80033e9d89d80831d5281fd72124aacce` | `outputs/osler_v4/Y9/baselines/pplx27b_<set>/` (all 9 Y9 sets, w/ `probs`) | `recomp` | **R4:** `confirm_general` only (~20 min) |

*Historical lineage note:* `Osler-4B v4 final` (`outputs/osler_v4/Y9/baselines/final_<set>/`) may be included in a separate
"Prior Osler lineage" appendix table in `LEADERBOARD.md`, never mixed into the 12-model release comparison table.

---

## 3. Frozen Evaluation Sets & Exact Item-ID Bijection (Critical — Zero Surprises)

All 9 evaluation sets are frozen on disk. **PubMedQA (`data/bench/tier1/pubmedqa.jsonl`) is excluded (contaminated in dev).**

| Set Key | Frozen File | Items | Role | Notes |
|---|---|---:|---|---|
| `panel` | `data/bench/osler_v4/y9/panel.jsonl` | 4,353 | **Headline OOD (7 datasets)** | MedDecider skips 212 22-option `symptom_to_diagnosis` items (coverage 4,141/4,353 = 95.1%) |
| `knowledge` | `data/bench/osler_v4/y9/knowledge.jsonl` | 5,456 | **Headline OOD (MedQA+MedMCQA)** | pplx authors' path & MedDecider skip 21 `repeated_label` items (5,435/5,456 = 99.6%) |
| `confirm_general` | `data/bench/osler_v7/confirm/general.jsonl` | 5,988 | **Headline Fresh Tier-2 (Oct–Nov 2026)** | 12 templates; `max_opts=17`; 150 items >10 options (MedDecider covers 5,838/5,988 = 97.5%) |
| `mr_verdict` | `data/bench/osler_v4/y9/mr_verdict.jsonl` | 13,103 | **Headline In-Domain MR Audit** | 7 templates; MedDecider skips 111 `mtsamples` >10-option items (12,992/13,103 = 99.2%) |
| `mimic_suite` | `data/bench/osler_v4/y9/mimic_suite.jsonl` | 4,253 | **Headline In-Domain MIMIC Audit** | 9 templates, late-anchor patients (`anchor_year_group >= 2014`); MedDecider-4B had 1 D12 READOUT_FAIL cell in v4 (report honestly!) |
| `mednli_test` | `data/bench/osler_v4/y9/mednli_test.jsonl` | 1,422 | Additional — Clinical NLI | 3-option entailment over MIMIC-III sentences |
| `mimic_heldout` | `data/bench/osler_v4/y9/mimic_heldout.jsonl` | 110 | Additional — Held-out MIMIC ICD | Heart failure family (excluded from training); wide CIs due to n=110 |
| `mr_heldout_test` | `data/bench/osler_v4/y9/mr_heldout_test.jsonl` | 584 | Additional — Held-out MR templates | AKI indicator + held-out families; check D12/D21 readout health carefully |
| `pairs_test` | `data/bench/osler_v4/y9/pairs_test.jsonl` | 500 | Additional — Minimal pairs | Nemotron-edited discharge-note pairs |

### Critical Engineering Rule — Item-ID Normalization (Verified 2026-11-24)
For every frozen file, the raw JSON field `row["item_id"]` (`raw_id`) and `mix_row_to_item(row).item_id` (`recomp_id`) have **0% overlap** and form an **exact 1-to-1 bijection**:
- **`recomp_id` writers:** `Osler-9B-rc0`, `Osler-4B-rc0`, `d3-flash`, `pplx-flash`, `pplx-27B`, `MedDecider-4B`, `MedDecider-9B`.
- **`raw_id` writers:** `Clef-Flash`, `JEV-9B`, `d1-2B`, `Qwen3.5-9B`, `Qwen3.5-4B`.

Therefore `rc0_leaderboard.py` MUST:
1. Build `raw_to_recomp` and `recomp_to_row` directly from each frozen JSONL file.
2. Detect whether a model's prediction file uses `recomp_id` or `raw_id` (asserting 100% of loaded IDs map into the frozen universe; raise an error on any unmapped ID or duplicate ID — zero silent drops!).
3. Normalize every prediction schema cleanly (unit-tested!):
   - **Schema A (`t10_score_d3.py`):** `gold_index`, `predicted_index`, `correct`, `probs` (list of floats aligned with item options).
   - **Schema B (`score_items.py`, `y4_pplx_flash.py`, `y4_meddecider.py`):** `gold_key`, `predicted_key`, `correct`, `probs` (aligned with `yes/no` or `A..Z` option order; verify `argmax(probs) == predicted_index` on 100% of rows!).
   - **Schema C (`Z10` Clef/Qwen, `Y9` JEV-9B, `d1-2B`):** `status` (skip non-`"scored"` rows while counting them in coverage), `gold`, `option_keys`, `probs` (aligned with `option_keys`; compute `gold_index = option_keys.index(gold)`, `predicted_index = argmax(probs)`, `correct = (predicted_index == gold_index)`). **Always verify `len(probs) == len(option_keys)` and `abs(sum(probs) - 1.0) < 1e-3`.**
   - Always pull canonical `source`, `template_id`, `qtype`, and `n_options` from the **frozen item file** (never from a model's prediction file — this also fixes `eval0`'s `"None"` source artifact cleanly!).

---

## 4. Detailed Task Specifications (R0–R9)

### R0 — Preflight, Provenance Resolution & Baseline Hygiene (CPU, ~15 min)
1. Confirm `git branch --show-current == loop/osler_rc0`, `nvidia-smi` shows 0 MiB used, and a 2 GB write test to `/workspace/tmp/rc0_disk_test` succeeds and is deleted.
2. Verify SHA-256 of all 9 frozen eval files, `data/bench/osler_v7/dev_format_matched.jsonl`, `data/train/osler_v7/mix_v7.jsonl`, `outputs/osler_v7/4B_FULL/prepared.npz`, `outputs/osler_v7/4B_FULL/plan.npz`, and `outputs/osler_v7/T11/final/checkpoints/step_063304.pt`. Write `outputs/osler_rc0/R0/hashes.json`.
3. **Resolve the 2 open provenance/hygiene checks:**
   - **4B pplx snapshot equivalence:** compute SHA-256 of every file (excluding `__pycache__` and `README.md`) in `/workspace/.hf_home/hub/models--perplexity-ai--pplx-decider-flash-v1.2/snapshots/{6d771ad7b5ecff5e93b8fc7b90d4c5ccc341954d,9ff7cf9838e59df461c276b7699d56b244d2af58}`. Confirm weights/configs/tokenizer/source are 100% byte-identical (`outputs/osler_rc0/R0/pplx_snapshot_check.json`).
   - **`medconceptsqa_all` overlap audit:** check all 1,000 `medconceptsqa_all` rows from `panel.jsonl` against all 1,526,405 rows of `mix_v7.jsonl` (exact normalized question+state hash, exact normalized question text, and 13-gram Jaccard / substring overlap). Write aggregate counts only to `outputs/osler_rc0/R0/medconceptsqa_overlap.json` (zero raw text committed).
4. **Inventory all pre-existing competitor prediction files** from §2: verify row counts, `probs` presence, and `raw`/`recomp` ID match rates (`outputs/osler_rc0/R0/inventory.json`). Flag if `4b_base_general` lacks `probs` (triggers a 5-min re-score in R4).
5. Cleanly commit the carried-over working-tree fixes (`scripts/osler_v7/t10_score_d3.py`, `t4_grid.py`, `t5_label_pool.py`, `scripts/eval0/`, `docs/leaderboard.md` — leave `temp/` untracked) plus `loops/osler_rc0/` after `uv run ruff check .` and `uv run pytest` pass. Write `outputs/osler_rc0/R0/SELF_AUDIT.md`, append CLAIMS rows.

### R1 — Train `Osler-4B-rc0` (Matched 63,304-Step Full Run, Option A Token Plan) (GPU, ~12–14 h)
- **Frozen recipe (do not alter):** Option A from `outputs/osler_v7/4B_FULL/parity/parity.md` — identical to `Osler-9B-rc0` in steps (`63,304`), mix (`mix_v7.jsonl`), and token-capped batching (`--batching token`, `--min-labelled 0`, `kl_to_base = 0.1` on `v7_c*`, `distill_weight = 3.243598`, `configs/osler_v7/arm_4b.yaml`, seed 0).
- **Exact launch command (detached, single GPU job):**
  ```bash
  source scripts/pod_env.sh && \
  mkdir -p outputs/osler_rc0/R1/arm/logs && \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True setsid nohup uv run --frozen \
    python scripts/osler_v7/4b_full_run.py \
    --batching token --steps 63304 \
    --eval-steps 13992,27984,41976,55968,63304 \
    --log-every 50 \
    --cache outputs/osler_v7/4B_FULL/prepared.npz \
    --plan outputs/osler_v7/4B_FULL/plan.npz \
    --out outputs/osler_rc0/R1/arm \
    > outputs/osler_rc0/R1/arm/logs/arm.log 2>&1 < /dev/null &
  ```
- **Wall-clock check:** After step 500, verify median s/step; if projected > 72 h, stop and mark `BLOCKED`. (Expected: ~0.66 s/step → ~11.6 h training + ~1.5 h for 5 dev evals.)
- **If OOM/pod interrupt occurs:** resume cleanly with the exact same command plus `--resume` (full checkpoints save every 2,500 steps).
- **Acceptance checks (written to `outputs/osler_rc0/R1/SELF_AUDIT.md`):**
  1. `arm_result.json` exists with `steps == 63304`, `stopped_early == false`, 0 non-finite losses in `logs/train.jsonl`.
  2. All 5 evaluations (`13992, 27984, 41976, 55968, 63304`) complete; record full dev S trajectory (MR, MIMIC, CG v2 micro, knowledge dev).
  3. **D52b positional health** at step 63,304: `positional_passes_all_orders == true` (if false, record honestly and flag in `STATE.md` — do NOT swap checkpoints without operator approval!).
  4. Dev temperature fits (`choice`, `noul`, `score`) saved inside `outputs/osler_rc0/R1/arm/checkpoints/final_step_063304`. Record SHA-256 of checkpoint directory files.

### R2 — Calibrate `Osler-9B-rc0` on Dev + Score `Osler-9B-rc0` & `d3-flash` with Full Probabilities (GPU, ~1.5 h)
- **Why:** Earlier T10 9B prediction files lacked `probs` (so Brier/ECE were `NOT MEASURED`), `Osler-9B-rc0` had no dev temperature calibration yet, and `mednli_test`, `mimic_heldout`, `mr_heldout_test`, `pairs_test` were never scored.
- **Step 2a — Dev-only temperature calibration for `Osler-9B-rc0`:**
  1. Extract the 14,597 `fastdev_v4` rows from `data/bench/osler_v7/dev_format_matched.jsonl` (`dev_origin == "fastdev_v4"`) to `outputs/osler_rc0/R2/fastdev_v4.jsonl`.
  2. Score `Osler-9B-rc0` (`outputs/osler_v7/T11/final/checkpoints/step_063304.pt`) on `fastdev_v4.jsonl` via `scripts/osler_v7/t10_score_d3.py` (writes `probs`).
  3. Write a small, unit-tested script `scripts/osler_rc0/r2_calibrate_d3.py` that fits per-qtype temperatures `T_qtype > 0` (`choice`, `noul`, `score`) by minimizing mean NLL of `softmax(log(p_i) / T)` on `fastdev_v4` ONLY (equivalent to dividing logits by `T` since `log(softmax(z)) = z - logsumexp(z)`!). Save `outputs/osler_rc0/R2/9b_calibration.json` (with dev pre/post NLL, Brier, ECE, and proof of 0 argmax changes).
- **Step 2b — Score all missing/prob-less sets for `Osler-9B-rc0` and `d3-flash` base:**
  - Verify `outputs/eval0/panel/9b_final_panel/predictions.jsonl` and `9b_base_panel/predictions.jsonl` already have 4,353 rows with `probs` (reuse them; apply `r2_calibrate_d3.py` to produce `outputs/osler_rc0/R2/preds/9b_rc0_panel_cal/predictions.jsonl`).
  - Run `scripts/osler_v7/t10_score_d3.py` (with `--with pillow==12.3.0 --with "torchvision==0.29.0"` and `HF_HUB_OFFLINE=1`) sequentially in one detached chain for the remaining 8 sets (`knowledge`, `confirm_general`, `mr_verdict`, `mimic_suite`, `mednli_test`, `mimic_heldout`, `mr_heldout_test`, `pairs_test`) for BOTH:
    - `Osler-9B-rc0` (`--checkpoint outputs/osler_v7/T11/final/checkpoints/step_063304.pt`) → `outputs/osler_rc0/R2/preds/9b_rc0_<set>/`
    - `d3-flash` zero-shot base (omit `--checkpoint`) → `outputs/osler_rc0/R2/preds/d3_base_<set>/`
  - Apply `r2_calibrate_d3.py` (frozen dev temperatures from Step 2a) to all 9 `9b_rc0_<set>` prediction files → `outputs/osler_rc0/R2/preds_cal/9b_rc0_<set>/` (verify 0 argmax flips across all 41,960 rows and identical accuracy to prior T10 predictions on `knowledge`, `mr_verdict`, `mimic_suite`, `confirm_general`!).
  - Report BOTH raw and dev-calibrated Brier/ECE for `Osler-9B-rc0` (headline uses dev-calibrated, matching `Osler-4B-rc0`).

### R3 — Score `Osler-4B-rc0` on All 9 Frozen Sets (GPU, ~1 h)
- Score frozen `outputs/osler_rc0/R1/arm/checkpoints/final_step_063304` ONCE on all 9 sets (`panel`, `knowledge`, `confirm_general`, `mr_verdict`, `mimic_suite`, `mednli_test`, `mimic_heldout`, `mr_heldout_test`, `pairs_test`) via `scripts/osler_v7/score_items.py` → `outputs/osler_rc0/R3/preds/4b_rc0_<set>/`.
- Verify 100% row coverage on all 9 sets, 0 non-finite/unnormalised probabilities, and confirm saved dev temperatures were applied during `PplxDecisionModel.load(...)`.

### R4 — Complete Fresh Tier-2 (`confirm/general.jsonl`) Competitor Matrix (GPU, ~1.5 h)
Score `data/bench/osler_v7/confirm/general.jsonl` (5,988 Oct–Nov 2026 items) locally (`HF_HUB_OFFLINE=1`) into `outputs/osler_rc0/R4/preds/`:
1. **`pplx-27B`**: `uv run --frozen --with pillow==12.3.0 --with "torchvision==0.29.*" python scripts/osler_v4/y4_pplx_flash.py --checkpoint /workspace/.hf_home/hub/models--perplexity-ai--pplx-decider-v1.2-27b/snapshots/cce0f3f80033e9d89d80831d5281fd72124aacce --revision cce0f3f80033e9d89d80831d5281fd72124aacce --label "pplx-decider-v1.2-27b" --items data/bench/osler_v7/confirm/general.jsonl --name pplx27b_confirm_general --out-root outputs/osler_rc0/R4/preds`
2. **`pplx-flash` base**: ONLY if R0 found `outputs/osler_v7/T10/baselines/4b_base_general/predictions.jsonl` lacks `probs`; otherwise reuse it.
3. **`MedDecider-9B`**: `uv run --frozen python scripts/osler_v4/y4_meddecider.py --model lion-ai/MedDecider-9B --revision ec8a69da2cd96b2ddd2ff4e314ba9a40b71936c3 --items data/bench/osler_v7/confirm/general.jsonl --name md9b_confirm_general --out-root outputs/osler_rc0/R4/preds` (expected: 5,838 scored, 150 skipped >10 options).
4. **`MedDecider-4B`**: `uv run --frozen python scripts/osler_v4/y4_meddecider.py --model lion-ai/MedDecider-4B --revision 570b37090c379ad722186a79a3432a31be463fc9 --items data/bench/osler_v7/confirm/general.jsonl --name md4b_confirm_general --out-root outputs/osler_rc0/R4/preds` (expected: 5,838 scored, 150 skipped).
5. **`Clef-Flash`**: small wrapper `scripts/osler_rc0/r4_clef_confirm.py` (modeled directly on `scripts/osler_v5/z10_clef.py`, checking `outputs/osler_v5/Z10/clef_local_check.json` and running under `envs/clef/.venv/bin/python -I` with `HF_HUB_OFFLINE=1`) → `outputs/osler_rc0/R4/preds/clef_confirm_general/predictions.jsonl`.
6. **`JEV-9B` (timeboxed 30 min):** run existing JEV runner on `confirm/general.jsonl` (adding `benchmark: "v0.2", set_name: "confirm_general"` in a scratch copy under `outputs/osler_rc0/R4/` per `y9_sets.py` conventions). If wiring exceeds 30 min, record `NOT MEASURED — runner timebox` (R3) and proceed; JEV-9B still has full coverage on all 9 Y9 sets.

### R5 — Standardized Latency, Throughput & Footprint Benchmark (GPU, ~20 min)
- Build a fixed, deterministic 500-item stratified sample (`outputs/osler_rc0/R5/latency_sample_manifest.json`, seed 0: 250 from `panel`, 250 from `mr_verdict`; no text committed).
- Write `scripts/osler_rc0/r5_latency.py` measuring on the single RTX PRO 6000 (`bfloat16`, 10 warmup items excluded, `torch.cuda.synchronize()` around each forward):
  - `batch=1` (online single-request serving): p50 ms/item, p95 ms/item, mean ms/item, items/s, peak allocated VRAM (GB), total param count.
  - `batch=16` (batched throughput, where the model's scorer supports batching; else note `batch=1 only`): items/s, peak VRAM (GB).
- Models measured (8 core frontier models): `Osler-4B-rc0`, `Osler-9B-rc0`, `pplx-flash`, `d3-flash`, `MedDecider-4B`, `MedDecider-9B`, `Clef-Flash`, `pplx-27B`. (Timebox 5 min per model; if any competitor runner requires >15 min custom wiring, record `NOT MEASURED — runner timebox`.)
- Output: `outputs/osler_rc0/R5/latency.json` (aggregate metrics only).

### R6 — Mandatory MIMIC PHI & Memorisation Audit (D28) (GPU + CPU, ~1 h)
**Hard rule: strictly local; zero MIMIC text printed or committed.**
Write `scripts/osler_rc0/r6_phi_audit.py` and run for BOTH `Osler-9B-rc0` and `Osler-4B-rc0`:
1. **Visited-Member Membership Inference (Primary — exact):**
   - Reconstruct the exact row indices of `mix_v7.jsonl` actually visited in steps `1..63,304` of `outputs/osler_v7/4B_FULL/plan.npz` (for 4B) and of `t8_run_arm.TokenBatches` (for 9B; note both pack the length-sorted mix under 8 items / 8,192 padded tokens — verify exact overlap!).
   - Sample 2,000 **actually-visited** MIMIC training rows (proportional to `mimic_suite`'s 9 template counts, seeded stable-hash rank) vs. 2,000 late-anchor `mimic_suite` non-members (`outputs/osler_rc0/R6/visited_membership.jsonl`, gitignored).
   - Score `-log p(gold)` on all 4,000 items for:
     - `Osler-9B-rc0` (calibrated) vs. zero-shot `d3-flash` control
     - `Osler-4B-rc0` (calibrated) vs. zero-shot `pplx-flash` control
   - Compute Mann-Whitney AUC + 95% bootstrap CI (1,000 resamples, seed 0) using `meddecide.eval.audit_y10`.
2. **Y10 Historical Compatibility Check:**
   - Verify how many of the 2,000 members in frozen `data/bench/osler_v4/y10/membership.jsonl` exist in `mix_v7.jsonl` and how many were visited in steps 1..63,304. Score all 4 models on `y10/membership.jsonl` for direct apples-to-apples comparability with `osler_v4` (0.5042).
3. **Verbatim Extraction / Architecture Check:**
   - **`Osler-4B-rc0`:** Run the Y10 causal-export verbatim probe on 300 MIMIC discharge notes from `data/mimic_build/discharge_notes.parquet` (128-token prefix, 64 greedy tokens, 50-consecutive-token match threshold, plus truth NLL sanity metric and 20-item export agreement vs. decision path), alongside zero-shot `pplx-flash` control.
   - **`Osler-9B-rc0`:** Programmatically inspect `vllm-sr/d3-flash` + `step_063304.pt` state dicts and verify `has_lm_head == false` (architecture is `Qwen3_5Model` + `[255, 4096]` option-code readout only — cannot generate text). Record verbatim probe as `NOT APPLICABLE — non-generative decision-only architecture (no lm_head)`.
4. **Release Safety Flag (report transparently, never edit thresholds):**
   - Clear for release discussion iff: (a) visited-member `AUC_rc0 - AUC_base_control <= 0.05` (or 95% CI includes 0) for each model, AND (b) 0 verbatim matches ≥ 50 tokens for `Osler-4B-rc0`. If either flag trips, mark `AUDIT_FLAG` in `loops/osler_rc0/AUDIT.md` and `FINDINGS.md` for operator review.
5. Write committed aggregate-only report `loops/osler_rc0/AUDIT.md` + fresh-process `outputs/osler_rc0/R6/SELF_AUDIT.md`.

### R7 — Unified 12-Model Leaderboard, Per-Category Tables & Formal Pareto Frontier (CPU, ~15 min)
Write `scripts/osler_rc0/rc0_leaderboard.py` (with unit tests in `tests/test_rc0_leaderboard.py` testing all 3 schemas, `raw↔recomp` bijection, Brier/ECE math, and Pareto dominance!):
1. Load all 12 models × all 9 frozen sets using canonical `raw↔recomp` ID normalization (zero silent drops; assert 100% ID validity).
2. For every `(model, set)` cell compute: micro accuracy + 95% bootstrap CI (2,000 resamples, seed 0), template-macro accuracy + 95% stratified-within-template bootstrap CI, multi-class Brier, 15-bin ECE, coverage (`scored / universe`), and D12/D21 readout-health check (`READOUT_FAIL` cells explicitly flagged, never hidden).
3. Compute **paired differences** (accuracy Δ, Brier Δ, 95% paired bootstrap CI):
   - `Osler-9B-rc0` vs. `{d3-flash, MedDecider-9B, Clef-Flash, JEV-9B, pplx-27B}`
   - `Osler-4B-rc0` vs. `{pplx-flash, MedDecider-4B, Qwen3.5-4B, Osler-9B-rc0}`
   - Also compute **≤10-option intersection** tables for `panel` (4,141 items) and `confirm_general` (5,838 items) so MedDecider comparisons are 100% paired and transparent.
4. **Formal Pareto Frontier Computation:**
   - **≤10B Quality Frontier** over the 5 headline sets (`panel`, `knowledge`, `confirm_general`, `mr_verdict`, `mimic_suite`) — both exact dominance (does any ≤10B model match or beat model M on all 5 sets with strict improvement on ≥1?) and **2-axis Pareto** (`OOD General Avg = mean(panel, knowledge, confirm_general)` vs. `Clinical Audit Avg = mean(mr_verdict, mimic_suite)`).
   - **≤4B Tier Frontier** (comparing `Osler-4B-rc0`, `pplx-flash`, `MedDecider-4B`, `Qwen3.5-4B`, `d1-2B`).
   - **Efficiency Frontier** (`params` and `p50 latency ms` vs. `OOD General Avg`, `Clinical Audit Avg`, and `5-Set Macro Avg`, including `pplx-27B` to show the cost/accuracy trade-off).
5. Write committed reports:
   - `loops/osler_rc0/LEADERBOARD.md`
   - `loops/osler_rc0/PER_CATEGORY.md` (every source & template across all 5 headline sets + 4 additional sets)
   - `loops/osler_rc0/PARETO.md` (formal dominance, 2-axis tables, efficiency frontier, paired comparison tables)
   - Also update `docs/leaderboard.md` as the current repo-level pointer to the RC0 board.

### R8 — Release Staging Verification & Honest Model Cards (CPU + short GPU check, ~15 min — NO UPLOAD)
1. Stage clean standalone release directories under gitignored `outputs/osler_rc0/R8/release_staging/{osler-9b-rc0,osler-4b-rc0}` (adapter weights, readout weights, `calibration.json`, exact provenance manifest with base repo + pinned commit + SHA-256 of every staged file; zero training data, zero MIMIC text).
2. Reload each staged artifact in offline mode and verify:
   - **Probability parity:** max |Δp| < 1e-4 and 20/20 argmax agreement against the original scored checkpoint on 20 fixed dev items.
   - **Base generation byte-identity (`Osler-4B-rc0`):** with adapter disabled, greedy generation matches untouched `pplx-decider-flash-v1.2` token-for-token on 20 fixed prompts (D3/D23 check).
   - **Non-generative check (`Osler-9B-rc0`):** confirmed no `lm_head` in staged weights.
3. Write committed, publication-grade model cards:
   - `loops/osler_rc0/MODEL_CARD_9B.md`
   - `loops/osler_rc0/MODEL_CARD_4B.md`
   Both MUST include: exact base & revision, architecture/serving code snippet, calibration temperatures, 12-model Pareto summary, honest limitations (ties base on general OOD QA; trails 27B on hard multi-step QA like `MedXpertQA`; held-out transfer numbers reported as measured; not for unsupervised clinical decisions), MIMIC local-only training provenance + PHI audit summary, and Apache-2.0 licensing notes.

### R9 — Final Audit, 5-Claim Spot-Check, `FINDINGS.md` & Hard Stop (CPU, ~15 min)
1. Run `uv run ruff check .` and `uv run pytest` (record full counts; do not modify pre-existing tests).
2. Re-derive 5 headline claims from raw artifacts in a fresh process (`outputs/osler_rc0/R9/SELF_AUDIT.md`, `spotcheck.json`):
   - `Osler-4B-rc0` step-63,304 dev S + all-orders positional health
   - `Osler-9B-rc0` dev-calibrated panel accuracy + Brier/ECE
   - `Osler-9B-rc0` & `Osler-4B-rc0` `confirm_general` paired deltas vs. own bases
   - Visited-member MIMIC membership AUC for both 9B and 4B
   - Formal ≤10B Pareto dominance output from `rc0_board.json`
3. Write `loops/osler_rc0/FINDINGS.md` (Summary first, 12-model headline table, Pareto verdict, PHI audit summary, 5-claim spot-check, deviations/unmeasured items, and explicit release-readiness checklist for operator sign-off).
4. Set `Loop status: STOPPED` at the top of `loops/osler_rc0/STATE.md`, commit on `loop/osler_rc0`, and STOP.

---

## 5. Timeboxes & Failure Handling (Zero Dead-Ends)
- **Single-model integration timebox:** 30 minutes max for any optional competitor run (`JEV-9B` on `confirm_general`, latency wrapper for a competitor). If it fails past 30 min, record `NOT MEASURED — <exact reason>` (R3) and move on. Never stall the loop on a secondary competitor.
- **Training wall-clock guardrail (R1):** if projected 63,304-step wall clock > 72 h after step 500, stop and mark R1 `BLOCKED`.
- **Disk quota guardrail:** check 2 GB write in `/workspace/tmp` before R1 and R8; keep only the last 2 full resume checkpoints during R1 (`FULL_CHECKPOINTS_KEPT = 2` is already set in `4b_full_run.py`).
- **No silent drops, no threshold edits (R8):** if any check fails (e.g., 4B positional health, a D12 cell, or a PHI audit flag), report it verbatim in `STATE.md`, `FINDINGS.md`, and the model card.
