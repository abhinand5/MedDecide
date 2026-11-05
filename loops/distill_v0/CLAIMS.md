# distill_v0 — claims (aggregates only; no item text, predictions, teacher labels or MIMIC text)

Format: `id | claim | value | artifact | recompute command | task`. Artifacts are gitignored; paths are given for the operator's machine. A number without a row here is not a result (R1).

| id | claim | value | artifact | recompute command | task |
|---|---|---|---|---|---|
| C-D1-01 | Teacher ceiling, external panel, micro accuracy | 0.6825 (2,971 / 4,353) | `outputs/distill_v0/D1/ceiling.json` | `uv run --frozen python scripts/distill_v0/d1_rederive.py --out outputs/distill_v0/D1` | D1 |
| C-D1-02 | Teacher ceiling, external panel, template-macro accuracy | 0.7416 | `outputs/distill_v0/D1/ceiling.json` | same as C-D1-01 | D1 |
| C-D1-03 | Teacher ceiling, knowledge, micro accuracy | 0.8167 (4,456 / 5,456) | `outputs/distill_v0/D1/ceiling.json` | same as C-D1-01 | D1 |
| C-D1-04 | Teacher ceiling, MedNLI test, micro accuracy | 0.8812 (1,253 / 1,422) | `outputs/distill_v0/D1/ceiling.json` | same as C-D1-01 | D1 |
| C-D1-05 | Teacher ceiling, MR verdict, micro accuracy | 0.7387 (9,678 / 13,102 labelled; 1 of 13,103 failed) | `outputs/distill_v0/D1/ceiling.json` | same as C-D1-01 | D1 |
| C-D1-06 | Teacher ceiling, MR reported page type, micro accuracy | 1.0000 (6,191 / 6,191; cue-solvable, not a gate set) | `outputs/distill_v0/D1/ceiling.json` | same as C-D1-01 | D1 |
| C-D1-07 | Teacher ceiling, MR held-out test, micro accuracy | 0.8322 (486 / 584; majority baseline 0.8510) | `outputs/distill_v0/D1/ceiling.json` | same as C-D1-01 | D1 |
| C-D1-08 | Teacher ceiling, MIMIC suite, micro accuracy | 0.6901 (2,935 / 4,253) | `outputs/distill_v0/D1/ceiling.json` | same as C-D1-01 | D1 |
| C-D1-09 | Teacher ceiling, MIMIC doc pairs, micro accuracy | 0.9360 (468 / 500) | `outputs/distill_v0/D1/ceiling.json` | same as C-D1-01 | D1 |
| C-D1-10 | Teacher ceiling, MIMIC held-out, micro accuracy | 0.7364 (81 / 110) | `outputs/distill_v0/D1/ceiling.json` | same as C-D1-01 | D1 |
| C-D1-11 | D1 teacher-ceiling API spend | $1.270910 for 30,259,759 input tokens, 35,971 calls, 1 failure (http_422) | `outputs/distill_v0/D1/manifest.json` | same as C-D1-01 (checks cost and token sums against the ledger) | D1 |
| C-D1-12 | Teacher non-determinism, 40 panel items labelled twice | 37/40 argmax agreement; max abs prob diff 0.2083; mean 0.0183 | `outputs/distill_v0/D1/parity.json` | same as C-D1-01 (recomputes both parity checks) | D1 |
| C-D1-13 | Frozen D1 label file SHA-256 | `aca410959a477e59…` (full value in the manifest) | `outputs/distill_v0/D1/labels.jsonl`, `manifest.json` | same as C-D1-01 (re-checks the file hash) | D1 |
| — | (rows are appended as tasks finish) | — | — | — | — |

D3 (Clef-Flash port check; artifact `outputs/distill_v0/D3/port_check.json`)

| id | claim | value | artifact | recompute command | task |
|---|---|---|---|---|---|
| C-D3-01 | Clef-Flash text-tower port verdict | `PORT_OK` (2026-11-03T20:44:39Z; pinned `fde727a2…`) | `outputs/distill_v0/D3/port_check.json` | `source scripts/pod_env.sh && uv run --frozen python scripts/distill_v0/d3_clef_port.py --out outputs/distill_v0/D3` (GPU) | D3 |
| C-D3-02 | Option-code readout initialised from the Clef `lm_head` | 255 codes; `rows_equal_lm_head: true`; text tower hidden 4096, 32 layers, vocab 248,320; vision tower dropped (456,010,480 parameters) | `outputs/distill_v0/D3/port_check.json` | same as C-D3-01 | D3 |
| C-D3-03 | Trainable parameters and a gradient pass on the 9B port | 8,908,800 trainable; 20-item smoke probabilities sum to 1 (min 0.99999993); LoRA grad norm 130.2, head grad norm 71.0; peak GPU 28.1 GB | `outputs/distill_v0/D3/port_check.json` | same as C-D3-01 | D3 |

D2 (basket + hosted label run; artifacts in `outputs/distill_v0/D2/`)

| id | claim | value | artifact | recompute command | task |
|---|---|---|---|---|---|
| C-D2-01 | Pool: corrected mix minus the rule-lookup templates | 1,268,815 rows (matches the pre-registration); rule-lookup drop 187,623 rows, exactly the recorded per-template counts; text-key SHA-256 `11d1a068d14a04ee…`, item-id SHA-256 `9f3a6faad38f6b4a…` | `outputs/distill_v0/D2/composition.json` | `uv run --frozen python scripts/distill_v0/d2_rederive.py --out outputs/distill_v0/D2` | D2 |
| C-D2-02 | Teacher basket T | 494,803 rows / 494,748 unique text keys (pre-registered 508,516; deviation −13,713 from nine stale per-template counts, documented); teacher share 39.00 %, MIMIC share 39.34 % | `outputs/distill_v0/D2/composition.json` | same as C-D2-01 | D2 |
| C-D2-03 | Frozen d1-ultra labels for training | 494,748 rows, 0 failures, 0 skipped; SHA-256 `06679bcd0899eec5…` | `outputs/distill_v0/D2/labels.jsonl`, `label_manifest.json` | same as C-D2-01 (re-checks the hash and the row count) | D2 |
| C-D2-04 | D2 label-run spend | $6.628504 over three attempts (label-file cost $6.621052); 157,644,106 input tokens; loop spend after D2 $7.899430 of the $15 cap | `outputs/distill_v0/D2/label_manifest.json`, `outputs/distill_v0/cost_ledger.jsonl` | same as C-D2-01 (ledger and label sums) | D2 |
| C-D2-05 | Pool key structure | 5,232 duplicated item ids / 27,340 rows (Winogrande twins); 1,826 duplicate text keys / 8,478 rows; 71 keys with conflicting gold, 0 with conflicting option keys | `outputs/distill_v0/D2/composition.json` | same as C-D2-01 | D2 |
| C-D2-06 | Frozen-set exclusion of the training basket | 0 labelled text-key hits and 0 labelled item-id hits against the 14 frozen files | `outputs/distill_v0/D2/composition.json`, `label_manifest.json` | same as C-D2-01 | D2 |

D4 (Osler-4B distillation run; artifacts in `outputs/distill_v0/D4/`)

| id | claim | value | artifact | recompute command | task |
|---|---|---|---|---|---|
| C-D4-01 | 4B distillation run completed | 63,304 steps (step budget), 10 evaluations at the Y8 grid, positional health passing in all 10 | `outputs/distill_v0/D4/arm_result.json` | `uv run --frozen python scripts/distill_v0/d4_rederive.py --out outputs/distill_v0/D4 --steps 63304` | D4 |
| C-D4-02 | λ calibration (pre-registered) | λ = 3.2067, frozen at step 200, from 72 measured steps; predicted distillation gradient share 0.40 | `outputs/distill_v0/D4/logs/lambda_calibration.json` | same as C-D4-01 | D4 |
| C-D4-03 | Realised dose | teacher 38.94 % of item slots (band 38–45 %), MIMIC 39.35 % (37–42 %); mean teacher items per batch 1.455 (batches mean 3.74 items; 3.115 per 8 slots); 76.2 % of batches below 3 | `outputs/distill_v0/D4/dose_summary.json`, `logs/dose.jsonl` | same as C-D4-01 | D4 |
| C-D4-04 | Final dev figures at step 63,304 | S 0.8486, MR 0.9684, MIMIC 0.7851, CG micro 0.7922, knowledge dev 0.7718, positional health pass | `outputs/distill_v0/D4/arm_result.json` | same as C-D4-01 | D4 |
| C-D4-05 | Dev-fitted temperatures | choice 1.345, noul 1.709, score 1.645 | `outputs/distill_v0/D4/arm_result.json` | same as C-D4-01 | D4 |
| C-D4-06 | Run provenance | pool text SHA-256 = D2's `11d1a068…`; labels SHA-256 = D2's `06679bcd…`; pplx base revision `6d771ad7…`; wall clock 16 h 41 m | `outputs/distill_v0/D4/arm_result.json`, `run_provenance.json` | same as C-D4-01 | D4 |

D5 (Osler-9B distillation run; artifacts in `outputs/distill_v0/D5/`)

| id | claim | value | artifact | recompute command | task |
|---|---|---|---|---|---|
| C-D5-01 | 9B distillation run completed | 63,304 steps (step budget), 10 evaluations at the Y8 grid, positional health passing in all 10 | `outputs/distill_v0/D5/arm_result.json` | `uv run --frozen python scripts/distill_v0/d4_rederive.py --out outputs/distill_v0/D5 --steps 63304` | D5 |
| C-D5-02 | λ calibration (pre-registered) | λ = 2.2077, frozen at step 200, from 72 measured steps; predicted distillation gradient share 0.40 | `outputs/distill_v0/D5/logs/lambda_calibration.json` | same as C-D5-01 | D5 |
| C-D5-03 | Realised dose | teacher 38.94 % of item slots (band 38–45 %), MIMIC 39.35 % (37–42 %); mean teacher items per batch 1.456 (3.115 per 8 slots); 76.2 % of batches below 3 | `outputs/distill_v0/D5/dose_summary.json`, `logs/dose.jsonl` | same as C-D5-01 | D5 |
| C-D5-04 | Final dev figures at step 63,304 | S 0.8357, MR 0.9560, MIMIC 0.7860, CG micro 0.7651, knowledge dev 0.7360, positional health pass | `outputs/distill_v0/D5/arm_result.json` | same as C-D5-01 | D5 |
| C-D5-05 | Dev-fitted temperatures | choice 1.389, noul 2.107, score 1.795 | `outputs/distill_v0/D5/arm_result.json` | same as C-D5-01 | D5 |
| C-D5-06 | Run provenance and the memory fallback | base `Cloudflare/clef-flash` @ `fde727a2…` loaded in memory; pool/label hashes equal D2's; every batch checkpointed (the first attempt OOM'd at 8,192-token batches without checkpointing); wall clock 27 h 49 m | `outputs/distill_v0/D5/arm_result.json`, `run_provenance.json`, `model.json` | same as C-D5-01 | D5 |
