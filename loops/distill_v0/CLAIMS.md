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
