# student_v0 — CLAIMS

Every number reported anywhere in this loop (STATE headlines, reports, FINDINGS) has a
row here (AGENTS.md R1). Append only. IDs are `S001`, `S002`, …

`recompute` must be a command that reproduces the value in a fresh process from a
committed script and an artifact under `outputs/student_v0/` or `data/` (paths relative to
repo root). Numbers carried over from earlier loops cite their original ID (`X0xx`, `C0xx`)
instead of getting a new row.

| id | claim | value | artifact | recompute | task |
|---|---|---|---|---|---|
| S001 | Qwen3.5-0.8B prefill throughput, 8,192-token prompt, fused kernels absent (bf16, batch 1, best of 5 after warmup) | 40,809 tok/s | `outputs/student_v0/S0/kernels_before.json` | `uv run python scripts/student/s0_kernels.py --kernels off --lengths 8192 16384 --reps 5 --warmup-at-length --out outputs/student_v0/S0/kernels_before.json` | S0 |
| S002 | same, 16,384-token prompt, kernels absent | 36,399 tok/s | `outputs/student_v0/S0/kernels_before.json` | as S001 | S0 |
| S003 | same, 8,192-token prompt, `causal_conv1d` 1.7.0 + `flash-linear-attention` 0.5.2 installed and bound | 124,526 tok/s (**3.05×**) | `outputs/student_v0/S0/kernels_after.json` | `uv run python scripts/student/s0_kernels.py --kernels auto --lengths 8192 16384 --reps 5 --warmup-at-length --out outputs/student_v0/S0/kernels_after.json` | S0 |
| S004 | same, 16,384-token prompt, kernels bound | 106,697 tok/s (**2.93×**) | `outputs/student_v0/S0/kernels_after.json` | as S003 | S0 |
| S005 | peak allocated GPU memory, 8k / 16k prompt, kernels absent → bound | 2.329 → 1.968 GiB / 3.225 → 2.505 GiB | `outputs/student_v0/S0/kernels.json` | `uv run python scripts/student/s0_snapshot.py --out outputs/student_v0/S0/snapshot.json` (merges both runs) | S0 |
| S006 | transformers fallback warnings emitted per run (kernels absent → bound) | 2 → **0**; bound implementations `causal_conv1d.causal_conv1d_interface.causal_conv1d_fn`, `fla.ops.gated_delta_rule.chunk.chunk_gated_delta_rule` | `outputs/student_v0/S0/kernels_{before,after}.json` | read `fallback_warnings` and `resolved_implementations` from both files | S0 |
| S007 | first-call Triton JIT cost at a new sequence length (single-shot minus warmed measurement, 8,192 tokens) | 7.46 s (1,069 vs 124,526 tok/s) | `outputs/student_v0/S0/kernels_after.json` + the superseded single-shot run in `outputs/student_v0/S0/logs/kernels_after.log` | compare `seconds_first` under `--warmup-at-length` with the earlier single-shot artifact | S0 |
| S008 | unit tests at the S0 commit | 84 passed, 0 failed, 0 skipped | `outputs/student_v0/S0/pytest.xml` | `uv run pytest --junitxml=outputs/student_v0/S0/pytest.xml` | S0 |
| S009 | v0.1 manifest hashes at loop start (tier-1 / fresh) | `e001bc05…26acc9` (22,594 items) / `bfba2f38…5bff739` (23,582 items) | `outputs/student_v0/S0/snapshot.json` | `uv run python scripts/student/s0_snapshot.py --out outputs/student_v0/S0/snapshot.json` | S0 |
| S010 | candidate-dataset catalog rows and verdicts (documentation only, nothing trained) | **62 rows**: 14 `train-candidate`, 20 `eval-candidate`, 28 `reject`; provenance 37 human / 15 structured / 6 llm / 4 unknown | `docs/benchmark/dataset_catalog.md` | `uv run python scripts/bench/catalog_datasets.py --out /workspace/tmp/s13/catalog.json --markdown-out /workspace/tmp/s13/regen_table.md` (or `--from-json` to regenerate offline) | S13 |
| S011 | ids fetched / Hub search queries / fetch failures for the catalog | 74 / 49 / 0 | `/workspace/tmp/s13/catalog.json` (scratch, gitignored) | as S010, then read `meta` from the JSON | S13 |
| S012 | catalog rows with no licence in the card's licence field (16 remain `UNKNOWN`, 2 are overrides declared elsewhere in the repo) | 18 of 62 | `docs/benchmark/dataset_catalog.md` | as S010; the table's licence column shows the resolved value | S13 |
| S013 | `bigbio/mednli` licence, which disqualifies it as a training source | `PHYSIONET_LICENSE_1p5` (credentialed data) | `docs/benchmark/dataset_catalog.md` | `uv run python -c "from huggingface_hub import HfApi; print(HfApi().dataset_info('bigbio/mednli').card_data)"` | S13 |

