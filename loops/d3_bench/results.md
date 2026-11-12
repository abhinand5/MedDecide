# d3_bench — vllm-sr/d3-ultra quick local benchmark (aggregates only)

Status: **BLOCKED — the model download hit the volume's disk quota; d3-ultra was not loaded or scored.**
Branch `bench/d3-ultra` (from `dev` at `1a188ba`). Run 2026-11-06, 05:03–05:20 UTC, about 17 minutes in all. No item text
and no predictions are committed. No MIMIC-derived file was read. No data went to a hosted service: nothing was scored.

## Step 1: download

| field | value |
|---|---|
| repo | `vllm-sr/d3-ultra` |
| revision (pinned) | `ccdcfa611649a59e095e0b5e1732a15cedafadd6` |
| licence (LICENSE file) | Apache License, Version 2.0 |
| full size on the Hub | 52,221,058,140 bytes (69 files; 11 safetensors shards) |
| on disk when it stopped | 19,909,031,711 bytes (52 files present; shards 1, 2 and 6 of 11 complete) |
| free-space check | `df` shows 553 TB free on `/workspace`; a 2 GB test write (`dd`, 2,048 MiB) succeeded before the download |
| result | `RuntimeError: Task error: File reconstruction error: IO Error: Disk quota exceeded (os error 122)` |

The `df` figure does not show the per-pod quota on this network volume (distill_v0 D3 hit the same `EDQUOT` with an 18 GB copy).
The 2 GB test passed, but the remaining ~32 GB did not fit. The partial snapshot (19.9 GB) is still in `HF_HOME`. It was not
deleted, because the guardrails allow deletes only under `outputs/` and `/workspace/tmp`. A retry was not made: filling the
quota again could break other jobs that write to this volume.

## Step 2: load and validation: NOT MEASURED

The checkpoint is incomplete, so the README's Quickstart load (`AutoModel.from_pretrained(..., trust_remote_code=True)`) was
not attempted, and the 7-item synthetic validation (`scripts/d3_bench/run_d3.py --validate`) was not run. The README's Quickstart
names `transformers==5.17.0`, torch, torchvision, pillow, opencv-python-headless, safetensors and accelerate, plus optional
flash-linear-attention. These were installed into the worktree `.venv` only (`uv pip install`; `uv.lock` and `pyproject.toml`
are unchanged and their md5 sums were checked): transformers 5.18.0 → 5.17.0, torchvision 0.29.2+cu132, pillow 13.0.0,
opencv-python-headless 5.0.0.93, flash-linear-attention / fla-core 0.6.0, einops 0.8.2. A later `uv run --frozen` resynced the
venv to the lock, so they must be installed again before a run. The remote code was read before any load. Its only network
path is fetching image or video inputs given as http(s) URLs (`d3_runtime.py`). The runner sends text only and sets
`HF_HUB_OFFLINE=1`.

## Step 3–4: aggregates on the seeded subsamples

Subsamples (`scripts/d3_bench/make_subsamples.py`, seed 0; ids and hashes in `subsamples_manifest.json`). They are drawn from
the frozen Y9 files that distill_v0 D1 used, and each file's SHA-256 was checked:

| set | k of N | method | ids sha256 (prefix) |
|---|---|---|---|
| knowledge (MedQA 241 + MedMCQA 759, test) | 1,000 of 5,456 | simple random | `a2e103c9fb9ea15c` |
| external clinical panel (7 datasets) | 500 of 4,353 | stratified by dataset, proportional | `a908c3b49702f461` |
| MR verdict v1 (7 templates) | 500 of 13,103 | simple random | `58eeeb1cd8c88181` |

Accuracy is item micro / template macro. The gate is D12/D21 per (model, template) cell. All comparisons are
**subsample-to-subsample only**: each model is scored on the same item ids, and none of this is a gate verdict.

| set | model | items | micro | template macro | readout gate |
|---|---|---:|---:|---:|---|
| knowledge | **d3-ultra** | 0 | NOT MEASURED | NOT MEASURED | — (BLOCKED: download) |
| knowledge | d1-ultra teacher (hosted, distill_v0 D1) | 1,000 | 0.8230 | 0.8551 | PASS |
| knowledge | zero-shot pplx-decider-flash-v1.2 | 998 (2 not in source) | 0.6774 | 0.7067 | PASS |
| knowledge | zero-shot Clef-Flash | 1,000 | 0.6970 | 0.7310 | PASS |
| panel | **d3-ultra** | 0 | NOT MEASURED | NOT MEASURED | — (BLOCKED: download) |
| panel | d1-ultra teacher | 500 | 0.6460 | 0.7182 | PASS |
| panel | zero-shot pplx-decider-flash-v1.2 | 500 | 0.5440 | 0.6495 | PASS |
| panel | zero-shot Clef-Flash | 500 | 0.5600 | 0.6588 | PASS |
| MR verdict | **d3-ultra** | 0 | NOT MEASURED | NOT MEASURED | — (BLOCKED: download) |
| MR verdict | d1-ultra teacher | 500 | 0.7520 | 0.7406 | PASS |
| MR verdict | zero-shot pplx-decider-flash-v1.2 | 500 | 0.7600 | 0.6442 | READOUT_FAIL (`mtsamples_page_category_v1`, 2 items, constant answer) — not a result |
| MR verdict | zero-shot Clef-Flash | 500 | 0.6920 | 0.6734 | PASS |

The paired comparisons d3-ultra minus each model are **NOT MEASURED** because there are no d3-ultra rows. The pplx MR-verdict
gate failure comes from a 2-item cell in this subsample: only 2 of 111 `mtsamples_page_category_v1` items were drawn. Nothing
about the full set should be read from it.

**What the subsample can and cannot show.** On 500–1,000 items, a gap of a few points between two models on the same items is
detectable, but template-macro figures on small cells (MR: 2 to 19 items for 5 of 7 templates) are noisy and no gate was run.
These are not the frozen-set gate numbers (Gate O7 / O4 use the full sets), and they cannot rank models within ~3 points or
stand in for any gate.

## To unblock

Free about 33 GB of quota on the volume, or raise the quota. That is the operator's call: nothing outside `outputs/` was
deleted. Then:
1. Resume the download: `scripts/d3_bench/download.py`.
2. Reinstall the step-2 deps into `.venv`.
3. Run `.venv/bin/python scripts/d3_bench/run_d3.py --snapshot <snapshot> --validate`, then the same command without
   `--validate`, detached, with logs in `outputs/d3_bench/logs/`.
4. Run `scripts/d3_bench/aggregate.py`.
