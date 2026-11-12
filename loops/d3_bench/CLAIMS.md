# d3_bench — claims

All artifacts are gitignored under `outputs/d3_bench/` (worktree) unless stated. Recompute: `uv run --frozen python
scripts/d3_bench/make_subsamples.py` then `uv run --frozen python scripts/d3_bench/aggregate.py` (writes
`outputs/d3_bench/aggregate.json`; `sets.<set>.models.<model>.{items,correct,accuracy_micro,accuracy_template_macro,gate}`).
Every comparison row is subsample-to-subsample only (same seeded item ids), not a gate.

| id | claim | value | artifact | recompute command | task |
|---|---|---|---|---|---|
| C-D3-01 | d3-ultra revision downloaded | `ccdcfa611649a59e095e0b5e1732a15cedafadd6` | `outputs/d3_bench/logs/download.log`; HF_HOME snapshot dir | `uv run --frozen python -c "from huggingface_hub import HfApi; print(HfApi().model_info('vllm-sr/d3-ultra').sha)"` | step 1 |
| C-D3-02 | d3-ultra full repo size on the Hub | 52,221,058,140 bytes, 69 files | Hub metadata | `HfApi().model_info('vllm-sr/d3-ultra', files_metadata=True)` sum of sibling sizes | step 1 |
| C-D3-03 | on disk when the download stopped | 19,909,031,711 bytes, 52 files | `$HF_HOME/hub/models--vllm-sr--d3-ultra/snapshots/ccdcfa61…` | `du -sbL <snapshot>` | step 1 |
| C-D3-04 | download BLOCKED | `RuntimeError: Task error: File reconstruction error: IO Error: Disk quota exceeded (os error 122)` | `outputs/d3_bench/logs/download.log` | `grep "Disk quota" outputs/d3_bench/logs/download.log` | step 1 |
| C-D3-05 | load and 7-item synthetic validation | NOT MEASURED (checkpoint incomplete) | — | — | step 2 |
| C-D3-06 | d3-ultra on knowledge / panel / MR verdict subsamples | NOT MEASURED (BLOCKED, C-D3-04) | `outputs/d3_bench/aggregate.json` | aggregate.py | step 3 |
| C-D3-07 | subsample ids (seed 0): knowledge 1,000/5,456, panel 500/4,353 stratified by 7 datasets, MR verdict 500/13,103 | ids sha256 `a2e103c9…`, `a908c3b4…`, `58eeeb1c…` | `loops/d3_bench/subsamples_manifest.json` (committed) | make_subsamples.py | step 3 |
| C-D3-08 | d1-ultra teacher, knowledge subsample | micro 0.8230 (823/1,000), macro 0.8551, gate PASS | aggregate.json (source `outputs/distill_v0/D1/labels.jsonl`) | aggregate.py | step 4 |
| C-D3-09 | zero-shot pplx-flash, knowledge subsample | micro 0.6774 (676/998; 2 not in source), macro 0.7067, gate PASS | aggregate.json (source `outputs/osler_v4/Y9/baselines/pplx_knowledge`) | aggregate.py | step 4 |
| C-D3-10 | zero-shot Clef-Flash, knowledge subsample | micro 0.6970 (697/1,000), macro 0.7310, gate PASS | aggregate.json (source `outputs/osler_v5/Z10/clef_flash`) | aggregate.py | step 4 |
| C-D3-11 | d1-ultra teacher, panel subsample | micro 0.6460 (323/500), macro 0.7182, gate PASS | aggregate.json | aggregate.py | step 4 |
| C-D3-12 | zero-shot pplx-flash, panel subsample | micro 0.5440 (272/500), macro 0.6495, gate PASS | aggregate.json | aggregate.py | step 4 |
| C-D3-13 | zero-shot Clef-Flash, panel subsample | micro 0.5600 (280/500), macro 0.6588, gate PASS | aggregate.json | aggregate.py | step 4 |
| C-D3-14 | d1-ultra teacher, MR verdict subsample | micro 0.7520 (376/500), macro 0.7406, gate PASS | aggregate.json | aggregate.py | step 4 |
| C-D3-15 | zero-shot pplx-flash, MR verdict subsample | READOUT_FAIL on `mtsamples_page_category_v1` (2-item cell); micro 0.7600 / macro 0.6442 not a result (D12) | aggregate.json | aggregate.py | step 4 |
| C-D3-16 | zero-shot Clef-Flash, MR verdict subsample | micro 0.6920 (346/500), macro 0.6734, gate PASS | aggregate.json | aggregate.py | step 4 |
| C-D3-17 | d3-ultra minus each model (paired) | NOT MEASURED (no d3-ultra rows) | aggregate.json `comparisons` | aggregate.py | step 4 |
