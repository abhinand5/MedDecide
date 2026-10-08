# student_v1 — CLAIMS

Every number reported anywhere in this loop (STATE headlines, reports, FINDINGS) has a
row here (AGENTS.md R1). Append only. IDs are `V001`, `V002`, …

`recompute` must be a command that reproduces the value in a fresh process from a
committed script and an artifact under `outputs/student_v1/` or `data/` (paths relative to
repo root). Numbers carried over from earlier loops cite their original ID (`S0xx`, `X0xx`,
`C0xx`) instead of getting a new row.

| id | claim | value | artifact | recompute | task |
|---|---|---|---|---|---|
| V001 | v0.2 benchmark manifest is the frozen input (unchanged from student_v0) | sha256 `363c037e3b3e4e119f26ad5c3ee86123b234f5b67522b57ede74a50430243641` | outputs/student_v1/V0/snapshot.json | `sha256sum data/bench/v0.2/manifest.json` | V0 |
| V002 | student_v0 training mix (frozen comparator) has 212,481 rows; train.jsonl sha256 `2535d46d…79ff6` | 212481 rows | outputs/student_v1/V0/snapshot.json | `wc -l data/train/student_v0/train.jsonl` | V0 |
| V003 | Test suite at the V0 snapshot | 283 passed | outputs/student_v1/V0/snapshot.json | `uv run pytest -q` | V0 |
| V004 | Unsloth 2026.10.2 is the first release with `FastDecisionModel` (2026.9.9 lacks it) | 2026.10.2 | envs/unsloth/uv.lock | `envs/unsloth/.venv/bin/python -I -c "from unsloth import FastDecisionModel"` | V0 |
| V005 | Qwen3.5-0.8B loads via `FastDecisionModel` with `load_in_4bit=False`: 344 bf16 weight tensors, 251 fp32 norm/head tensors, 0 fp32 tensors outside norm/head; PASS | PASS | outputs/student_v1/V0/unsloth_load.json | `envs/unsloth/.venv/bin/python -I scripts/student/v1_unsloth_load.py` | V0 |
| V006 | Unsloth `predict` probabilities are finite and sum to 1 (noul 1.0, choice 0.9999) | 1.0 / 0.9999 | outputs/student_v1/V0/unsloth_load.json | `envs/unsloth/.venv/bin/python -I scripts/student/v1_unsloth_load.py` | V0 |
| V007 | Environment versions: main transformers 5.18.0 vs unsloth env 5.17.0; torch 2.14.1+cu130 in both; unsloth env unsloth 2026.10.2, peft 0.21.2 | see artifact | outputs/student_v1/V0/envs.json | `uv run python scripts/student/v1_snapshot.py` and `envs/unsloth/.venv/bin/python -I scripts/student/v1_snapshot.py --env unsloth` | V0 |
