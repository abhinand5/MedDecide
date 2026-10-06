# bench_v0_fix0 — CLAIMS

Every number reported anywhere in this loop (STATE headlines, reports, CORRECTIONS,
FINDINGS) has a row here (AGENTS.md R1). Append only. IDs continue the program's
sequence in a loop-local namespace: `X001`, `X002`, … (bench_v0 used `C001`–`C053`;
`CORRECTIONS.md` maps old `C` ids to new `X` ids).

`recompute` must be a command that reproduces the value in a fresh process from a
committed script and an artifact under `outputs/bench_v0_fix0/` or `data/bench/v0.1/`
(paths relative to repo root).

| id | claim | value | artifact | recompute | task |
|---|---|---|---|---|---|
| X001 | Loop-start snapshot: v0 benchmark file hashes and totals, git commit, test result | 11 v0 files hashed (8 tier-1 + 3 fresh); tier-1 21,202 items, fresh 41,502 items; commit `88dea18`; tests 74 passed / 0 failed | `outputs/bench_v0_fix0/F0/start.json`, `outputs/bench_v0_fix0/F0/pytest.xml` | `jq '{commit: .git_commit, tier1: .v0_manifests.tier1.n_items, fresh: .v0_manifests.fresh.n_items, pytest: .pytest_summary}' outputs/bench_v0_fix0/F0/start.json` | F0 |
