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
| X002 | MedMCQA `cop` is 0-based — proven two independent ways | raw validation `cop=0` occurs 1,348 times; the record's own `exp` names option[cop] in 1,462/2,194 = 0.666 of usable rows vs 253/2,194 = 0.115 for option[cop+1] | `outputs/bench_v0_fix0/F1/gold_verification.json` (`cop_semantics_proof`) | `uv run python scripts/bench/verify_gold.py --prove-cop --out /tmp/gv.json` then `jq '.cop_semantics_proof' /tmp/gv.json` | F1 |
| X003 | v0.1 MedMCQA test set and gold distribution after the fix | 4,183 test items (was 2,835 in v0); gold A 1,348 / B 1,085 / C 925 / D 825 — identical to the raw `cop` counts | `data/bench/v0.1/tier1/manifest.json`, `outputs/bench_v0_fix0/F1/gold_verification.json` | `jq -r '.files.medmcqa | {n_rows, count_by_split}' data/bench/v0.1/tier1/manifest.json; jq '.sources.medmcqa.gold_class_counts' outputs/bench_v0_fix0/F1/gold_verification.json` | F1 |
| X004 | Independent gold verification: every tier-1 source, 0 mismatches | 0 mismatched of 15,915 checked across 8 sources (2,679 pairs unverifiable because the (query, passage) pair is unjudged) | `outputs/bench_v0_fix0/F1/gold_verification.json` | `uv run python scripts/bench/verify_gold.py --tier1 data/bench/v0.1/tier1 --out outputs/bench_v0_fix0/F1/gold_verification.json && jq '.summary' outputs/bench_v0_fix0/F1/gold_verification.json` | F1 |
| X005 | nfcorpus score sampler bug found by verification and fixed | 3 items claimed `qrel_grade: 2` while their own (query, passage) pair is grade 1; pools are now strictly per query, so nfcorpus is 3,076 items (was 3,616) and 0/752 checked items mismatch | `src/meddecide/bench/tier1/relevance.py`, `outputs/bench_v0_fix0/F1/gold_verification.json` | `jq '.sources.nfcorpus.n_mismatched, .sources.nfcorpus.n_checked' outputs/bench_v0_fix0/F1/gold_verification.json` | F1 |
| X006 | tier-1 v0.1 build integrity | 22,594 items (test 13,907 / dev 8,687), 0 split leaks, 0 duplicate item ids, all hashes match, 8/8 sources | `data/bench/v0.1/tier1/{manifest,acceptance,audit}.json` | `uv run python scripts/bench/audit_tier1.py --tier1 data/bench/v0.1/tier1 --out data/bench/v0.1/tier1/audit.json && jq '.checks, .totals' data/bench/v0.1/tier1/audit.json` | F1 |
