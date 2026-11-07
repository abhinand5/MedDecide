# d3_distill — STATE

Branch `distill/d3_distill`, created from `bench/d3-ultra` at `597d605`. Updated 2026-11-07T15:44Z.
Session stopped by the coordinator (the main loop was handed to another agent) before any GPU job was started.

## Steps

| step | status | note |
|---|---|---|
| 0 — teacher throughput gate (≥ 25 items/s on 1,000 MedMCQA-train items) | PENDING — not started | no batched-inference code written, no GPU job run |
| 1 — port parity, d3-mini and d3-flash (40 items, 1e-4) | PENDING — not started | the coordinator said not to start it |
| 2 — label set and labelling | PENDING — not started | depends on step 0; the coordinator said not to start it |

Running jobs: none. The GPU was idle (0 MiB used) when the session stopped.

## Pre-flight findings (read-only checks; nothing was downloaded, loaded or deleted)

- D61 is recorded in `docs/plans/PROGRAM.md` on `loop/osler_v6` (`c885e9a`), not on this branch. This branch's PLAN.md still
  says "D61 pending".
- The teacher snapshot `vllm-sr/d3-ultra@ccdcfa61` in `HF_HOME` now has all 11 safetensors shards plus `readout.safetensors`
  (sizes checked with `stat -L`; shards 3–5, 7, 8, 10 and 11 dated 2026-11-07 13:33 UTC). The SHA-256 of each shard against the
  Hub was not checked.
- Students `vllm-sr/d3-mini` and `vllm-sr/d3-flash`: no snapshot in `HF_HOME`. Step 1 needs a download first, after a 2 GB
  write check (the bench hit the volume quota).
- Provenance gap: the committed `loops/d3_bench/results.md` (`b110b0f`) says d3-ultra was **not loaded or scored** (BLOCKED,
  disk quota). The PLAN's "3.5 items/s, sequential" figure and its d3-ultra benchmark comparison have no committed artifact or
  CLAIMS row on this branch. Uncommitted outputs of a later run appear to exist in another worktree
  (`.claude/worktrees/agent-a068285ca278d228f/outputs/d3_bench/`, including `run/run_score.json` and `logs/full_run.log`). They
  were not read in this session. They should be committed as aggregates before the PLAN cites them (R1/R2).
- The bench's `transformers==5.17.0` and remote-code deps were installed into a worktree `.venv` only. A `uv run --frozen`
  resync removes them, so the next session must reinstall them before loading d3 models.

## Questions for the operator

1. Should the d3_bench run in `agent-a068285ca278d228f` (which the 3.5 items/s and the benchmark comparison seem to come from)
   be committed as aggregates before step 0? Then the step 0 gate has a recorded baseline.
