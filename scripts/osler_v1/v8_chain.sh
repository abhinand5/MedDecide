#!/usr/bin/env bash
# V8 GPU queue (osler_v1 ADVISORY V8). One GPU job at a time, in this order:
#   1. wait for the V6 baselines chain to finish;
#   2. zero-shot Qwen3.5-4B on the fast dev knowledge items (the comparison of B0's knowledge check);
#   3. a smoke run of B0 (8 steps, 200 dev items, every step logged); the queue stops if its pipeline check fails;
#   4. B0 and B1 at the replay's step count. B2 and B3 read the teacher prose and the real-text files: they are built with
#      v7_blocks.py --teacher-final, replayed and run by hand once V4 has finished or been stopped by the budget (STATE §5-§7).
# A block directory from an earlier attempt must be moved aside before a job is started again.
# Launch detached:  setsid nohup bash scripts/osler_v1/v8_chain.sh > /dev/null 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/../.."
source scripts/pod_env.sh
LOG=outputs/osler_v1/V8/logs/chain.log
mkdir -p outputs/osler_v1/V8/logs
echo "CHAIN WAITS for the V6 chain $(date -u +%FT%TZ)" >> "$LOG"
until grep -q "V6 BASELINES CHAIN DONE" outputs/osler_v1/V6/logs/baselines_chain.log 2>/dev/null; do sleep 60; done

job() {  # job <name> <command...>: logs START and END lines and returns the command's exit code
  local name=$1
  shift
  echo "START $name $(date -u +%FT%TZ)" >> "$LOG"
  "$@" >> "outputs/osler_v1/V8/logs/$name.log" 2>&1
  local rc=$?
  echo "END $name rc=$rc $(date -u +%FT%TZ)" >> "$LOG"
  return $rc
}

job knowledge_zs uv run --frozen python scripts/osler/o2_run_zeroshot.py --model Qwen/Qwen3.5-4B --scope mix_v2 \
  --items data/bench/osler_mr_v0/fastdev_knowledge.jsonl --out-root outputs/osler_v1/V8/knowledge_zs \
  || { echo "CHAIN STOPPED at knowledge_zs $(date -u +%FT%TZ)" >> "$LOG"; exit 1; }

job smoke_B0 uv run --frozen python scripts/osler_v1/v8_block_run.py --block B0 --smoke 8 \
  || { echo "CHAIN STOPPED at smoke_B0 $(date -u +%FT%TZ)" >> "$LOG"; exit 1; }
uv run --frozen python -I - >> "$LOG" 2>&1 <<'PY' || { echo "CHAIN STOPPED: smoke check failed $(date -u +%FT%TZ)" >> "$LOG"; exit 1; }
import json, sys
r = json.load(open("outputs/osler_v1/V8/smoke/B0/block_result.json"))
v = r["validation"]
ok = (v["steps_as_planned"] and v["evaluation_steps_as_planned"] and v["eval_subset_is_full_dev"]
      and v["logged_steps"]["logged_steps_checked"] > 0 and v["logged_steps"]["mismatches"] == 0)
print("SMOKE OK" if ok else f"SMOKE FAILED {v}")
sys.exit(0 if ok else 1)
PY

for block in B0 B1; do
  job "$block" uv run --frozen python scripts/osler_v1/v8_block_run.py --block "$block" \
    || { echo "CHAIN STOPPED at $block $(date -u +%FT%TZ)" >> "$LOG"; exit 1; }
done
echo "V8 CHAIN DONE $(date -u +%FT%TZ)" >> "$LOG"
