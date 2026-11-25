#!/usr/bin/env bash
# V8 B2 and B3 (ADVISORY V8; operator instruction on V4). The queue waits for the V4 closure record
# (outputs/osler_v1/V4/FINAL.json, written when V4 is complete or finished at the weekly budget). It then builds B2 and B3 with
# the teacher prose as built (--teacher-final), replays them at the replay's step count (the same for every block), and runs
# B2 and B3 one after the other (one GPU job at a time). The keep decisions are run by hand afterwards.
# Launch detached:  setsid nohup bash scripts/osler_v1/v8_chain_b2b3.sh > /dev/null 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/../.."
source scripts/pod_env.sh
LOG=outputs/osler_v1/V8/logs/chain.log
mkdir -p outputs/osler_v1/V8/logs
echo "CHAIN B2B3 WAITS for the V4 closure record $(date -u +%FT%TZ)" >> "$LOG"
until [ -f outputs/osler_v1/V4/FINAL.json ]; do sleep 60; done
echo "START B2B3 build $(date -u +%FT%TZ)" >> "$LOG"
uv run --frozen python scripts/osler_v1/v7_blocks.py --blocks B2 B3 --teacher-final \
  >> outputs/osler_v1/V7/logs/blocks_build_b2b3.log 2>&1 \
  || { echo "CHAIN STOPPED: V7 B2 and B3 build $(date -u +%FT%TZ)" >> "$LOG"; exit 1; }
uv run --frozen python scripts/osler_v1/v8_replay_steps.py --blocks B0 B1 B2 B3 \
  >> outputs/osler_v1/V8/logs/replay_steps_b2b3.log 2>&1 \
  || { echo "CHAIN STOPPED: replay of B2 and B3 $(date -u +%FT%TZ)" >> "$LOG"; exit 1; }
for block in B2 B3; do
  echo "START $block $(date -u +%FT%TZ)" >> "$LOG"
  uv run --frozen python scripts/osler_v1/v8_block_run.py --block "$block" >> "outputs/osler_v1/V8/logs/$block.log" 2>&1
  rc=$?
  echo "END $block rc=$rc $(date -u +%FT%TZ)" >> "$LOG"
  [ "$rc" -eq 0 ] || { echo "CHAIN STOPPED at $block $(date -u +%FT%TZ)" >> "$LOG"; exit 1; }
done
echo "V8 B2B3 DONE $(date -u +%FT%TZ)" >> "$LOG"
