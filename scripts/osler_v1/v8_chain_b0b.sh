#!/usr/bin/env bash
# V8 alternative run (ADVISORY V8): B0b is B0 with the LoRA restricted to the attention projections. B0's knowledge check
# failed (CLAIMS V104), so B0b runs once the main queue has finished (one GPU job at a time): it starts when the main queue
# writes "V8 CHAIN DONE" to outputs/osler_v1/V8/logs/chain.log. Its knowledge check and the keep rule run afterwards.
# Launch detached:  setsid nohup bash scripts/osler_v1/v8_chain_b0b.sh > /dev/null 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/../.."
source scripts/pod_env.sh
LOG=outputs/osler_v1/V8/logs/chain.log
echo "CHAIN B0b WAITS for the main V8 queue $(date -u +%FT%TZ)" >> "$LOG"
until grep -q "V8 CHAIN DONE" "$LOG" 2>/dev/null; do sleep 60; done
echo "START B0b $(date -u +%FT%TZ)" >> "$LOG"
uv run --frozen python scripts/osler_v1/v8_block_run.py --block B0b >> outputs/osler_v1/V8/logs/B0b.log 2>&1
rc=$?
echo "END B0b rc=$rc $(date -u +%FT%TZ)" >> "$LOG"
