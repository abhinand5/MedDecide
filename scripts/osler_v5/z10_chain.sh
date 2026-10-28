#!/usr/bin/env bash
# osler_v5 Z10 chain (ADVISORY §10 Z10): the Clef local-only check, then zero-shot Qwen3.5-4B and -9B, then Clef-Flash.
# One GPU job at a time, run detached under setsid nohup; each step logs its exit code to outputs/osler_v5/Z10/logs/z10_chain.log.
# The Clef scorer refuses to start unless the local-only check passed (z10_clef.py), so a failed check leaves the Clef cells NOT MEASURED
# and the other steps still run. Resumable: each step skips the rows it already wrote.
#
# Usage (after Z9 has exited and the test suite has passed on the committed code):
#     cd /workspace/MedDecide && setsid nohup bash scripts/osler_v5/z10_chain.sh > /dev/null 2>&1 < /dev/null &
set -u
cd /workspace/MedDecide
source scripts/pod_env.sh
LOG=outputs/osler_v5/Z10/logs
CHAIN=$LOG/z10_chain.log
SNAP=/workspace/.hf_home/hub/models--Cloudflare--clef-flash/snapshots/fde727a287004204b7518dcc983fe64379776712
mkdir -p "$LOG"

step() {
    local name=$1; shift
    echo "[$(date -u +%FT%TZ)] $name start" >> "$CHAIN"
    "$@" > "$LOG/$name.log" 2>&1 < /dev/null
    local rc=$?
    echo "[$(date -u +%FT%TZ)] $name exit $rc" >> "$CHAIN"
}

step clef_local_check envs/clef/.venv/bin/python -I scripts/osler_v5/z10_clef_local_check.py --path "$SNAP"
step zeroshot_qwen35_4b uv run --frozen python scripts/osler_v5/z10_zeroshot.py \
    --model Qwen/Qwen3.5-4B --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a --slug qwen35_4b
step zeroshot_qwen35_9b uv run --frozen python scripts/osler_v5/z10_zeroshot.py \
    --model Qwen/Qwen3.5-9B --revision c202236235762e1c871ad0ccb60c8ee5ba337b9a --slug qwen35_9b
step clef_flash envs/clef/.venv/bin/python -I scripts/osler_v5/z10_clef.py --path "$SNAP"
echo "[$(date -u +%FT%TZ)] chain complete" >> "$CHAIN"
