#!/usr/bin/env bash
# osler_v3 X10 (operator ruling, 2026-10-20): score the four Stage A runs and Stage B block B1 on the V10 items, one GPU job at a
# time, each with its own dev calibration. Additional rows only; no Gate O3 verdict.
# Usage (detached):  setsid nohup bash scripts/osler_v3/x10_chain.sh > outputs/osler_v3/X10/logs/chain.out 2>&1 < /dev/null &
set -uo pipefail
cd /workspace/MedDecide
source scripts/pod_env.sh
LOG=outputs/osler_v3/X10/logs
mkdir -p "$LOG"

used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
if [ "$used" -gt 1000 ]; then
  echo "GPU busy ($used MiB used); refusing to start" | tee -a "$LOG/chain.log"
  exit 1
fi

run() {  # run <name> <run dir>
  local name=$1 dir=$2
  echo "$(date -u +%FT%TZ) start $name" >> "$LOG/chain.log"
  uv run --frozen python scripts/osler_v3/x10_score.py --checkpoint "$dir/checkpoints/best" --name "$name" \
      --temperatures "$dir/arm_result.json" > "$LOG/$name.log" 2>&1
  local status=$?
  echo "$(date -u +%FT%TZ) end $name exit=$status" >> "$LOG/chain.log"
}

run A0 outputs/osler_v3/stage_a/A0
run A1 outputs/osler_v3/stage_a/A1
run A2 outputs/osler_v3/stage_a/A2
run A3 outputs/osler_v3/stage_a/A3
run B1 outputs/osler_v3/stage_b/B1
echo "$(date -u +%FT%TZ) chain done" >> "$LOG/chain.log"
