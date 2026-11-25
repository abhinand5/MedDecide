#!/usr/bin/env bash
# O6-O8 sequence, one GPU job at a time (ADVISORY O6-O8). Waits until arm L has exited, starts arm P, then arm N.
# Stops at the first arm that exits without arm_result.json, and records the reason in the chain log.
# Launch detached:  setsid nohup bash scripts/osler/o6_arm_chain.sh > /dev/null 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/../.."
source scripts/pod_env.sh
LOG=outputs/osler_v0/O6/logs/arm_chain.log
mkdir -p outputs/osler_v0/O6/logs
echo "CHAIN WAIT for arm L exit $(date -u +%FT%TZ)" >> "$LOG"
while ! grep -q "EXIT CODE" outputs/osler_v0/O6/logs/arm_L.log 2>/dev/null; do sleep 60; done
if [ ! -f outputs/osler_v0/O6/arm_L/arm_result.json ]; then
  echo "STOP: arm L exited without arm_result.json $(date -u +%FT%TZ)" >> "$LOG"; exit 1
fi
echo "arm L exited with a result $(date -u +%FT%TZ)" >> "$LOG"
for arm in P N; do
  echo "START arm $arm $(date -u +%FT%TZ)" >> "$LOG"
  PYTHONFAULTHANDLER=1 uv run --frozen python scripts/osler/o6_train_arm.py --arm "$arm" > "outputs/osler_v0/O6/logs/arm_$arm.log" 2>&1
  rc=$?
  echo "END arm $arm rc=$rc $(date -u +%FT%TZ)" >> "$LOG"
  if [ "$rc" -ne 0 ] || [ ! -f "outputs/osler_v0/O6/arm_$arm/arm_result.json" ]; then
    echo "STOP after arm $arm: rc=$rc or no arm_result.json $(date -u +%FT%TZ)" >> "$LOG"; exit 1
  fi
done
echo "ARMS P AND N DONE $(date -u +%FT%TZ)" >> "$LOG"
