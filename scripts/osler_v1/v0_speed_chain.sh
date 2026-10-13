#!/usr/bin/env bash
# V0 step 4 chain (osler_v1 ADVISORY V0): configurations a, b, c in order; d only if b ran out of memory; then the
# aggregate that applies the pre-registered rule. One GPU job at a time. Each configuration is its own process.
# Launch detached:  setsid nohup bash scripts/osler_v1/v0_speed_chain.sh > /dev/null 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/../.."
source scripts/pod_env.sh
LOG=outputs/osler_v1/V0/logs/speed_chain.log
mkdir -p outputs/osler_v1/V0/logs
run() {
  echo "START config $1 $(date -u +%FT%TZ)" >> "$LOG"
  uv run --frozen python scripts/osler_v1/v0_speed.py --config "$1" > "outputs/osler_v1/V0/logs/speed_$1.log" 2>&1
  local rc=$?
  echo "END config $1 rc=$rc $(date -u +%FT%TZ)" >> "$LOG"
  return $rc
}
for name in a b c; do
  run "$name" || echo "config $name exited nonzero; see speed_$name.log" >> "$LOG"
done
if uv run --frozen python -c 'import json,sys; sys.exit(0 if json.load(open("outputs/osler_v1/V0/speed/b.json")).get("oom") else 1)' 2>/dev/null; then
  echo "config b ran out of memory: running d" >> "$LOG"
  run d || echo "config d exited nonzero; see speed_d.log" >> "$LOG"
fi
uv run --frozen python scripts/osler_v1/v0_speed.py --aggregate > outputs/osler_v1/V0/logs/speed_aggregate.log 2>&1
echo "CHAIN DONE $(date -u +%FT%TZ)" >> "$LOG"
