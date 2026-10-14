#!/usr/bin/env bash
# V4 build chain (osler_v1 ADVISORY V4): indicator prose, then Synthea narrative pages, one teacher process at a time (the
# two-requests-per-second limit is per process). Each step resumes from its progress file. A step that stops on the spend
# guard waits 15 minutes and resumes; a step that stops for any other reason (for example a refused endpoint) ends the chain.
# The ICD prose is not built: its pilot pass rate stayed below 60 % after the one permitted prompt fix (STATE, blocked items).
# Launch detached:  setsid nohup bash scripts/osler_v1/v4_build_chain.sh > /dev/null 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/../.."
source scripts/pod_env.sh
LOG=outputs/osler_v1/V4/logs/build_chain.log
mkdir -p outputs/osler_v1/V4/logs
run_step() {
  local gen=$1 start=$2 count=$3
  while true; do
    echo "BUILD $gen start=$start count=$count $(date -u +%FT%TZ)" >> "$LOG"
    uv run --frozen python scripts/osler_v1/v4_teacher_build.py --mode build --gen "$gen" --start "$start" --count "$count" \
      >> "outputs/osler_v1/V4/logs/build_$gen.log" 2>&1
    local rc=$?
    local stop
    stop=$(uv run --frozen python -I -c "import json; print(json.load(open('outputs/osler_v1/V4/build_${gen}_stats.json')).get('stop_reason',''))" 2>/dev/null)
    echo "END $gen rc=$rc stop=$stop $(date -u +%FT%TZ)" >> "$LOG"
    if [[ "$stop" == spend\ guard* ]]; then
      sleep 900
      continue
    fi
    return "$rc"
  done
}
run_step indicator 0 21000 || { echo "CHAIN STOPPED at indicator $(date -u +%FT%TZ)" >> "$LOG"; exit 1; }
run_step narrative 0 28600 || { echo "CHAIN STOPPED at narrative $(date -u +%FT%TZ)" >> "$LOG"; exit 1; }
echo "V4 BUILD CHAIN DONE $(date -u +%FT%TZ)" >> "$LOG"
