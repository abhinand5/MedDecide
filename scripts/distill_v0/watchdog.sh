#!/usr/bin/env bash
# distill_v0 training watchdog: run the D4/D5 driver, resume from the newest full checkpoint after a crash.
#
#   scripts/distill_v0/watchdog.sh <config> <out-dir> <steps>
#
# Restarts at most MAX_RESTARTS times, never after a completed run (arm_result.json), the loop STOP
# file, or the dose tripwire. The driver itself refuses to run twice and resumes from the newest full
# checkpoint, so a restart repeats no step.
set -u
CONFIG="${1:?config}"
OUT="${2:?out dir}"
STEPS="${3:-63304}"
MAX_RESTARTS="${MAX_RESTARTS:-20}"
LOG="$OUT/logs/watchdog.log"
mkdir -p "$OUT/logs"
cd "$(dirname "$0")/../.." || exit 1
source scripts/pod_env.sh >/dev/null 2>&1 || true
restarts=0
while true; do
  if [ -f "$OUT/arm_result.json" ]; then
    echo "$(date -u +%FT%TZ) arm_result.json present; done" >> "$LOG"
    exit 0
  fi
  if [ -f "outputs/distill_v0/STOP" ]; then
    echo "$(date -u +%FT%TZ) STOP file present; not restarting" >> "$LOG"
    exit 0
  fi
  if [ "$restarts" -gt "$MAX_RESTARTS" ]; then
    echo "$(date -u +%FT%TZ) restart limit reached; stopping" >> "$LOG"
    exit 1
  fi
  resume=""
  if [ -d "$OUT/full" ] && [ -n "$(ls -A "$OUT/full" 2>/dev/null)" ]; then
    resume="--resume"
  fi
  echo "$(date -u +%FT%TZ) launch $restarts resume='$resume'" >> "$LOG"
  uv run --frozen python scripts/distill_v0/d4_train.py --config "$CONFIG" --out "$OUT" --steps "$STEPS" \
    $resume >> "$OUT/logs/run.log" 2>&1
  code=$?
  echo "$(date -u +%FT%TZ) driver exit $code" >> "$LOG"
  if [ "$code" -eq 0 ]; then
    exit 0
  fi
  if tail -n 40 "$OUT/logs/run.log" 2>/dev/null | grep -qE "dose tripwire|QA failed|the teacher attachment is not complete|refuse-twice|not complete"; then
    echo "$(date -u +%FT%TZ) marked failure in the log; not restarting" >> "$LOG"
    exit 1
  fi
  restarts=$((restarts + 1))
  sleep 60
done
