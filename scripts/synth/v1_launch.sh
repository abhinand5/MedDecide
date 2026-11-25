#!/usr/bin/env bash
# synth_v1 final run (ADVISORY step 3 = osler_v5 F2 recipe, 63,304 steps, ten evaluations) with the watchdog of osler_v5 ADVISORY 3.7:
# a job whose run.log has not advanced for 15 minutes is treated as dead and resumed from its newest full checkpoint.
# Modes: launch | watchdog. The watchdog runs detached and stops when arm_result.json exists or after MAX_RESUMES resumes.
set -u
CODE=/workspace/tmp/synth_train_code
OUT=outputs/synth_v1/train
LOG=/workspace/MedDecide/$OUT/logs/run.log
WLOG=/workspace/MedDecide/$OUT/logs/watchdog.log
PIDFILE=/workspace/MedDecide/$OUT/logs/run.pid
MAX_RESUMES=5
STALL_S=900
GROUNDED=/workspace/MedDecide/outputs/synth_v1/train/grounded_rows.jsonl
export CODE OUT GROUNDED

run_cmd() { # $1 = "" or "--resume"
  cd "$CODE" || exit 1
  source scripts/pod_env.sh
  PYTHONPATH="$CODE/src" exec /workspace/MedDecide/.venv/bin/python scripts/osler_v5/run_arm.py \
    --config configs/osler_v5/arm_F2.yaml --out "$OUT" --steps 63304 --evaluations 10 --grounded "$GROUNDED" $1
}

start() { # $1 = "" or "--resume"; appends to run.log
  ( cd "$CODE" && setsid nohup bash -c "$(declare -f run_cmd); run_cmd '$1'" >> "$LOG" 2>&1 < /dev/null & echo $! > "$PIDFILE" )
  sleep 2
  cat "$PIDFILE"
}

case "${1:-}" in
  launch) start "" ;;
  watchdog)
    resumes=0
    while true; do
      sleep 60
      [ -f "/workspace/MedDecide/$OUT/arm_result.json" ] && { echo "$(date -u +%FT%TZ) arm_result present; watchdog exits" >> "$WLOG"; exit 0; }
      pid=$(cat "$PIDFILE" 2>/dev/null)
      newest=$(stat -c %Y "$LOG" "/workspace/MedDecide/$OUT/logs/train.jsonl" 2>/dev/null | sort -n | tail -1)
      age=$(( $(date +%s) - ${newest:-0} ))
      if ! kill -0 "$pid" 2>/dev/null || [ "$age" -gt "$STALL_S" ]; then
        kill -9 "$pid" 2>/dev/null
        if [ "$resumes" -ge "$MAX_RESUMES" ]; then echo "$(date -u +%FT%TZ) resume limit reached; watchdog exits" >> "$WLOG"; exit 1; fi
        resumes=$((resumes+1))
        echo "$(date -u +%FT%TZ) stall or death (log age ${age}s, pid $pid); resume $resumes" >> "$WLOG"
        start "--resume" >> "$WLOG"
        sleep 120
      fi
    done ;;
  *) echo "usage: $0 launch|watchdog"; exit 2 ;;
esac
