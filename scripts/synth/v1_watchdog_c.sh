#!/usr/bin/env bash
# synth_v1 watchdog (osler_v5 ADVISORY 3.7): resumes the 63,304-step F2 run when its logs stall for 15 minutes or the process dies.
# Finds the python PID itself (pgrep on the driver command line), so the recorded PID is always the job's.
set -u
CODE=/workspace/tmp/synth_train_code
OUT=outputs/synth_v1/train
ROOT=/workspace/MedDecide/$OUT
LOG=$ROOT/logs/run.log
WLOG=$ROOT/logs/watchdog.log
MAX_RESUMES=5
STALL_S=900
GROUNDED=/workspace/MedDecide/outputs/synth_v1/train/grounded_rows.jsonl

job_pid() { pgrep -f "^/workspace/MedDecide/.venv/bin/python scripts/osler_v5/run_arm.py --config configs/osler_v5/arm_F2.yaml --out $OUT " | head -1; }

start_job() { # $1 = "" or "--resume"
  (
    cd "$CODE" || exit 1
    source scripts/pod_env.sh
    export PYTHONPATH="$CODE/src"
    exec setsid nohup /workspace/MedDecide/.venv/bin/python scripts/osler_v5/run_arm.py \
      --config configs/osler_v5/arm_F2.yaml --out "$OUT" --steps 63304 --evaluations 10 --grounded "$GROUNDED" $1 \
      >> "$LOG" 2>&1 < /dev/null
  ) &
  sleep 20
  job_pid > "$ROOT/logs/run.pid"
  echo "$(date -u +%FT%TZ) started job pid $(cat "$ROOT/logs/run.pid") mode '${1:-fresh}'" >> "$WLOG"
}

resumes=0
while true; do
  sleep 60
  [ -f "$ROOT/arm_result.json" ] && { echo "$(date -u +%FT%TZ) arm_result present; watchdog exits" >> "$WLOG"; exit 0; }
  pid=$(job_pid)
  newest=$(stat -c %Y "$LOG" "$ROOT/logs/train.jsonl" 2>/dev/null | sort -n | tail -1)
  age=$(( $(date +%s) - ${newest:-0} ))
  util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1)
  util=${util:-0}
  # GPU-aware stall rule (coordinator 2026-10-29): an evaluation writes no log lines for more than 15 min but keeps the GPU busy;
  # a stall is a silent log AND an idle GPU (utilisation below 5 %), or a dead process.
  if [ -z "$pid" ] || { [ "$age" -gt "$STALL_S" ] && [ "$util" -lt 5 ]; }; then
    [ -n "$pid" ] && kill -9 "$pid" 2>/dev/null
    if [ "$resumes" -ge "$MAX_RESUMES" ]; then echo "$(date -u +%FT%TZ) resume limit reached; watchdog exits" >> "$WLOG"; exit 1; fi
    resumes=$((resumes+1))
    echo "$(date -u +%FT%TZ) stall or death (log age ${age}s, pid '${pid:-none}'); resume $resumes" >> "$WLOG"
    start_job "--resume"
    sleep 120
  fi
done
