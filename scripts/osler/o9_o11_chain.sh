#!/usr/bin/env bash
# O9-O11 after arms P and N (ADVISORY O9-O11), one GPU job at a time. Stops at the first failure and logs the reason.
# Launch detached once the O6-O8 chain is running (it waits for the O6-O8 chain's "ARMS P AND N DONE"):
#   setsid nohup bash scripts/osler/o9_o11_chain.sh > /dev/null 2>&1 < /dev/null &
# Stages: O9 head choice (CPU) -> O10 configs from the choice (CPU) -> O10 Osler-9B (GPU) -> O10 Osler-0.8B (GPU) ->
# O11 scoring of arms L, P, N and both O10 models (GPU, one at a time) -> O11 Gate O1 (CPU).
# The chain commits nothing. The agent commits the generated configs and reports with their CLAIMS rows.
# Dry run (prints the sequence, runs nothing):  DRY_RUN=1 DRY_CHOSEN=P bash scripts/osler/o9_o11_chain.sh
set -u
cd "$(dirname "$0")/../.."
source scripts/pod_env.sh
LOG=outputs/osler_v0/O9/logs/chain.log
mkdir -p outputs/osler_v0/O9/logs outputs/osler_v0/O10/logs outputs/osler_v0/O11/logs

log() {
  if [ -n "${DRY_RUN:-}" ]; then echo "DRY log: $1"; return 0; fi
  echo "$1 $(date -u +%FT%TZ)" >> "$LOG"
}
fail() {
  log "STOP: $1"
  exit 1
}
run() {  # run <log file> <command...>: the command's output goes to the log file; its exit code is returned
  local logfile=$1
  shift
  if [ -n "${DRY_RUN:-}" ]; then echo "DRY run: $* > $logfile"; return 0; fi
  "$@" > "$logfile" 2>&1
}
need() {  # need <file> <reason>: the file must exist and be non-empty
  if [ -n "${DRY_RUN:-}" ]; then echo "DRY check: $1"; return 0; fi
  [ -s "$1" ] || fail "$2 ($1 is missing or empty)"
}

log "CHAIN START O9-O11"

# 0. wait for the O6-O8 chain to finish both matched arms (the O6-O8 chain stops itself with a STOP line on failure)
if [ -z "${DRY_RUN:-}" ]; then
  while ! grep -q "ARMS P AND N DONE" outputs/osler_v0/O6/logs/arm_chain.log 2>/dev/null; do
    if grep -q "^STOP" outputs/osler_v0/O6/logs/arm_chain.log 2>/dev/null; then
      fail "the O6-O8 chain stopped before arms P and N finished"
    fi
    sleep 60
  done
fi
log "ARMS P AND N DONE seen"

# 1. O9: the pre-registered head choice on dev (no GPU)
run outputs/osler_v0/O9/logs/o9.log uv run --frozen python scripts/osler/o9_head_choice.py || fail "O9 exited non-zero (outputs/osler_v0/O9/logs/o9.log)"
need outputs/osler_v0/O9/head_choice.json "O9 wrote no head choice"
need loops/osler_v0/head_choice.md "O9 wrote no report"
if [ -n "${DRY_RUN:-}" ]; then
  chosen=${DRY_CHOSEN:-L}
else
  chosen=$(uv run --frozen python -c "import json; print(json.load(open('outputs/osler_v0/O9/head_choice.json'))['chosen'])")
fi
log "O9 DONE chosen=$chosen"

# 2. the O10 configs from arm L's recipe and the choice (no GPU; the driver checks them again before the model loads)
run outputs/osler_v0/O10/logs/configs.log uv run --frozen python scripts/osler/o10_write_configs.py || fail "O10 configs were not written"
need configs/osler_v0/osler_9b.yaml "O10 config (9B)"
need configs/osler_v0/osler_0p8b.yaml "O10 config (0.8B)"
log "O10 configs written"

# 3. O10: Osler-9B, then Osler-0.8B (the reference), the same recipe and budget (GPU, one job at a time)
run outputs/osler_v0/O10/logs/osler_9b.log uv run --frozen python scripts/osler/o10_train_model.py --config configs/osler_v0/osler_9b.yaml \
  || fail "O10 Osler-9B exited non-zero (outputs/osler_v0/O10/logs/osler_9b.log)"
need outputs/osler_v0/O10/osler_9b/arm_result.json "O10 Osler-9B wrote no arm_result.json"
log "O10 osler_9b DONE"
run outputs/osler_v0/O10/logs/osler_0p8b.log uv run --frozen python scripts/osler/o10_train_model.py --config configs/osler_v0/osler_0p8b.yaml \
  || fail "O10 Osler-0.8B exited non-zero (outputs/osler_v0/O10/logs/osler_0p8b.log)"
need outputs/osler_v0/O10/osler_0p8b/arm_result.json "O10 Osler-0.8B wrote no arm_result.json"
log "O10 osler_0p8b DONE"

# 4. O11: score every arm and both O10 models on the O2 common set, both orders and single order (GPU)
score() {  # score <name> <checkpoint directory>
  run "outputs/osler_v0/O11/logs/$1.log" uv run --frozen python scripts/osler/o11_score.py --checkpoint "$2" --name "$1" \
    || fail "O11 scoring of $1 exited non-zero (outputs/osler_v0/O11/logs/$1.log)"
  need "outputs/osler_v0/O11/$1/run.json" "O11 scoring of $1 wrote no run.json"
  log "O11 score $1 DONE"
}
score osler_4b_L outputs/osler_v0/O6/arm_L/checkpoints/best
score osler_4b_P outputs/osler_v0/O6/arm_P/checkpoints/best
score osler_4b_N outputs/osler_v0/O6/arm_N/checkpoints/best
score osler_9b outputs/osler_v0/O10/osler_9b/checkpoints/best
score osler_0p8b outputs/osler_v0/O10/osler_0p8b/checkpoints/best

# 5. O11: Gate O1 for the chosen 4B arm and the 9B (CPU). Writes loops/osler_v0/gate_o1.md; no CLAIMS rows yet.
run outputs/osler_v0/O11/logs/gate.log uv run --frozen python scripts/osler/o11_gate.py --osler-4b "osler_4b_$chosen" --osler-9b osler_9b \
  || fail "O11 gate exited non-zero (outputs/osler_v0/O11/logs/gate.log)"
need loops/osler_v0/gate_o1.md "O11 gate wrote no report"
log "O11 GATE DONE (4B arm $chosen)"
log "CHAIN DONE O9-O11"
