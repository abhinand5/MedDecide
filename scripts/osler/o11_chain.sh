#!/usr/bin/env bash
# O11 scoring and Gate O1 (ADVISORY O11), resumed after the O9-O11 chain stopped at O11 on the converter failure logged at
# 2026-10-13T03:06:49Z (fixed in osler_scoring.o2_row_to_item; pre-flight in o11_score.py). One GPU job at a time. Each scoring
# run resumes from the rows already written, and each stage checks its own output. Stops at the first failure and logs why.
# Launch detached:
#   setsid nohup bash scripts/osler/o11_chain.sh > /dev/null 2>&1 < /dev/null &
# This chain does not repeat O9 or O10: the O9-O11 chain would retrain the two O10 models. It commits nothing.
set -u
cd "$(dirname "$0")/../.."
source scripts/pod_env.sh
LOG=outputs/osler_v0/O11/logs/chain.log
mkdir -p outputs/osler_v0/O11/logs

log() {
  echo "$1 $(date -u +%FT%TZ)" >> "$LOG"
}
fail() {
  log "STOP: $1"
  exit 1
}
need() {  # need <file> <reason>: the file must exist and be non-empty
  [ -s "$1" ] || fail "$2 ($1 is missing or empty)"
}
complete() {  # complete <name>: both files hold every in-scope row, the count run.json records
  local expected both single
  expected=$(uv run --frozen python -c "import json; print(json.load(open('outputs/osler_v0/O11/$1/run.json'))['items_in_scope'])") \
    || fail "O11 $1: run.json could not be read"
  both=$(wc -l < "outputs/osler_v0/O11/$1/both_orders.jsonl")
  single=$(wc -l < "outputs/osler_v0/O11/$1/single_order.jsonl")
  [ "$both" -eq "$expected" ] && [ "$single" -eq "$expected" ] \
    || fail "O11 $1: $both and $single rows written, $expected in scope"
  log "O11 $1 complete: $expected rows in both files"
}
score() {  # score <name> <checkpoint directory>
  uv run --frozen python scripts/osler/o11_score.py --checkpoint "$2" --name "$1" >> "outputs/osler_v0/O11/logs/$1.log" 2>&1 \
    || fail "O11 scoring of $1 exited non-zero (outputs/osler_v0/O11/logs/$1.log)"
  need "outputs/osler_v0/O11/$1/run.json" "O11 scoring of $1 wrote no run.json"
  complete "$1"
  log "O11 score $1 DONE"
}

log "CHAIN START O11 (resume)"
chosen=$(uv run --frozen python -c "import json; print(json.load(open('outputs/osler_v0/O9/head_choice.json'))['chosen'])") \
  || fail "the O9 head choice could not be read"
log "O9 head choice read: chosen=$chosen"

# 1. O11 scoring of arms L, P, N and both O10 models (GPU, one job at a time; both orders and the single order)
score osler_4b_L outputs/osler_v0/O6/arm_L/checkpoints/best
score osler_4b_P outputs/osler_v0/O6/arm_P/checkpoints/best
score osler_4b_N outputs/osler_v0/O6/arm_N/checkpoints/best
score osler_9b outputs/osler_v0/O10/osler_9b/checkpoints/best
score osler_0p8b outputs/osler_v0/O10/osler_0p8b/checkpoints/best

# 2. O11 Gate O1 for the chosen 4B arm and the 9B (CPU). Writes loops/osler_v0/gate_o1.md; no CLAIMS rows.
uv run --frozen python scripts/osler/o11_gate.py --osler-4b "osler_4b_$chosen" --osler-9b osler_9b \
  >> outputs/osler_v0/O11/logs/gate.log 2>&1 || fail "O11 gate exited non-zero (outputs/osler_v0/O11/logs/gate.log)"
need loops/osler_v0/gate_o1.md "O11 gate wrote no report"
log "O11 GATE DONE (4B arm $chosen)"
log "CHAIN DONE O11"
