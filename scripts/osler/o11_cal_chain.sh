#!/usr/bin/env bash
# O11 calibrated scoring and Gate O1 (ADVISORY O11), after deviation 59: every Osler probability is calibrated with its own arm's
# dev per-question-type temperatures (arm_result.json temperature_fits), pass by pass before the two option orders are averaged,
# as the model's predict() does. The saved checkpoints do not carry the fits, so the scorer applies them from the arm result.
# One GPU job at a time. Each stage resumes and checks its own output. The chain stops at the first failure and logs why.
# Order: osler_4b_L and osler_9b, then Gate O1 (the verdict's models), then their held-out items, then osler_4b_P, osler_4b_N,
# osler_0p8b and their held-out items (additional rows, never in a verdict). Launch detached:
#   setsid nohup bash scripts/osler/o11_cal_chain.sh > /dev/null 2>&1 < /dev/null &
# The chain commits nothing. The uncalibrated outputs in outputs/osler_v0/O11/<name>/ are not used by this chain.
set -u
cd "$(dirname "$0")/../.."
source scripts/pod_env.sh
ROOT=outputs/osler_v0/O11/calibrated
LOG=outputs/osler_v0/O11/logs/cal_chain.log
mkdir -p "$ROOT/heldout" outputs/osler_v0/O11/logs

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
complete() {  # complete <dir> <label>: both files hold every row that run.json counts as in scope
  local expected both single
  expected=$(uv run --frozen python -c "import json; print(json.load(open('$1/run.json'))['items_in_scope'])") \
    || fail "$2: run.json could not be read"
  both=$(wc -l < "$1/both_orders.jsonl")
  single=$(wc -l < "$1/single_order.jsonl")
  [ "$both" -eq "$expected" ] && [ "$single" -eq "$expected" ] \
    || fail "$2: $both and $single rows written, $expected in scope"
  log "$2 complete: $expected rows in both files"
}
score() {  # score <name> <checkpoint directory> <arm result>: calibrated O11 scoring of the O2 common set, then the self-audit
  uv run --frozen python scripts/osler/o11_score.py --checkpoint "$2" --name "$1" --temperatures "$3" --out-root "$ROOT" \
    >> "outputs/osler_v0/O11/logs/cal_$1.log" 2>&1 \
    || fail "calibrated scoring of $1 exited non-zero (outputs/osler_v0/O11/logs/cal_$1.log)"
  need "$ROOT/$1/run.json" "calibrated scoring of $1 wrote no run.json"
  complete "$ROOT/$1" "O11 calibrated $1"
  uv run --frozen python scripts/osler/o11_audit.py --name "$1" --out-root "$ROOT" >> "outputs/osler_v0/O11/logs/cal_$1.log" 2>&1 \
    || fail "the self-audit of $1 failed (outputs/osler_v0/O11/logs/cal_$1.log)"
  log "O11 calibrated score $1 DONE"
}
score_heldout() {  # score_heldout <name> <checkpoint directory> <arm result>: calibrated held-out scoring
  uv run --frozen python scripts/osler/o11_heldout_score.py --checkpoint "$2" --name "$1" --temperatures "$3" \
    >> "outputs/osler_v0/O11/logs/cal_heldout_$1.log" 2>&1 \
    || fail "held-out scoring of $1 exited non-zero (outputs/osler_v0/O11/logs/cal_heldout_$1.log)"
  need "$ROOT/heldout/$1/run.json" "held-out scoring of $1 wrote no run.json"
  complete "$ROOT/heldout/$1" "held-out $1"
  log "held-out score $1 DONE"
}

L_CKPT=outputs/osler_v0/O6/arm_L/checkpoints/best
L_ARM=outputs/osler_v0/O6/arm_L/arm_result.json
P_CKPT=outputs/osler_v0/O6/arm_P/checkpoints/best
P_ARM=outputs/osler_v0/O6/arm_P/arm_result.json
N_CKPT=outputs/osler_v0/O6/arm_N/checkpoints/best
N_ARM=outputs/osler_v0/O6/arm_N/arm_result.json
NINE_CKPT=outputs/osler_v0/O10/osler_9b/checkpoints/best
NINE_ARM=outputs/osler_v0/O10/osler_9b/arm_result.json
ZERO8_CKPT=outputs/osler_v0/O10/osler_0p8b/checkpoints/best
ZERO8_ARM=outputs/osler_v0/O10/osler_0p8b/arm_result.json

log "CHAIN START O11 CALIBRATED (deviation 59)"

# 1. the verdict's models: osler_4b_L (the chosen arm) and osler_9b, calibrated
score osler_4b_L "$L_CKPT" "$L_ARM"
score osler_9b "$NINE_CKPT" "$NINE_ARM"

# 2. Gate O1 (CPU) on the calibrated outputs; writes loops/osler_v0/gate_o1.md and calibrated/gate.json
uv run --frozen python scripts/osler/o11_gate.py --osler-4b osler_4b_L --osler-9b osler_9b --o11-root "$ROOT" \
  >> outputs/osler_v0/O11/logs/cal_gate.log 2>&1 || fail "the O11 gate exited non-zero (outputs/osler_v0/O11/logs/cal_gate.log)"
need loops/osler_v0/gate_o1.md "the O11 gate wrote no report"
log "O11 GATE DONE (calibrated; 4B arm L)"

# 3. the held-out generator items for the two verdict models (reported separately; never in a verdict)
score_heldout osler_4b_L "$L_CKPT" "$L_ARM"
score_heldout osler_9b "$NINE_CKPT" "$NINE_ARM"

# 4. additional rows (never in a verdict): the other arms and the 0.8B reference, on the O2 common set and the held-out items
score osler_4b_P "$P_CKPT" "$P_ARM"
score osler_4b_N "$N_CKPT" "$N_ARM"
score osler_0p8b "$ZERO8_CKPT" "$ZERO8_ARM"
score_heldout osler_4b_P "$P_CKPT" "$P_ARM"
score_heldout osler_4b_N "$N_CKPT" "$N_ARM"
score_heldout osler_0p8b "$ZERO8_CKPT" "$ZERO8_ARM"
log "CHAIN DONE O11CAL"
