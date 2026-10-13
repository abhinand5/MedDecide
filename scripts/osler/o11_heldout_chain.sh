#!/usr/bin/env bash
# O11 held-out generator scoring (ADVISORY O11): the five Osler models on the 600 held-out test items, both orders and single
# order, after the O11 chain (scripts/osler/o11_chain.sh) finishes. It waits for "CHAIN DONE O11" in the O11 chain log; it stops
# if that chain writes STOP or is no longer running without a CHAIN DONE line. One GPU job at a time. Each scoring run resumes
# from the rows already written, and each is checked for completeness. Stops at the first failure and logs why.
# Launch detached:
#   setsid nohup bash scripts/osler/o11_heldout_chain.sh > /dev/null 2>&1 < /dev/null &
# The chain commits nothing. The held-out rows are reported in their own table (heldout.md rule 3).
set -u
cd "$(dirname "$0")/../.."
source scripts/pod_env.sh
CHAIN_LOG=outputs/osler_v0/O11/logs/chain.log
LOG=outputs/osler_v0/O11/logs/heldout_chain.log
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
complete() {  # complete <name>: both files hold every held-out row, the count run.json records
  local expected both single
  expected=$(uv run --frozen python -c "import json; print(json.load(open('outputs/osler_v0/O11/heldout/$1/run.json'))['items_in_scope'])") \
    || fail "held-out $1: run.json could not be read"
  both=$(wc -l < "outputs/osler_v0/O11/heldout/$1/both_orders.jsonl")
  single=$(wc -l < "outputs/osler_v0/O11/heldout/$1/single_order.jsonl")
  [ "$both" -eq "$expected" ] && [ "$single" -eq "$expected" ] \
    || fail "held-out $1: $both and $single rows written, $expected in scope"
  log "held-out $1 complete: $expected rows in both files"
}
score() {  # score <name> <checkpoint directory>
  uv run --frozen python scripts/osler/o11_heldout_score.py --checkpoint "$2" --name "$1" >> "outputs/osler_v0/O11/logs/heldout_$1.log" 2>&1 \
    || fail "held-out scoring of $1 exited non-zero (outputs/osler_v0/O11/logs/heldout_$1.log)"
  need "outputs/osler_v0/O11/heldout/$1/run.json" "held-out scoring of $1 wrote no run.json"
  complete "$1"
  log "held-out score $1 DONE"
}

log "HELDOUT CHAIN START (waits for the O11 chain)"
until grep -q "CHAIN DONE O11" "$CHAIN_LOG" 2>/dev/null; do
  grep -q "^STOP" "$CHAIN_LOG" 2>/dev/null && fail "the O11 chain stopped; the held-out scoring does not start"
  pgrep -f "[o]11_chain.sh" > /dev/null || fail "the O11 chain is not running and wrote no CHAIN DONE line"
  sleep 60
done
log "O11 chain done; held-out scoring starts"

# 1. held-out scoring of arms L, P, N and both O10 models (GPU, one job at a time)
score osler_4b_L outputs/osler_v0/O6/arm_L/checkpoints/best
score osler_4b_P outputs/osler_v0/O6/arm_P/checkpoints/best
score osler_4b_N outputs/osler_v0/O6/arm_N/checkpoints/best
score osler_9b outputs/osler_v0/O10/osler_9b/checkpoints/best
score osler_0p8b outputs/osler_v0/O10/osler_0p8b/checkpoints/best
log "CHAIN DONE HELDOUT"
