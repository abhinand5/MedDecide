#!/usr/bin/env bash
# osler_v3 Stage A (ADVISORY X7): the four matched recipe runs, one GPU job at a time, each scored on the clinical-general partition.
# A0 control (KL 0) and A1/A2 (KL 0.1 / 0.5) train on osler_v2's C1 block with osler_v2's replay of it; A3 (KL 0) trains on the A3
# block (C1 + the broad replay, 8 % cap) with its own replay. A finished arm (block_result.json present) is skipped, so the chain
# can be restarted after a crash without repeating work. Launch detached:
#   setsid nohup bash scripts/osler_v3/x7_stage_a.sh > outputs/osler_v3/X7/logs/stage_a_chain.out 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/../.."
source scripts/pod_env.sh
LOG=outputs/osler_v3/X7/logs/stage_a_chain.log
mkdir -p outputs/osler_v3/X7/logs
until [ -f data/bench/osler_v3/fastdev_v3.jsonl ]; do echo "WAIT fast dev v3 $(date -u +%FT%TZ)" >> "$LOG"; sleep 300; done
run_arm() {
  arm=$1; kl=$2; train=$3; replay=$4
  if [ -f "outputs/osler_v3/stage_a/$arm/block_result.json" ]; then echo "SKIP $arm (done) $(date -u +%FT%TZ)" >> "$LOG"; return; fi
  echo "START $arm kl=$kl $(date -u +%FT%TZ)" >> "$LOG"
  uv run --frozen python scripts/osler_v3/x7_block_run.py --stage stage_a --arm "$arm" --train "$train" --replay "$replay" --kl-to-base "$kl" >> "outputs/osler_v3/X7/logs/$arm.log" 2>&1
  echo "END block $arm rc=$? $(date -u +%FT%TZ)" >> "$LOG"
  uv run --frozen python scripts/osler_v3/x7_score.py --stage stage_a --arm "$arm" >> "outputs/osler_v3/X7/logs/$arm.log" 2>&1
  echo "END score $arm rc=$? $(date -u +%FT%TZ)" >> "$LOG"
}
run_arm A0 0 data/train/osler_v2/block_C1.jsonl outputs/osler_v2/W6/replay/C1_pplx.json
run_arm A1 0.1 data/train/osler_v2/block_C1.jsonl outputs/osler_v2/W6/replay/C1_pplx.json
run_arm A2 0.5 data/train/osler_v2/block_C1.jsonl outputs/osler_v2/W6/replay/C1_pplx.json
run_arm A3 0 data/train/osler_v3/block_A3.jsonl outputs/osler_v3/replays/A3_pplx.json
echo "STAGE A DONE $(date -u +%FT%TZ)" >> "$LOG"
