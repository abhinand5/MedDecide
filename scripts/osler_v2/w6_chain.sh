#!/usr/bin/env bash
# osler_v2 W6 GPU queue (ADVISORY W6; one GPU job at a time): the C0 pair, then the base choice and the knowledge checks.
#   1. smoke C0 on Qwen3.5-4B and on pplx-decider-flash (8 steps, 200 dev items, every step logged);
#   2. C0q and C0p, the full 13,992 steps each (the same data, adapter and steps; D39);
#   3. the base choice (CPU, D39) and the knowledge checks of C0q and C0p (CPU, on the zero-shot predictions of W5).
# C1 and C2 are started by hand after the base choice is written (blocks.md), so that the operator can see it first.
# Launch detached:  setsid nohup bash scripts/osler_v2/w6_chain.sh > /dev/null 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/../.."
source scripts/pod_env.sh
LOG=outputs/osler_v2/W6/logs/chain.log
mkdir -p outputs/osler_v2/W6/logs
job() {
  local name=$1
  shift
  echo "START $name $(date -u +%FT%TZ)" >> "$LOG"
  "$@" >> "outputs/osler_v2/W6/logs/$name.log" 2>&1
  local rc=$?
  echo "END $name rc=$rc $(date -u +%FT%TZ)" >> "$LOG"
  return $rc
}
job smoke_C0_qwen uv run --frozen python scripts/osler_v2/w6_block_run.py --block C0 --backbone qwen --smoke 8 \
  || { echo "CHAIN STOPPED at smoke_C0_qwen $(date -u +%FT%TZ)" >> "$LOG"; exit 1; }
job smoke_C0_pplx uv run --frozen --with "torchvision==0.29.*" --with pillow python scripts/osler_v2/w6_block_run.py --block C0 --backbone pplx --smoke 8 \
  || { echo "CHAIN STOPPED at smoke_C0_pplx $(date -u +%FT%TZ)" >> "$LOG"; exit 1; }
job C0_qwen uv run --frozen python scripts/osler_v2/w6_block_run.py --block C0 --backbone qwen \
  || { echo "CHAIN STOPPED at C0_qwen $(date -u +%FT%TZ)" >> "$LOG"; exit 1; }
job C0_pplx uv run --frozen --with "torchvision==0.29.*" --with pillow python scripts/osler_v2/w6_block_run.py --block C0 --backbone pplx \
  || { echo "CHAIN STOPPED at C0_pplx $(date -u +%FT%TZ)" >> "$LOG"; exit 1; }
job base_choice uv run --frozen python scripts/osler_v2/w6_keep.py --base
job kcheck_C0q uv run --frozen python scripts/osler_v2/w6_keep.py --knowledge-check C0_qwen --zs qwen
job kcheck_C0p uv run --frozen python scripts/osler_v2/w6_keep.py --knowledge-check C0_pplx --zs pplx
echo "W6 C0 STAGE DONE $(date -u +%FT%TZ)" >> "$LOG"
