#!/usr/bin/env bash
# osler_v3 X8 (ADVISORY X8): the Stage B blocks on the Stage A choice. One GPU job at a time.
#   B0 = the chosen Stage A run (no new run; its mix is C1 for A0-A2 and the A3 block for A3; its KL is the chosen arm's).
#   B1 = B0 + the X1 real sources (RAGTruth, DrugProt); B3 = the last kept block + the X4 charts (x4_to_mix.py output).
#   B2 (X3 Gemma items) is NOT MEASURED (X3 blocked, no rows); B4 (MIMIC) is BLOCKED (official files absent).
# Each candidate: build (CPU), replay (CPU), train 13,992 steps (GPU), CG score (GPU), keep rule (CPU).
# Launch detached after X7's choice:  setsid nohup bash scripts/osler_v3/x8_chain.sh > outputs/osler_v3/X8/logs/chain.out 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/../.."
source scripts/pod_env.sh
LOG=outputs/osler_v3/X8/logs/chain.log
STEPS=13992
mkdir -p outputs/osler_v3/X8/logs
if pgrep -f "[v]llm serve" > /dev/null; then
  echo "ABORT: the vLLM Gemma server still holds the GPU $(date -u +%FT%TZ)" >> "$LOG"
  exit 1
fi
CHOSEN=$(uv run --frozen python -c "import json;print(json.load(open('outputs/osler_v3/X7/choice.json'))['chosen'])")
case "$CHOSEN" in
  A0) KL=0 ;;
  A1) KL=0.1 ;;
  A2) KL=0.5 ;;
  A3) KL=0 ;;
  *) echo "ABORT: no Stage A choice ($CHOSEN) $(date -u +%FT%TZ)" >> "$LOG"; exit 1 ;;
esac
echo "X8 START chosen=$CHOSEN kl=$KL $(date -u +%FT%TZ)" >> "$LOG"
echo "B2 NOT MEASURED: X3 blocked, no rows $(date -u +%FT%TZ)" >> "$LOG"
echo "B4 BLOCKED: official MIMIC files absent $(date -u +%FT%TZ)" >> "$LOG"

kept() {  # kept <name>: prints true or false from the keep rule's report
  uv run --frozen python -c "import json;print(str(json.load(open('outputs/osler_v3/X8/keep_$1.json'))['decision']).lower())"
}

candidate() {  # candidate <name> <base mix> <last kept name> <last kept dir> <new source>
  local name=$1 base=$2 last=$3 last_dir=$4 add=$5
  local out="outputs/osler_v3/X8/logs/$name.log"
  echo "START build $name $(date -u +%FT%TZ)" >> "$LOG"
  uv run --frozen python scripts/osler_v3/x8_build_block.py --name "$name" --base "$base" --add "$add" >> "$out" 2>&1 \
    || { echo "BUILD $name FAILED $(date -u +%FT%TZ)" >> "$LOG"; return 1; }
  echo "START replay $name $(date -u +%FT%TZ)" >> "$LOG"
  uv run --frozen python scripts/osler_v3/x7_replay.py --train "data/train/osler_v3/block_$name.jsonl" --name "$name" \
    --steps "$STEPS" >> "$out" 2>&1 || { echo "REPLAY $name FAILED $(date -u +%FT%TZ)" >> "$LOG"; return 1; }
  echo "START train $name kl=$KL $(date -u +%FT%TZ)" >> "$LOG"
  uv run --frozen python scripts/osler_v3/x7_block_run.py --stage stage_b --arm "$name" \
    --train "data/train/osler_v3/block_$name.jsonl" --replay "outputs/osler_v3/replays/${name}_pplx.json" \
    --kl-to-base "$KL" >> "$out" 2>&1 || { echo "TRAIN $name FAILED $(date -u +%FT%TZ)" >> "$LOG"; return 1; }
  echo "END train $name $(date -u +%FT%TZ)" >> "$LOG"
  uv run --frozen python scripts/osler_v3/x7_score.py --stage stage_b --arm "$name" >> "$out" 2>&1 \
    || { echo "CG SCORE $name FAILED $(date -u +%FT%TZ)" >> "$LOG"; return 1; }
  uv run --frozen python scripts/osler_v3/x8_keep.py --candidate "$name" --candidate-dir "outputs/osler_v3/stage_b/$name" \
    --current "$last" --current-dir "$last_dir" >> "$out" 2>&1 || { echo "KEEP $name FAILED $(date -u +%FT%TZ)" >> "$LOG"; return 1; }
  echo "END keep $name $(kept "$name") $(date -u +%FT%TZ)" >> "$LOG"
}

if [ "$CHOSEN" = "A3" ]; then BASE=data/train/osler_v3/block_A3.jsonl; else BASE=data/train/osler_v2/block_C1.jsonl; fi
LAST_FILE=$BASE; LAST=B0; LAST_DIR=outputs/osler_v3/stage_a/$CHOSEN
candidate B1 "$LAST_FILE" "$LAST" "$LAST_DIR" data/train/osler_v3/x1_sources_train.jsonl
if [ "$(kept B1)" = "true" ]; then LAST_FILE=data/train/osler_v3/block_B1.jsonl; LAST=B1; LAST_DIR=outputs/osler_v3/stage_b/B1; fi
if [ -f data/train/osler_v3/x4_train.jsonl ]; then
  candidate B3 "$LAST_FILE" "$LAST" "$LAST_DIR" data/train/osler_v3/x4_train.jsonl
  if [ "$(kept B3)" = "true" ]; then LAST=B3; fi
else
  echo "B3 NOT MEASURED: data/train/osler_v3/x4_train.jsonl absent $(date -u +%FT%TZ)" >> "$LOG"
fi
echo "$LAST" > outputs/osler_v3/X8/last_kept.txt
echo "X8 DONE last kept: $LAST $(date -u +%FT%TZ)" >> "$LOG"
