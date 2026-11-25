#!/usr/bin/env bash
# osler_rc0 R2 — dev-only calibration for Osler-9B-rc0 and full-probability scoring of Osler-9B-rc0 + d3-flash.
#
# One GPU job at a time; run detached with
#   setsid nohup bash scripts/osler_rc0/r2_score_chain.sh > outputs/osler_rc0/R2/logs/chain.log 2>&1 < /dev/null &
#
# Order: score fastdev_v4 (fit set) -> score the 8 prob-less/missing sets for the release candidate and its base ->
# fit the three dev temperatures -> apply them to all 9 sets -> verify argmax parity with the pre-existing T10 files.
set -u
cd "$(dirname "$0")/../.." || exit 1
source scripts/pod_env.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
OUT=outputs/osler_rc0/R2
CKPT=outputs/osler_v7/T11/final/checkpoints/step_063304.pt
WITH=(--with pillow==12.3.0 --with "torchvision==0.29.0")
mkdir -p "$OUT/logs" "$OUT/preds"
SETS=(knowledge confirm_general mr_verdict mimic_suite mednli_test mimic_heldout mr_heldout_test pairs_test)
ITEMS=(
  data/bench/osler_v4/y9/knowledge.jsonl
  data/bench/osler_v7/confirm/general.jsonl
  data/bench/osler_v4/y9/mr_verdict.jsonl
  data/bench/osler_v4/y9/mimic_suite.jsonl
  data/bench/osler_v4/y9/mednli_test.jsonl
  data/bench/osler_v4/y9/mimic_heldout.jsonl
  data/bench/osler_v4/y9/mr_heldout_test.jsonl
  data/bench/osler_v4/y9/pairs_test.jsonl
)

echo "R2_CHAIN_START $(date -u +%FT%TZ)"
uv run --frozen "${WITH[@]}" python scripts/osler_v7/t10_score_d3.py \
  --checkpoint "$CKPT" --items "$OUT/fastdev_v4.jsonl" --name 9b_rc0_fastdev_v4 --out "$OUT/preds" \
  >> "$OUT/logs/score_9b_rc0_fastdev_v4.log" 2>&1 || { echo "FAILED 9b_rc0_fastdev_v4"; exit 1; }
echo "R2_FASTDEV_DONE $(date -u +%FT%TZ)"

for i in "${!SETS[@]}"; do
  set_name="${SETS[$i]}"; items="${ITEMS[$i]}"
  uv run --frozen "${WITH[@]}" python scripts/osler_v7/t10_score_d3.py \
    --checkpoint "$CKPT" --items "$items" --name "9b_rc0_$set_name" --out "$OUT/preds" \
    >> "$OUT/logs/score_9b_rc0_$set_name.log" 2>&1 || { echo "FAILED 9b_rc0_$set_name"; exit 1; }
  echo "R2_SCORED 9b_rc0_$set_name $(date -u +%FT%TZ)"
  uv run --frozen "${WITH[@]}" python scripts/osler_v7/t10_score_d3.py \
    --items "$items" --name "d3_base_$set_name" --out "$OUT/preds" \
    >> "$OUT/logs/score_d3_base_$set_name.log" 2>&1 || { echo "FAILED d3_base_$set_name"; exit 1; }
  echo "R2_SCORED d3_base_$set_name $(date -u +%FT%TZ)"
done

uv run --frozen python scripts/osler_rc0/r2_calibrate_d3.py fit \
  --items "$OUT/fastdev_v4.jsonl" --predictions "$OUT/preds/9b_rc0_fastdev_v4/predictions.jsonl" \
  --out "$OUT/9b_calibration.json" >> "$OUT/logs/calibrate_fit.log" 2>&1 || { echo "FAILED fit"; exit 1; }
echo "R2_FIT_DONE $(date -u +%FT%TZ)"

for i in "${!SETS[@]}"; do
  set_name="${SETS[$i]}"; items="${ITEMS[$i]}"
  uv run --frozen python scripts/osler_rc0/r2_calibrate_d3.py apply \
    --items "$items" --predictions "$OUT/preds/9b_rc0_$set_name/predictions.jsonl" \
    --calibration "$OUT/9b_calibration.json" --out "$OUT/preds_cal/9b_rc0_$set_name/predictions.jsonl" \
    --summary "$OUT/preds_cal/9b_rc0_$set_name/calibration_apply.json" \
    >> "$OUT/logs/calibrate_apply_$set_name.log" 2>&1 || { echo "FAILED apply $set_name"; exit 1; }
  echo "R2_CALIBRATED $set_name $(date -u +%FT%TZ)"
done
uv run --frozen python scripts/osler_rc0/r2_calibrate_d3.py apply \
  --items data/bench/osler_v4/y9/panel.jsonl --predictions outputs/eval0/panel/9b_final_panel/predictions.jsonl \
  --calibration "$OUT/9b_calibration.json" --out "$OUT/preds_cal/9b_rc0_panel/predictions.jsonl" \
  --summary "$OUT/preds_cal/9b_rc0_panel/calibration_apply.json" \
  >> "$OUT/logs/calibrate_apply_panel.log" 2>&1 || { echo "FAILED apply panel"; exit 1; }
echo "R2_CHAIN_DONE $(date -u +%FT%TZ)"
