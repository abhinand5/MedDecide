#!/bin/bash
# Score the frozen external panel (once) for the Osler models and the d3-flash zero-shot base. One GPU job at a time.
set -u
cd /workspace/MedDecide
source scripts/pod_env.sh
OUT=outputs/eval0/panel
LOG=$OUT/logs/queue.log
PANEL=data/bench/osler_v4/y9/panel.jsonl
echo "PANEL_QUEUE_START $(date -u +%FT%TZ)" >> $LOG
uv run --frozen --with pillow==12.3.0 --with "torchvision==0.29.0" python scripts/osler_v7/t10_score_d3.py \
  --checkpoint outputs/osler_v7/T11/final/checkpoints/step_063304.pt --items $PANEL --name 9b_final_panel --out $OUT >> $LOG 2>&1 || echo "FAILED 9b_final_panel" >> $LOG
uv run --frozen --with pillow==12.3.0 --with "torchvision==0.29.0" python scripts/osler_v7/t10_score_d3.py \
  --checkpoint outputs/osler_v7/T8/arm/checkpoints/step_013992.pt --items $PANEL --name 9b_arm_panel --out $OUT >> $LOG 2>&1 || echo "FAILED 9b_arm_panel" >> $LOG
uv run --frozen --with pillow==12.3.0 --with "torchvision==0.29.0" python scripts/osler_v7/t10_score_d3.py \
  --items $PANEL --name 9b_base_panel --out $OUT >> $LOG 2>&1 || echo "FAILED 9b_base_panel" >> $LOG
uv run --frozen python scripts/osler_v7/score_items.py --checkpoint outputs/osler_v7/T7/chosen \
  --items $PANEL --name 4b_arm_panel --out $OUT >> $LOG 2>&1 || echo "FAILED 4b_arm_panel" >> $LOG
echo "PANEL_QUEUE_DONE $(date -u +%FT%TZ)" >> $LOG
