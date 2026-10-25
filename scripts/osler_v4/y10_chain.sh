#!/usr/bin/env bash
# Y10 (osler_v4): the membership scores of the final model (calibrated) and of its zero-shot control, then the verbatim probe.
# One GPU job at a time. Run after `y10_audit.py members` has written data/bench/osler_v4/y10/membership.jsonl.
#
# Corrections to the draft in outputs/osler_v4/Y10/logs/y10_chain.sh (never run): (1) each step's exit code is taken right after its
# own command (the draft expanded $(date ...) before $?, which records date's status); (2) the final model's membership scores take
# the saved calibration (--arm-result), as the Y9 chain does and the Y10 docstring states; (3) the pplx steps take the torchvision
# flags of Y4 and Y9's repair (the scorer's processor needs torchvision). Row counts are recorded at the end; check them, not the exits.
set -u
source /workspace/MedDecide/scripts/pod_env.sh
cd /workspace/MedDecide
L=outputs/osler_v4/Y10/logs
LOG=$L/y10_chain.log
ITEMS=data/bench/osler_v4/y10/membership.jsonl
FINAL=outputs/osler_v4/Y8/checkpoints/best
FLAGS=(--with pillow==12.3.0 --with "torchvision==0.29.*")
mkdir -p "$L"
step() {
  local name="$1"
  shift
  echo "$(date -u +%FT%TZ) start $name" >> "$LOG"
  uv run --frozen "$@" >> "$L/$name.log" 2>&1
  local rc=$?
  echo "$(date -u +%FT%TZ) end $name exit $rc" >> "$LOG"
}
step final_membership python scripts/osler_v4/y4_checkpoint_score.py --checkpoint "$FINAL" --items "$ITEMS" --name final_membership \
  --out-root outputs/osler_v4/Y10/scores --arm-result outputs/osler_v4/Y8/arm_result.json \
  --label "osler_v4 Osler-4B final (Y8 selected checkpoint, calibrated; membership items)"
step base_membership "${FLAGS[@]}" python scripts/osler_v4/y4_pplx_flash.py --items "$ITEMS" --name base_membership \
  --out-root outputs/osler_v4/Y10/scores
step probe "${FLAGS[@]}" python scripts/osler_v4/y10_audit.py probe
echo "$(date -u +%FT%TZ) rows: final $(wc -l < outputs/osler_v4/Y10/scores/final_membership/predictions.jsonl 2>/dev/null)" \
  "base $(wc -l < outputs/osler_v4/Y10/scores/base_membership/predictions.jsonl 2>/dev/null) items $(wc -l < "$ITEMS")" >> "$LOG"
echo "$(date -u +%FT%TZ) chain done" >> "$LOG"
