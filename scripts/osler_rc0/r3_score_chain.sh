#!/usr/bin/env bash
# osler_rc0 R3 — score the frozen Osler-4B-rc0 checkpoint on all 9 frozen evaluation sets with full probabilities.
#
# One GPU job at a time; run detached with
#   setsid nohup bash scripts/osler_rc0/r3_score_chain.sh > outputs/osler_rc0/R3/logs/chain.log 2>&1 < /dev/null &
#
# The checkpoint is loaded ONCE per set through `PplxDecisionModel.load`, which applies the dev temperatures saved in the
# checkpoint directory (R1 acceptance item 4). No checkpoint selection happens here: the step is frozen at 63,304.
set -u
cd "$(dirname "$0")/../.." || exit 1
source scripts/pod_env.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
OUT=outputs/osler_rc0/R3
CKPT=outputs/osler_rc0/R1/arm/checkpoints/final_step_063304
mkdir -p "$OUT/logs" "$OUT/preds"
SETS=(panel knowledge confirm_general mr_verdict mimic_suite mednli_test mimic_heldout mr_heldout_test pairs_test)
ITEMS=(
  data/bench/osler_v4/y9/panel.jsonl
  data/bench/osler_v4/y9/knowledge.jsonl
  data/bench/osler_v7/confirm/general.jsonl
  data/bench/osler_v4/y9/mr_verdict.jsonl
  data/bench/osler_v4/y9/mimic_suite.jsonl
  data/bench/osler_v4/y9/mednli_test.jsonl
  data/bench/osler_v4/y9/mimic_heldout.jsonl
  data/bench/osler_v4/y9/mr_heldout_test.jsonl
  data/bench/osler_v4/y9/pairs_test.jsonl
)
if [ ! -d "$CKPT" ]; then echo "FATAL no checkpoint $CKPT"; exit 1; fi
python - "$CKPT" "$OUT/4b_calibration.json" <<'PY' >> "$OUT/logs/calibration_check.log" 2>&1
import json, sys
from pathlib import Path
meta = json.loads((Path(sys.argv[1]) / "model.json").read_text())
print(json.dumps({"checkpoint": sys.argv[1], "base_model_id": meta.get("base_model_id"),
                  "calibration": meta.get("calibration"), "backbone": meta.get("backbone"),
                  "dtype": meta.get("dtype"), "step": meta.get("step")}, indent=1))
if not meta.get("calibration"):
    raise SystemExit("the checkpoint carries no dev temperatures: stop")
Path(sys.argv[2]).write_text(json.dumps({"method": "dev-only per-qtype temperature, saved by the R1 trainer",
                                         "temperatures": meta["calibration"],
                                         "source": str(Path(sys.argv[1]) / "model.json")}, indent=1) + "\n")
PY
echo "R3_CHAIN_START $(date -u +%FT%TZ)"
for i in "${!SETS[@]}"; do
  set_name="${SETS[$i]}"; items="${ITEMS[$i]}"
  uv run --frozen python scripts/osler_v7/score_items.py \
    --checkpoint "$CKPT" --items "$items" --name "4b_rc0_$set_name" --out "$OUT/preds" \
    >> "$OUT/logs/score_4b_rc0_$set_name.log" 2>&1 || { echo "FAILED 4b_rc0_$set_name"; exit 1; }
  uv run --frozen python scripts/osler_rc0/r2_calibrate_d3.py apply \
    --items "$items" --predictions "$OUT/preds/4b_rc0_$set_name/predictions.jsonl" \
    --calibration "$OUT/4b_calibration.json" --out "$OUT/preds_cal/4b_rc0_$set_name/predictions.jsonl" \
    --summary "$OUT/preds_cal/4b_rc0_$set_name/calibration_apply.json" \
    >> "$OUT/logs/calibrate_4b_$set_name.log" 2>&1 || { echo "FAILED apply 4b_rc0_$set_name"; exit 1; }
  echo "R3_SCORED_AND_CALIBRATED 4b_rc0_$set_name $(date -u +%FT%TZ)"
done
echo "R3_CHAIN_DONE $(date -u +%FT%TZ)"
