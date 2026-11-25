#!/usr/bin/env bash
# osler_rc0 R6 — GPU part of the MIMIC PHI / memorisation audit (D28): membership scores and the 4B verbatim probe.
#
# One GPU job at a time; run detached with
#   setsid nohup bash scripts/osler_rc0/r6_chain.sh > outputs/osler_rc0/R6/logs/chain.log 2>&1 < /dev/null &
#
# Prerequisites (CPU): `r6_phi_audit.py --step visited`, `--step sample`, and the R2 calibration JSON.
# Local only: MIMIC item text is read from data/ and written only to gitignored files under outputs/osler_rc0/R6/.
set -u
cd "$(dirname "$0")/../.." || exit 1
source scripts/pod_env.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
OUT=outputs/osler_rc0/R6
CKPT4=outputs/osler_rc0/R1/arm/checkpoints/final_step_063304
CKPT9=outputs/osler_v7/T11/final/checkpoints/step_063304.pt
FLAGS=(--with pillow==12.3.0 --with "torchvision==0.29.0")
mkdir -p "$OUT/logs" "$OUT/scores" "$OUT/scores/y10"
run() {
  local name="$1"; shift
  echo "R6_STEP_START $name $(date -u +%FT%TZ)"
  "$@" >> "$OUT/logs/$name.log" 2>&1
  local rc=$?
  echo "R6_STEP_END $name exit=$rc $(date -u +%FT%TZ)"
  return $rc
}
echo "R6_CHAIN_START $(date -u +%FT%TZ)"

# --- visited-member items (4,000) --------------------------------------------------------------------------------------
run visit_4b_rc0 uv run --frozen python scripts/osler_v7/score_items.py \
  --checkpoint "$CKPT4" --items "$OUT/visited_membership.jsonl" --name 4b_rc0 --out "$OUT/scores"
run visit_9b_rc0 uv run --frozen "${FLAGS[@]}" python scripts/osler_v7/t10_score_d3.py \
  --checkpoint "$CKPT9" --items "$OUT/visited_membership.jsonl" --name 9b_rc0 --out "$OUT/scores"
run visit_d3_base uv run --frozen "${FLAGS[@]}" python scripts/osler_v7/t10_score_d3.py \
  --items "$OUT/visited_membership.jsonl" --name d3_base --out "$OUT/scores"
run visit_pplx_base uv run --frozen "${FLAGS[@]}" python scripts/osler_v4/y4_pplx_flash.py \
  --items "$OUT/visited_membership.jsonl" --name pplx_base --out-root "$OUT/scores"

# --- Y10 compatibility items (4,000) -----------------------------------------------------------------------------------
run y10_4b_rc0 uv run --frozen python scripts/osler_v7/score_items.py \
  --checkpoint "$CKPT4" --items data/bench/osler_v4/y10/membership.jsonl --name 4b_rc0 --out "$OUT/scores/y10"
run y10_9b_rc0 uv run --frozen "${FLAGS[@]}" python scripts/osler_v7/t10_score_d3.py \
  --checkpoint "$CKPT9" --items data/bench/osler_v4/y10/membership.jsonl --name 9b_rc0 --out "$OUT/scores/y10"
run y10_d3_base uv run --frozen "${FLAGS[@]}" python scripts/osler_v7/t10_score_d3.py \
  --items data/bench/osler_v4/y10/membership.jsonl --name d3_base --out "$OUT/scores/y10"
run y10_pplx_base uv run --frozen "${FLAGS[@]}" python scripts/osler_v4/y4_pplx_flash.py \
  --items data/bench/osler_v4/y10/membership.jsonl --name pplx_base --out-root "$OUT/scores/y10"

# --- 4B verbatim probe (300 MIMIC discharge notes, zero-shot control) ---------------------------------------------------
run verbatim uv run --frozen python scripts/osler_rc0/r6_phi_audit.py --step verbatim
run arch uv run --frozen python scripts/osler_rc0/r6_phi_audit.py --step arch
echo "R6_CHAIN_DONE $(date -u +%FT%TZ)"
