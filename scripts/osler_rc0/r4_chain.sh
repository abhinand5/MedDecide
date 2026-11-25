#!/usr/bin/env bash
# osler_rc0 R4 — complete the fresh tier-2 (`confirm/general.jsonl`, 5,988 Oct–Nov 2026 items) competitor matrix.
#
# One GPU job at a time; run detached with
#   setsid nohup bash scripts/osler_rc0/r4_chain.sh > outputs/osler_rc0/R4/logs/chain.log 2>&1 < /dev/null &
#
# Covered here: pplx-27B (authors' code), MedDecider-9B, MedDecider-4B (authors' code), Clef-Flash (authors' stack), and a
# timeboxed JEV-9B attempt. `pplx-flash` is NOT re-scored: the R0 inventory found `4b_base_general` has full `probs`.
set -u
cd "$(dirname "$0")/../.." || exit 1
source scripts/pod_env.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
OUT=outputs/osler_rc0/R4
ITEMS=data/bench/osler_v7/confirm/general.jsonl
PPLX27=/workspace/.hf_home/hub/models--perplexity-ai--pplx-decider-v1.2-27b/snapshots/cce0f3f80033e9d89d80831d5281fd72124aacce
CLEF=/workspace/.hf_home/hub/models--Cloudflare--clef-flash/snapshots/fde727a287004204b7518dcc983fe64379776712
FLAGS=(--with pillow==12.3.0 --with "torchvision==0.29.*")
mkdir -p "$OUT/logs" "$OUT/preds"
echo "R4_CHAIN_START $(date -u +%FT%TZ)"

uv run --frozen "${FLAGS[@]}" python scripts/osler_v4/y4_pplx_flash.py \
  --checkpoint "$PPLX27" --revision cce0f3f80033e9d89d80831d5281fd72124aacce \
  --label "pplx-decider-v1.2-27b (zero-shot, authors' code)" \
  --items "$ITEMS" --name pplx27b_confirm_general --out-root "$OUT/preds" \
  >> "$OUT/logs/pplx27b.log" 2>&1
rc=$?; [ $rc -ne 0 ] && echo "FAILED pplx27b"
echo "R4_STEP pplx27b exit=$rc $(date -u +%FT%TZ)"

uv run --frozen "${FLAGS[@]}" python scripts/osler_v4/y4_meddecider.py \
  --model lion-ai/MedDecider-9B --revision ec8a69da2cd96b2ddd2ff4e314ba9a40b71936c3 \
  --items "$ITEMS" --name md9b_confirm_general --out-root "$OUT/preds" \
  >> "$OUT/logs/md9b.log" 2>&1
rc=$?; [ $rc -ne 0 ] && echo "FAILED md9b"
echo "R4_STEP md9b exit=$rc $(date -u +%FT%TZ)"

uv run --frozen "${FLAGS[@]}" python scripts/osler_v4/y4_meddecider.py \
  --model lion-ai/MedDecider-4B --revision 570b37090c379ad722186a79a3432a31be463fc9 \
  --items "$ITEMS" --name md4b_confirm_general --out-root "$OUT/preds" \
  >> "$OUT/logs/md4b.log" 2>&1
rc=$?; [ $rc -ne 0 ] && echo "FAILED md4b"
echo "R4_STEP md4b exit=$rc $(date -u +%FT%TZ)"

if [ -d envs/clef/.venv/bin ]; then
  envs/clef/.venv/bin/python -I scripts/osler_rc0/r4_clef_confirm.py --path "$CLEF" --items "$ITEMS" \
    --out "$OUT/preds/clef_confirm_general/predictions.jsonl" \
    >> "$OUT/logs/clef.log" 2>&1
  rc=$?; [ $rc -ne 0 ] && echo "FAILED clef"
  echo "R4_STEP clef exit=$rc $(date -u +%FT%TZ)"
else
  echo "R4_STEP clef skipped: envs/clef/.venv is absent"
fi

# JEV-9B: the existing runner reads benchmark-scoped items; give it a scratch copy tagged v0.2 (30-minute timebox).
mkdir -p "$OUT/jev"
python - "$ITEMS" "$OUT/jev/confirm_general_jev.jsonl" <<'PY'
import json, sys
src, dst = sys.argv[1], sys.argv[2]
with open(src, encoding="utf-8") as handle, open(dst, "w", encoding="utf-8") as out:
    for line in handle:
        if not line.strip():
            continue
        row = json.loads(line)
        row["benchmark"] = "v0.2"
        row["set_name"] = "confirm_general"
        out.write(json.dumps(row) + "\n")
print("wrote", dst)
PY
timeout 1800 uv run --frozen "${FLAGS[@]}" python scripts/osler/o2_run_jev.py --repo autotrust/JEV-9B --scope all \
  --items "$OUT/jev/confirm_general_jev.jsonl" --out-root "$OUT/jev/preds" \
  >> "$OUT/logs/jev.log" 2>&1 || echo "FAILED jev (timebox or runner)"
if [ -f "$OUT/jev/preds/jev-9b/predictions.jsonl" ]; then
  mkdir -p "$OUT/preds/jev9b_confirm_general"
  cp "$OUT/jev/preds/jev-9b/predictions.jsonl" "$OUT/preds/jev9b_confirm_general/predictions.jsonl"
  wc -l "$OUT/preds/jev9b_confirm_general/predictions.jsonl" >> "$OUT/logs/chain.log"
fi
echo "R4_STEP jev $(date -u +%FT%TZ)"
echo "R4_CHAIN_DONE $(date -u +%FT%TZ)"
