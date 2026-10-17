#!/usr/bin/env bash
# osler_v2 W5 GPU queue (ADVISORY W5 steps 2-3). One GPU job at a time, in this order:
#   1. zero-shot Qwen3.5-4B on knowledge dev v2 (the C0 knowledge checks need it);
#   2. zero-shot pplx-decider-flash-v1.2 on knowledge dev v2, through its authors' code (torchvision and pillow for this run);
#   3. the five baselines on the EHR-Rel test set (373 items): zero-shot Qwen3.5-4B and -9B, MedDecider-4B and -9B, JEV-9B.
# Each run resumes from its predictions file. Revisions are those of osler_v1's V6 chain.
# Launch detached:  setsid nohup bash scripts/osler_v2/w5_chain.sh > /dev/null 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/../.."
source scripts/pod_env.sh
LOG=outputs/osler_v2/W5/logs/chain.log
KDEV=data/bench/osler_v2/knowledge_dev_v2.jsonl
EHR=data/bench/osler_v2/ehr_rel_test_mrv0.jsonl
OUT=outputs/osler_v2/W5/baselines
mkdir -p outputs/osler_v2/W5/logs "$OUT"
step() {
  local name=$1
  shift
  echo "START $name $(date -u +%FT%TZ)" >> "$LOG"
  "$@" >> "outputs/osler_v2/W5/logs/run_$name.log" 2>&1
  local rc=$?
  echo "END $name rc=$rc $(date -u +%FT%TZ)" >> "$LOG"
  return $rc
}
step kdev_zs4b uv run --frozen python scripts/osler/o2_run_zeroshot.py --model Qwen/Qwen3.5-4B --scope mix_v2 \
  --items "$KDEV" --out-root outputs/osler_v2/W5/zeroshot_knowledge || echo "kdev_zs4b failed" >> "$LOG"
step kdev_pplx uv run --frozen --with pillow --with "torchvision==0.29.*" python scripts/osler_v2/o2_run_pplx_zeroshot.py \
  --scope mix_v2 --items "$KDEV" --out-root outputs/osler_v2/W5/zeroshot_knowledge || echo "kdev_pplx failed" >> "$LOG"
step ehr_zs4b uv run --frozen python scripts/osler/o2_run_zeroshot.py --model Qwen/Qwen3.5-4B --scope mr_v0 \
  --items "$EHR" --out-root "$OUT" || echo "ehr_zs4b failed" >> "$LOG"
step ehr_zs9b uv run --frozen python scripts/osler/o2_run_zeroshot.py --model Qwen/Qwen3.5-9B --scope mr_v0 \
  --items "$EHR" --out-root "$OUT" || echo "ehr_zs9b failed" >> "$LOG"
step ehr_md4b uv run --frozen python scripts/osler/o2_run_meddecider.py --model lion-ai/MedDecider-4B \
  --revision 570b37090c379ad722186a79a3432a31be463fc9 --scope mr_v0 --items "$EHR" --out-root "$OUT" --verify 50 \
  || echo "ehr_md4b failed" >> "$LOG"
step ehr_md9b uv run --frozen python scripts/osler/o2_run_meddecider.py --model lion-ai/MedDecider-9B \
  --revision ec8a69da2cd96b2ddd2ff4e314ba9a40b71936c3 --scope mr_v0 --items "$EHR" --out-root "$OUT" --verify 50 \
  || echo "ehr_md9b failed" >> "$LOG"
step ehr_jev9b uv run --frozen python scripts/osler/o2_run_jev.py --repo autotrust/JEV-9B --scope mr_v0 --items "$EHR" \
  --out-root "$OUT" || echo "ehr_jev9b failed" >> "$LOG"
echo "W5 CHAIN DONE $(date -u +%FT%TZ)" >> "$LOG"
