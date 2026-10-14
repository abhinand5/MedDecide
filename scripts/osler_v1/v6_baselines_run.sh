#!/usr/bin/env bash
# V6 baselines (osler_v1 ADVISORY V6, step 2): the O2 runners, unchanged protocol, on the MR evaluation file (test and held-out
# rows). One GPU job at a time, in order: zero-shot Qwen3.5-4B and -9B; MedDecider-4B and -9B (authors' code, both orders);
# JEV-9B (the card's decision head). Each run resumes from its predictions file. The additional d1-3B row is not in this chain.
# Revisions are the full commit hashes from configs/osler_v0/competitors.json.
# Launch detached:  setsid nohup bash scripts/osler_v1/v6_baselines_run.sh > /dev/null 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/../.."
source scripts/pod_env.sh
LOG=outputs/osler_v1/V6/logs/baselines_chain.log
ITEMS=data/bench/osler_mr_v0/mr_eval.jsonl
OUT=outputs/osler_v1/V6/baselines
mkdir -p outputs/osler_v1/V6/logs "$OUT"
step() {
  local name=$1
  shift
  echo "START $name $(date -u +%FT%TZ)" >> "$LOG"
  uv run --frozen python "$@" >> "outputs/osler_v1/V6/logs/run_$name.log" 2>&1
  local rc=$?
  echo "END $name rc=$rc $(date -u +%FT%TZ)" >> "$LOG"
  return $rc
}
step zs4b scripts/osler/o2_run_zeroshot.py --model Qwen/Qwen3.5-4B --scope mr_v0 --items "$ITEMS" --out-root "$OUT" || echo "zs4b failed" >> "$LOG"
step zs9b scripts/osler/o2_run_zeroshot.py --model Qwen/Qwen3.5-9B --scope mr_v0 --items "$ITEMS" --out-root "$OUT" || echo "zs9b failed" >> "$LOG"
step md4b scripts/osler/o2_run_meddecider.py --model lion-ai/MedDecider-4B --revision 570b37090c379ad722186a79a3432a31be463fc9 --scope mr_v0 --items "$ITEMS" --out-root "$OUT" --verify 50 || echo "md4b failed" >> "$LOG"
step md9b scripts/osler/o2_run_meddecider.py --model lion-ai/MedDecider-9B --revision ec8a69da2cd96b2ddd2ff4e314ba9a40b71936c3 --scope mr_v0 --items "$ITEMS" --out-root "$OUT" --verify 50 || echo "md9b failed" >> "$LOG"
step jev9b scripts/osler/o2_run_jev.py --repo autotrust/JEV-9B --scope mr_v0 --items "$ITEMS" --out-root "$OUT" || echo "jev9b failed" >> "$LOG"
echo "V6 BASELINES CHAIN DONE $(date -u +%FT%TZ)" >> "$LOG"
