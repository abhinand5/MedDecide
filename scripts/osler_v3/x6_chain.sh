#!/usr/bin/env bash
# osler_v3 X6 GPU queue (ADVISORY X6 steps 3-4), then Stage A (X7). One GPU job at a time, in this order:
#   1. zero-shot pplx-decider-flash-v1.2 on the clinical-general partition (cg_v3.jsonl, scope mix_v2; its authors' code);
#   2. the new test pool (X1 test and the kept X4 test templates; additional rows only, never in a verdict): osler_v2 W5's five
#      baselines (zero-shot Qwen3.5-4B and -9B, MedDecider-4B and -9B, JEV-9B) and zero-shot pplx-decider-flash-v1.2;
#   3. Stage A (scripts/osler_v3/x7_stage_a.sh), which waits for fast dev v3 (built before this queue starts).
# Each runner resumes from its predictions file. The queue refuses to start while the Gemma server (vLLM) holds the GPU.
# Launch detached:  setsid nohup bash scripts/osler_v3/x6_chain.sh > outputs/osler_v3/X6/logs/chain.out 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/../.."
source scripts/pod_env.sh
LOG=outputs/osler_v3/X6/logs/chain.log
OUT=outputs/osler_v3/X6
POOL=data/bench/osler_v3/new_test_o2.jsonl
CG=data/bench/osler_v3/cg_v3.jsonl
mkdir -p "$OUT/logs"
if pgrep -f "[v]llm serve" > /dev/null; then
  echo "ABORT: the vLLM Gemma server still holds the GPU $(date -u +%FT%TZ)" >> "$LOG"
  exit 1
fi
step() {
  local name=$1
  shift
  echo "START $name $(date -u +%FT%TZ)" >> "$LOG"
  "$@" >> "$OUT/logs/$name.log" 2>&1
  local rc=$?
  echo "END $name rc=$rc $(date -u +%FT%TZ)" >> "$LOG"
  return $rc
}
PPLX="uv run --frozen --with pillow --with torchvision==0.29.* python scripts/osler_v2/o2_run_pplx_zeroshot.py"
step cg_pplx $PPLX --scope mix_v2 --items "$CG" --out-root "$OUT/zeroshot_pplx_cg" || echo "cg_pplx failed" >> "$LOG"
step pool_pplx $PPLX --benchmarks v0.2 --items "$POOL" --out-root "$OUT/new_test/zeroshot_pplx" || echo "pool_pplx failed" >> "$LOG"
step pool_zs4b uv run --frozen python scripts/osler/o2_run_zeroshot.py --model Qwen/Qwen3.5-4B --scope all \
  --items "$POOL" --out-root "$OUT/new_test/zeroshot_qwen4b" || echo "pool_zs4b failed" >> "$LOG"
step pool_zs9b uv run --frozen python scripts/osler/o2_run_zeroshot.py --model Qwen/Qwen3.5-9B --scope all \
  --items "$POOL" --out-root "$OUT/new_test/zeroshot_qwen9b" || echo "pool_zs9b failed" >> "$LOG"
step pool_md4b uv run --frozen python scripts/osler/o2_run_meddecider.py --model lion-ai/MedDecider-4B \
  --revision 570b37090c379ad722186a79a3432a31be463fc9 --scope all --items "$POOL" \
  --out-root "$OUT/new_test/meddecider4b" --verify 50 || echo "pool_md4b failed" >> "$LOG"
step pool_md9b uv run --frozen python scripts/osler/o2_run_meddecider.py --model lion-ai/MedDecider-9B \
  --revision ec8a69da2cd96b2ddd2ff4e314ba9a40b71936c3 --scope all --items "$POOL" \
  --out-root "$OUT/new_test/meddecider9b" --verify 50 || echo "pool_md9b failed" >> "$LOG"
step pool_jev9b uv run --frozen python scripts/osler/o2_run_jev.py --repo autotrust/JEV-9B --scope all --items "$POOL" \
  --out-root "$OUT/new_test/jev9b" || echo "pool_jev9b failed" >> "$LOG"
echo "X6 QUEUE DONE $(date -u +%FT%TZ)" >> "$LOG"
bash scripts/osler_v3/x7_stage_a.sh
