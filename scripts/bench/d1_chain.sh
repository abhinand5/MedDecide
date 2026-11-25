#!/usr/bin/env bash
# d1-2B over the five frozen sets, one GPU job at a time (bench/d1-2b). Each step must exit 0; a failure stops the chain.
# Usage: setsid nohup bash scripts/bench/d1_chain.sh > outputs/d1_bench/logs/chain.log 2>&1 < /dev/null &
set -euo pipefail
cd "$(dirname "$0")/../.."
export HF_HOME=/workspace/.hf_home UV_CACHE_DIR=/workspace/.uv_cache TMPDIR=/workspace/tmp HF_HUB_OFFLINE=1
for set in mednli_test knowledge panel mr_verdict mimic_suite; do
  echo "[$(date -u +%FT%TZ)] START $set"
  uv run --frozen --no-sync python scripts/bench/d1_score.py --set "$set" > "outputs/d1_bench/logs/$set.log" 2>&1
  echo "[$(date -u +%FT%TZ)] END $set exit=0"
done
echo "[$(date -u +%FT%TZ)] CHAIN COMPLETE"
