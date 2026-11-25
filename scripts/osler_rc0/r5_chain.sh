#!/usr/bin/env bash
# osler_rc0 R5 — standardized latency / throughput / VRAM benchmark, one model per invocation (GPU; one job at a time).
#
# Run detached:
#   setsid nohup bash scripts/osler_rc0/r5_chain.sh > outputs/osler_rc0/R5/logs/chain.log 2>&1 < /dev/null &
#
# Models reachable through the two native decision-model paths are measured (Osler-4B-rc0, Osler-9B-rc0, pplx-flash,
# d3-flash, pplx-27B). The authors' single-item runners for MedDecider-4B/9B and Clef-Flash are outside the wiring
# timebox and are recorded NOT MEASURED by the merge step; no latency is imputed.
set -u
cd "$(dirname "$0")/../.." || exit 1
source scripts/pod_env.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
OUT=outputs/osler_rc0/R5
mkdir -p "$OUT/logs" "$OUT/parts"
echo "R5_CHAIN_START $(date -u +%FT%TZ)"
for model in Osler-4B-rc0 Osler-9B-rc0 pplx-flash d3-flash pplx-27B; do
  uv run --frozen python scripts/osler_rc0/r5_latency.py --model "$model" \
    --sample "$OUT/latency_sample_manifest.json" --out "$OUT/parts/$model.json" \
    >> "$OUT/logs/latency_$model.log" 2>&1
  echo "R5_MODEL_DONE $model exit=$? $(date -u +%FT%TZ)"
done
uv run --frozen python scripts/osler_rc0/r5_latency.py --merge --out "$OUT/latency.json" \
  >> "$OUT/logs/merge.log" 2>&1
echo "R5_CHAIN_DONE $(date -u +%FT%TZ)"
