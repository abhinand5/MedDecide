#!/usr/bin/env bash
# Reproduce every O0 output from the repo (AGENTS.md R7). Downloads are idempotent: entries already
# DONE in outputs/osler_v0/O0/fetch.json are skipped, and their pinned revisions are re-checked
# against the Hub. Run from anywhere; GPU jobs run one at a time, in this order.
#
#   source scripts/pod_env.sh && setsid nohup bash scripts/osler/o0_reproduce.sh \
#       > outputs/osler_v0/O0/logs/reproduce.log 2>&1 < /dev/null &
set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/pod_env.sh
O0=outputs/osler_v0/O0
mkdir -p "$O0/smoke" "$O0/logs"
H=/workspace/.hf_home/hub

echo "[$(date -u +%FT%TZ)] fetch (idempotent)"
uv run --frozen python scripts/osler/o0_fetch.py --only all

echo "[$(date -u +%FT%TZ)] snapshot"
uv run --frozen python scripts/osler/o0_snapshot.py

echo "[$(date -u +%FT%TZ)] smoke: MedDecider cards (their own code, main env)"
uv run --frozen python scripts/osler/o0_smoke_card.py --id md4b \
  --card "$H/models--lion-ai--MedDecider-4B/snapshots/570b37090c379ad722186a79a3432a31be463fc9/README.md" \
  --out "$O0/smoke/md4b.json"
uv run --frozen python scripts/osler/o0_smoke_card.py --id md9b \
  --card "$H/models--lion-ai--MedDecider-9B/snapshots/ec8a69da2cd96b2ddd2ff4e314ba9a40b71936c3/README.md" \
  --out "$O0/smoke/md9b.json"
uv run --frozen python scripts/osler/o0_smoke_card.py --id md27b \
  --card "$H/models--lion-ai--MedDecider-27B/snapshots/c30e188118471138300c79f2f963cbdac530e658/README.md" \
  --out "$O0/smoke/md27b.json"
uv run --frozen python scripts/osler/o0_smoke_card.py --id md31b \
  --card "$H/models--lion-ai--MedDecider-31B/snapshots/035f4543edf46202347515cda8201f1457b1d5fb/README.md" \
  --out "$O0/smoke/md31b.json"

echo "[$(date -u +%FT%TZ)] smoke: JEV (card decision-head protocol, main env)"
uv run --frozen python scripts/osler/o0_smoke_jev.py --repo autotrust/JEV-27B --card-values \
  --out "$O0/smoke/jev27b.json"
uv run --frozen python scripts/osler/o0_smoke_jev.py --repo autotrust/JEV-9B \
  --out "$O0/smoke/jev9b.json"

echo "[$(date -u +%FT%TZ)] smoke: pplx-decider (authors' code, envs/pplx27b)"
envs/pplx27b/.venv/bin/python -I scripts/osler/o0_smoke_pplx.py \
  --checkpoint "$H/models--perplexity-ai--pplx-decider-v1.1-27b/snapshots/5cd25e3f8ae59ba65d1ebea74d8a06fa4a5f60fd" \
  --out "$O0/smoke/pplx27b.json"

echo "[$(date -u +%FT%TZ)] smoke: Clef and Clef-Flash (authors' code, envs/clef)"
envs/clef/.venv/bin/python -I scripts/osler/o0_smoke_clef.py \
  --path "$H/models--Cloudflare--clef/snapshots/ed3eed331870db2eff4b0db01237128ede8a00ce" \
  --id clef --out "$O0/smoke/clef.json"
envs/clef/.venv/bin/python -I scripts/osler/o0_smoke_clef.py \
  --path "$H/models--Cloudflare--clef-flash/snapshots/fde727a287004204b7518dcc983fe64379776712" \
  --id clef_flash --out "$O0/smoke/clef_flash.json"

echo "[$(date -u +%FT%TZ)] throughput: Qwen3.5-4B, Qwen3.5-9B"
uv run --frozen python scripts/osler/o0_throughput.py --model Qwen/Qwen3.5-4B --out "$O0/throughput_4b.json"
uv run --frozen python scripts/osler/o0_throughput.py --model Qwen/Qwen3.5-9B --out "$O0/throughput_9b.json"

echo "[$(date -u +%FT%TZ)] aggregate: envs.json, throughput.json, smoke_summary.json"
uv run --frozen python scripts/osler/o0_envs.py

echo "[$(date -u +%FT%TZ)] re-derive headline numbers in a fresh process"
uv run --frozen python scripts/osler/o0_rederive.py

echo "[$(date -u +%FT%TZ)] O0 REPRODUCE DONE"
