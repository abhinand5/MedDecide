#!/usr/bin/env bash
# O2 chain: every remaining scoreboard run, one GPU job at a time, in a fixed order (ADVISORY O2).
# Each step is resumable (its prediction file is appended, never rewritten); a failed step is recorded in
# chain.log and the chain continues, so one blocked model does not block the others (GOAL: BLOCKED is per task).
#
#   source scripts/pod_env.sh && setsid nohup bash scripts/osler/o2_chain.sh \
#       > outputs/osler_v0/O2/logs/chain_stdout.log 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")/../.."
source scripts/pod_env.sh
O2=outputs/osler_v0/O2
H=/workspace/.hf_home/hub
mkdir -p "$O2/logs"
LOG="$O2/logs/chain.log"
# one GPU job at a time: wait for the zero-shot 4B job started by hand (PID passed as $1, optional)
if [ -n "${1:-}" ]; then
  while kill -0 "$1" 2>/dev/null; do sleep 15; done
  echo "[$(date -u +%FT%TZ)] PID $1 finished; chain starts" | tee -a "$LOG"
fi

run() {
  local name=$1; shift
  echo "[$(date -u +%FT%TZ)] START $name" | tee -a "$LOG"
  "$@" > "$O2/logs/$name.log" 2>&1
  local rc=$?
  echo "[$(date -u +%FT%TZ)] END $name rc=$rc" | tee -a "$LOG"
}

run md4b uv run --frozen python scripts/osler/o2_run_meddecider.py \
  --model lion-ai/MedDecider-4B --revision 570b37090c379ad722186a79a3432a31be463fc9 --verify 50
run qwen9b uv run --frozen python scripts/osler/o2_run_zeroshot.py \
  --model Qwen/Qwen3.5-9B --scope panel_robustness
run jev9b uv run --frozen python scripts/osler/o2_run_jev.py \
  --repo autotrust/JEV-9B --scope panel_robustness --verify 20
run md9b uv run --frozen python scripts/osler/o2_run_meddecider.py \
  --model lion-ai/MedDecider-9B --revision ec8a69da2cd96b2ddd2ff4e314ba9a40b71936c3 --verify 50
run clef_flash envs/clef/.venv/bin/python -I scripts/osler/o2_run_clef.py \
  --path "$H/models--Cloudflare--clef-flash/snapshots/fde727a287004204b7518dcc983fe64379776712" \
  --id clef_flash --scope all
run pplx27b envs/pplx27b/.venv/bin/python -I scripts/osler/o2_run_pplx.py \
  --checkpoint "$H/models--perplexity-ai--pplx-decider-v1.1-27b/snapshots/5cd25e3f8ae59ba65d1ebea74d8a06fa4a5f60fd" \
  --scope all
run jev27b uv run --frozen python scripts/osler/o2_run_jev.py \
  --repo autotrust/JEV-27B --scope all --verify 20
run clef envs/clef/.venv/bin/python -I scripts/osler/o2_run_clef.py \
  --path "$H/models--Cloudflare--clef/snapshots/ed3eed331870db2eff4b0db01237128ede8a00ce" \
  --id clef --scope all
run md27b uv run --frozen python scripts/osler/o2_run_meddecider.py \
  --model lion-ai/MedDecider-27B --revision c30e188118471138300c79f2f963cbdac530e658 --verify 50
run md31b uv run --frozen python scripts/osler/o2_run_meddecider.py \
  --model lion-ai/MedDecider-31B --revision 035f4543edf46202347515cda8201f1457b1d5fb --verify 50
run metrics uv run --frozen python scripts/osler/o2_metrics.py

echo "[$(date -u +%FT%TZ)] O2 CHAIN DONE" | tee -a "$LOG"
