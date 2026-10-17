#!/usr/bin/env bash
# osler_v2: the EHR-Rel score keys changed to 1..4 (the bench schema). Rebuild everything that reads them, in order, then
# run the EHR-Rel baselines and the W6 C0 chain. One GPU job at a time. Superseded outputs are moved, not deleted.
set -u
cd "$(dirname "$0")/../.."
source scripts/pod_env.sh
LOG=outputs/osler_v2/W6/logs/rebuild.log
mkdir -p outputs/osler_v2/W6/logs
echo "REBUILD START $(date -u +%FT%TZ)" >> "$LOG"
uv run --frozen python scripts/osler_v2/w2_ehr_rel.py >> "$LOG" 2>&1 || { echo "REBUILD STOPPED at ehr_rel" >> "$LOG"; exit 1; }
uv run --frozen python scripts/osler_v2/w5_ehr_scope.py >> "$LOG" 2>&1 || { echo "REBUILD STOPPED at ehr_scope" >> "$LOG"; exit 1; }
uv run --frozen python scripts/osler_v2/w5_assemble.py >> "$LOG" 2>&1 || { echo "REBUILD STOPPED at assemble" >> "$LOG"; exit 1; }
uv run --frozen python scripts/osler_v2/w6_replay.py --blocks C1 C2 --backbones qwen pplx >> "$LOG" 2>&1 || { echo "REBUILD STOPPED at replay" >> "$LOG"; exit 1; }
mv outputs/osler_v2/W5/baselines outputs/osler_v2/W5/superseded_ehr_keys/baselines_keys_0_3 2>/dev/null
mv outputs/osler_v2/W5/logs/chain.log outputs/osler_v2/W5/superseded_ehr_keys/chain_keys_0_3.log 2>/dev/null
echo "REBUILD DATA DONE $(date -u +%FT%TZ)" >> "$LOG"
bash scripts/osler_v2/w5_chain.sh
echo "REBUILD W5 CHAIN DONE $(date -u +%FT%TZ)" >> "$LOG"
mv outputs/osler_v2/W6/logs/chain.log outputs/osler_v2/W6/logs/chain_before_rebuild.log 2>/dev/null
bash scripts/osler_v2/w6_chain.sh
echo "REBUILD W6 CHAIN DONE $(date -u +%FT%TZ)" >> "$LOG"
