#!/usr/bin/env bash
# Z1 chain (osler_v6 ADVISORY §2.2): after the detached anchor scoring job (python PID 301463) exits, join the scored records
# into the soft-label file and write the Z1 summary. CPU only; does not start anything that needs Z1 to be marked DONE
# (the mix build is Z2's first step and waits for that decision). Logs under outputs/osler_v6/Z1/logs/.
set -u
source /workspace/MedDecide/scripts/pod_env.sh
cd /workspace/MedDecide
LOG=outputs/osler_v6/Z1/logs/chain.log
PID=301463
echo "$(date -u +%FT%TZ) chain waiting for anchor python PID $PID" >> "$LOG"
while kill -0 "$PID" 2>/dev/null; do sleep 60; done
echo "$(date -u +%FT%TZ) anchor job gone" >> "$LOG"
uv run --frozen python scripts/osler_v6/z1_anchor_scoring.py join >> outputs/osler_v6/Z1/logs/join.log 2>&1
echo "$(date -u +%FT%TZ) join exit $?" >> "$LOG"
uv run --frozen python scripts/osler_v6/z1_anchor_scoring.py summary > outputs/osler_v6/Z1/logs/summary.log 2>&1
echo "$(date -u +%FT%TZ) summary exit $?" >> "$LOG"
echo "$(date -u +%FT%TZ) chain done" >> "$LOG"
