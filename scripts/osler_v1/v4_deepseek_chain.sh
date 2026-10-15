#!/usr/bin/env bash
# V4 second endpoint (osler_v1 ADVISORY V4; operator instruction; STATE V4-D7).
# 1. The OpenCode Go narrative build runs until its weekly guard ($36) binds. A stop on the 5-hour window waits 15 minutes
#    and resumes; any other stop ends the chain. A build already running when this script starts is waited for (its pid).
# 2. When the weekly guard binds, the narrative build resumes on the DeepSeek API (api.deepseek.com, model deepseek-flash:
#    the operator's name for the V4.1 Flash model) with the same generator, prompts and verification, ONLY if the approval file
#    outputs/osler_v1/V4/DEEPSEEK_APPROVED exists (D30 requires a zero-data-retention hosted endpoint; that is an open question).
#    Without the file, V4 closes on the OpenCode pages at the guard. With it, the build runs in chunks of 200 encounter indices,
#    with its own ledger (spend_ledger_deepseek.jsonl, total cap $9.00). Before each chunk the account balance is read
#    (GET /user/balance) and the phase stops below $0.40.
# 3. The chain then closes V4 (v4_closure.py --finalize), which writes FINAL.json and releases the B2/B3 queue.
# The key is read from the DEEPSEEK_API_KEY export in ~/.bashrc; it is never printed or written to a file.
# Launch detached (pass the pid of a running OpenCode build, or nothing):
#     setsid nohup bash scripts/osler_v1/v4_deepseek_chain.sh [PID] > /dev/null 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/../.."
source scripts/pod_env.sh
TOTAL=28600
CHUNK=200
FLOOR_USD=0.40
DS_CAP_USD=9.00
LOG=outputs/osler_v1/V4/logs/deepseek_chain.log
OC_LOG=outputs/osler_v1/V4/logs/build_narrative.log
DS_LOG=outputs/osler_v1/V4/logs/build_narrative_deepseek.log
DS_LEDGER=outputs/osler_v1/V4/spend_ledger_deepseek.jsonl
APPROVAL=outputs/osler_v1/V4/DEEPSEEK_APPROVED
mkdir -p outputs/osler_v1/V4/logs
say() { echo "$* $(date -u +%FT%TZ)" >> "$LOG"; }

# the stop reason, which a build prints as the last line of its log ('' when the run did not finish)
stop_reason() {
  tail -n 1 "$1" | uv run --frozen python -c '
import json, sys
try:
    print(json.loads(sys.stdin.read()).get("stop_reason", ""))
except ValueError:
    print("")
'
}

# the account balance in USD (the API's total_balance); prints the number, or nothing when it cannot be read
balance() {
  uv run --frozen python -c '
from meddecide.teacher.client import TeacherClient
body = TeacherClient.from_env().get("user/balance").json()
print((body.get("balance_infos") or [{}])[0].get("total_balance", ""))
' 2>/dev/null
}

# exit 0 when the balance in $1 is at least the floor
above_floor() {
  uv run --frozen python -c '
import sys
try:
    sys.exit(0 if float(sys.argv[1]) >= float(sys.argv[2]) else 1)
except ValueError:
    sys.exit(1)
' "$1" "$FLOOR_USD"
}

# one OpenCode Go build pass (resumable); prints its stop reason
run_opencode() {
  say "BUILD narrative (OpenCode Go) start=0 count=$TOTAL"
  uv run --frozen python scripts/osler_v1/v4_teacher_build.py --mode build --gen narrative --start 0 --count "$TOTAL" \
    >> "$OC_LOG" 2>&1
  local reason
  reason=$(stop_reason "$OC_LOG")
  say "END narrative (OpenCode Go) stop=$reason"
  echo "$reason"
}

# closes V4 with a reason; FINAL.json releases the B2 and B3 queue
finalize() {
  say "CLOSING V4: $1"
  uv run --frozen python scripts/osler_v1/v4_closure.py --finalize --reason "$1" >> outputs/osler_v1/V4/logs/closure.log 2>&1
  local rc=$?
  say "CLOSURE rc=$rc (outputs/osler_v1/V4/logs/closure.log)"
}

PID=${1:-}
if [[ -n "$PID" ]]; then
  say "WAIT for the running OpenCode build (pid $PID)"
  while kill -0 "$PID" 2>/dev/null; do sleep 60; done
  reason=$(stop_reason "$OC_LOG")
  say "OpenCode build (pid $PID) exited; stop=$reason"
else
  reason=$(run_opencode)
fi
while [[ "$reason" != *"week at"* ]]; do
  case "$reason" in
    "spend guard"*) say "5-hour window: waiting 900 s before the next OpenCode pass"; sleep 900; reason=$(run_opencode) ;;
    completed) finalize "narrative build complete on OpenCode Go (28,600 encounters); the DeepSeek API was not needed"; exit 0 ;;
    *) say "CHAIN STOPPED on OpenCode Go: '$reason'"; exit 1 ;;
  esac
done

# the DeepSeek API is used only with an explicit approval file: its zero-data-retention status (D30) is an open question
if [[ ! -f "$APPROVAL" ]]; then
  say "HOLD: the OpenCode Go weekly guard is reached ($reason); no approval file for the DeepSeek API; V4 closes on the OpenCode pages"
  finalize "OpenCode Go weekly guard reached ($reason). The DeepSeek API continuation (operator instruction, V4-D7) is held: its zero-data-retention status under D30 is an open operator question; V4 closes on the verified OpenCode pages (operator instruction Q8)"
  exit 0
fi

say "SWITCH: the OpenCode Go weekly guard is reached ($reason); the narrative build continues on the DeepSeek API (V4-D7; approval file present)"
eval "$(grep '^export DEEPSEEK_API_KEY' ~/.bashrc)"
if [[ -z "${DEEPSEEK_API_KEY:-}" ]]; then
  say "STOPPED: DEEPSEEK_API_KEY is not set"
  exit 1
fi
unset OPENAI_API_BASE OPENAI_BASE_URL OPENAI_API_KEY OPENAI_MODEL TEACHER_LOCAL TEACHER_EXTRA_HEADERS
export TEACHER_BASE_URL=https://api.deepseek.com TEACHER_MODEL=deepseek-flash TEACHER_API_KEY="$DEEPSEEK_API_KEY"
export TEACHER_THINKING=disabled

start=0
reason="not started"
bal=$(balance)
say "DeepSeek balance before the phase: ${bal:-unreadable} USD (ledger cap \$$DS_CAP_USD)"
while (( start < TOTAL )); do
  bal=$(balance)
  if [[ -z "$bal" ]] || ! above_floor "$bal"; then
    reason="the balance ${bal:-unreadable} USD is below the floor \$$FLOOR_USD"
    break
  fi
  uv run --frozen python scripts/osler_v1/v4_teacher_build.py --mode build --gen narrative --start "$start" --count "$CHUNK" \
    --ledger "$DS_LEDGER" --total-cap "$DS_CAP_USD" --tag _deepseek >> "$DS_LOG" 2>&1
  reason=$(stop_reason "$DS_LOG")
  say "DeepSeek chunk start=$start count=$CHUNK stop=$reason balance-before-chunk=$bal USD"
  [[ "$reason" == "completed" ]] || break
  start=$((start + CHUNK))
done
bal=$(balance)
say "DeepSeek balance after the phase: ${bal:-unreadable} USD"
if (( start >= TOTAL )); then
  finalize "narrative build reached the 28,600-encounter target: OpenCode Go to its weekly guard, then the DeepSeek API (V4-D7)"
else
  finalize "OpenCode Go weekly guard reached; the DeepSeek API phase stopped in the chunk at index $start: $reason"
fi
say "DEEPSEEK PHASE END"
