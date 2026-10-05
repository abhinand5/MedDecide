#!/usr/bin/env bash
# Re-invoke the loop agent until loops/<loop>/STATE.md says `Loop status: STOPPED`.
#
# Usage (inside tmux on the pod):
#   source scripts/pod_env.sh
#   AGENT_CMD='<your agent CLI, taking the prompt as its last argument>' \
#     bash scripts/run_loop.sh bench_v0
#
# AGENT_CMD is the operator's harness invocation; the prompt (from KICKOFF.md) is
# appended as the final argument. Each session's output goes to
# /workspace/loop_logs/<loop>/session_<n>.log.
#
# Env knobs: MAX_SESSIONS (default 60), PAUSE_SECONDS between sessions (default 30).
set -uo pipefail

LOOP="${1:?usage: run_loop.sh <loop-name>}"
: "${AGENT_CMD:?set AGENT_CMD to your agent CLI invocation}"
MAX_SESSIONS="${MAX_SESSIONS:-60}"
PAUSE_SECONDS="${PAUSE_SECONDS:-30}"

REPO="$(cd "$(dirname "$0")/.." && pwd)"
STATE="$REPO/loops/$LOOP/STATE.md"
KICKOFF="$REPO/loops/$LOOP/KICKOFF.md"
LOGDIR="${LOGDIR:-/workspace/loop_logs/$LOOP}"
mkdir -p "$LOGDIR"

[ -f "$STATE" ] || { echo "missing $STATE"; exit 1; }
[ -f "$KICKOFF" ] || { echo "missing $KICKOFF"; exit 1; }

# The prompt is the first fenced block under "## Prompt" in KICKOFF.md.
PROMPT="$(awk '/^## Prompt/{p=1} p&&/^```/{n++; next} p&&n==1{print} n==2{exit}' "$KICKOFF")"
[ -n "$PROMPT" ] || { echo "could not extract prompt from $KICKOFF"; exit 1; }

stopped() { grep -q 'Loop status: `STOPPED`' "$STATE"; }

for ((i = 1; i <= MAX_SESSIONS; i++)); do
  cd "$REPO" && git pull --ff-only -q || true
  if stopped; then
    echo "[$(date -u +%FT%TZ)] $LOOP is STOPPED — exiting after $((i - 1)) session(s)."
    exit 0
  fi
  log="$LOGDIR/session_$(printf '%03d' "$i").log"
  echo "[$(date -u +%FT%TZ)] session $i/$MAX_SESSIONS → $log"
  # shellcheck disable=SC2086
  $AGENT_CMD "$PROMPT" >"$log" 2>&1
  echo "[$(date -u +%FT%TZ)] session $i exited with $?"
  sleep "$PAUSE_SECONDS"
done
echo "[$(date -u +%FT%TZ)] hit MAX_SESSIONS=$MAX_SESSIONS without STOPPED — check $STATE."
