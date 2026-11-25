#!/usr/bin/env bash
# V2 measurement matrix (student_v1). Waits for a scoring job to write its "exit=" marker, then
# runs one GPU job at a time:
#   1. padding probe on the committed HEAD code (exported copy, d3cc79c) and on the fixed working
#      tree, each in strict fp32 and in the as-run mode (cuDNN TF32 convolutions, PyTorch default)
#   2. shape-noise control on the fixed code, in both precision modes
#   3. batch-16 test scoring of the fixed code, with the same arguments as the student_v0 run
#      (its batch plan depends on real token counts only, so the batches are the same)
# Usage: scripts/student/v2_run_matrix.sh <log file whose "exit=" line ends the wait>
set -u
ROOT=/workspace/MedDecide
cd "$ROOT" || exit 1
source scripts/pod_env.sh

PY="$ROOT/.venv/bin/python"
OUT=outputs/student_v1/V2
CKPT="$ROOT/outputs/student_v0/S9_run3/checkpoints/step_1000"
BENCH="$ROOT/data/bench/v0.2"
HEAD_COPY=/workspace/tmp/head_d3cc79c
WAIT_LOG="${1:?usage: v2_run_matrix.sh <log to wait for>}"

until grep -q '^exit=' "$WAIT_LOG" 2>/dev/null; do sleep 30; done
echo "$(date -u +%FT%TZ) matrix start (after $WAIT_LOG)"

# Provenance for the fixed-code runs: HEAD, dirty flag and a hash of the uncommitted diff (R7).
"$PY" - "$OUT/code_state.json" <<'EOF'
import hashlib, json, subprocess, sys
from datetime import datetime, timezone
def git(*args):
    return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout
# hash the tracked diff and the bytes of every untracked file, so the code state is fully pinned
untracked = sorted(git("ls-files", "--others", "--exclude-standard").splitlines())
digest = hashlib.sha256(git("diff").encode())
for name in untracked:
    digest.update(name.encode() + b"\0" + open(name, "rb").read())
state = {
    "head": git("rev-parse", "HEAD").strip(),
    "dirty_files": sorted(git("status", "--porcelain").splitlines()),
    "code_sha256": digest.hexdigest(),
    "recorded_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
}
open(sys.argv[1], "w").write(json.dumps(state, indent=2) + "\n")
print(json.dumps({k: state[k] for k in ("head", "code_sha256")}))
EOF

run() {
  local name=$1; shift
  echo "$(date -u +%FT%TZ) begin $name"
  "$@" > "$OUT/logs/matrix_$name.log" 2>&1
  echo "exit=$?" >> "$OUT/logs/matrix_$name.log"
  echo "$(date -u +%FT%TZ) end $name"
}

# The HEAD copy has no .git; it runs the same probe script, copied in, against its own src/.
cp scripts/student/v2_padding_probe.py "$HEAD_COPY/scripts/student/"

run probe_head_strict "$PY" "$HEAD_COPY/scripts/student/v2_padding_probe.py" \
  --checkpoint "$CKPT" --bench "$BENCH" --code-label d3cc79c --out "$OUT/probe_head_strict_fp32.json"
run probe_head_tf32conv "$PY" "$HEAD_COPY/scripts/student/v2_padding_probe.py" \
  --checkpoint "$CKPT" --bench "$BENCH" --code-label d3cc79c --tf32-conv \
  --out "$OUT/probe_head_tf32conv_fp32.json"
run probe_fixed_strict "$PY" scripts/student/v2_padding_probe.py \
  --checkpoint "$CKPT" --bench "$BENCH" --out "$OUT/probe_fixed_strict_fp32.json"
run probe_fixed_tf32conv "$PY" scripts/student/v2_padding_probe.py \
  --checkpoint "$CKPT" --bench "$BENCH" --tf32-conv --out "$OUT/probe_fixed_tf32conv_fp32.json"
run shape_fixed_strict "$PY" scripts/student/v2_shape_control.py \
  --checkpoint "$CKPT" --bench "$BENCH" --out "$OUT/shape_control_strict_fp32.json"
run shape_fixed_tf32conv "$PY" scripts/student/v2_shape_control.py \
  --checkpoint "$CKPT" --bench "$BENCH" --tf32-conv --out "$OUT/shape_control_tf32conv_fp32.json"
run score_fixed_b16 uv run python scripts/bench/run_student.py \
  --checkpoint outputs/student_v0/S9_run3/best --split test --out "$OUT/fixed_b16" --tag test

echo "$(date -u +%FT%TZ) matrix done"
