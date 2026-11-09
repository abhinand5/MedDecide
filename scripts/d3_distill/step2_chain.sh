#!/usr/bin/env bash
# d3_distill step 2 + 3 chain: port parity, lambda probe, two arms, selection, final, and the diagnostics scores.
# One GPU job at a time: this chain runs the 4B student to completion, then the 9B student. Every stage skips when its
# result file already exists, so an interrupted chain can be restarted without repeating a completed stage.
#
# Usage (GPU; detached):
#   source scripts/pod_env.sh && setsid nohup scripts/d3_distill/step2_chain.sh \
#       > outputs/d3_distill/step2/logs/chain.log 2>&1 < /dev/null &
set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/pod_env.sh
PY=/workspace/venvs/d3/bin/python
OUT=outputs/d3_distill/step2
mkdir -p "$OUT/logs"
log() { echo "[$(date -u +%FT%TZ)] $*"; }

MINI=/workspace/.hf_home/hub/models--vllm-sr--d3-mini/snapshots/df99bf4ca4418bc6908ee68e9b7479af1f3b8800
FLASH=/workspace/.hf_home/hub/models--vllm-sr--d3-flash/snapshots/67e03e1bc157961c582ee7c6a35d9b15fada3dec

ARMS=13992
FINAL=63304
EVAL_ARM=3500
EVAL_FINAL=6330

student() {  # student <name> <snapshot>
  local name=$1 snap=$2 dir="$OUT/$1"
  mkdir -p "$dir/logs"
  log "=== $name: port parity ==="
  if [ ! -f "$dir/parity.json" ]; then
    "$PY" scripts/d3_distill/step2_parity.py --snapshot "$snap" --out "$dir/parity.json"
  fi
  "$PY" - "$dir/parity.json" <<'EOF'
import json, sys
d = json.load(open(sys.argv[1]))
if not d.get("gate_within_1e-4"):
    raise SystemExit(f"PARITY FAIL: max |harness-reference| = {d['single_row']['max_abs_diff']}")
print(f"parity PASS: max |harness-reference| = {d['single_row']['max_abs_diff']:.3e}")
EOF

  log "=== $name: lambda probe (200 steps, lambda=1) ==="
  if [ ! -f "$dir/probe/probe_result.json" ]; then
    "$PY" scripts/d3_distill/step2_arm.py --snapshot "$snap" --probe --steps 200 --schedule-steps "$ARMS" --seed 0 \
        --max-train-rows 0 --out "$dir/probe"
  fi
  "$PY" - "$dir/probe/probe_result.json" "$dir/lambda.json" <<'EOF'
import json, sys
probe = json.load(open(sys.argv[1]))
s0 = probe["probe_share"]
if s0 is None or not (0.0 < s0 < 1.0):
    raise SystemExit(f"the probe share {s0!r} is outside (0, 1): lambda is undefined; BLOCKED")
lam = (0.4 / 0.6) * (1.0 - s0) / s0
out = {"probe_share_s0": s0, "lambda": lam, "formula": "(0.4/0.6)*(1-s0)/s0",
       "band": [0.30, 0.50], "student": probe["cfg"]["student"]}
json.dump(out, open(sys.argv[2], "w"), indent=1)
print(f"lambda = {lam:.6f} (s0 = {s0:.6f})")
EOF
  local lam
  lam=$("$PY" -c "import json;print(json.load(open('$dir/lambda.json'))['lambda'])")

  log "=== $name: arms (13,992 steps, seeds 0 and 1, lambda $lam) ==="
  for seed in 0 1; do
    local arm="$dir/arm_s$seed"
    if [ ! -f "$arm/arm_result.json" ]; then
      "$PY" scripts/d3_distill/step2_arm.py --snapshot "$snap" --steps "$ARMS" --seed "$seed" \
          --lam "$lam" --eval-every "$EVAL_ARM" --out "$arm"
    fi
  done
  "$PY" scripts/d3_distill/step2_select.py --student "$name" \
      --arms "$dir/arm_s0" "$dir/arm_s1" --out "$dir/selection.json"

  local chosen
  chosen=$("$PY" -c "import json;d=json.load(open('$dir/selection.json'));print('' if d['selected'] is None else d['selected']['seed'])")
  log "=== $name: selection = ${chosen:-none} ==="
  if [ -z "$chosen" ]; then
    log "$name: no arm qualified; no final is trained"
    return 0
  fi

  log "=== $name: final (63,304 steps, seed $chosen) ==="
  if [ ! -f "$dir/final/arm_result.json" ]; then
    "$PY" scripts/d3_distill/step2_arm.py --snapshot "$snap" --steps "$FINAL" --seed "$chosen" \
        --lam "$lam" --eval-every "$EVAL_FINAL" --out "$dir/final"
  fi

  log "=== $name: scores (zero-shot and final on the public test splits) ==="
  if [ ! -f "$dir/zero_shot_score/aggregate.json" ]; then
    "$PY" scripts/d3_distill/score_students.py --snapshot "$snap" --sets medmcqa_val,medqa_test,knowledge_subsample \
        --out "$dir/zero_shot_score"
  fi
  if [ ! -f "$dir/final_score/aggregate.json" ]; then
    "$PY" scripts/d3_distill/score_students.py --snapshot "$snap" --checkpoint "$dir/final/best.pt" \
        --sets medmcqa_val,medqa_test,knowledge_subsample --out "$dir/final_score"
  fi
}

log "chain start; labels: $(wc -l < outputs/d3_distill/step1/labels.jsonl) rows"
if [ ! -f outputs/d3_distill/step1/label_manifest.json ]; then
  log "label_manifest.json is missing: the labelling run has not finished; wait for it, then rerun this chain"
  exit 2
fi
"$PY" - <<'EOF'
import json
m = json.load(open("outputs/d3_distill/step1/label_manifest.json"))
assert m["labels_rows"] == 200000, m["labels_rows"]
assert m["invalid_rate"] is not None and m["invalid_rate"] <= 0.02, m["invalid_rate"]
print(f"labels OK: {m['labels_rows']} rows, invalid rate {m['invalid_rate']:.4%}, sha256 {m['labels_sha256'][:16]}...")
EOF

log "=== Gate O7 baseline: zero-shot pplx-decider-flash-v1.2 on the frozen Y9 sets ==="
RUNS=outputs/d3_distill/step3/baselines
for s in knowledge panel mr_verdict; do
  if [ ! -f "$RUNS/pplx_$s/manifest.json" ]; then
    "$PY" scripts/osler_v4/y4_pplx_flash.py --items "data/bench/osler_v4/y9/$s.jsonl" --name "pplx_$s" --out-root "$RUNS"
  fi
done

student mini "$MINI"
student flash "$FLASH"
log "=== Gate O7 report ==="
"$PY" scripts/d3_distill/step3_gate.py --out loops/d3_distill/GATE_O7.md
log "chain done"
