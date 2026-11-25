#!/usr/bin/env bash
# Launch one osler_v6 arm/final at the frozen Z2 λ, detached (one GPU job at a time). Reads λ from the probe record,
# refuses when the probe is not DONE or the arm already has an arm_result.json, and leaves the log and the PID on stdout.
#
# Usage: bash scripts/osler_v6/launch_arm.sh <name> <config> <mix> <mix-manifest> <out-dir> <steps> <evaluations>
#   e.g. bash scripts/osler_v6/launch_arm.sh G1s0 configs/osler_v5/arm_F2.yaml \
#            data/train/osler_v6/mix_f2_distill.jsonl outputs/osler_v6/Z2/mix_manifest.json \
#            outputs/osler_v6/Z3/G1s0 13992 2
set -eu
source /workspace/MedDecide/scripts/pod_env.sh
cd /workspace/MedDecide
NAME=$1; CONFIG=$2; MIX=$3; MIX_MANIFEST=$4; OUT=$5; STEPS=$6; EVALUATIONS=$7
PROBE=outputs/osler_v6/Z2/probe/probe.json
if [ ! -f "$PROBE" ]; then echo "no Z2 probe at $PROBE: run the probe first" >&2; exit 1; fi
LAMBDA=$(uv run --frozen python -c "import json; d = json.load(open('$PROBE')); assert d.get('status') == 'DONE', 'the probe is not DONE: ' + str(d.get('status')); print(d['lambda']['value'])")
test -n "$LAMBDA"
echo "arm $NAME: frozen lambda $LAMBDA; config $CONFIG; mix $MIX; steps $STEPS; evaluations $EVALUATIONS"
if [ -e "$OUT/arm_result.json" ]; then echo "refuse: $OUT/arm_result.json exists; an arm is not run twice" >&2; exit 1; fi
if [ ! -f "$MIX" ] || [ ! -f "$MIX_MANIFEST" ]; then echo "the mix or its manifest is missing" >&2; exit 1; fi
mkdir -p "$OUT/logs"
setsid nohup uv run --frozen python scripts/osler_v6/run_arm.py \
    --config "$CONFIG" --mix "$MIX" --mix-manifest "$MIX_MANIFEST" --distill-weight "$LAMBDA" --min-labelled 3 \
    --out "$OUT" --steps "$STEPS" --evaluations "$EVALUATIONS" \
    > "$OUT/logs/run.log" 2>&1 < /dev/null &
echo "launched PID $!; log $OUT/logs/run.log"
