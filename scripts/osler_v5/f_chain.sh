#!/usr/bin/env bash
# osler_v5 corrected arms (ADVISORY §8.3, D56b): F2, then F0, then F2s1, one after another on the GPU (one GPU job at a time), each on the
# corrected mix (fda_route_choice_v1 dropped; the driver checks the recorded hash). Before each arm: a 2 GB headroom write. A finished
# arm keeps its arm_result.json and loses its full/ resume states. A failed arm stops the chain with its exit code (the watchdog resumes by
# hand). An arm with a full checkpoint but no arm_result.json is resumed with --resume.
#
# Usage (detached):
#   source scripts/pod_env.sh && setsid nohup bash scripts/osler_v5/f_chain.sh > outputs/osler_v5/Z2/f_chain.out 2>&1 < /dev/null &
set -u
cd /workspace/MedDecide
source scripts/pod_env.sh
LOG=outputs/osler_v5/Z2/f_chain.log
mkdir -p outputs/osler_v5/Z2
for arm in F2 F0 F2s1; do
  out=outputs/osler_v5/Z2/$arm
  mkdir -p "$out/logs"
  if [ -f "$out/arm_result.json" ]; then
    echo "$(date -u +%FT%TZ) $arm already has arm_result.json; skipped" >> "$LOG"
    continue
  fi
  if ! dd if=/dev/zero of=/workspace/tmp/osler_v5_headroom_2GB.bin bs=1M count=2048 conv=fsync 2>/dev/null; then
    echo "$(date -u +%FT%TZ) headroom test FAILED before $arm; chain stopped" >> "$LOG"
    exit 1
  fi
  rm -f /workspace/tmp/osler_v5_headroom_2GB.bin
  resume=""
  if ls "$out"/full/step_* >/dev/null 2>&1; then resume="--resume"; fi
  echo "$(date -u +%FT%TZ) $arm start $resume" >> "$LOG"
  uv run --frozen python scripts/osler_v5/run_arm.py --config "configs/osler_v5/arm_$arm.yaml" --out "$out" \
    --steps 13992 --evaluations 2 $resume >> "$out/logs/run.log" 2>&1
  code=$?
  echo "$(date -u +%FT%TZ) $arm exit $code" >> "$LOG"
  if [ "$code" -ne 0 ]; then
    echo "$(date -u +%FT%TZ) chain stopped at $arm" >> "$LOG"
    exit "$code"
  fi
  rm -rf "$out/full"
done
echo "$(date -u +%FT%TZ) chain complete" >> "$LOG"
