#!/usr/bin/env bash
# Y9 repair (osler_v4): the zero-shot pplx-decider steps (step 2: seven non-MIMIC sets; step 7: the pplx-27B ceiling on nine sets),
# rerun with the flags the Y4 chain used (--with pillow==12.3.0 --with "torchvision==0.29.*"; the venv has no torchvision, and the
# lock does not declare it). The first chain's pplx steps failed at processor load (ImportError, no outputs), and the first chain's
# step lines recorded exit 0 for every step (bash expands $(date) before $?); this script captures the exit code right after the
# command and checks every output's row count against its item file. One GPU job at a time; started only after the first chain exits.
set -u
source /workspace/MedDecide/scripts/pod_env.sh
cd /workspace/MedDecide
LOG=outputs/osler_v4/Y9/logs/y9_chain_repair.log
SETS=data/bench/osler_v4/y9
RUNS=outputs/osler_v4/Y9/baselines
PPLX27_REV=cce0f3f80033e9d89d80831d5281fd72124aacce
PPLX27="$HF_HOME/hub/models--perplexity-ai--pplx-decider-v1.2-27b/snapshots/$PPLX27_REV"
NOT_MIMIC="mr_verdict mr_reported mr_heldout_test panel knowledge mednli_test pairs_test"
ALL="mr_verdict mr_reported mr_heldout_test panel knowledge mednli_test pairs_test mimic_suite mimic_heldout"

step() {
  local name="$1"; shift
  echo "$(date -u +%FT%TZ) start $name" >> "$LOG"
  uv run --frozen --with pillow==12.3.0 --with "torchvision==0.29.*" python "$@" >> "outputs/osler_v4/Y9/logs/$name.log" 2>&1
  local rc=$?
  echo "$(date -u +%FT%TZ) end $name exit $rc" >> "$LOG"
}

check() {
  # rows written against rows expected (one JSON line per item); a mismatch is recorded, never skipped
  local name="$1" items="$2"
  local want got
  want=$(wc -l < "$SETS/$items.jsonl")
  got=$(wc -l < "$RUNS/$name/predictions.jsonl" 2>/dev/null || echo 0)
  echo "$(date -u +%FT%TZ) rows $name got $got want $want" >> "$LOG"
}

echo "$(date -u +%FT%TZ) repair start" >> "$LOG"

# 2. zero-shot pplx-decider-flash-v1.2 on the seven non-MIMIC sets (the Y4 MIMIC runs exist and are not repeated)
for s in $NOT_MIMIC; do
  step "pplx_$s" scripts/osler_v4/y4_pplx_flash.py --items "$SETS/$s.jsonl" --name "pplx_$s" --out-root "$RUNS"
  check "pplx_$s" "$s"
done

# 7. the pplx-27B ceiling on the nine sets (additional; no verdict)
if [ -d "$PPLX27" ]; then
  for s in $ALL; do
    step "pplx27b_$s" scripts/osler_v4/y4_pplx_flash.py --checkpoint "$PPLX27" --revision "$PPLX27_REV" \
      --label "pplx-decider-v1.2-27b (zero-shot, authors' code; ceiling, additional)" --items "$SETS/$s.jsonl" --name "pplx27b_$s" --out-root "$RUNS"
    check "pplx27b_$s" "$s"
  done
else
  echo "$(date -u +%FT%TZ) pplx-27B snapshot missing: step 7 NOT RUN ($PPLX27)" >> "$LOG"
fi
echo "$(date -u +%FT%TZ) repair done" >> "$LOG"
