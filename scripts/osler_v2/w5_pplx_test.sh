#!/usr/bin/env bash
# osler_v2 W5 step 3 (ADVISORY W5): zero-shot pplx-decider-flash-v1.2 on the MR suite (test and held-out), the external panel and
# the knowledge test items (MedQA and MedMCQA, v0.2), through its authors' code. Used only at W8, and only as the substitute
# baseline if D39 chooses pplx. Runs after the W6 C0 stage so that one GPU job runs at a time.
set -u
cd "$(dirname "$0")/../.."
source scripts/pod_env.sh
LOG=outputs/osler_v2/W5/logs/pplx_test.log
until grep -q "W6 C0 STAGE DONE" outputs/osler_v2/W6/logs/chain.log 2>/dev/null; do sleep 60; done
PPLX="uv run --frozen --with pillow --with torchvision==0.29.* python scripts/osler_v2/o2_run_pplx_zeroshot.py"
echo "START pplx_mr $(date -u +%FT%TZ)" >> "$LOG"
$PPLX --benchmarks mr_v0 --items data/bench/osler_mr_v0/mr_eval.jsonl --out-root outputs/osler_v2/W5/pplx_test/mr >> "$LOG" 2>&1
echo "END pplx_mr rc=$? $(date -u +%FT%TZ)" >> "$LOG"
echo "START pplx_panel $(date -u +%FT%TZ)" >> "$LOG"
$PPLX --benchmarks ext_panel --items data/bench/v0.3_ext/o2_items.jsonl --out-root outputs/osler_v2/W5/pplx_test/panel >> "$LOG" 2>&1
echo "END pplx_panel rc=$? $(date -u +%FT%TZ)" >> "$LOG"
echo "START pplx_knowledge $(date -u +%FT%TZ)" >> "$LOG"
$PPLX --benchmarks v0.2 --sets medqa medmcqa --items data/bench/v0.3_ext/o2_items.jsonl --out-root outputs/osler_v2/W5/pplx_test/knowledge >> "$LOG" 2>&1
echo "END pplx_knowledge rc=$? $(date -u +%FT%TZ)" >> "$LOG"
echo "W5 PPLX TEST DONE $(date -u +%FT%TZ)" >> "$LOG"
