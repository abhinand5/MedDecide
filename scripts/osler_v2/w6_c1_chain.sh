#!/usr/bin/env bash
# osler_v2 W6, after the base choice (pplx, D39): C1 on pplx-decider-flash-v1.2 once the pplx test-side scoring has finished;
# the keep rule C1 against C0p; then C2 on the last kept block only when C1 is kept (C2 = C1 + prose). If C1 is not kept, the
# chain stops and records it: C2 would then be C0 + prose, a file not built yet (ADVISORY W6: "last kept + prose").
set -u
cd "$(dirname "$0")/../.."
source scripts/pod_env.sh
LOG=outputs/osler_v2/W6/logs/chain.log
until grep -q "W5 PPLX TEST DONE" outputs/osler_v2/W5/logs/pplx_test.log 2>/dev/null; do sleep 60; done
PPLX_RUN="uv run --frozen --with torchvision==0.29.* --with pillow python scripts/osler_v2/w6_block_run.py"
echo "START C1_pplx $(date -u +%FT%TZ)" >> "$LOG"
$PPLX_RUN --block C1 --backbone pplx >> outputs/osler_v2/W6/logs/C1_pplx.log 2>&1
echo "END C1_pplx rc=$? $(date -u +%FT%TZ)" >> "$LOG"
uv run --frozen python scripts/osler_v2/w6_keep.py --candidate C1_pplx --current C0_pplx >> outputs/osler_v2/W6/logs/keep_C1.log 2>&1
KEPT=$(python3 -c "import json; print(json.load(open('outputs/osler_v2/W6/keep/C1_pplx_vs_C0_pplx.json'))['verdict']['kept'])")
echo "KEEP C1 vs C0p: $KEPT $(date -u +%FT%TZ)" >> "$LOG"
if [ "$KEPT" = "True" ]; then
  echo "START C2_pplx $(date -u +%FT%TZ)" >> "$LOG"
  $PPLX_RUN --block C2 --backbone pplx >> outputs/osler_v2/W6/logs/C2_pplx.log 2>&1
  echo "END C2_pplx rc=$? $(date -u +%FT%TZ)" >> "$LOG"
  uv run --frozen python scripts/osler_v2/w6_keep.py --candidate C2_pplx --current C1_pplx >> outputs/osler_v2/W6/logs/keep_C2.log 2>&1
  echo "W6 C2 DONE $(date -u +%FT%TZ)" >> "$LOG"
else
  echo "W6 C1 NOT KEPT: C2 needs C0 + prose (not built); see the chain log and the keep report $(date -u +%FT%TZ)" >> "$LOG"
fi
