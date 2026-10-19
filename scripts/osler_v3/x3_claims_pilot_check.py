"""osler_v3 X3 claims pilot check (ADVISORY X3 step 3; D44(b); the operator's ruling of 2026-10-19, D-X3-17).

Reads a claims pilot's candidate file (x3_generate.py output), keeps the kept candidates, and runs the label-tracking check of the
assembler (its phrase test and thresholds, with the edit words of the labelled fact exempt, as in x3_assemble.py). The pilot PASSES
when no label-tracking phrase is flagged. It also reports the reasons (including family_filter) and the kept count, all as counts.

Writes outputs/osler_v3/X3/<name>.json (aggregates only; the flagged phrases are kept in the gitignored file, never printed here).
Usage:  source scripts/pod_env.sh && uv run --frozen python scripts/osler_v3/x3_claims_pilot_check.py \
            --pilot data/gen/osler_v3/claims_v2_pilot_train.jsonl --name claims_pilot_v2
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from meddecide.utils.io import write_json
from meddecide.utils.provenance import git_commit, utcnow

REPO = Path(__file__).resolve().parents[2]
EXEMPT = {"contradicted", "supported", "not", "mentioned", "passage", "claim"}  # the assembler's claims exemptions


def _assembler() -> Any:
    spec = importlib.util.spec_from_file_location("x3_assemble", REPO / "scripts" / "osler_v3" / "x3_assemble.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pilot", required=True, type=Path)
    ap.add_argument("--name", required=True)
    args = ap.parse_args()
    asm = _assembler()
    pilot = args.pilot if args.pilot.is_absolute() else REPO / args.pilot
    rows = asm.read_candidates(pilot)
    kept = [r for r in rows if r.get("kept")]
    texts: dict[str, list[str]] = {}
    for r in kept:
        texts.setdefault(str(r["label"]), []).append(str(r["claim"]))
    result = asm.phrase_check(texts, EXEMPT)
    report: dict[str, Any] = {
        "built_utc": utcnow(), "git_commit": git_commit(REPO), "pilot_file": str(pilot.relative_to(REPO)),
        "candidates": len(rows), "kept": len(kept), "kept_by_label": dict(Counter(str(r["label"]) for r in kept)),
        "reasons": dict(Counter(str(r.get("reason", "missing")) for r in rows)),
        "label_tracking_flagged": result["flagged_count"], "label_tracking_exempt_words": sorted(EXEMPT),
        "passes_d44b": result["flagged_count"] == 0,
    }
    write_json(REPO / "outputs" / "osler_v3" / "X3" / f"{args.name}.json", report)
    print({k: report[k] for k in ("candidates", "kept", "kept_by_label", "label_tracking_flagged", "passes_d44b")}, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
