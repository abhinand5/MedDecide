"""V8 keep rule and knowledge check (ADVISORY V8). Dev only: paired bootstrap within templates, 1,000 resamples, seed 0.

  --candidate X --current C   keep X against the last kept block C (margins: meddecide.train.blocks.KEEP_MARGINS)
  --knowledge-check B         B's knowledge dev minus zero-shot Qwen3.5-4B on the same items (lower bound above -0.02)

Reads the fast dev predictions the block runs wrote (outputs/osler_v1/V8/<block>/dev_predictions.jsonl) and, for the knowledge
check, the zero-shot predictions on data/bench/osler_mr_v0/fastdev_knowledge.jsonl. The zero-shot cells pass the D12 readout
gate or the check is reported as NOT MEASURED; no zero-shot accuracy is written for a failing cell. The decisions are computed
in meddecide.train.blocks. Writes aggregates to outputs/osler_v1/V8/keep/. No test set, the external panel or the MR test
suite is read.

Usage:
    source scripts/pod_env.sh && uv run --frozen python scripts/osler_v1/v8_keep.py --candidate B1 --current B0
    source scripts/pod_env.sh && uv run --frozen python scripts/osler_v1/v8_keep.py --knowledge-check B0
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

from meddecide.eval.o2_metrics import cell_health
from meddecide.train.blocks import keep_comparison, knowledge_comparison
from meddecide.utils.io import iter_jsonl, write_json
from meddecide.utils.provenance import git_commit, utcnow

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "outputs" / "osler_v1" / "V8"
ZS_PREDICTIONS = OUT / "knowledge_zs" / "qwen35-4b" / "predictions.jsonl"
ZS_LABEL = "zero-shot Qwen3.5-4B"


def keep_report(candidate: str, current: str) -> dict[str, Any]:
    x_rows = list(iter_jsonl(OUT / candidate / "dev_predictions.jsonl"))
    c_rows = list(iter_jsonl(OUT / current / "dev_predictions.jsonl"))
    return {"kind": "osler_v1_V8_keep_rule", "written_utc": utcnow(), "git_commit": git_commit(REPO),
            **keep_comparison(x_rows, c_rows, candidate=candidate, current=current)}


def knowledge_report(block: str) -> dict[str, Any]:
    block_rows = list(iter_jsonl(OUT / block / "dev_predictions.jsonl"))
    zs_rows = list(iter_jsonl(ZS_PREDICTIONS))
    skipped = [r["item_id"] for r in zs_rows if r["status"] == "skipped"]
    if skipped:  # refused before the readout gate reads the rows (a skipped row has no probabilities)
        raise ValueError(f"the zero-shot run skipped {len(skipped)} knowledge items: decide before comparing")
    by_template: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in zs_rows:
        by_template[r["template_id"]].append(r)
    health = {t: cell_health(ZS_LABEL, t, rows, zero_shot=True)["status"] for t, rows in by_template.items()}
    return {"kind": "osler_v1_V8_knowledge_check", "written_utc": utcnow(), "git_commit": git_commit(REPO),
            **knowledge_comparison(block_rows, zs_rows, block=block, comparison=ZS_LABEL, health=health)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--candidate")
    parser.add_argument("--current")
    parser.add_argument("--knowledge-check")
    args = parser.parse_args()
    keep_dir = OUT / "keep"
    if args.knowledge_check:
        report = knowledge_report(args.knowledge_check)
        path = keep_dir / f"knowledge_check_{args.knowledge_check}.json"
    elif args.candidate and args.current:
        report = keep_report(args.candidate, args.current)
        path = keep_dir / f"{args.candidate}_vs_{args.current}.json"
    else:
        raise SystemExit("give --candidate with --current, or --knowledge-check")
    write_json(path, report)
    print(f"wrote {path.relative_to(REPO)}", flush=True)


if __name__ == "__main__":
    main()
