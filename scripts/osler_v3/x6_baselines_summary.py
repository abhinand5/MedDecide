"""osler_v3 X6 step 4 (ADVISORY X6): the five baselines and zero-shot pplx-decider-flash-v1.2 on the new test pool. ADDITIONAL ROWS ONLY.

The pool is data/bench/osler_v3/new_test_o2.jsonl (X1 test and the kept X4 test templates; 9,016 items). For each model this reads
its predictions file and reports the coverage (rows, scored, skipped and the recorded reasons), the accuracy on the scored items
(meddecide.eval.o2_metrics.score_row), and the accuracy per template. A skipped item is never counted as wrong: it is counted as
skipped, and the comparison of a model on a template with skipped items is labelled. D44(a): a comparison that cannot pair more
than 1 % of its items is NOT MEASURED for the Gate O3 verdict; this summary is additional, never a verdict.

Writes outputs/osler_v3/X6/baselines_summary.json (aggregates only; the predictions stay gitignored) and prints a table.

Usage:  source scripts/pod_env.sh && uv run --frozen python scripts/osler_v3/x6_baselines_summary.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from meddecide.eval.o2_metrics import score_row
from meddecide.utils.io import write_json
from meddecide.utils.provenance import git_commit, utcnow

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "outputs" / "osler_v3" / "X6" / "new_test"
OUT = REPO / "outputs" / "osler_v3" / "X6" / "baselines_summary.json"
MODELS = {
    "zero-shot pplx-decider-flash-v1.2": "zeroshot_pplx/pplx-decider-flash-v1.2/predictions.jsonl",
    "zero-shot Qwen3.5-4B": "zeroshot_qwen4b/qwen35-4b/predictions.jsonl",
    "zero-shot Qwen3.5-9B": "zeroshot_qwen9b/qwen35-9b/predictions.jsonl",
    "MedDecider-4B": "meddecider4b/meddecider-4b/predictions.jsonl",
    "MedDecider-9B": "meddecider9b/meddecider-9b/predictions.jsonl",
    "JEV-9B": "jev9b/jev-9b/predictions.jsonl",
}


def summarise(path: Path, pool_ids: set[str]) -> dict[str, Any]:
    rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    ids = {str(r["item_id"]) for r in rows}
    scored = [r for r in rows if r["status"] == "scored"]
    correct = {str(r["item_id"]): bool(score_row(r).correct) for r in scored}
    by_template: dict[str, list[bool]] = defaultdict(list)
    for r in scored:
        by_template[str(r["template_id"])].append(correct[str(r["item_id"])])
    return {
        "file": str(path.relative_to(REPO)), "rows": len(rows), "unique_items": len(ids),
        "items_missing_from_pool": len(pool_ids - ids), "scored": len(scored),
        "skipped": len(rows) - len(scored), "skipped_reasons": dict(Counter(str(r.get("reason")) for r in rows
                                                                          if r["status"] != "scored")),
        "accuracy_on_scored": round(sum(correct.values()) / len(correct), 4) if correct else None,
        "by_template": {t: {"scored": len(v), "accuracy": round(sum(v) / len(v), 4)}
                        for t, v in sorted(by_template.items())},
    }


def main() -> int:
    pool = [json.loads(line) for line in (REPO / "data" / "bench" / "osler_v3" / "new_test_o2.jsonl").open(encoding="utf-8")
            if line.strip()]
    pool_ids = {str(r["item_id"]) for r in pool}
    report: dict[str, Any] = {"built_utc": utcnow(), "git_commit": git_commit(REPO), "pool_items": len(pool_ids),
                              "note": "additional rows only; never in a Gate O3 verdict (ADVISORY X6, X10)", "models": {}}
    for name, rel in MODELS.items():
        report["models"][name] = summarise(ROOT / rel, pool_ids)
        m = report["models"][name]
        print(f"{name}: rows {m['rows']}, scored {m['scored']}, skipped {m['skipped']}, "
              f"accuracy on scored {m['accuracy_on_scored']}", flush=True)
    write_json(OUT, report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
