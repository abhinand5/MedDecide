#!/usr/bin/env python
"""Report model accuracy on the long-record slice separately (loop task S3).

A long-record item is one whose rendered prompt exceeds a token threshold (default 8,192 under
the Qwen3.5 tokenizer). The slice is a *reported subset*: every accuracy is given for the slice
and for the rest of the same template's test items, so a reader can see whether a model's number
is carried by short records.

The per-item correctness comes from the prediction logs (deduplicated by `(run_id, item_id)` via
`meddecide.eval.predlog`), and the slice membership from `long_record_items.json`. Nothing here
re-scores anything.

Usage:
    uv run python scripts/bench/report_long_slice.py --dir outputs/student_v0/S3 \
        --slice data/bench/v0.2/long_record_items.json --out outputs/student_v0/S3/long_slice.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from meddecide.eval.predlog import read_prediction_log
from meddecide.utils.io import write_json
from meddecide.utils.provenance import utcnow


def _accuracy(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"n": 0, "accuracy": None}
    n_correct = sum(1 for row in rows if row.get("correct"))
    return {"n": len(rows), "n_correct": n_correct, "accuracy": n_correct / len(rows)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, required=True)
    parser.add_argument("--slice", type=Path,
                        default=Path("data/bench/v0.2/long_record_items.json"))
    parser.add_argument("--threshold", type=int, default=8192)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    slice_data = json.loads(args.slice.read_text())
    tokens = {item_id: int(n) for item_id, n in slice_data["prompt_tokens"].items()}
    long_ids = {item_id for item_id, n in tokens.items() if n > args.threshold}

    report: dict[str, Any] = {
        "generated_at_utc": utcnow(),
        "threshold_tokens": args.threshold,
        "n_items_with_counts": len(tokens),
        "n_long_items": len(long_ids),
        "models": {},
    }
    for path in sorted(args.dir.glob("preds_*.jsonl")):
        log = read_prediction_log(path)
        run_id = log.run_ids[-1] if log.run_ids else ""
        rows = log.rows_for_run(run_id)
        per_template: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
            lambda: {"long": [], "short": []}
        )
        unknown = 0
        for row in rows:
            item_id = str(row.get("item_id"))
            n_tokens = tokens.get(item_id)
            if n_tokens is None:
                unknown += 1
                continue
            bucket = "long" if n_tokens > args.threshold else "short"
            per_template[str(row.get("template_id"))][bucket].append(row)
        report["models"][path.stem.replace("preds_", "")] = {
            "run_id": run_id,
            "n_rows_raw": log.n_raw,
            "n_rows_unique": log.n_unique,
            "n_rows_without_token_count": unknown,
            "per_template": {
                template: {
                    "long": _accuracy(buckets["long"]),
                    "short": _accuracy(buckets["short"]),
                    "all": _accuracy(buckets["long"] + buckets["short"]),
                }
                for template, buckets in sorted(per_template.items())
            },
        }
    out = args.out or args.dir / "long_slice.json"
    write_json(out, report)
    print(json.dumps(report, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
