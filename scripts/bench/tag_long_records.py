#!/usr/bin/env python
"""Tag the long-record slice of MedDecide-Bench v0.2 (loop task S3).

A *long record* is a benchmark item whose rendered prompt exceeds 8,192 tokens under the
Qwen3.5 tokenizer — the length at which the previous loop's 9B baseline started allocating
30 GB and skipping items. Real records are long and carry many checks each, so the slice is
reported separately rather than being averaged away.

The flag is recorded **alongside** the items, not inside them: `data/bench/v0.2/long_record.json`
lists the item ids per template and the manifest gets a `long_record` block. Rewriting the items
would break the v0.1 carried-identity property that S1 established (meta is part of the item
content hash), and the slice is a reported subset, not a change to the benchmark.

Usage:
    uv run python scripts/bench/tag_long_records.py --threshold 8192
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from meddecide.bench.schema import Item
from meddecide.eval.readout import canonicalise_options, render_prompt
from meddecide.utils.io import read_jsonl, write_json
from meddecide.utils.provenance import utcnow

TOKENIZER_ID = "Qwen/Qwen3.5-0.8B"
FRESH_DIR = Path("data/bench/v0.2/fresh")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench", type=Path, default=Path("data/bench/v0.2"))
    parser.add_argument("--tokenizer", default=TOKENIZER_ID)
    parser.add_argument("--threshold", type=int, default=8192)
    parser.add_argument("--out", type=Path, default=Path("data/bench/v0.2/long_record.json"))
    parser.add_argument("--manifest", type=Path, default=Path("data/bench/v0.2/manifest.json"))
    parser.add_argument("--per-item-out", type=Path,
                        default=Path("data/bench/v0.2/long_record_items.json"))
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    items: list[Item] = []
    for path in sorted((args.bench / "fresh").glob("*.jsonl")):
        rows, report = read_jsonl(path, Item)
        report.check_closes()
        if report.n_dropped:
            raise SystemExit(f"{path}: {report.n_dropped} unreadable rows")
        items.extend(row for row in rows if str(row.split) == "test")

    per_template_total: Counter = Counter()
    per_template_long: Counter = Counter()
    long_ids: dict[str, list[str]] = defaultdict(list)
    all_tokens: list[int] = []
    per_template_tokens: dict[str, list[int]] = defaultdict(list)
    for item in items:
        canonical, _ = canonicalise_options(item)
        prompt = render_prompt(canonical, tokenizer, "space")
        n_tokens = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
        all_tokens.append(n_tokens)
        per_template_total[item.template_id] += 1
        per_template_tokens[item.template_id].append(n_tokens)
        if n_tokens > args.threshold:
            per_template_long[item.template_id] += 1
            long_ids[item.template_id].append(item.item_id)

    tokens = np.asarray(all_tokens, dtype=np.int64)
    report: dict[str, Any] = {
        "generated_at_utc": utcnow(),
        "benchmark": str(args.bench),
        "tokenizer": args.tokenizer,
        "threshold_tokens": args.threshold,
        "n_fresh_test_items": len(items),
        "n_long_items": int((tokens > args.threshold).sum()),
        "share_long": float((tokens > args.threshold).mean()),
        "prompt_tokens": {
            "min": int(tokens.min()),
            "p50": int(np.percentile(tokens, 50)),
            "p90": int(np.percentile(tokens, 90)),
            "p99": int(np.percentile(tokens, 99)),
            "max": int(tokens.max()),
        },
        "per_template": {
            template_id: {
                "n_test": per_template_total[template_id],
                "n_long": per_template_long.get(template_id, 0),
                "share_long": (
                    per_template_long.get(template_id, 0) / per_template_total[template_id]
                ),
                "p50_tokens": int(np.percentile(per_template_tokens[template_id], 50)),
                "max_tokens": int(max(per_template_tokens[template_id])),
                "item_ids": sorted(long_ids.get(template_id, [])),
            }
            for template_id in sorted(per_template_total)
        },
        "note": (
            "the flag is recorded here and in the manifest, not inside the items: rewriting the "
            "items would change their content hash and break the v0.1 carried-identity check"
        ),
    }
    write_json(args.out, report)
    # per-item prompt-token counts: the artifact that lets any consumer recompute the slice at a
    # different threshold, and that quantifies how many items exceed the harness prompt cap
    write_json(
        args.per_item_out,
        {
            "generated_at_utc": report["generated_at_utc"],
            "tokenizer": args.tokenizer,
            "n_items": len(items),
            "prompt_tokens": {item.item_id: n for item, n in zip(items, all_tokens, strict=True)},
            "n_over_16384": int((tokens > 16384).sum()),
            "n_over_8192": int((tokens > args.threshold).sum()),
        },
    )

    manifest = json.loads(args.manifest.read_text())
    manifest.setdefault("long_record", {}).update(
        {
            "path": str(args.out),
            "threshold_tokens": args.threshold,
            "tokenizer": args.tokenizer,
            "n_fresh_test_items": report["n_fresh_test_items"],
            "n_long_items": report["n_long_items"],
            "per_item_path": str(args.per_item_out),
            "per_template_n_long": {
                template: entry["n_long"] for template, entry in report["per_template"].items()
            },
            "generated_at_utc": report["generated_at_utc"],
        }
    )
    write_json(args.manifest, manifest)
    write_json(args.bench / "fresh" / "manifest.json", manifest)

    print(json.dumps({k: v for k, v in report.items() if k != "per_template"}, indent=2))
    for template, entry in report["per_template"].items():
        print(f"{template}: n_test={entry['n_test']} n_long={entry['n_long']} "
              f"p50={entry['p50_tokens']} max={entry['max_tokens']}")
    print(f"wrote {args.out} and updated {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
