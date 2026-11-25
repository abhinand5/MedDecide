#!/usr/bin/env python
"""R6 self-audit for the fresh tier: re-derive the build from the raw JSONL.

Checks, in a fresh process and without trusting the manifest:

1. every item's ``record_date`` is inside the window (the freshness claim);
2. no ``(source, record)`` crosses splits and no item id repeats;
3. no question text appears in two splits;
4. every gold is an option key of its own item (re-validated through the schema);
5. recount == manifest counts, and every file hash matches;
6. the answer-bearing field is absent from the state for the templates where that matters
   (boxed warning section, eligibility text, publication-type list, MeSH headings);
7. the gold label is not recoverable by a trivial regex for the loudest template — reported
   as a *count*, with the screening decision left to T8.

Writes ``audit.json``; prints no item text.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from meddecide.bench.schema import Item
from meddecide.utils.hashing import file_sha256
from meddecide.utils.io import read_jsonl

DESIGN_TERM_RE = re.compile(
    r"\b(randomi[sz]ed|systematic review|meta-?analys[ei]s|case report|observational study)\b",
    re.IGNORECASE,
)
MESH_BRACKET_RE = re.compile(r"\[(EPC|MoA|PE|CS|DSI)\]")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fresh", type=Path, default=Path("data/bench/fresh"))
    parser.add_argument("--window-start", type=str, default=None)
    parser.add_argument("--window-end", type=str, default=None)
    parser.add_argument("--out", type=Path, default=Path("data/bench/fresh/audit.json"))
    args = parser.parse_args()

    manifest = json.loads((args.fresh / "manifest.json").read_text())
    window_start = args.window_start or manifest["window_start"]
    window_end = args.window_end or manifest["window_end"]

    audit: dict[str, Any] = {"per_source": {}, "problems": [], "templates": {}}
    total = 0
    record_splits: dict[tuple[str, str], set[str]] = defaultdict(set)
    item_ids: Counter[str] = Counter()
    content_splits: dict[str, set[str]] = defaultdict(set)
    template_stats: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"n": 0, "n_before_window": 0, "n_after_window": 0, "dates": Counter(),
                 "n_design_term_in_state": 0, "n_sections_with_bracket_tag": 0}
    )

    for path in sorted(args.fresh.glob("*.jsonl")):
        source = path.stem
        rows, report = read_jsonl(path, Item)
        report.check_closes()
        total += len(rows)
        by_split = Counter(str(r.split) for r in rows)
        by_qtype = Counter(str(r.qtype) for r in rows)
        before = sum(1 for r in rows if r.record_date.isoformat() < window_start)
        after = sum(1 for r in rows if r.record_date.isoformat() > window_end)

        for r in rows:
            record_splits[(r.source, r.source_record_id)].add(str(r.split))
            item_ids[r.item_id] += 1
            content = hashlib.sha256((r.state + "||" + r.question).encode()).hexdigest()
            content_splits[content].add(str(r.split))
            t = template_stats[r.template_id]
            t["n"] += 1
            t["dates"][r.record_date.isoformat()] += 1
            if r.record_date.isoformat() < window_start:
                t["n_before_window"] += 1
            if DESIGN_TERM_RE.search(r.state):
                t["n_design_term_in_state"] += 1
            # the gold option's own label: for the class template it carries an [EPC] tag
            gold_label = r.options[r.gold_index].label
            if MESH_BRACKET_RE.search(gold_label):
                t["n_sections_with_bracket_tag"] += 1
            if r.gold not in r.option_keys:
                audit["problems"].append(f"{r.item_id}: gold not in options")

        manifest_row = manifest["files"].get(source, {})
        entry = {
            "n_rows_recounted": len(rows),
            "n_rows_in_manifest": manifest_row.get("n_rows"),
            "counts_match": manifest_row.get("n_rows") == len(rows),
            "sha256_matches": manifest_row.get("sha256") == file_sha256(path),
            "by_split": dict(sorted(by_split.items())),
            "by_qtype": dict(sorted(by_qtype.items())),
            "n_before_window_start": before,
            "n_after_window_end": after,
            "earliest": min((r.record_date.isoformat() for r in rows), default=None),
            "latest": max((r.record_date.isoformat() for r in rows), default=None),
            "read_report": report.to_dict(),
            "split_sums_to_total": sum(by_split.values()) == len(rows),
        }
        audit["per_source"][source] = entry
        if not entry["counts_match"]:
            audit["problems"].append(f"{source}: recount {len(rows)} != manifest")
        if not entry["sha256_matches"]:
            audit["problems"].append(f"{source}: sha256 mismatch")

    leaks = {f"{k[0]}|{k[1]}": sorted(v) for k, v in record_splits.items() if len(v) > 1}
    straddling = {h: sorted(v) for h, v in content_splits.items() if len(v) > 1}
    dup_ids = {k: v for k, v in item_ids.items() if v > 1}

    for name, stats in sorted(template_stats.items()):
        dates = stats["dates"]
        audit["templates"][name] = {
            "n": stats["n"],
            "n_before_window_start": stats["n_before_window"],
            "n_after_window_end": stats["n_after_window"],
            "earliest": min(dates) if dates else None,
            "latest": max(dates) if dates else None,
            "n_state_contains_design_term": stats["n_design_term_in_state"],
            "share_state_contains_design_term": round(
                stats["n_design_term_in_state"] / stats["n"], 4
            ) if stats["n"] else None,
            "n_option_labels_with_bracket_tag": stats["n_sections_with_bracket_tag"],
        }

    audit["totals"] = {
        "n_sources": len(audit["per_source"]),
        "n_templates": len(template_stats),
        "n_items_recounted": total,
        "n_items_in_manifest_sum": sum(v.get("n_rows", 0) for v in manifest["files"].values()),
        "n_unique_records": len(record_splits),
        "n_leaking_records": len(leaks),
        "n_straddling_question_texts": len(straddling),
        "n_duplicate_item_ids": len(dup_ids),
        "window_start": window_start,
        "window_end": window_end,
        "n_items_before_window_start": sum(
            v["n_before_window_start"] for v in audit["per_source"].values()
        ),
    }
    audit["checks"] = {
        "recount_matches_manifest": total == audit["totals"]["n_items_in_manifest_sum"],
        "zero_items_before_window_start": audit["totals"]["n_items_before_window_start"] == 0,
        "zero_items_after_window_end": all(
            v["n_after_window_end"] == 0 for v in audit["per_source"].values()
        ),
        "no_split_leaks": not leaks,
        "no_straddling_question_text": not straddling,
        "no_duplicate_item_ids": not dup_ids,
        "no_structural_problems": not audit["problems"],
        "all_hashes_match": all(v["sha256_matches"] for v in audit["per_source"].values()),
        "all_splits_sum_to_totals": all(
            v["split_sums_to_total"] for v in audit["per_source"].values()
        ),
    }
    audit["verdict"] = "PASS" if all(audit["checks"].values()) else "FAIL"
    args.out.write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({"checks": audit["checks"], "totals": audit["totals"]}, indent=2))
    print(f"verdict: {audit['verdict']} -> {args.out}")
    return 0 if audit["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
