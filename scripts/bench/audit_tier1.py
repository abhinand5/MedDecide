#!/usr/bin/env python
"""T3 acceptance + R6 self-audit: re-derive the tier-1 build numbers from the raw JSONL.

Re-reads every ``data/bench/tier1/*.jsonl`` with a *fresh* validator run, recomputes the
counts (not from the manifest), recomputes file hashes, re-checks the split-disjointness
invariant and the structural invariants (gold in options, qtype shape), and compares
everything against ``manifest.json``. Writes ``audit.json`` for SELF_AUDIT.md to cite.

It deliberately does **not** print item text: only counts, ids and hashes.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from meddecide.bench.schema import Item, QuestionType
from meddecide.utils.hashing import file_sha256
from meddecide.utils.io import read_jsonl


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier1", type=Path, default=Path("data/bench/tier1"))
    parser.add_argument("--out", type=Path, default=Path("data/bench/tier1/audit.json"))
    args = parser.parse_args()

    manifest = json.loads((args.tier1 / "manifest.json").read_text())
    audit: dict[str, Any] = {"per_source": {}, "problems": [], "checks": {}}

    total_rows = 0
    seen_records: dict[tuple[str, str], set[str]] = defaultdict(set)
    seen_item_ids: Counter[str] = Counter()

    for path in sorted(args.tier1.glob("*.jsonl")):
        source = path.stem
        rows, report = read_jsonl(path, Item)
        n = len(rows)
        total_rows += n
        report.check_closes()

        # independent recount of the split/qtype/template breakdown
        by_split = Counter(str(r.split) for r in rows)
        by_qtype = Counter(str(r.qtype) for r in rows)
        by_template = Counter(r.template_id for r in rows)

        for r in rows:
            seen_records[(r.source, r.source_record_id)].add(str(r.split))
            seen_item_ids[r.item_id] += 1
            # structural invariants, re-derived (not trusting the loader)
            if r.gold not in r.option_keys:
                audit["problems"].append(f"{r.item_id}: gold not in options")
            if r.qtype is QuestionType.SCORE and len(r.options) > 10:
                audit["problems"].append(f"{r.item_id}: score with >10 levels")
            if r.qtype is QuestionType.NOUL and r.option_keys != ["yes", "no"]:
                audit["problems"].append(f"{r.item_id}: noul keys {r.option_keys}")

        manifest_row = manifest["files"].get(source, {})
        audit["per_source"][source] = {
            "n_rows_recounted": n,
            "n_rows_in_manifest": manifest_row.get("n_rows"),
            "counts_match_manifest": manifest_row.get("n_rows") == n,
            "sha256_recomputed": file_sha256(path),
            "sha256_matches_manifest": manifest_row.get("sha256") == file_sha256(path),
            "by_split": dict(sorted(by_split.items())),
            "by_qtype": dict(sorted(by_qtype.items())),
            "n_templates": len(by_template),
            "templates": dict(sorted(by_template.items())),
            "read_report": report.to_dict(),
            "split_sums_to_total": sum(by_split.values()) == n,
            "qtype_sums_to_total": sum(by_qtype.values()) == n,
        }
        if not audit["per_source"][source]["counts_match_manifest"]:
            audit["problems"].append(f"{source}: recount {n} != manifest {manifest_row.get('n_rows')}")
        if not audit["per_source"][source]["sha256_matches_manifest"]:
            audit["problems"].append(f"{source}: sha256 differs from manifest")

    leaks = {f"{k[0]}|{k[1]}": sorted(v) for k, v in seen_records.items() if len(v) > 1}
    dup_item_ids = {k: v for k, v in seen_item_ids.items() if v > 1}

    audit["totals"] = {
        "n_sources": len(audit["per_source"]),
        "n_items_recounted": total_rows,
        "n_items_in_manifest_sum": sum(v.get("n_rows", 0) for v in manifest["files"].values()),
        "n_unique_records": len(seen_records),
        "n_leaking_records": len(leaks),
        "n_duplicate_item_ids": len(dup_item_ids),
        "items_per_split": dict(
            sorted(
                Counter(
                    split
                    for info in audit["per_source"].values()
                    for split, count in info["by_split"].items()
                    for _ in range(count)
                ).items()
            )
        ),
    }
    audit["duplicate_item_id_examples"] = dict(sorted(dup_item_ids.items())[:10])
    audit["checks"] = {
        "recount_matches_manifest": audit["totals"]["n_items_recounted"]
        == audit["totals"]["n_items_in_manifest_sum"],
        "no_split_leaks": not leaks,
        "no_duplicate_item_ids": not dup_item_ids,
        "no_structural_problems": not audit["problems"],
        "all_splits_sum_to_totals": all(
            info["split_sums_to_total"] and info["qtype_sums_to_total"]
            for info in audit["per_source"].values()
        ),
        "all_hashes_match": all(
            info["sha256_matches_manifest"] for info in audit["per_source"].values()
        ),
    }
    audit["verdict"] = "PASS" if all(audit["checks"].values()) else "FAIL"
    args.out.write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({"checks": audit["checks"], "totals": audit["totals"]}, indent=2))
    print(f"verdict: {audit['verdict']} -> {args.out}")
    return 0 if audit["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
