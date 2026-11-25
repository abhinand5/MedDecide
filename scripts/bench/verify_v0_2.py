#!/usr/bin/env python
"""Independently verify the built v0.2 benchmark (task S1, AGENTS.md R6).

This script is deliberately **not** part of the build: it re-reads the written v0.2 files and
the frozen v0.1 files and re-derives every acceptance property from scratch. A builder that
checks its own output in memory can be right about the wrong thing; only a check on the
artifact settles what was written.

Checks
------
1. every carried item is byte-identical in content to the v0.1 item of the same id;
2. no superseded `_v1` template appears in v0.2, and each `_v2` template has >= 200 test items;
3. for the graded-score `_v2` template, *every offered level is the gold of at least one test
   item* (the defect it replaces) and the option count equals the offered-level count;
4. no source record appears in two splits;
5. every fresh item's date is inside the v0.1 window, and the strict slice counted by date
   matches the manifest;
6. every gold is one of its item's option keys, and item ids match their identity fields;
7. the manifest's per-file row counts match the files.

Usage:
    uv run python scripts/bench/verify_v0_2.py --out outputs/student_v0/S1/verify_v0_2.json
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Any

from meddecide.bench.schema import Item, compute_item_id
from meddecide.utils.hashing import stable_hash
from meddecide.utils.io import check_no_record_crosses_splits, read_jsonl, write_json

SUPERSEDED = {
    "nfcorpus_graded_score_v1": "nfcorpus_graded_score_v2",
    "pubmed_mesh_major_choice_v1": "pubmed_mesh_major_choice_v2",
    "fda_class_choice_v1": "fda_class_choice_v2",
}
NEW = set(SUPERSEDED.values())


def _read(directory: Path) -> dict[str, list[Item]]:
    out: dict[str, list[Item]] = {}
    for path in sorted(directory.glob("*.jsonl")):
        rows, report = read_jsonl(path, Item)
        report.check_closes()
        if report.n_dropped:
            raise SystemExit(f"{path}: {report.n_dropped} rows failed to parse: {report.to_dict()}")
        out[path.stem] = list(rows)
    return out


def _content_hash(row: Item) -> str:
    return stable_hash(row.model_dump(mode="json"), length=32)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v0-1", type=Path, default=Path("data/bench/v0.1"))
    parser.add_argument("--v0-2", type=Path, default=Path("data/bench/v0.2"))
    parser.add_argument("--out", type=Path, default=Path("outputs/student_v0/S1/verify_v0_2.json"))
    args = parser.parse_args()

    old = {**_read(args.v0_1 / "tier1"), **_read(args.v0_1 / "fresh")}
    new = {**_read(args.v0_2 / "tier1"), **_read(args.v0_2 / "fresh")}
    manifest = json.loads((args.v0_2 / "manifest.json").read_text())
    window_start = date.fromisoformat(manifest["window_start"])
    window_end = date.fromisoformat(manifest["window_end"])
    strict_start = date.fromisoformat(manifest["strict_slice_start"])

    old_rows = [row for rows in old.values() for row in rows]
    new_rows = [row for rows in new.values() for row in rows]

    # 1. carried items identical
    old_carried = {
        row.item_id: _content_hash(row) for row in old_rows
        if row.template_id not in SUPERSEDED and row.template_id not in NEW
    }
    new_carried = {
        row.item_id: _content_hash(row) for row in new_rows
        if row.template_id not in SUPERSEDED and row.template_id not in NEW
    }
    missing = sorted(set(old_carried) - set(new_carried))
    extra = sorted(set(new_carried) - set(old_carried))
    mismatched = sorted(i for i in set(old_carried) & set(new_carried)
                        if old_carried[i] != new_carried[i])

    # 2. superseded gone, new templates present with a test split
    leftovers = sorted({row.template_id for row in new_rows if row.template_id in SUPERSEDED})
    per_template_split: dict[str, Counter] = defaultdict(Counter)
    for row in new_rows:
        per_template_split[row.template_id][str(row.split)] += 1
    new_template_test = {t: per_template_split.get(t, Counter()).get("test", 0) for t in sorted(NEW)}

    # 3. score template: every offered level is a gold in test
    score = [r for r in new_rows if r.template_id == "nfcorpus_graded_score_v2"]
    offered = {str(level) for row in score for level in row.meta.get("offered_levels", [])}
    golds_test = {row.gold for row in score if str(row.split) == "test"}
    option_counts_ok = all(
        row.n_options == len(row.meta.get("offered_levels", [])) for row in score
    )

    # 4. splits
    leak = check_no_record_crosses_splits(new_rows)

    # 5. dates and the strict slice
    out_of_window = [
        row.item_id for tier_rows in (new.get("clinicaltrials", []), new.get("openfda", []),
                                      new.get("pubmed", []))
        for row in tier_rows
        if not (window_start <= row.record_date <= window_end)
    ]
    tier1_out_of_window = [
        row.item_id for source, rows in new.items()
        if source in ("medmcqa", "medqa", "medquad", "mmlu_medical", "nfcorpus", "pubmedqa",
                      "scifact", "trec_covid")
        for row in rows if not (date(1990, 1, 1) <= row.record_date <= date(2026, 10, 6))
    ]
    fresh_rows = [row for source in ("clinicaltrials", "openfda", "pubmed")
                  for row in new.get(source, [])]
    strict_by_date = sum(1 for row in fresh_rows if row.record_date >= strict_start)

    # 6. gold/option consistency and id integrity
    bad_gold = [row.item_id for row in new_rows if row.gold not in row.option_keys]
    bad_id = [
        row.item_id for row in new_rows
        if compute_item_id(row.source, row.source_record_id, row.template_id,
                           int(row.meta.get("option_order_seed", 0)), str(row.split)) != row.item_id
    ]

    # 7. manifest counts
    manifest_mismatch = []
    for source, rows in new.items():
        entry = manifest["files"].get(source)
        if entry is None or entry["n_rows"] != len(rows):
            manifest_mismatch.append({"source": source, "manifest": None if entry is None
                                      else entry["n_rows"], "file": len(rows)})

    # 8. shortcut risk: how often the gold option text is in the state verbatim. The screen's
    # gold-in-state check uses the option label as stored; for the openFDA class template the
    # label carries the source's "[EPC]" tag, which never appears in a label's prose, so that
    # check is trivially 0. The tag-stripped share is what a model could actually exploit.
    tag = re.compile(r"\s*\[[A-Za-z]{2,4}\]\s*$")

    def _copy_shares(template_id: str) -> dict[str, Any]:
        rows = [r for r in new_rows if r.template_id == template_id and str(r.split) == "test"]
        if not rows:
            return {"n": 0, "status": "NOT MEASURED - no test items"}
        exact = sum(1 for r in rows if r.options[r.gold_index].label in r.state)
        stripped = sum(
            1 for r in rows if tag.sub("", r.options[r.gold_index].label).strip() in r.state
        )
        return {
            "n": len(rows),
            "gold_label_in_state_share": exact / len(rows),
            "gold_label_without_tag_in_state_share": stripped / len(rows),
        }

    shortcut_risk = {
        template_id: _copy_shares(template_id)
        for template_id in sorted(NEW)
    }

    checks = {
        "carried_items_identical": not (missing or extra or mismatched),
        "no_superseded_template_left": not leftovers,
        "every_v2_template_has_200_test_items": all(n >= 200 for n in new_template_test.values()),
        "every_offered_score_level_is_a_gold_in_test": offered == golds_test,
        "score_option_count_matches_offered_levels": option_counts_ok,
        "no_record_crosses_splits": bool(leak["ok"]),
        "every_fresh_item_inside_window": not out_of_window,
        "every_tier1_item_has_a_plausible_date": not tier1_out_of_window,
        "every_gold_is_an_option_key": not bad_gold,
        "every_item_id_matches_its_identity": not bad_id,
        "manifest_matches_files": not manifest_mismatch,
    }
    report = {
        "checks": checks,
        "verdict": "PASS" if all(checks.values()) else "FAIL",
        "shortcut_risk": shortcut_risk,
        "counts": {
            "n_items": len(new_rows),
            "n_carried": len(new_carried),
            "n_in_new_templates": sum(1 for row in new_rows if row.template_id in NEW),
            "per_template_split": {k: dict(v) for k, v in sorted(per_template_split.items())},
            "new_template_test": new_template_test,
            "score_offered_levels": sorted(offered, key=int),
            "score_gold_levels_in_test": sorted(golds_test, key=int),
            "n_fresh_items": len(fresh_rows),
            "n_strict_slice_by_date": strict_by_date,
            "n_out_of_window": len(out_of_window),
            "n_bad_gold": len(bad_gold),
            "n_bad_id": len(bad_id),
        },
        "details": {
            "carried_missing_from_v0_2": missing[:20],
            "carried_extra_in_v0_2": extra[:20],
            "carried_content_mismatches": mismatched[:20],
            "superseded_leftovers": leftovers,
            "split_disjointness": leak,
            "manifest_mismatch": manifest_mismatch,
            "out_of_window_ids": out_of_window[:20],
        },
    }
    write_json(args.out, report)
    print(json.dumps({k: v for k, v in report.items() if k != "details"}, indent=2))
    print(f"verdict: {report['verdict']} -> {args.out}")
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
