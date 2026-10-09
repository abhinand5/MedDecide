#!/usr/bin/env python
"""Fetch and report the openFDA pharmacologic-class index used by ``fda_class_choice_v2``.

The index records, for every established pharmacologic class in ``openfda.pharm_class_epc``,
the MoA / physiologic-effect / route values seen on the labels that carry it, which is what
lets the ``_v2`` template draw *near-miss* distractors instead of unrelated classes.

Usage:
    uv run python scripts/bench/fetch_fda_class_index.py --out /workspace/tmp/openfda/class_index.json
    uv run python scripts/bench/fetch_fda_class_index.py --out /workspace/tmp/openfda/class_index.json \\
        --max-labels 2000 --examples "Penicillin-class Antibacterial" "Proton Pump Inhibitor"
    uv run python scripts/bench/fetch_fda_class_index.py --index /workspace/tmp/openfda/class_index.json \\
        --examples "Biguanides"
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from meddecide.bench.fresh import fda_classes

# How many shared values to print per rule before eliding (the counts stay exact).
MAX_SHARED_PRINTED = 12
# How many ranked candidates to show per example class (a readability aid, not a metric).
MAX_CANDIDATES_PRINTED = 5


def _counts_by_rule(index: fda_classes.ClassIndex) -> dict[str, dict[str, int]]:
    """Per-class candidate counts per rule (uncapped: ``max_candidates`` is lifted)."""
    limit = max(len(index["classes"]), 1)
    out: dict[str, dict[str, int]] = {}
    for name in index["classes"]:
        candidates = fda_classes.near_miss_candidates(name, index, max_candidates=limit)
        out[name] = fda_classes.rule_counts(candidates)
    return out


def _distribution(counts: list[int]) -> dict[str, float | int]:
    """min / median / max plus the 0, 1-2 and >=3 buckets, with the denominator."""
    if not counts:
        return {"n_classes": 0}
    return {
        "n_classes": len(counts),
        "min": min(counts),
        "median": statistics.median(counts),
        "max": max(counts),
        "n_zero": sum(1 for c in counts if c == 0),
        "n_1_2": sum(1 for c in counts if 1 <= c <= 2),
        "n_ge_3": sum(1 for c in counts if c >= 3),
    }


def _print_summary(index: fda_classes.ClassIndex) -> None:
    meta = index["meta"]
    by_class = _counts_by_rule(index)
    n_classes = len(by_class)
    print("\nmeta:")
    print(json.dumps(meta, indent=2, sort_keys=True))
    print(f"\nrule availability over {n_classes} classes:")
    for rule in fda_classes.RULES:
        with_rule = sum(1 for counts in by_class.values() if counts[rule] > 0)
        total = sum(counts[rule] for counts in by_class.values())
        share = 100.0 * with_rule / n_classes if n_classes else 0.0
        print(f"  {rule:11s} {with_rule:5d} classes ({share:5.1f}%) {total:7d} candidates")
    per_class = [sum(counts.values()) for counts in by_class.values()]
    usable = sum(1 for c in per_class if c >= 3)
    share = 100.0 * usable / n_classes if n_classes else 0.0
    print(f"  {'any rule':11s} {sum(1 for c in per_class if c > 0):5d} classes have >=1 candidate")
    print(f"  {'usable':11s} {usable:5d} classes ({share:5.1f}%) have >=3 candidates "
          "(a 4-option item needs 3 distractors)")
    for rule in (*fda_classes.RULES, "any"):
        counts = (
            per_class if rule == "any" else [c[rule] for c in by_class.values()]
        )
        dist = _distribution(counts)
        if not dist.get("n_classes"):
            print(f"\n{rule}: no classes")
            continue
        print(f"\n{rule} candidate-count distribution:")
        print(f"  min {dist['min']}  median {dist['median']}  max {dist['max']}  "
              f"(n={dist['n_classes']})")
        print(f"  0: {dist['n_zero']}   1-2: {dist['n_1_2']}   >=3: {dist['n_ge_3']}")


def _print_examples(index: fda_classes.ClassIndex, names: list[str]) -> None:
    limit = max(len(index["classes"]), 1)
    print("\nexamples:")
    for name in names:
        key = fda_classes.find_class(name, index)
        if key is None:
            print(f"  {name!r}: NOT IN INDEX")
            continue
        entry = index["classes"][key]
        print(f"  {key!r}  (n_labels={entry['n_labels']}, "
              f"moa={len(entry['moa'])}, pe={len(entry['pe'])}, routes={entry['routes']})")
        candidates = fda_classes.near_miss_candidates(key, index, max_candidates=limit)
        counts = fda_classes.rule_counts(candidates)
        for rule in fda_classes.RULES:
            shared = sorted({value for c in candidates if c["rule"] == rule for value in c["shared"]})
            shown = ", ".join(shared[:MAX_SHARED_PRINTED])
            if len(shared) > MAX_SHARED_PRINTED:
                shown += f", ... (+{len(shared) - MAX_SHARED_PRINTED} more)"
            print(f"    {rule:11s} {counts[rule]:4d} candidates"
                  + (f"  shared: {shown}" if shared else ""))
        top = candidates[:MAX_CANDIDATES_PRINTED]
        for candidate in top:
            print(f"      - {candidate['class']!r} [{candidate['rule']}] "
                  f"shared={candidate['shared']}")
        if len(candidates) > len(top):
            print(f"      ... {len(candidates) - len(top)} more")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("/workspace/tmp/openfda/class_index.json"),
                        help="where to write the index JSON (parent dirs are created)")
    parser.add_argument("--index", type=Path, default=None,
                        help="reuse an index already on disk instead of fetching")
    parser.add_argument("--max-labels", type=int, default=fda_classes.MAX_LABELS,
                        help="cap on labels fetched; api_total shows the full corpus size")
    parser.add_argument("--page-size", type=int, default=fda_classes.PAGE_SIZE)
    parser.add_argument("--examples", nargs="*", default=None,
                        help="class names (tagged or untagged) to report in detail")
    args = parser.parse_args()

    t0 = time.perf_counter()
    if args.index is not None:
        index = fda_classes.load_index(args.index)
        source = f"loaded {args.index}"
    else:
        index = fda_classes.fetch_class_index(max_labels=args.max_labels, page_size=args.page_size)
        source = f"fetched with max_labels={args.max_labels} page_size={args.page_size}"
    elapsed = time.perf_counter() - t0
    path = fda_classes.save_index(index, args.out)
    print(f"index: {path}  ({source}, {elapsed:.1f}s wall clock)")
    _print_summary(index)
    if args.examples:
        _print_examples(index, args.examples)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
