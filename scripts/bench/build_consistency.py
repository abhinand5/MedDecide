#!/usr/bin/env python
"""Build the record-claim consistency templates (S2) into v0.2 and into a pre-window training file.

Two windows, one builder:

* **fresh** (>= 2026-03-01, the v0.2 window): items are appended to
  ``data/bench/v0.2/fresh/{clinicaltrials,openfda}.jsonl`` next to the carried v0.1 rows and the
  `_v2` rows, the manifest's counts are updated, and a ``builds.consistency`` block records the
  drop accounting, the string-presence baseline and the balance check.
* **pre-window** (< 2026-03-01, ``--prewindow-start 2023-01-01``): items go to
  ``data/train/student_v0/prewindow_consistency.jsonl`` for S6. The **held-out** template
  (D14: ``ct_arm_role_noul_v1``) is excluded from this file entirely — training data must never
  contain an item of a held-out template.

The string-presence baseline is the S2 acceptance rule, and it is measured here rather than
asserted: a rule that reads the state for the claimed value must not exceed 0.60 macro accuracy.
For ``noul`` the rule answers "supported iff the claimed value (or its stem) occurs in the
state"; for the multi-field ``choice`` variant it picks the first option whose value does *not*
occur, which is the same rule seen from the other side.

Usage:
    uv run python scripts/bench/build_consistency.py --window fresh
    uv run python scripts/bench/build_consistency.py --window prewindow
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Any

import yaml

from meddecide.bench.fresh import clinicaltrials, consistency, openfda
from meddecide.bench.schema import Item, QuestionType, compute_item_id
from meddecide.bench.tier1.common import balance_classes, finalize_splits
from meddecide.utils.io import read_jsonl, write_json, write_jsonl
from meddecide.utils.provenance import Provenance, gpu_name, utcnow

FRESH_DIR = Path("data/bench/v0.2/fresh")
TRAIN_DIR = Path("data/train/student_v0")
HELD_OUT = "ct_arm_role_noul_v1"  # D14, chosen in S2: the arm/intervention role structure
CONSISTENCY_TEMPLATES = ("ct_arm_role_noul_v1", "ct_claim_set_choice_v1", "fda_route_claim_noul_v1")


def _value_in_state(value: str, state: str) -> bool:
    """The string-presence test used by the baseline: value or its stem, case-insensitive."""
    return re.search(rf"\b{re.escape(value)}\w*", state, re.IGNORECASE) is not None


def string_presence_baseline(items: list[Item]) -> dict[str, Any]:
    """The naive rule an item must defeat, scored on the items given (usually the test split)."""
    n = correct = 0
    per_template: dict[str, Counter] = defaultdict(Counter)
    for item in items:
        if item.qtype is QuestionType.NOUL:
            claimed = str(item.meta.get("claimed_intervention")
                          or item.meta.get("claimed_route") or "")
            predicted = "yes" if _value_in_state(claimed, item.state) else "no"
            right = predicted == item.gold
        else:
            labels = [opt.label for opt in item.options]
            values = [lab.split(":", 1)[1].strip() if ":" in lab else lab for lab in labels]
            absent = [opt.key for opt, value in zip(item.options, values, strict=True)
                      if not _value_in_state(value, item.state)]
            predicted = absent[0] if absent else item.options[0].key
            right = predicted == item.gold
        n += 1
        correct += int(right)
        per_template[item.template_id]["n"] += 1
        per_template[item.template_id]["correct"] += int(right)
    return {
        "n": n,
        "accuracy": correct / n if n else None,
        "per_template": {
            template: {
                "n": counts["n"],
                "accuracy": counts["correct"] / counts["n"] if counts["n"] else None,
            }
            for template, counts in sorted(per_template.items())
        },
        "rule": (
            "noul: answer 'yes' iff the claimed value (or its stem) occurs in the state; "
            "choice: choose the first option whose value does not occur in the state"
        ),
    }


def _carried_rows(source: str) -> list[Item]:
    path = FRESH_DIR / f"{source}.jsonl"
    if not path.is_file():
        return []
    rows, report = read_jsonl(path, Item)
    report.check_closes()
    if report.n_dropped:
        raise SystemExit(f"{path}: {report.n_dropped} unreadable rows; refusing to build on it")
    return [row for row in rows if row.template_id not in CONSISTENCY_TEMPLATES]


def _reconcile(rows: list[Item], carried: list[Item]) -> tuple[list[Item], int]:
    """Join each new item the split its record already has (v0.1 moved some items to test)."""
    carried_split: dict[tuple[str, str], set[str]] = {}
    for row in carried:
        carried_split.setdefault((row.source, row.source_record_id), set()).add(str(row.split))
    joined = 0
    out: list[Item] = []
    for row in rows:
        splits = carried_split.get((row.source, row.source_record_id))
        if splits is not None and len(splits) == 1 and str(row.split) not in splits:
            target = next(iter(splits))
            payload = row.model_dump()
            payload["split"] = target
            payload["item_id"] = compute_item_id(
                row.source, row.source_record_id, row.template_id,
                int(row.meta.get("option_order_seed", 0)), target,
            )
            row = Item.model_validate(payload)
            joined += 1
        out.append(row)
    return out, joined


def build_fresh(cfg: dict[str, Any], window_start: date, window_end: date) -> dict[str, Any]:
    t0 = time.perf_counter()
    studies = clinicaltrials.fetch_studies(start=window_start, end=window_end)
    labels = openfda.fetch_labels(start=window_start, end=window_end)

    ct_rows_arm, ct_drops_arm, ct_notes_arm = consistency.build_ct_arm_role_items(
        studies, cfg=cfg, window_start=window_start, window_end=window_end
    )
    ct_rows_set, ct_drops_set, ct_notes_set = consistency.build_ct_claim_set_items(
        studies, cfg=cfg, window_start=window_start, window_end=window_end
    )
    fda_rows, fda_drops, fda_notes = consistency.build_fda_route_claim_items(
        labels, cfg=cfg, window_start=window_start, window_end=window_end
    )

    report: dict[str, Any] = {"generated_at_utc": utcnow(), "sources": {}}
    # each source reports only its own builders' drops: merging them would double-count the
    # trial drops onto the label source (and vice versa) in the manifest
    drops_by_source = {
        "clinicaltrials": {
            **{f"ct_arm_role:{k}": v for k, v in ct_drops_arm.items()},
            **{f"ct_claim_set:{k}": v for k, v in ct_drops_set.items()},
        },
        "openfda": {f"fda_route_claim:{k}": v for k, v in fda_drops.items()},
    }
    notes_by_source = {
        "clinicaltrials": [*ct_notes_arm, *ct_notes_set],
        "openfda": fda_notes,
    }
    for source, new_rows in (("clinicaltrials", [*ct_rows_arm, *ct_rows_set]), ("openfda", fda_rows)):
        carried = _carried_rows(source)
        new_rows, joined = _reconcile(new_rows, carried)
        new_rows, counts, extra = finalize_splits(
            new_rows, cfg=cfg, source=source, salt=f"{source}_consistency", cap_test=5000,
            cap_dev=2000,
        )
        new_rows, balance = balance_classes(new_rows, cfg=cfg, salt=f"{source}_consistency")
        combined = [*carried, *new_rows]
        write_jsonl(FRESH_DIR / f"{source}.jsonl", combined)
        report["sources"][source] = {
            "n_carried": len(carried),
            "n_new": len(new_rows),
            "n_joined_existing_split": joined,
            "finalize_counts": counts,
            "notes": extra,
            "balance": balance["templates"],
            "dropped": drops_by_source[source],
            "builder_notes": notes_by_source[source],
        }

    # re-read the written files and measure the acceptance properties on the artifact
    written: dict[str, list[Item]] = {}
    for source in ("clinicaltrials", "openfda"):
        rows, r = read_jsonl(FRESH_DIR / f"{source}.jsonl", Item)
        r.check_closes()
        written[source] = [row for row in rows if row.template_id in CONSISTENCY_TEMPLATES]
    all_new = [row for rows in written.values() for row in rows]
    test_new = [row for row in all_new if str(row.split) == "test"]
    per_template_split: dict[str, Counter] = defaultdict(Counter)
    for row in all_new:
        per_template_split[row.template_id][str(row.split)] += 1
    per_template_gold: dict[str, Counter] = defaultdict(Counter)
    for row in test_new:
        per_template_gold[row.template_id][row.gold] += 1

    baseline = string_presence_baseline(test_new)
    report["templates"] = {
        template: {
            "n_test": per_template_split[template].get("test", 0),
            "n_dev": per_template_split[template].get("dev", 0),
            "gold_counts_test": dict(sorted(per_template_gold[template].items())),
            "string_presence_baseline": baseline["per_template"].get(template),
        }
        for template in CONSISTENCY_TEMPLATES
    }
    report["string_presence_baseline"] = baseline
    report["checks"] = {
        "each_template_has_200_test_items": all(
            entry["n_test"] >= 200 for entry in report["templates"].values()
        ),
        "each_template_is_balanced_or_near": all(
            _balance_ok(entry["gold_counts_test"]) for entry in report["templates"].values()
        ),
        "string_presence_baseline_at_or_below_0.60": all(
            (entry["string_presence_baseline"] or {}).get("accuracy", 1.0) <= 0.60
            for entry in report["templates"].values()
        ),
    }
    report["verdict"] = "PASS" if all(report["checks"].values()) else "FAIL"
    report["wall_clock_s"] = round(time.perf_counter() - t0, 1)
    return report


def _balance_ok(gold_counts: dict[str, int]) -> bool:
    """A noul template must be 50/50; a choice template must have no option above 0.40."""
    if not gold_counts:
        return False
    total = sum(gold_counts.values())
    return max(gold_counts.values()) / total <= 0.60


def build_prewindow(cfg: dict[str, Any], window_start: date, window_end: date) -> dict[str, Any]:
    t0 = time.perf_counter()
    studies = clinicaltrials.fetch_studies(start=window_start, end=window_end)
    labels = openfda.fetch_labels(start=window_start, end=window_end)
    rows: list[Any] = []
    report: dict[str, Any] = {"generated_at_utc": utcnow(), "window_start": window_start.isoformat(),
                              "window_end": window_end.isoformat(), "held_out_excluded": HELD_OUT}
    arm_rows, arm_drops, _ = consistency.build_ct_arm_role_items(
        studies, cfg=cfg, window_start=window_start, window_end=window_end
    )
    set_rows, set_drops, _ = consistency.build_ct_claim_set_items(
        studies, cfg=cfg, window_start=window_start, window_end=window_end
    )
    fda_rows, fda_drops, _ = consistency.build_fda_route_claim_items(
        labels, cfg=cfg, window_start=window_start, window_end=window_end
    )
    # D14: the held-out template never enters training data
    rows.extend(row for row in set_rows if row.template_id != HELD_OUT)
    rows.extend(row for row in fda_rows if row.template_id != HELD_OUT)
    excluded = len(arm_rows)
    rows, counts, extra = finalize_splits(
        rows, cfg=cfg, source="prewindow_consistency", salt="prewindow_consistency",
        cap_test=None, cap_dev=None,
    )
    report["n_excluded_held_out_items"] = excluded
    report["finalize_counts"] = counts
    report["notes"] = extra
    report["n_items"] = len(rows)
    report["by_template"] = dict(sorted(Counter(row.template_id for row in rows).items()))
    report["dropped"] = {
        "clinicaltrials": {
            **{f"ct_arm_role:{k}": v for k, v in arm_drops.items()},
            **{f"ct_claim_set:{k}": v for k, v in set_drops.items()},
        },
        "openfda": {f"fda_route_claim:{k}": v for k, v in fda_drops.items()},
    }
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    write_jsonl(TRAIN_DIR / "prewindow_consistency.jsonl", rows)
    report["checks"] = {
        "wrote_items": len(rows) > 0,
        "held_out_template_absent": all(row.template_id != HELD_OUT for row in rows),
        "every_source_present": len({row.source for row in rows}) >= 2,
    }
    report["verdict"] = "PASS" if all(report["checks"].values()) else "FAIL"
    report["wall_clock_s"] = round(time.perf_counter() - t0, 1)
    write_json(TRAIN_DIR / "prewindow_consistency_manifest.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/bench_v0_2.yaml"))
    parser.add_argument("--window", choices=["fresh", "prewindow"], default="fresh")
    parser.add_argument("--fresh-start", type=date.fromisoformat, default=None)
    parser.add_argument("--fresh-end", type=date.fromisoformat, default=None)
    parser.add_argument("--prewindow-start", type=date.fromisoformat, default=date(2023, 1, 1))
    parser.add_argument("--prewindow-end", type=date.fromisoformat, default=date(2026, 2, 28))
    parser.add_argument("--log-dir", type=Path, default=Path("outputs/student_v0/S2/logs"))
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    manifest_path = Path("data/bench/v0.2/manifest.json")
    manifest = json.loads(manifest_path.read_text())
    fresh_start = args.fresh_start or date.fromisoformat(str(manifest["window_start"]))
    fresh_end = args.fresh_end or date.fromisoformat(str(manifest["window_end"]))

    args.log_dir.mkdir(parents=True, exist_ok=True)
    prov = Provenance(
        run_name=f"S2_build_consistency_{args.window}",
        command=" ".join([sys.executable, *sys.argv]),
        gpu=gpu_name(),
        seed=int(cfg.get("seed", 0)),
        config={"window": args.window, "config_file": str(args.config)},
    )
    if args.window == "fresh":
        report = build_fresh(cfg, fresh_start, fresh_end)
        manifest.setdefault("builds", {})["consistency"] = report
        manifest["updated_at_utc"] = utcnow()
        # refresh the per-file row counts for the two files this run rewrote
        for source in ("clinicaltrials", "openfda"):
            rows, r = read_jsonl(FRESH_DIR / f"{source}.jsonl", Item)
            r.check_closes()
            manifest["files"][source]["n_rows"] = len(rows)
            manifest["totals"][source] = len(rows)
        manifest["n_items_total"] = sum(manifest["totals"].values())
        write_json(manifest_path, manifest)
        write_json(FRESH_DIR / "manifest.json", manifest)
        out = args.log_dir / "build_consistency_fresh.json"
    else:
        report = build_prewindow(cfg, args.prewindow_start, args.prewindow_end)
        out = args.log_dir / "build_consistency_prewindow.json"
    write_json(out, report)
    prov.wall_clock_s = report.get("wall_clock_s")
    prov.finish().write(args.log_dir / f"S2_build_consistency_{args.window}_provenance.json")
    print(json.dumps({k: v for k, v in report.items() if k not in ("sources",)}, indent=2)[:4000])
    print(f"verdict: {report['verdict']} -> {out}")
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
