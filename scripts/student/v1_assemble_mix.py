#!/usr/bin/env python
"""V4: assemble the student_v1 training mix and its selection dev set, then run the leakage check.

Inputs
  data/train/student_v0/train.jsonl        the student_v0 training mix (tier-1 train + pre-window
                                           structured items; checked against the v0.2 index first)
  data/train/student_v1/components/*.jsonl  the V4 pre-window components (train and dev rows)
  data/train/student_v1/kept_templates.json the screen's kept list (outputs/student_v1/V4/screen.json)
  data/bench/v0.2/{tier1,fresh}/*.jsonl     the benchmark (seen-template dev, for selection only)

Steps
  1. leakage index of v0.2 test/dev; the student_v0 base must already be leak-free
  2. new pre-window rows (train and dev) that collide with the index are removed and counted
     (record id, normalised state hash, state+question hash, date, held-out template)
  3. train pool = base rows + kept new train rows; per-template cap at 8 % of the capped mix
  4. selection dev = up to ``--dev-per-template`` seen-template v0.2 dev items per template plus the
     same for the kept new templates' pre-window dev; deterministic by item id; must reach
     ``--min-dev`` (ADVISORY V4: 4,000)
  5. final leakage check on the written rows: training rows against the index and the window, dev
     rows against held-out templates and questions (their seen v0.2 dev items are the benchmark's
     own dev split, by design, so they are not checked against the index)
  6. write train.jsonl, dev.jsonl, the data manifest, the outputs report and a committed summary

Every removal and every cap is counted in the manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "bench"))

from build_training_mix import (  # noqa: E402
    LEAKAGE_REASONS,
    build_leakage_index,
    remove_leaks,
    verify_no_leaks,
)

from meddecide.bench.schema import Item  # noqa: E402
from meddecide.train.mix_v1 import (  # noqa: E402
    balance_classes,
    balanced_sample,
    cap_fixed_point,
    cap_items,
    describe,
    max_share,
    stratified_sample,
)
from meddecide.train.templates_v1 import HELD_OUT_TEMPLATES  # noqa: E402

WINDOW_START = date(2026, 3, 1)
BASE = ROOT / "data" / "train" / "student_v0" / "train.jsonl"
COMPONENTS = ROOT / "data" / "train" / "student_v1" / "components"
KEPT = ROOT / "data" / "train" / "student_v1" / "kept_templates.json"
BENCH = ROOT / "data" / "bench" / "v0.2"
OUT_DIR = ROOT / "data" / "train" / "student_v1"
REPORT = ROOT / "outputs" / "student_v1" / "V4" / "mix_report.json"
SUMMARY = ROOT / "loops" / "student_v1" / "mix_summary.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_items(path: Path) -> list[Item]:
    with path.open(encoding="utf-8") as handle:
        return [Item.model_validate_json(line) for line in handle if line.strip()]


def group(items: list[Item]) -> dict[str, list[Item]]:
    out: dict[str, list[Item]] = defaultdict(list)
    for item in items:
        out[str(item.template_id)].append(item)
    return dict(out)


def removed_by_template(before: list[Item], after: list[Item]) -> dict[str, int]:
    kept_ids = {i.item_id for i in after}
    return dict(Counter(str(i.template_id) for i in before if i.item_id not in kept_ids))


def item_key(item: Item) -> str:
    return item.item_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev-per-template", type=int, default=250)
    parser.add_argument("--min-dev", type=int, default=4000)
    args = parser.parse_args()

    kept = set(json.loads(KEPT.read_text(encoding="utf-8"))["kept"])
    if kept & HELD_OUT_TEMPLATES:
        raise SystemExit("a held-out template passed the screen; refusing to assemble")

    # ---- 1. inputs and index ------------------------------------------------
    index = build_leakage_index(BENCH)
    base_rows = read_items(BASE)
    base_check = verify_no_leaks(base_rows, index, window_start=WINDOW_START)
    if any(base_check.get("by_reason", {}).values()):
        raise SystemExit(f"the student_v0 base mix is not leak-free: {base_check}")
    base_by_t = group(base_rows)

    comp_files = sorted(COMPONENTS.glob("*_v1.jsonl"))
    new_train_all: list[Item] = []
    new_dev_all: list[Item] = []
    for path in comp_files:
        for item in read_items(path):
            if str(item.template_id) not in kept:
                continue
            if str(item.split) == "train":
                new_train_all.append(item)
            elif str(item.split) == "dev":
                new_dev_all.append(item)

    # ---- 2. counted removal of colliding pre-window rows ----------------------
    new_train, train_report = remove_leaks(new_train_all, index, window_start=WINDOW_START)
    new_dev, dev_report = remove_leaks(new_dev_all, index, window_start=WINDOW_START)
    removed_train_t = removed_by_template(new_train_all, new_train)
    removed_dev_t = removed_by_template(new_dev_all, new_dev)
    new_train_by_t = group(new_train)
    new_dev_by_t = group(new_dev)

    # Class balance for the new binary (noul) templates. The source yes-rates are 4 % to 74 %, and a
    # template with a few per cent positives would teach its prior. Decided before any model was trained,
    # from the pre-window counts alone. Recorded per template in the manifest.
    noul_new = {tid for tid, rows in new_train_by_t.items() if rows and str(rows[0].qtype) == "noul"}
    class_balance: dict[str, dict[str, dict[str, int]]] = {}
    for tid in sorted(noul_new):
        before = Counter(str(i.gold) for i in new_train_by_t[tid])
        new_train_by_t[tid] = balance_classes(new_train_by_t[tid], label=lambda i: str(i.gold), key=item_key)
        class_balance[tid] = {"train_before": dict(before),
                              "train_after": dict(Counter(str(i.gold) for i in new_train_by_t[tid]))}

    # ---- 3. cap ---------------------------------------------------------------
    pool: dict[str, list[Item]] = {t: list(v) for t, v in base_by_t.items()}
    for tid, rows in new_train_by_t.items():
        pool.setdefault(tid, []).extend(rows)
    pool_counts = {t: len(v) for t, v in pool.items()}
    cap = cap_fixed_point(pool_counts)
    capped = cap_items(pool, cap, key=item_key)
    train_rows = [item for tid in sorted(capped) for item in capped[tid]]
    train_counts = {t: len(v) for t, v in capped.items()}
    if max_share(train_counts) > 0.08 + 1e-12:
        raise SystemExit("cap failed: a template is above 8 % of the mix")

    # ---- 4. selection dev set ---------------------------------------------
    bench_dev: list[Item] = []
    bench_all_questions: set[str] = set()
    for directory in ("tier1", "fresh"):
        for path in sorted((BENCH / directory).glob("*.jsonl")):
            for item in read_items(path):
                if str(item.template_id) in HELD_OUT_TEMPLATES:
                    bench_all_questions.add(item.question)
                elif str(item.split) == "dev":
                    bench_dev.append(item)
    seen_dev_sample = stratified_sample(group(bench_dev), args.dev_per_template, key=item_key)
    new_dev_sample: dict[str, list[Item]] = {}
    for tid in sorted(new_dev_by_t):
        if tid in noul_new:
            pool_dev = new_dev_by_t[tid]
            new_dev_sample[tid] = balanced_sample(pool_dev, args.dev_per_template // 2,
                                                  label=lambda i: str(i.gold), key=item_key)
            class_balance.setdefault(tid, {})["dev_before"] = dict(Counter(str(i.gold) for i in pool_dev))
            class_balance[tid]["dev_after"] = dict(Counter(str(i.gold) for i in new_dev_sample[tid]))
        else:
            new_dev_sample.update(stratified_sample({tid: new_dev_by_t[tid]}, args.dev_per_template, key=item_key))
    dev_rows = [item for t in sorted(seen_dev_sample) for item in seen_dev_sample[t]]
    dev_rows += [item for t in sorted(new_dev_sample) for item in new_dev_sample[t]]
    if len(dev_rows) < args.min_dev:
        raise SystemExit(f"selection dev has {len(dev_rows)} items; the ADVISORY requires {args.min_dev}")
    # a template can appear in both samples (pubmed_pubtype_choice_v1 is a seen benchmark template and
    # a new pre-window template), so the counts add rather than merge
    dev_counter: Counter[str] = Counter()
    for sample in (seen_dev_sample, new_dev_sample):
        for tid, items in sample.items():
            dev_counter[tid] += len(items)
    dev_counts = dict(sorted(dev_counter.items()))

    # ---- 5. final leakage check on the rows to be written -------------------
    train_final = verify_no_leaks(train_rows, index, window_start=WINDOW_START)
    train_question_hits = sum(1 for i in train_rows if i.question in bench_all_questions)
    dev_held_hits = sum(1 for i in dev_rows if str(i.template_id) in HELD_OUT_TEMPLATES)
    dev_question_hits = sum(
        1 for i in dev_rows if i.question in bench_all_questions and str(i.template_id) not in HELD_OUT_TEMPLATES
    )
    new_dev_rows = [item for t in sorted(new_dev_sample) for item in new_dev_sample[t]]
    dev_new_index_hits = verify_no_leaks(new_dev_rows, index, window_start=WINDOW_START)
    train_split_ok = all(str(i.split) == "train" for i in train_rows)
    dev_split_ok = all(str(i.split) == "dev" for i in dev_rows)
    failures = {
        "train_index_or_date_hits": sum(train_final.get("by_reason", {}).values()),
        "train_held_out_question_text": train_question_hits,
        "dev_held_out_template": dev_held_hits,
        "dev_held_out_question_text": dev_question_hits,
        "new_dev_index_hits": sum(dev_new_index_hits.get("by_reason", {}).values()),
        "split_labels": 0 if (train_split_ok and dev_split_ok) else 1,
    }
    if any(failures.values()):
        raise SystemExit(f"leakage check failed: {failures}")

    # ---- 6. write -------------------------------------------------------------
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_path = OUT_DIR / "train.jsonl"
    dev_path = OUT_DIR / "dev.jsonl"
    with train_path.open("w", encoding="utf-8") as handle:
        for item in sorted(train_rows, key=item_key):
            handle.write(item.model_dump_json() + "\n")
    with dev_path.open("w", encoding="utf-8") as handle:
        for item in sorted(dev_rows, key=item_key):
            handle.write(item.model_dump_json() + "\n")

    blocks: Counter[str] = Counter()
    for item in train_rows:
        blocks[f"{item.source}|{item.template_id}|{item.qtype}|train"] += 1
    for item in dev_rows:
        blocks[f"{item.source}|{item.template_id}|{item.qtype}|dev"] += 1
    licences = Counter(f"{i.source}|{i.source_license}" for i in train_rows + dev_rows)
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, OSError):
        head = "unknown"
    manifest = {
        "kind": "student_v1_training_mix",
        "git_commit": head,
        "window_start": WINDOW_START.isoformat(),
        "held_out_templates": sorted(HELD_OUT_TEMPLATES),
        "cap": {"share": 0.08, "per_template_cap": cap,
                "rule": "fixed point of cap = floor(0.08 * sum(min(n_t, cap)))"},
        "base": {"path": str(BASE.relative_to(ROOT)), "sha256": sha256(BASE), "rows": len(base_rows),
                 "templates": len(base_by_t), "leak_check": "clean (verify_no_leaks before use)"},
        "new_components": {p.name: sha256(p) for p in comp_files},
        "new_templates_kept": sorted(kept),
        "new_train_rows_read": len(new_train_all),
        "new_train_removed_by_leak_check": {"total": len(new_train_all) - len(new_train),
                                            "report": train_report, "by_template": removed_train_t},
        "new_dev_rows_read": len(new_dev_all),
        "new_dev_removed_by_leak_check": {"total": len(new_dev_all) - len(new_dev),
                                          "report": dev_report, "by_template": removed_dev_t},
        "pool_counts_before_cap": dict(sorted(pool_counts.items())),
        "train_counts_after_cap": dict(sorted(train_counts.items())),
        "train": {"rows": len(train_rows), "templates": len(train_counts), **describe(train_counts)},
        "dev": {"rows": len(dev_rows), "templates": len(dev_counts),
                "per_template_request": args.dev_per_template, "counts": dev_counts},
        "distinct_training_templates_student_v0": len(base_by_t),
        "distinct_training_templates_student_v1": len(train_counts),
        "ratio_to_student_v0": len(train_counts) / len(base_by_t),
        "class_balance_new_noul_templates": class_balance,
        "by_source_template_qtype_split": dict(sorted(blocks.items())),
        "licences": dict(sorted(licences.items())),
        "leakage": {
            "reasons": list(LEAKAGE_REASONS),
            "index": index.to_dict(),
            "final_train_check": train_final,
            "final_train_held_out_question_text": train_question_hits,
            "final_dev_held_out_template": dev_held_hits,
            "final_dev_held_out_question_text": dev_question_hits,
            "final_new_dev_index_check": dev_new_index_hits,
            "train_split_is_train": train_split_ok,
            "dev_split_is_dev": dev_split_ok,
            "rule_note": "seen-template v0.2 dev items are the selection set by design and are not "
                         "checked against the index",
        },
        "outputs": {
            "train": {"path": str(train_path.relative_to(ROOT)), "sha256": sha256(train_path)},
            "dev": {"path": str(dev_path.relative_to(ROOT)), "sha256": sha256(dev_path)},
        },
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                                           encoding="utf-8")
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = {
        "kind": manifest["kind"],
        "git_commit": head,
        "cap": manifest["cap"],
        "train": manifest["train"],
        "dev": {"rows": manifest["dev"]["rows"], "templates": manifest["dev"]["templates"]},
        "distinct_training_templates_student_v0": manifest["distinct_training_templates_student_v0"],
        "distinct_training_templates_student_v1": manifest["distinct_training_templates_student_v1"],
        "ratio_to_student_v0": manifest["ratio_to_student_v0"],
        "new_templates_kept": manifest["new_templates_kept"],
        "train_counts_after_cap": manifest["train_counts_after_cap"],
        "new_train_removed_by_leak_check_total": manifest["new_train_removed_by_leak_check"]["total"],
        "new_dev_removed_by_leak_check_total": manifest["new_dev_removed_by_leak_check"]["total"],
        "final_leakage": {"train_index_or_date_hits": failures["train_index_or_date_hits"],
                          "held_out_hits": failures["train_held_out_question_text"]
                          + failures["dev_held_out_template"] + failures["dev_held_out_question_text"]},
    }
    SUMMARY.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"train": len(train_rows), "dev": len(dev_rows), "cap": cap,
                      "templates": len(train_counts), "max_share": round(max_share(train_counts), 4),
                      "removed_train": len(new_train_all) - len(new_train),
                      "removed_dev": len(new_dev_all) - len(new_dev)}))


if __name__ == "__main__":
    main()
