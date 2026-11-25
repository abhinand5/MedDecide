#!/usr/bin/env python
"""Fresh-process re-derivation of the student_v1 mix numbers (R6). Plain Python: no numpy, no
meddecide import, so it cannot share a bug with v1_assemble_mix.py.

Re-reads data/train/student_v1/{train,dev}.jsonl and checks: row and per-template counts against the
manifest, the 8 % share bound, the dev size, held-out template and question absence, the window date
rule on train rows, and the split labels. Run as ``python -I scripts/student/v1_check_mix.py``.
Exit status 1 on any mismatch.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MIX = ROOT / "data" / "train" / "student_v1"
WINDOW_START = date(2026, 3, 1)
HELD_OUT = {
    "ct_phase_choice_v1", "fda_boxed_warning_noul_v1", "pubmed_humans_noul_v1", "ct_arm_role_noul_v1",
}


def rows(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            yield json.loads(line)


def main() -> int:
    manifest = json.loads((MIX / "manifest.json").read_text(encoding="utf-8"))
    failures: list[str] = []
    checks: list[str] = []

    def check(label: str, got, want) -> None:
        ok = got == want
        checks.append(f"{'OK  ' if ok else 'FAIL'} {label}: fresh={got!r} manifest={want!r}")
        if not ok:
            failures.append(label)

    # held-out question texts come from the benchmark's own held-out items (both splits)
    held_questions: set[str] = set()
    for folder in ("tier1", "fresh"):
        for path in sorted((ROOT / "data" / "bench" / "v0.2" / folder).glob("*.jsonl")):
            for row in rows(path):
                if row["template_id"] in HELD_OUT:
                    held_questions.add(row["question"])

    train_counts: Counter[str] = Counter()
    train_split: Counter[str] = Counter()
    train_held_template = train_held_question = train_late = 0
    for line_row in rows(MIX / "train.jsonl"):
        tid = line_row["template_id"]
        train_counts[tid] += 1
        train_split[line_row["split"]] += 1
        if tid in HELD_OUT:
            train_held_template += 1
        if line_row["question"] in held_questions:
            train_held_question += 1
        if line_row["record_date"] >= WINDOW_START.isoformat():
            train_late += 1
    dev_counts: Counter[str] = Counter()
    dev_split: Counter[str] = Counter()
    dev_held_template = dev_held_question = 0
    for line_row in rows(MIX / "dev.jsonl"):
        tid = line_row["template_id"]
        dev_counts[tid] += 1
        dev_split[line_row["split"]] += 1
        if tid in HELD_OUT:
            dev_held_template += 1
        if line_row["question"] in held_questions:
            dev_held_question += 1

    total_train = sum(train_counts.values())
    total_dev = sum(dev_counts.values())
    check("train rows", total_train, manifest["train"]["rows"])
    check("dev rows", total_dev, manifest["dev"]["rows"])
    check("train templates", len(train_counts), manifest["train"]["templates"])
    check("dev templates", len(dev_counts), manifest["dev"]["templates"])
    check("train per-template counts equal the manifest",
          dict(sorted(train_counts.items())), dict(sorted(manifest["train_counts_after_cap"].items())))
    check("dev per-template counts equal the manifest",
          dict(sorted(dev_counts.items())), dict(sorted(manifest["dev"]["counts"].items())))
    max_share = max(train_counts.values()) / total_train
    check("max template share <= 0.08", max_share <= 0.08 + 1e-12, True)
    check("dev rows >= 4000", total_dev >= 4000, True)
    check("held-out template rows in train", train_held_template, 0)
    check("held-out template rows in dev", dev_held_template, 0)
    check("held-out question text in train", train_held_question, 0)
    check("held-out question text in dev", dev_held_question, 0)
    check("train rows dated on or after the window", train_late, 0)
    check("train rows are labelled train", train_split.get("train", 0), total_train)
    check("dev rows are labelled dev", dev_split.get("dev", 0), total_dev)
    ratio = len(train_counts) / manifest["distinct_training_templates_student_v0"]
    check("distinct training templates >= 1.5 x student_v0 (14)", ratio >= 1.5, True)

    print("\n".join(checks))
    print(f"checked {len(checks)} items; failures: {len(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
