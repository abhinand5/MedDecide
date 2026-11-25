#!/usr/bin/env python
"""T11: draw the operator audit sample and write it as JSONL for the audit page.

150 fresh **test** items, stratified 50 per source (ClinicalTrials.gov, openFDA, PubMed),
spread across the kept templates with a fixed seed. Each row carries the item as the model
sees it plus the structured source field(s) the gold was derived from, so the operator can
judge whether the label is right without leaving the page.

The sample is written under ``outputs/`` and is **never committed** (the repo is public).

Usage:
    uv run python scripts/bench/make_audit_sample.py --out outputs/bench_v0/T11/audit_sample.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

from meddecide.bench.schema import Item, Split
from meddecide.utils.hashing import file_sha256
from meddecide.utils.io import read_jsonl
from meddecide.utils.provenance import Provenance, utcnow

PER_SOURCE = 50
# Fields that explain the gold, per template. Only structured source fields — no model output.
GOLD_PROVENANCE_KEYS = {
    "ct_phase_choice_v1": ["phases", "source_field", "first_posted", "nct_id"],
    "ct_randomised_noul_v1": ["allocation", "study_type", "source_field", "nct_id"],
    "ct_primary_purpose_choice_v1": ["raw_primary_purpose", "source_field", "nct_id"],
    "ct_intervention_type_choice_v1": ["raw_intervention_types", "source_field", "nct_id"],
    "fda_boxed_warning_noul_v1": ["source_field", "set_id", "effective_time"],
    "fda_class_choice_v1": ["pharm_class_epc", "source_field", "set_id"],
    "fda_route_choice_v1": ["routes", "source_field", "set_id"],
    "pubmed_pubtype_choice_v1": ["publication_types", "source_field", "journal", "doi"],
    "pubmed_humans_noul_v1": ["check_tags", "source_field", "journal"],
    "pubmed_mesh_major_choice_v1": ["major_topics", "source_field", "journal"],
    "pubmed_observational_noul_v1": ["publication_types", "source_field", "journal"],
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fresh", type=Path, default=Path("data/bench/fresh"))
    parser.add_argument("--screen", type=Path, default=Path("outputs/bench_v0/T8/screen.json"))
    parser.add_argument("--config", type=Path, default=Path("configs/bench_v0.yaml"))
    parser.add_argument("--out", type=Path, default=Path("outputs/bench_v0/T11/audit_sample.jsonl"))
    parser.add_argument("--n-total", type=int, default=150)
    parser.add_argument("--per-source", type=int, default=PER_SOURCE)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    seed = int(cfg.get("seed", 0)) if args.seed is None else args.seed

    dropped_templates: set[str] = set()
    if args.screen.is_file():
        screen = json.loads(args.screen.read_text())
        dropped_templates = {
            tid for tid, entry in screen["templates"].items() if entry.get("drop")
        }

    by_source: dict[str, list[Item]] = defaultdict(list)
    for path in sorted(args.fresh.glob("*.jsonl")):
        rows, report = read_jsonl(path, Item)
        report.check_closes()
        for row in rows:
            if row.split is not Split.TEST:
                continue
            if row.template_id in dropped_templates:
                continue
            by_source[row.source].append(row)

    rng = random.Random(f"{seed}:audit_sample")
    sample: list[dict[str, Any]] = []
    per_source_counts: dict[str, int] = {}
    for source in sorted(by_source):
        pool = sorted(by_source[source], key=lambda i: i.item_id)
        # stratify by template, round-robin so every kept template is represented
        by_template: dict[str, list[Item]] = defaultdict(list)
        for item in pool:
            by_template[item.template_id].append(item)
        for items in by_template.values():
            rng.shuffle(items)
        ordered: list[Item] = []
        while any(by_template.values()):
            for template_id in sorted(by_template):
                if by_template[template_id]:
                    ordered.append(by_template[template_id].pop())
        picked = ordered[: args.per_source]
        per_source_counts[source] = len(picked)
        for item in picked:
            provenance_keys = GOLD_PROVENANCE_KEYS.get(item.template_id, ["source_field"])
            sample.append(
                {
                    "item_id": item.item_id,
                    "template_id": item.template_id,
                    "source": item.source,
                    "skill": item.skill,
                    "qtype": str(item.qtype),
                    "record_date": item.record_date.isoformat(),
                    "source_url": item.source_url,
                    "source_record_id": item.source_record_id,
                    "state": item.state,
                    "question": item.question,
                    "options": [
                        {"key": o.key, "label": o.label, "description": o.description}
                        for o in item.options
                    ],
                    "gold": item.gold,
                    "gold_label": item.options[item.gold_index].label,
                    "gold_provenance": {
                        k: item.meta.get(k) for k in provenance_keys if k in item.meta
                    },
                }
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for row in sample:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "generated_at_utc": utcnow(),
        "seed": seed,
        "n_items": len(sample),
        "per_source": per_source_counts,
        "per_template": {
            t: sum(1 for r in sample if r["template_id"] == t)
            for t in sorted({r["template_id"] for r in sample})
        },
        "dropped_templates_excluded": sorted(dropped_templates),
        "out": str(args.out),
        "out_sha256": file_sha256(args.out),
    }
    (args.out.with_name("audit_sample_manifest.json")).write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    prov = Provenance(
        run_name="T11_audit_sample", command="uv run python scripts/bench/make_audit_sample.py",
        seed=seed, config={"per_source": args.per_source},
    )
    prov.finish().write(args.out.with_name("audit_sample_provenance.json"))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
