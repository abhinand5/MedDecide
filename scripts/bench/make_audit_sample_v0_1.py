#!/usr/bin/env python
"""F10: draw the operator audit sample from MedDecide-Bench v0.1 fresh test items.

150 fresh **test** items from the templates that survived the F3 screen, stratified across the
three sources and spread across their templates, with a fixed seed. Each row carries the item as
the model sees it plus the structured source field(s) the gold was derived from, so the operator
can judge whether each label is right without leaving the audit page (`tools/audit/audit.html`).

The sample is written under `outputs/` and is **never committed** — the repo is public and the
rows contain benchmark item text.

Usage:
    uv run python scripts/bench/make_audit_sample_v0_1.py \
        --out outputs/bench_v0_fix0/F10/audit_sample.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from meddecide.bench.schema import Item
from meddecide.utils.hashing import file_sha256
from meddecide.utils.io import read_jsonl
from meddecide.utils.provenance import Provenance, utcnow

PER_SOURCE = 50

# Structured source fields that explain each template's gold. Only fields of the source record —
# never a model output, and never the gold itself in text form.
GOLD_PROVENANCE_KEYS = {
    "ct_phase_choice_v1": ["phases", "source_field", "first_posted", "nct_id"],
    "ct_randomised_noul_v1": ["allocation", "study_type", "source_field", "nct_id"],
    "ct_primary_purpose_choice_v1": ["raw_primary_purpose", "source_field", "nct_id"],
    "ct_intervention_type_choice_v1": ["raw_intervention_types", "source_field", "nct_id"],
    "ct_healthy_volunteers_noul_v1": ["healthy_volunteers", "source_field", "nct_id"],
    "fda_boxed_warning_noul_v1": ["source_field", "set_id", "effective_time"],
    "fda_class_choice_v1": ["pharm_class_epc", "source_field", "set_id"],
    "fda_route_choice_v1": ["routes", "source_field", "set_id"],
    "pubmed_pubtype_choice_v1": ["publication_types", "source_field", "journal", "doi"],
    "pubmed_humans_noul_v1": ["check_tags", "source_field", "journal"],
    "pubmed_mesh_major_choice_v1": ["major_topics", "source_field", "journal"],
    "pubmed_observational_noul_v1": ["publication_types", "source_field", "journal"],
}


def _provenance(item: Item) -> dict[str, Any]:
    keys = GOLD_PROVENANCE_KEYS.get(item.template_id, ["source_field"])
    out: dict[str, Any] = {}
    for key in keys:
        if key in item.meta:
            out[key] = item.meta[key]
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fresh", type=Path, default=Path("data/bench/v0.1/fresh"))
    parser.add_argument("--screen", type=Path, default=Path("data/bench/v0.1/screen.json"))
    parser.add_argument("--out", type=Path,
                        default=Path("outputs/bench_v0_fix0/F10/audit_sample.jsonl"))
    parser.add_argument("--n-total", type=int, default=150)
    parser.add_argument("--per-source", type=int, default=PER_SOURCE)
    parser.add_argument("--seed", type=int, default=20261006)
    args = parser.parse_args()

    screen = json.loads(args.screen.read_text())
    kept = sorted(t["template_id"] for t in screen["templates"]
                  if not t["drop"] and t["tier"] == "fresh")

    rows: list[Item] = []
    for path in sorted(args.fresh.glob("*.jsonl")):
        loaded, report = read_jsonl(path, Item)
        report.check_closes()
        rows.extend(i for i in loaded if str(i.split) == "test" and i.template_id in kept)

    by_source: dict[str, list[Item]] = defaultdict(list)
    for item in rows:
        by_source[item.source].append(item)

    rng = random.Random(args.seed)
    sample: list[Item] = []
    per_source_selected: dict[str, int] = {}
    for source in sorted(by_source):
        pool = sorted(by_source[source], key=lambda i: i.item_id)
        # round-robin across the source's templates so the sample is spread, not dominated by the
        # largest template, then random within each template
        by_template: dict[str, list[Item]] = defaultdict(list)
        for item in pool:
            by_template[item.template_id].append(item)
        for template_rows in by_template.values():
            rng.shuffle(template_rows)
        queues = [by_template[t] for t in sorted(by_template)]
        picked: list[Item] = []
        while queues and len(picked) < args.per_source:
            for queue in list(queues):
                if not queue:
                    queues.remove(queue)
                    continue
                picked.append(queue.pop())
                if len(picked) >= args.per_source:
                    break
        sample.extend(picked)
        per_source_selected[source] = len(picked)

    sample = sample[: args.n_total]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for item in sample:
            payload = {
                "item_id": item.item_id,
                "source": item.source,
                "template_id": item.template_id,
                "qtype": str(item.qtype),
                "skill": str(item.skill),
                "split": str(item.split),
                "record_date": str(item.record_date),
                "source_record_id": item.source_record_id,
                "source_url": item.source_url,
                "source_license": item.source_license,
                "state": item.state,
                "question": item.question,
                "options": [{"key": o.key, "label": o.label} for o in item.options],
                "gold_key": item.gold,
                "gold_label": item.options[item.gold_index].label,
                "gold_provenance": _provenance(item),
                "strict_post_teacher": bool(item.meta.get("strict_post_teacher")),
            }
            fh.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")

    by_template_counts: dict[str, int] = defaultdict(int)
    for item in sample:
        by_template_counts[item.template_id] += 1
    manifest = {
        "generated_at_utc": utcnow(),
        "n_rows": len(sample),
        "seed": args.seed,
        "benchmark": str(args.fresh),
        "kept_templates": kept,
        "per_source": per_source_selected,
        "per_template": dict(sorted(by_template_counts.items())),
        "n_strict_slice": sum(1 for i in sample if i.meta.get("strict_post_teacher")),
        "sample_sha256": file_sha256(args.out),
        "note": (
            "Item text stays in outputs/ (gitignored): this file is never committed. Only this "
            "manifest's counts and hash are committed."
        ),
    }
    manifest_path = args.out.with_name("audit_sample_manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    prov = Provenance(
        run_name="F10_audit_sample",
        command=" ".join([__import__("sys").executable, *__import__("sys").argv]),
        seed=args.seed,
        config={"fresh": str(args.fresh), "n_total": args.n_total, "per_source": args.per_source},
    )
    prov.finish().write(args.out.with_name("audit_sample_provenance.json"))
    print(json.dumps({k: v for k, v in manifest.items()
                      if k in ("n_rows", "per_source", "per_template", "n_strict_slice",
                               "sample_sha256")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
