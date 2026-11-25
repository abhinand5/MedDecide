#!/usr/bin/env python
"""Additional analysis (S1): does a higher near-miss option count change the accuracy?

`pubmed_mesh_major_choice_v2` replaced unrelated distractors with MeSH tree siblings and the
8B model still scored 0.9580 (0.9830 for 9B), i.e. the template stayed **saturated** by the
S1 acceptance rule (9B <= 0.90). The obvious next lever is the number of options, and the
PROGRAM lists "collapse at high option counts" as a failure mode worth measuring rather than
assuming.

This script builds an N-option variant **offline** from the frozen v0.2 items plus the MeSH
index (no PubMed re-parse), writes it to a scratch benchmark directory, and leaves scoring to
the ordinary harness. It is explicitly an **additional analysis**:

* it is not part of `data/bench/v0.2/` and nothing in v0.2 depends on it;
* the variant carries template id `<source template>_options<N>`, so it can never be confused
  with the benchmark template;
* the extra distractors are sampled with a per-item seeded RNG, so the variant is reproducible.

Usage:
    uv run python scripts/bench/mesh_options_experiment.py --n-options 8 \
        --out outputs/student_v0/S1/extra/options8
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from meddecide.bench.mesh import load_index
from meddecide.bench.schema import Item, compute_item_id
from meddecide.utils.hashing import stable_hash
from meddecide.utils.io import read_jsonl, write_json, write_jsonl

SOURCE_TEMPLATE = "pubmed_mesh_major_choice_v2"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench", type=Path, default=Path("data/bench/v0.2"))
    parser.add_argument("--mesh-index", type=Path, default=Path("/workspace/tmp/mesh/mesh_index.json"))
    parser.add_argument("--n-options", type=int, default=8)
    parser.add_argument("--out", type=Path, default=Path("outputs/student_v0/S1/extra/options8"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rows, report = read_jsonl(args.bench / "fresh" / "pubmed.jsonl", Item)
    report.check_closes()
    base = [row for row in rows if row.template_id == SOURCE_TEMPLATE]
    if not base:
        raise SystemExit(f"no {SOURCE_TEMPLATE} items under {args.bench}")
    index = load_index(args.mesh_index)

    template_id = f"{SOURCE_TEMPLATE}_options{args.n_options}"
    out_rows: list[Item] = []
    rule_counts: Counter[str] = Counter()
    dropped = 0
    for row in base:
        gold = str(row.meta["gold_topic"])
        own = set(row.meta.get("major_topics") or [])
        existing = list(row.meta.get("distractors") or [])
        need = args.n_options - 1 - len(existing)
        rng = random.Random(f"{args.seed}:{stable_hash({'item': row.item_id}, length=16)}")
        extra: list[str] = []
        if need > 0:
            pool = [
                name for name in index.siblings(gold)
                if name != gold and name not in own and name not in existing
            ]
            # never offer a second correct answer, and never a sibling of one
            unsafe: set[str] = set()
            for other in own:
                if other != gold:
                    unsafe.update(index.siblings(other))
            pool = [name for name in pool if name not in unsafe]
            if len(pool) < need:
                dropped += 1
                continue
            extra = rng.sample(pool, need)
        labels = [gold, *existing, *extra]
        order = list(range(len(labels)))
        rng.shuffle(order)
        options = [
            {"key": chr(ord("A") + i), "label": labels[j]} for i, j in enumerate(order)
        ]
        gold_key = chr(ord("A") + order.index(0))
        payload = row.model_dump()
        payload["template_id"] = template_id
        payload["options"] = options
        payload["gold"] = gold_key
        payload["item_id"] = compute_item_id(
            row.source, row.source_record_id, template_id,
            int(row.meta.get("option_order_seed", 0)), str(row.split),
        )
        payload["meta"] = {
            **row.meta,
            "additional_analysis": "mesh option-count experiment (S1)",
            "base_template": SOURCE_TEMPLATE,
            "n_options": args.n_options,
            "extra_distractors": extra,
        }
        out_rows.append(Item.model_validate(payload))
        rule_counts["mesh_sibling"] += 1

    fresh_dir = args.out / "fresh"
    tier1_dir = args.out / "tier1"
    fresh_dir.mkdir(parents=True, exist_ok=True)
    tier1_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(fresh_dir / "pubmed.jsonl", out_rows)
    write_json(tier1_dir / "manifest.json", {"note": "empty tier 1 for the option-count experiment"})
    write_json(fresh_dir / "acceptance.json", {"strict_slice_start": "2026-09-10"})
    summary = {
        "template_id": template_id,
        "n_options": args.n_options,
        "n_items": len(out_rows),
        "n_test": sum(1 for r in out_rows if str(r.split) == "test"),
        "n_dev": sum(1 for r in out_rows if str(r.split) == "dev"),
        "n_dropped_insufficient_siblings": dropped,
        "gold_letter_counts": dict(sorted(Counter(r.gold for r in out_rows).items())),
        "distractor_rules": dict(rule_counts),
        "base_template": SOURCE_TEMPLATE,
        "note": "additional analysis; not part of data/bench/v0.2",
    }
    write_json(args.out / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
