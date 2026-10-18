"""osler_v3 X6 partitions (ADVISORY X6 steps 1-2): the clinical-general (CG) dev proxy and fast dev v3, fixed before any Stage A run.

CG (the dev proxy for the external panel; never the panel itself):
- the knowledge dev v2 (4,000 MedQA and MedMCQA train hold-outs, osler_v2);
- the fast dev v2 items of the clinical, non-domain template families of mix v2 (medqa, medmcqa, pubmedqa, mmlu_medical,
  medquad, pubhealth, nfcorpus, trec_covid), minus any whose normalised text already appears in the knowledge dev v2 (osler_v2's
  fast dev repeats 4,122 of its knowledge items under other item ids; the duplicates are counted, not dropped silently).

Fast dev v3 = fast dev v2 (all 11,323 items) + up to 400 dev items per new template (X1 sources, X3 restyled pages, X4 charts),
drawn in a stable-hash order. The partitions are written to data/bench/osler_v3/ (gitignored) and the counts to
outputs/osler_v3/X6/manifest.json.

Usage:  source scripts/pod_env.sh && uv run --frozen python scripts/osler_v3/x6_partitions.py [--with-new-templates]
The new-template pools (x1_dev, x3 dev, chart dev) are read when present and recorded as absent otherwise.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from meddecide.panel.overlap import text_key
from meddecide.utils.hashing import stable_hash
from meddecide.utils.io import write_json, write_jsonl
from meddecide.utils.provenance import git_commit, utcnow

REPO = Path(__file__).resolve().parents[2]
FASTDEV_V2 = REPO / "data" / "bench" / "osler_v2" / "fastdev_v2.jsonl"
KNOWLEDGE_DEV_V2 = REPO / "data" / "bench" / "osler_v2" / "knowledge_dev_v2.jsonl"
POOLS = {  # new-template dev pools, if built
    "x1": REPO / "data" / "bench" / "osler_v3" / "x1_dev.jsonl",
    "x3": REPO / "data" / "bench" / "osler_v3" / "x3_dev.jsonl",
    "x4": REPO / "data" / "bench" / "osler_v3" / "x4_dev.jsonl",  # mix rows of x4_to_mix.py (blocked templates excluded)
}
CG_TEMPLATES = (
    "medqa_usmle4_v1", "medmcqa_4opt_v1", "pubmedqa_ynm_v1", "mmlu_mc_v1", "medquad_routing_v1",
    "pubhealth_claim_true_noul_v1", "nfcorpus_relevant_noul_v1", "trec_covid_graded_score_v1", "trec_covid_relevant_noul_v1",
)
OUT_MANIFEST = REPO / "outputs" / "osler_v3" / "X6" / "manifest.json"
CG_OUT = REPO / "data" / "bench" / "osler_v3" / "cg_v3.jsonl"
FASTDEV_V3_OUT = REPO / "data" / "bench" / "osler_v3" / "fastdev_v3.jsonl"
PER_TEMPLATE_NEW = 400  # ADVISORY X6 step 1


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def key_of(row: dict[str, Any]) -> str:
    return text_key(row["state"], row["question"], [o["label"] for o in row["options"]])


def build_cg(fastdev: list[dict[str, Any]], knowledge: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    known_text = {key_of(r) for r in knowledge}
    counts: Counter[str] = Counter()
    cg = [dict(r, cg_partition="knowledge_dev_v2") for r in knowledge]
    counts["knowledge_dev_v2"] = len(knowledge)
    seen_text = set(known_text)
    for row in fastdev:
        if row["template_id"] not in CG_TEMPLATES:
            continue
        counts["fastdev_clinical_family_items"] += 1
        text = key_of(row)
        if text in known_text:
            counts["dropped_text_already_in_knowledge_dev_v2"] += 1
            continue
        if text in seen_text:
            counts["dropped_text_duplicate_within_cg"] += 1
            continue
        seen_text.add(text)
        cg.append(dict(row, cg_partition="fastdev_v2_clinical_family"))
        counts["kept_fastdev_clinical_family"] += 1
    counts["cg_total"] = len(cg)
    return cg, {"counts": dict(counts), "templates": list(CG_TEMPLATES)}


def build_fastdev_v3(fastdev: list[dict[str, Any]], pools: dict[str, list[dict[str, Any]]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = [dict(r, fastdev_partition="fastdev_v2") for r in fastdev]
    per_template: dict[str, dict[str, int]] = {}
    by_template: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pool in pools.values():
        for row in pool:
            by_template[row["template_id"]].append(row)
    for template in sorted(by_template):
        items = sorted(by_template[template], key=lambda r: stable_hash({"fd3": r["item_id"]}, length=16))
        keep = items[:PER_TEMPLATE_NEW]
        rows.extend(dict(r, fastdev_partition="new_template_dev") for r in keep)
        per_template[template] = {"pool": len(items), "kept": len(keep)}
    return rows, {"fastdev_v2_items": len(fastdev), "new_template_items": sum(v["kept"] for v in per_template.values()),
                  "per_template": per_template, "pools_present": {k: len(v) for k, v in pools.items()}}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--with-new-templates", action="store_true", help="also build fast dev v3 from the new-template dev pools")
    args = ap.parse_args()
    fastdev = read_jsonl(FASTDEV_V2)
    knowledge = read_jsonl(KNOWLEDGE_DEV_V2)
    cg, cg_report = build_cg(fastdev, knowledge)
    if len(cg) < 4000:
        raise SystemExit(f"CG holds {len(cg)} items; ADVISORY X6 requires at least 4,000")
    CG_OUT.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(CG_OUT, cg)
    manifest: dict[str, Any] = {"built_utc": utcnow(), "git_commit": git_commit(REPO), "command": "scripts/osler_v3/x6_partitions.py",
                                "cg": cg_report | {"cg_items": len(cg),
                                                   "cg_by_partition": dict(Counter(r["cg_partition"] for r in cg)),
                                                   "cg_by_template": dict(Counter(r["template_id"] for r in cg))},
                                "sources": {"fastdev_v2": str(FASTDEV_V2.relative_to(REPO)),
                                            "knowledge_dev_v2": str(KNOWLEDGE_DEV_V2.relative_to(REPO))}}
    if args.with_new_templates:
        pools = {name: read_jsonl(path) for name, path in POOLS.items() if path.exists()}
        fastdev_v3, fd3 = build_fastdev_v3(fastdev, pools)
        write_jsonl(FASTDEV_V3_OUT, fastdev_v3)
        manifest["fastdev_v3"] = fd3 | {"items": len(fastdev_v3), "file": str(FASTDEV_V3_OUT.relative_to(REPO))}
    write_json(OUT_MANIFEST, manifest)
    print(f"CG: {len(cg)} items ({cg_report['counts']})", flush=True)
    if "fastdev_v3" in manifest:
        print(f"fast dev v3: {manifest['fastdev_v3']['items']} items", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
