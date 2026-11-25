"""V8 preparation: the fixed fast dev set in the mix v2 row schema, read by every block run (ADVISORY V6 step 1 and V8).

data/bench/osler_mr_v0/mr_fastdev.jsonl (V6, common-set shape) becomes data/bench/osler_mr_v0/fastdev_mix.jsonl: the MR items
converted to the mix v2 training schema (the source family and its licence as the V7 blocks give them) and the mix v2 items
copied unchanged from the osler_v0 dev rows by item id. The tier-1 knowledge items also go to fastdev_knowledge.jsonl in the
common-set shape, for the zero-shot knowledge check of B0 (ADVISORY V8). The set is written once, before the first block run,
and is not changed afterwards. Unknown templates and item mismatches raise; nothing is dropped silently.

Usage:
    source scripts/pod_env.sh && uv run --frozen python scripts/osler_v1/v8_prepare.py
"""

from __future__ import annotations

import importlib.util
import json
from collections import Counter
from pathlib import Path
from typing import Any

from meddecide.mix.leakage import bare_collisions
from meddecide.train.blocks import KNOWLEDGE_SOURCES, MR_SOURCES
from meddecide.utils.hashing import file_sha256
from meddecide.utils.io import write_json
from meddecide.utils.provenance import git_commit, utcnow

REPO = Path(__file__).resolve().parents[2]
FAST = REPO / "data" / "bench" / "osler_mr_v0" / "mr_fastdev.jsonl"
MIX_DEV = REPO / "data" / "train" / "osler_v0" / "dev.jsonl"
OUT = REPO / "data" / "bench" / "osler_mr_v0" / "fastdev_mix.jsonl"
KNOWLEDGE = REPO / "data" / "bench" / "osler_mr_v0" / "fastdev_knowledge.jsonl"
MANIFEST = REPO / "outputs" / "osler_v1" / "V8" / "fastdev_mix.json"
RECORD_DATE = "2026-10-13"
REAL_TEXT_FAMILIES = (("synthea_", "mr_synthea"), ("codiesp_", "mr_codiesp"), ("mtsamples_", "mr_mtsamples"),
                      ("pmc_", "mr_pmc"))


def _v7_sources() -> dict[str, dict[str, str]]:
    """The source table of the V7 blocks (URLs and licences), read from the script so the two cannot disagree."""
    spec = importlib.util.spec_from_file_location("v7_blocks", REPO / "scripts" / "osler_v1" / "v7_blocks.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SOURCES


def dev_bare_sets() -> dict[str, set[str]]:
    """The bare record ids of the protected sets for the dev check, as O4 applies them: every set but the v0.2 dev set."""
    spec = importlib.util.spec_from_file_location("o4_build_mix", REPO / "scripts" / "osler" / "o4_build_mix.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.protected_sets()["dev"]["bare"]


def family(template: str) -> str:
    """The mix v2 source family of an MR template; raises for a template that has no family."""
    for prefix, source in REAL_TEXT_FAMILIES:
        if template.startswith(prefix):
            return source
    if template.startswith("mr_"):
        return "mr_v1"
    raise ValueError(f"no source family for template {template!r}")


def main() -> None:
    sources = _v7_sources()
    dev_bare = dev_bare_sets()
    mix_dev: dict[str, dict[str, Any]] = {}
    with MIX_DEV.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                row = json.loads(line)
                mix_dev[row["item_id"]] = row
    with FAST.open(encoding="utf-8") as fh:
        raw_rows = [json.loads(line) for line in fh if line.strip()]

    counts: Counter[str] = Counter()
    out_rows: list[dict[str, Any]] = []
    knowledge_rows: list[dict[str, Any]] = []
    for r in raw_rows:
        if r["benchmark"] == "mr_v0":
            if r.get("split") != "dev":
                raise ValueError(f"MR fast dev item {r['item_id']} is not a dev item")
            if bare_collisions({"source_record_id": r["item_id"]}, dev_bare):
                raise ValueError(f"MR fast dev item {r['item_id']} collides with a protected record id")
            fam = family(r["template_id"])
            lic = sources[fam]
            out_rows.append({
                "item_id": r["item_id"], "tier": "training", "source": fam, "source_record_id": r["item_id"],
                "source_url": lic["url"], "source_license": lic["licence"], "record_date": RECORD_DATE, "split": "dev",
                "template_id": r["template_id"], "skill": "mr_audit", "qtype": r["qtype"], "state": r["state"],
                "question": r["question"], "options": r["options"], "gold": r["gold"],
            })
            counts[f"mr:{fam}"] += 1
        elif r["benchmark"] == "mix_v2":
            original = mix_dev.get(r["item_id"])
            if original is None:
                raise ValueError(f"mix item {r['item_id']} is not in the mix v2 dev rows")
            if original["gold"] != r["gold"] or original["template_id"] != r["template_id"]:
                raise ValueError(f"mix item {r['item_id']} disagrees with its dev row")
            hits = bare_collisions(original, dev_bare)
            if hits:
                counts["dropped: bare record id held by " + "+".join(hits) + " (V7-D4)"] += 1
                continue
            out_rows.append(original)
            counts[f"mix:{original['source']}"] += 1
            if original["source"] in KNOWLEDGE_SOURCES:
                knowledge_rows.append(r)
        else:
            raise ValueError(f"unexpected benchmark {r['benchmark']!r} in the fast dev set")

    if len({r["item_id"] for r in out_rows}) != len(out_rows):
        raise ValueError("duplicate item ids in the fast dev set")
    if len(knowledge_rows) != sum(n for key, n in counts.items() if key.split(":", 1)[1] in KNOWLEDGE_SOURCES):
        raise ValueError("the knowledge file does not match the knowledge items of the fast dev set")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    for path, rows in ((OUT, out_rows), (KNOWLEDGE, knowledge_rows)):
        with path.open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    partitions = {
        "mr": sum(1 for r in out_rows if r["source"] in MR_SOURCES),
        "general": sum(1 for r in out_rows if r["source"] not in MR_SOURCES),
        "knowledge": len(knowledge_rows),
    }
    write_json(MANIFEST, {
        "written_utc": utcnow(), "git_commit": git_commit(REPO), "rows": len(out_rows),
        "partitions": partitions, "knowledge_file_rows": len(knowledge_rows),
        "counts_by_source": dict(sorted(counts.items())),
        "mr_templates": len({r["template_id"] for r in out_rows if r["source"] in MR_SOURCES}),
        "general_templates": len({r["template_id"] for r in out_rows if r["source"] not in MR_SOURCES}),
        "file": str(OUT.relative_to(REPO)), "sha256": file_sha256(OUT),
        "knowledge_file": str(KNOWLEDGE.relative_to(REPO)), "knowledge_sha256": file_sha256(KNOWLEDGE),
        "source_fast_dev_sha256": file_sha256(FAST), "source_mix_dev_sha256": file_sha256(MIX_DEV),
    })
    print(json.dumps({"rows": len(out_rows), "partitions": partitions}), flush=True)


if __name__ == "__main__":
    main()
