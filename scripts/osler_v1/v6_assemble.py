"""V6: assemble the MR evaluation suite, its held-out file and the fast dev set (osler_v1 ADVISORY V6, step 1).

Rows are written in the common-set shape the O2 runners read (item_id, benchmark, set_name, template_id, qtype, state,
question, options, gold), with benchmark ``mr_v0`` for the MR items and ``mix_v2`` for the mix v2 dev part of the fast dev
set. The test and held-out files are the splits as built. The fast dev set is drawn from dev splits only: up to 400 items per
MR template, 3,000 items of the mix v2 dev split stratified by template, and every tier-1 knowledge dev item (MedQA and
MedMCQA) of that split. No test or held-out item is read for any choice.

Outputs (gitignored, under data/bench/osler_mr_v0/): mr_test.jsonl, mr_heldout.jsonl, mr_fastdev.jsonl, manifest.json.
The manifest holds counts, hashes and the source list only.

The V4 teacher items join the test suite when they exist; re-run this script then (ADVISORY §6).

Usage:
    source scripts/pod_env.sh && uv run --frozen python scripts/osler_v1/v6_assemble.py
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from meddecide.train.osler_arm import stratified_subset
from meddecide.utils.hashing import file_sha256
from meddecide.utils.io import write_json
from meddecide.utils.provenance import git_commit, utcnow

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "data" / "bench" / "osler_mr_v0"
V1 = REPO / "data" / "gen" / "osler_v1"
SYN = REPO / "data" / "gen" / "osler_v1_synthea"
REALTEXT = V1 / "realtext"
MIX_DEV = REPO / "data" / "train" / "osler_v0" / "dev.jsonl"
FAST_DEV_PER_TEMPLATE = 400
FAST_DEV_MIX = 3000
MR_BENCHMARK = "mr_v0"
MIX_BENCHMARK = "mix_v2"


class Keyed:
    """A row with the attribute ``stratified_subset`` groups by (its ``template``)."""

    def __init__(self, raw: dict[str, Any], template: str) -> None:
        self.raw = raw
        self.template = template
        self.item_id = raw["item_id"]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def realtext_files() -> list[Path]:
    return [p for p in sorted(REALTEXT.glob("*.jsonl")) if p.stem != "manifest"]


def mr_row(raw: dict[str, Any], *, set_name: str, template: str) -> dict[str, Any]:
    return {
        "item_id": raw["item_id"],
        "benchmark": MR_BENCHMARK,
        "set_name": set_name,
        "template_id": template,
        "qtype": raw["qtype"],
        "state": raw["state"],
        "question": raw["question"],
        "options": [{"key": o["key"], "label": o["label"]} for o in raw["options"]],
        "gold": raw["gold"],
        "split": raw.get("split"),
        "held_out": bool(raw.get("held_out", False)),
        "source": raw.get("source") or (raw.get("meta") or {}).get("source"),
    }


def mix_row(raw: dict[str, Any], *, set_name: str) -> dict[str, Any]:
    row = mr_row(raw, set_name=set_name, template=str(raw["template_id"]))
    row["benchmark"] = MIX_BENCHMARK
    row["source"] = raw.get("source")
    return row


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    sources: list[dict[str, Any]] = []

    # 1. the test suite: the test splits as built (V1, V5a and every V5b file), each item once
    test_rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for name, path in [("v1", V1 / "test.jsonl"), ("v5a_synthea", SYN / "test.jsonl"),
                       *[(f"v5b_{p.stem}", p) for p in realtext_files()]]:
        rows = read_jsonl(path)
        kept = 0
        for r in rows:
            if r.get("split") != "test" or r["item_id"] in seen:
                continue
            seen.add(r["item_id"])
            test_rows.append(mr_row(r, set_name="mr_test", template=r["generator"]))
            counts["test"][r["generator"]] += 1
            kept += 1
        sources.append({"name": name, "path": str(path.relative_to(REPO)), "test_rows_read": kept})

    # 2. the held-out file: the held-out indicator and family, reported separately and never in a verdict
    held_rows: list[dict[str, Any]] = []
    for path in (V1 / "heldout_test.jsonl", SYN / "heldout_test.jsonl"):
        for r in read_jsonl(path):
            held_rows.append(mr_row(r, set_name="mr_heldout", template=r["generator"]))
            counts["heldout"][r["generator"]] += 1

    # 3. the fast dev set: dev splits only
    dev_by_template: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in [V1 / "dev.jsonl", SYN / "dev.jsonl", *realtext_files()]:
        for r in read_jsonl(path):
            if r.get("split") == "dev":
                dev_by_template[r["generator"]].append(r)
    fast: list[dict[str, Any]] = []
    for template, rows in sorted(dev_by_template.items()):
        if len(rows) > FAST_DEV_PER_TEMPLATE:
            keyed = [Keyed(r, template) for r in rows]
            rows = [k.raw for k in stratified_subset(keyed, key="template", size=FAST_DEV_PER_TEMPLATE, seed=0)]
        for r in rows:
            fast.append(mr_row(r, set_name="mr_fastdev", template=template))
        counts["fastdev_mr"][template] = len(rows)

    mix_dev = read_jsonl(MIX_DEV)
    tier1 = [r for r in mix_dev if r.get("source") in ("medqa", "medmcqa")]
    keyed_mix = [Keyed(r, str(r["template_id"])) for r in mix_dev]
    sample = stratified_subset(keyed_mix, key="template", size=min(FAST_DEV_MIX, len(keyed_mix)), seed=0)
    for k in sample:
        fast.append(mix_row(k.raw, set_name="mix_fastdev"))
    for r in tier1:
        fast.append(mix_row(r, set_name="mix_tier1_dev"))
    counts["fastdev_mix"]["stratified_sample"] = len(sample)
    counts["fastdev_mix"]["tier1_all"] = len(tier1)
    # the fast dev set is the union: a tier-1 item drawn into the stratified sample is kept once (first occurrence)
    first_seen: set[str] = set()
    union: list[dict[str, Any]] = []
    for r in fast:
        if r["item_id"] not in first_seen:
            first_seen.add(r["item_id"])
            union.append(r)
    counts["fastdev_mix"]["duplicates_removed"] = len(fast) - len(union)
    fast = union

    for name, rows in (("mr_test.jsonl", test_rows), ("mr_heldout.jsonl", held_rows), ("mr_fastdev.jsonl", fast)):
        with (OUT / name).open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    manifest = {
        "written_utc": utcnow(), "git_commit": git_commit(REPO),
        "rows": {"mr_test": len(test_rows), "mr_heldout": len(held_rows), "mr_fastdev": len(fast)},
        "counts": {k: dict(v) for k, v in counts.items()},
        "sources": sources,
        "files_sha256": {n: file_sha256(OUT / n) for n in ("mr_test.jsonl", "mr_heldout.jsonl", "mr_fastdev.jsonl")},
        "rules": ("test and held-out files are the splits as built; the fast dev set is drawn from dev splits only; "
                  "held-out and test items never enter a choice"),
    }
    write_json(OUT / "manifest.json", manifest)
    print(json.dumps(manifest["rows"]), flush=True)


if __name__ == "__main__":
    main()
