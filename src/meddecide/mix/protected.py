"""Protected sets for the osler_v3 training rows: what no training row may contain (GOAL "Do not"; ADVISORY §3).

``load_protected`` returns three dictionaries (set name -> keys): the normalised text keys, the bare record ids and the
dataset-qualified record ids. The sets are osler_v0's training protected sets (the v0.2 test and dev, the external panel, the
robustness pack, the held-out generators and templates) plus the MR suite v1 test and held-out files, the knowledge dev v2,
the fast dev v2 and the EHR-Rel test. ``leakage_counts`` reports every hit by kind; a hit is a failure for the build, never a
silent filter (R4, R8).
"""

from __future__ import annotations

import importlib.util
import json
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from meddecide.mix.leakage import bare_collisions, leak_counts, qualified_collisions, row_text_key
from meddecide.utils.hashing import normalize_text, stable_hash

EXTRA_PROTECTED = {
    "mr_test": "data/bench/osler_mr_v0/mr_test.jsonl",
    "mr_heldout": "data/bench/osler_mr_v0/mr_heldout.jsonl",
    "mr_eval": "data/bench/osler_mr_v0/mr_eval.jsonl",
    "knowledge_dev_v2": "data/bench/osler_v2/knowledge_dev_v2.jsonl",
    "fastdev_v2": "data/bench/osler_v2/fastdev_v2.jsonl",
    "ehr_rel_test": "data/bench/osler_v2/ehr_rel_test.jsonl",
}
PROTECTED_SETS = dict[str, set[str]]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_protected(repo: Path) -> tuple[PROTECTED_SETS, PROTECTED_SETS, PROTECTED_SETS]:
    """Text keys, bare record ids and dataset-qualified record ids of every protected set, read from ``repo``."""
    spec = importlib.util.spec_from_file_location("o4_build_mix", repo / "scripts" / "osler" / "o4_build_mix.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load osler_v0's protected-set builder")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    prot = module.protected_sets()["train"]
    text = {f"v0_{k}": set(v) for k, v in prot["text"].items()}
    bare = {f"v0_{k}": set(v) for k, v in prot["bare"].items()}
    qualified = {f"v0_{k}": set(v) for k, v in prot["qualified"].items()}
    for name, rel in EXTRA_PROTECTED.items():
        text[name], bare[name], qualified[name] = set(), set(), set()
        for rec in read_jsonl(repo / rel):
            text[name].add(row_text_key(rec))
            bare[name].add(str(rec.get("source_record_id", "")))
            qualified[name].add(f"{rec.get('source', '')}/{rec.get('source_record_id', '')}")
    return text, bare, qualified


def leakage_counts(rows: Iterable[Mapping[str, Any]], text: PROTECTED_SETS, bare: PROTECTED_SETS,
                   qualified: PROTECTED_SETS) -> dict[str, int]:
    """Hits by kind over the rows: text keys and bare ids (``text:<set>``, ``record:<set>``), and rows whose qualified record
    id or PubMed id (``meta.pmid``, for abstracts) is in a protected set. Every set is reported, with zeros."""
    rows = list(rows)
    counts: Counter[str] = Counter(leak_counts(rows, text, bare))
    for row in rows:
        if bare_collisions(row, bare):
            counts["rows_bare_record_id_hit"] += 1
        if qualified_collisions(row, qualified):
            counts["rows_qualified_record_hit"] += 1
        pmid = str(row.get("meta", {}).get("pmid") or "")
        if pmid:
            hits = sorted(name for name, ids in bare.items() if pmid in ids)
            if hits:
                counts["rows_pmid_in_protected"] += 1
                for name in hits:
                    counts[f"rows_pmid_in_{name}"] += 1
    counts["rows_checked"] = len(rows)
    return dict(counts)


def hit_total(counts: Mapping[str, int]) -> int:
    """The number of hits in a ``leakage_counts`` result (every kind except the row count)."""
    return sum(v for k, v in counts.items() if k != "rows_checked" and k.startswith(("text:", "record:", "rows_")))


PARAGRAPH_MIN_CHARS = 60
PARAGRAPH_SOURCES = (
    "data/bench/v0.2/fresh/*.jsonl", "data/bench/v0.2/tier1/*.jsonl", "data/bench/v0.3_ext/panel.jsonl",
    "data/bench/v0.3_ext/robustness.jsonl",
)


def paragraph_key(text: str) -> str:
    """A stable key for one normalised paragraph (for document-level exclusion of real text)."""
    return stable_hash(normalize_text(text), length=16)


def protected_paragraph_keys(repo: Path) -> set[str]:
    """Keys of every paragraph of at least PARAGRAPH_MIN_CHARS characters in a protected state (the v0.2 and panel files, the
    MR suite files and the other protected bench files). A real document that shares one is excluded from the substrate."""
    files: list[Path] = []
    for pattern in PARAGRAPH_SOURCES:
        files.extend(sorted(repo.glob(pattern)))
    files.extend(repo / rel for rel in EXTRA_PROTECTED.values())
    keys: set[str] = set()
    for path in files:
        for rec in read_jsonl(path):
            for para in str(rec.get("state", "")).split("\n"):
                if len(para.strip()) >= PARAGRAPH_MIN_CHARS:
                    keys.add(paragraph_key(para))
            for block in str(rec.get("state", "")).split("\n\n"):
                if len(block.strip()) >= PARAGRAPH_MIN_CHARS:
                    keys.add(paragraph_key(block))
    return keys
