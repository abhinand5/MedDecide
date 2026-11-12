#!/usr/bin/env python
"""osler_v7 T5, cell C3 — code and concept items from the ICD-10-CM dictionary (CPU).

ADVISORY §2/C3: code-to-description, hierarchy and sibling distractors, gold from the dictionary. The only
description dictionary on disk for this build is ICD-10-CM (`data/mimic_build/icd10cm_titles.parquet`); the plan's
ATC and procedure-description parts have no dictionary on disk (the PCS table carries codes only), so they are
**NOT MEASURED** and counted below rather than replaced by another source.

Three question types, all with structured gold:
- `code_desc`: state is the code; four options are its title and three sibling titles (same 3-character category when
  possible, otherwise the same chapter);
- `desc_code`: state is a title; four options are codes (gold plus three siblings);
- `chapter`: state is a code; four options are chapter names (gold plus three other chapters).

Writes items (gitignored) and an aggregate manifest under `outputs/osler_v7/T5/`.

Run (CPU): ``uv run --frozen python scripts/osler_v7/c3_code_concept.py``
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from meddecide.utils.hashing import stable_hash  # noqa: E402

TITLES = ROOT / "data" / "mimic_build" / "icd10cm_titles.parquet"
OUT = ROOT / "data" / "train" / "osler_v7" / "c3_code_concept.jsonl"
MANIFEST = ROOT / "outputs" / "osler_v7" / "T5" / "c3_manifest.json"
TARGETS = {"code_desc": 10_000, "desc_code": 5_000, "chapter": 5_000}
SEED = 0
KEYS = ("A", "B", "C", "D")
CHAPTERS = {
    "A": "Infectious and parasitic diseases", "B": "Infectious and parasitic diseases",
    "C": "Neoplasms", "D": "Blood and neoplasms", "E": "Endocrine and metabolic",
    "F": "Mental and behavioural", "G": "Nervous system", "H": "Eye and ear",
    "I": "Circulatory system", "J": "Respiratory system", "K": "Digestive system", "L": "Skin",
    "M": "Musculoskeletal", "N": "Genitourinary", "O": "Pregnancy and childbirth", "P": "Perinatal",
    "Q": "Congenital malformations", "R": "Symptoms and signs", "S": "Injury and poisoning",
    "T": "Injury and poisoning", "V": "External causes", "W": "External causes", "X": "External causes",
    "Y": "External causes", "Z": "Factors influencing health",
}


def utcnow() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_dictionary() -> tuple[dict[str, str], dict[str, list[str]], dict[str, list[str]]]:
    """code -> title, family (first 3 chars) -> codes, chapter -> codes, with one code per normalised title."""
    frame = pd.read_parquet(TITLES)
    frame = frame[frame["icd_version"] == 10]
    by_title: dict[str, str] = {}
    for code, title in zip(frame["icd_code"].astype(str), frame["long_title"].astype(str), strict=True):
        clean = code.strip().upper().replace(".", "")
        text = " ".join(title.split())
        if not clean or not text or clean in by_title:
            continue
        by_title[clean] = text
    families: dict[str, list[str]] = defaultdict(list)
    chapters: dict[str, list[str]] = defaultdict(list)
    for code in by_title:
        families[code[:3]].append(code)
        chapter = CHAPTERS.get(code[:1], "Other")
        chapters[chapter].append(code)
    for values in families.values():
        values.sort()
    for values in chapters.values():
        values.sort()
    return by_title, dict(families), dict(chapters)


def make_row(item_id: str, template: str, state: str, question: str, options: list[str], gold_index: int) -> dict[str, Any]:
    keys = list(KEYS)
    return {
        "item_id": item_id,
        "tier": "established",
        "source": "icd10cm",
        "source_record_id": "",
        "source_url": "https://www.cms.gov/medicare/coding-billing/icd-10-codes",
        "source_license": "public-domain (CMS ICD-10-CM)",
        "record_date": "",
        "split": "train",
        "template_id": template,
        "skill": "medical_code",
        "qtype": "choice",
        "state": state,
        "question": question,
        "options": [{"key": key, "label": label, "description": None}
                    for key, label in zip(keys, options, strict=True)],
        "gold": keys[gold_index],
        "meta": {"n_options": len(options)},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    started = utcnow()
    by_title, families, chapters = load_dictionary()
    codes = sorted(by_title)
    rng = np.random.default_rng([SEED, 6351])
    rows: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    skipped: Counter[str] = Counter()
    chapter_names = sorted(chapters)

    # code -> description
    for code in rng.permutation(len(codes)):
        if counts["code_desc"] >= TARGETS["code_desc"]:
            break
        item = codes[int(code)]
        family = families.get(item[:3], [])
        pool = [other for other in family if other != item]
        if len(pool) < 3:
            pool = [other for other in chapters[CHAPTERS.get(item[:1], "Other")] if other != item]
        if len(pool) < 3:
            skipped["code_desc_pool_too_small"] += 1
            continue
        distractors = [pool[int(i)] for i in rng.choice(len(pool), size=3, replace=False)]
        options = [by_title[item]] + [by_title[other] for other in distractors]
        if len(set(options)) != 4:
            skipped["code_desc_duplicate_titles"] += 1
            continue
        gold = int(rng.integers(0, 4))
        options.insert(gold, options.pop(0))
        rows.append(make_row(stable_hash({"c3": "code_desc", "code": item}, length=16), "v7_c3_code_desc_v1",
                             f"ICD-10-CM code: {item}",
                             "Which description belongs to this ICD-10-CM code?", options, gold))
        counts["code_desc"] += 1

    # description -> code
    for index in rng.permutation(len(codes)):
        if counts["desc_code"] >= TARGETS["desc_code"]:
            break
        item = codes[int(index)]
        family = families.get(item[:3], [])
        pool = [other for other in family if other != item]
        if len(pool) < 3:
            skipped["desc_code_pool_too_small"] += 1
            continue
        distractors = [pool[int(i)] for i in rng.choice(len(pool), size=3, replace=False)]
        options = [item, *distractors]
        if len({by_title[code] for code in options}) != 4:
            skipped["desc_code_duplicate_titles"] += 1
            continue
        gold = int(rng.integers(0, 4))
        options.insert(gold, options.pop(0))
        rows.append(make_row(stable_hash({"c3": "desc_code", "code": item}, length=16), "v7_c3_desc_code_v1",
                             f"Description: {by_title[item]}",
                             "Which ICD-10-CM code has this description?",
                             [f"{code} - {by_title[code]}" for code in options], gold))
        counts["desc_code"] += 1

    # hierarchy: code -> chapter
    for index in rng.permutation(len(codes)):
        if counts["chapter"] >= TARGETS["chapter"]:
            break
        item = codes[int(index)]
        chapter = CHAPTERS.get(item[:1], "Other")
        others = [name for name in chapter_names if name != chapter]
        if len(others) < 3:
            skipped["chapter_pool_too_small"] += 1
            continue
        options = [chapter] + [others[int(i)] for i in rng.choice(len(others), size=3, replace=False)]
        gold = int(rng.integers(0, 4))
        options.insert(gold, options.pop(0))
        rows.append(make_row(stable_hash({"c3": "chapter", "code": item}, length=16), "v7_c3_chapter_v1",
                             f"ICD-10-CM code: {item} ({by_title[item]})",
                             "Which chapter of the ICD-10-CM classification contains this code?", options, gold))
        counts["chapter"] += 1

    for name, target in TARGETS.items():
        if counts[name] < target:
            raise SystemExit(f"C3 {name}: {counts[name]} rows, target {target}: stop")
    rows.sort(key=lambda row: row["item_id"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "generated_utc": utcnow(),
        "started_utc": started,
        "cell": "C3 code and concept",
        "seed": SEED,
        "rows": len(rows),
        "path": str(args.out.relative_to(ROOT)),
        "sha256": sha256_file(args.out),
        "counts": dict(sorted(counts.items())),
        "targets": TARGETS,
        "skipped": dict(sorted(skipped.items())),
        "dictionary": {"path": str(TITLES.relative_to(ROOT)), "sha256": sha256_file(TITLES),
                       "codes": len(codes), "families": len(families), "chapters": len(chapters)},
        "not_measured": [
            "ATC: no ATC dictionary is on disk for this build; that sub-part of C3 is NOT MEASURED",
            "procedure descriptions: the PCS table carries codes only (no title dictionary); NOT MEASURED",
        ],
        "notes": ["gold is a dictionary lookup (code/title/chapter), never a teacher label",
                  "only public CMS ICD-10-CM titles are read; no MIMIC text is used in C3"],
    }
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(rows), "sha256": manifest["sha256"][:12], "counts": manifest["counts"],
                      "skipped": manifest["skipped"]}, indent=1))


if __name__ == "__main__":
    main()
