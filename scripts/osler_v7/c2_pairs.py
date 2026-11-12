#!/usr/bin/env python
"""osler_v7 T5, cell C2 — balanced two-option equivalence pairs on MIMIC notes (CPU, local only).

ADVISORY §2/C2: balanced yes/no pairs on clinical text, paraphrase and minimal-edit, gold from the construction
(the edit type fixes the label). Construction used here, per MIMIC train-split discharge note:
- a term T1 that occurs in the note (not inside a negation window) gives a **paraphrase** yes item
  ("Does this note document T1?" -> yes);
- a same-chapter term T2 that does not occur in the note gives a **minimal-edit** no item
  ("Is T2 documented in this note?" -> no): the surface form is paraphrased and exactly one entity differs.
Each record contributes one yes and one no item, so the set is balanced by construction. Only the MIMIC train patient
split is read; dev/test/panel/audit items are untouched. The screening rules and every dropped record are counted.

Run (CPU): ``uv run --frozen python scripts/osler_v7/c2_pairs.py``
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
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

NOTES = ROOT / "data" / "mimic_build" / "discharge_notes.parquet"
TITLES = ROOT / "data" / "mimic_build" / "icd10cm_titles.parquet"
OUT = ROOT / "data" / "train" / "osler_v7" / "c2_pairs.jsonl"
MANIFEST = ROOT / "outputs" / "osler_v7" / "T5" / "c2_manifest.json"
TARGET_RECORDS = 5_000
SEED = 0
STATE_CHAR_CAP = 8_000
NEGATIONS = {"no", "not", "without", "denies", "denied", "negative", "absent", "never", "neither", "nor"}
NEGATION_WINDOW = 5
TOKEN = re.compile(r"[a-z]+")
PUNCT = re.compile(r"[^a-z ]+")


def utcnow() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def term_dictionary() -> tuple[dict[str, str], dict[str, str]]:
    """Normalised title -> display term, and display term -> chapter, for short 1-3 word ICD-10-CM titles."""
    frame = pd.read_parquet(TITLES)
    frame = frame[frame["icd_version"] == 10]
    terms: dict[str, str] = {}
    chapter: dict[str, str] = {}
    seen: set[str] = set()
    for code, title in zip(frame["icd_code"].astype(str), frame["long_title"].astype(str), strict=True):
        text = " ".join(PUNCT.sub(" ", title.lower()).split())
        words = text.split()
        if not 1 <= len(words) <= 3 or not 4 <= len(text) <= 30:
            continue
        if text in seen:
            continue
        seen.add(text)
        terms[text] = " ".join(word.capitalize() for word in words)
        chapter[terms[text]] = str(code)[:1]
    return terms, chapter


def negated(tokens: list[str], position: int) -> bool:
    start = max(0, position - NEGATION_WINDOW)
    return any(token in NEGATIONS for token in tokens[start:position])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-records", type=int, default=TARGET_RECORDS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    started = utcnow()
    notes = pd.read_parquet(NOTES, columns=["hadm_id", "text", "tokens", "split", "in_admissions"])
    notes = notes[(notes["split"] == "train") & notes["in_admissions"] & notes["hadm_id"].notna()]
    notes = notes.sort_values("hadm_id").drop_duplicates("hadm_id", keep="first").reset_index(drop=True)
    print(f"[c2] train discharge notes: {len(notes):,}", file=sys.stderr, flush=True)
    terms, chapter = term_dictionary()
    by_chapter: dict[str, list[str]] = defaultdict(list)
    for term, letter in chapter.items():
        by_chapter[letter].append(term)
    for values in by_chapter.values():
        values.sort()
    rng = np.random.default_rng([args.seed, 6251])
    rows: list[dict[str, Any]] = []
    drops: Counter[str] = Counter()
    pairs = 0
    order = rng.permutation(len(notes))
    for index in order:
        if pairs >= args.target_records:
            break
        record = notes.iloc[int(index)]
        text = str(record["text"])
        tokens = TOKEN.findall(text.lower())
        if len(tokens) < 20:
            drops["too_few_tokens"] += 1
            continue
        present: list[tuple[str, int]] = []
        present_strings: set[str] = set()
        for size in (3, 2, 1):
            for start in range(len(tokens) - size + 1):
                candidate = " ".join(tokens[start : start + size])
                present_strings.add(candidate)
                display = terms.get(candidate)
                if display is None or negated(tokens, start):
                    continue
                present.append((display, start))
        if not present:
            drops["no_known_term_present"] += 1
            continue
        display, _ = present[int(rng.integers(0, len(present)))]
        letter = chapter.get(display, "?")
        absent = [term for term in by_chapter.get(letter, []) if term.lower() not in present_strings]
        if not absent:
            drops["no_absent_same_chapter_term"] += 1
            continue
        swap = absent[int(rng.integers(0, len(absent)))]
        state = text if len(text) <= STATE_CHAR_CAP else text[:STATE_CHAR_CAP].rstrip() + "\n[... truncated]"
        yes_question = f"Does this note document {display}?"
        no_question = f"Is {swap} documented in this note?"
        base = {"tier": "fresh", "source": "mimic_iv", "source_record_id": str(record["hadm_id"]),
                "source_url": "https://physionet.org/content/mimiciv/3.1/",
                "source_license": "PhysioNet credentialed (D28)", "record_date": "", "split": "train",
                "qtype": "noul", "state": state,
                "options": [{"key": "yes", "label": "Yes", "description": None},
                            {"key": "no", "label": "No", "description": None}],
                "meta": {"edit_type": "paraphrase_present", "term": display}}
        rows.append({**base, "item_id": stable_hash({"c2": str(record["hadm_id"]), "kind": "yes",
                                                     "term": display, "seed": args.seed}, length=16),
                     "template_id": "v7_c2_paraphrase_present_v1", "skill": "record_consistency",
                     "question": yes_question, "gold": "yes"})
        rows.append({**base, "item_id": stable_hash({"c2": str(record["hadm_id"]), "kind": "no", "swap": swap,
                                                     "seed": args.seed}, length=16),
                     "template_id": "v7_c2_minimal_swap_absent_v1", "skill": "record_consistency",
                     "question": no_question, "gold": "no", "meta": {"edit_type": "minimal_swap_absent", "term": swap}})
        pairs += 1
        if pairs % 500 == 0:
            print(f"[c2] pairs {pairs:,}", file=sys.stderr, flush=True)
    if pairs < args.target_records:
        raise SystemExit(f"only {pairs} pairs built, target {args.target_records}: stop")
    rows.sort(key=lambda row: row["item_id"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "generated_utc": utcnow(),
        "started_utc": started,
        "cell": "C2 two-option equivalence",
        "seed": args.seed,
        "records": pairs,
        "rows": len(rows),
        "gold_counts": dict(sorted(Counter(row["gold"] for row in rows).items())),
        "templates": dict(sorted(Counter(row["template_id"] for row in rows).items())),
        "drops": dict(sorted(drops.items())),
        "path": str(args.out.relative_to(ROOT)),
        "sha256": sha256_file(args.out),
        "state_char_cap": STATE_CHAR_CAP,
        "notes": [
            "yes items: a term that occurs in the note (outside a 5-token negation window)",
            "no items: the same paraphrase with one entity swapped for a same-chapter term absent from the note",
            "balanced 1:1 by construction; only the MIMIC train patient split is read",
            "screening noise is possible (terms inside past-history sections); the rules are recorded, not tuned",
        ],
    }
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(rows), "sha256": manifest["sha256"][:12], "gold": manifest["gold_counts"],
                      "drops": manifest["drops"]}, indent=1))


if __name__ == "__main__":
    main()
