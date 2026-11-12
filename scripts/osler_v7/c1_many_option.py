#!/usr/bin/env python
"""osler_v7 T5, cell C1 — many-option diagnosis from MIMIC presentations (local only, CPU).

ADVISORY §2/C1: state = a MIMIC presentation (chief complaint, HPI); 5-50 options; same-chapter distractors; gold =
the admission's primary ICD-10 code from the structured field. Every gold is a structured lookup, never a teacher
label. Only the MIMIC **train** patient split is read (patient-level split from the osler_v4 build); no dev/test/panel/
audit-suite item is touched.

Rows are sampled deterministically from admissions whose `principal_icd10` exists and whose discharge note has both a
chief-complaint and an HPI section. Distractors are ICD-10-CM titles from the same chapter (the first-letter ranges of
the ICD-10-CM chapters); option keys follow the 255-code scheme (A..Z, AA..). Writes the items (gitignored) and an
aggregate manifest under `outputs/osler_v7/T5/`.

Run (CPU): ``uv run --frozen python scripts/osler_v7/c1_many_option.py``
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from meddecide.mr import mimic  # noqa: E402
from meddecide.utils.hashing import stable_hash  # noqa: E402

ADMISSIONS = ROOT / "data" / "mimic_build" / "admissions.parquet"
NOTES = ROOT / "data" / "mimic_build" / "discharge_notes.parquet"
TITLES = ROOT / "data" / "mimic_build" / "icd10cm_titles.parquet"
OUT = ROOT / "data" / "train" / "osler_v7" / "c1_many_option.jsonl"
MANIFEST = ROOT / "outputs" / "osler_v7" / "T5" / "c1_manifest.json"
TEMPLATE = "v7_c1_dx_many_v1"
TARGET = 20_000
SEED = 0
MIN_OPTIONS, MAX_OPTIONS = 5, 50
STATE_CHAR_CAP = 8_000
CHAPTERS = (
    ("A", "infections"), ("B", "infections"), ("C", "neoplasms"), ("D", "blood_neoplasm"),
    ("E", "endocrine_metabolic"), ("F", "mental"), ("G", "nervous"), ("H", "eye_ear"),
    ("I", "circulatory"), ("J", "respiratory"), ("K", "digestive"), ("L", "skin"),
    ("M", "musculoskeletal"), ("N", "genitourinary"), ("O", "pregnancy"), ("P", "perinatal"),
    ("Q", "congenital"), ("R", "symptoms_signs"), ("S", "injury_poisoning"), ("T", "injury_poisoning"),
    ("V", "external_causes"), ("W", "external_causes"), ("X", "external_causes"), ("Y", "external_causes"),
    ("Z", "factors"),
)
# ICD-10-CM chapters are not pure first-letter groups (C and D mix over two chapters, H over two); the ranges above
# are deliberately coarse and are used only to draw distractors from a nearby code neighbourhood.
CHAPTER_BY_LETTER = dict(CHAPTERS)


def utcnow() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalise_code(raw: Any) -> str:
    return str(raw or "").strip().upper().replace(".", "")


def chapter_of(code: str) -> str:
    return CHAPTER_BY_LETTER.get(code[:1], "other") if code else "other"


def option_key(index: int) -> str:
    """The readout code of an option position: A..Z then AA, AB, ... (the 255-code scheme)."""
    if index < 26:
        return chr(ord("A") + index)
    return chr(ord("A") + (index // 26) - 1) + chr(ord("A") + index % 26)


def title_pools() -> dict[str, list[tuple[str, str]]]:
    """chapter -> sorted (code, title) pools, deduplicated by title (first code wins)."""
    titles = pd.read_parquet(TITLES)
    titles = titles[titles["icd_version"] == 10]
    pools: dict[str, dict[str, str]] = {}
    seen_titles: set[str] = set()
    for code, title in zip(titles["icd_code"].astype(str), titles["long_title"].astype(str), strict=True):
        clean = normalise_code(code)
        text = " ".join(title.split())
        if not clean or not text or len(text) > 140:
            continue
        chapter = chapter_of(clean)
        if text in seen_titles:
            continue
        seen_titles.add(text)
        pools.setdefault(chapter, {})[clean] = text
    return {chapter: sorted(entries.items()) for chapter, entries in sorted(pools.items())}


def load_notes() -> pd.DataFrame:
    """One discharge note per train admission (the first by note id), with its parsed sections."""
    notes = pd.read_parquet(NOTES, columns=["note_id", "subject_id", "hadm_id", "text", "split", "in_admissions"])
    notes = notes[(notes["split"] == "train") & notes["in_admissions"] & notes["hadm_id"].notna()]
    notes = notes.sort_values(["hadm_id", "note_id"]).drop_duplicates("hadm_id", keep="first")
    return notes[["hadm_id", "note_id", "text"]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=int, default=TARGET)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    started = utcnow()
    admissions = pd.read_parquet(
        ADMISSIONS, columns=["hadm_id", "subject_id", "split", "anchor_year_group", "principal_icd10"]
    )
    admissions = admissions[(admissions["split"] == "train") & admissions["principal_icd10"].notna()]
    notes = load_notes()
    frame = admissions.merge(notes, on="hadm_id", how="inner")
    frame = frame.sort_values("hadm_id").reset_index(drop=True)
    print(f"[c1] train admissions with a principal ICD-10: {len(admissions):,}; with a linked note: {len(frame):,}",
          file=sys.stderr, flush=True)
    pools = title_pools()
    rng = np.random.default_rng([args.seed, 6151])
    order = rng.permutation(len(frame))
    rows: list[dict[str, Any]] = []
    drops: Counter[str] = Counter()
    chapter_counts: Counter[str] = Counter()
    option_counts: Counter[str] = Counter()
    for index in order:
        if len(rows) >= args.target:
            break
        record = frame.iloc[int(index)]
        code = normalise_code(record["principal_icd10"])
        chapter = chapter_of(code)
        pool = pools.get(chapter, [])
        gold_title = next((title for item_code, title in pool if item_code == code), None)
        if gold_title is None:
            drops["gold_title_missing_or_outside_chapter_pool"] += 1
            continue
        body = mimic.sections(str(record["text"]))
        complaint = body.get("chief_complaint", "").strip()
        hpi = body.get("history_of_present_illness", "").strip()
        if not complaint or not hpi:
            drops["missing_chief_complaint_or_hpi"] += 1
            continue
        state = f"Chief Complaint:\n{complaint}\nHistory of Present Illness:\n{hpi}"
        if len(state) > STATE_CHAR_CAP:
            state = state[:STATE_CHAR_CAP].rstrip() + "\n[... truncated]"
        n_options = int(rng.integers(MIN_OPTIONS, MAX_OPTIONS + 1))
        n_options = min(n_options, len(pool))
        distractors = [entry for entry in pool if entry[0] != code]
        if len(distractors) < n_options - 1:
            drops["chapter_pool_too_small"] += 1
            continue
        chosen = rng.choice(len(distractors), size=n_options - 1, replace=False)
        options = [{"code": code, "title": gold_title}] + [{"code": distractors[int(i)][0],
                                                             "title": distractors[int(i)][1]} for i in chosen]
        perm = rng.permutation(len(options))
        options = [options[int(i)] for i in perm]
        keys = [option_key(i) for i in range(len(options))]
        gold_position = next(i for i, option in enumerate(options) if option["code"] == code)
        gold_key = keys[gold_position]
        item_id = stable_hash({"c1": str(record["hadm_id"]), "code": code, "n": n_options, "seed": args.seed},
                              length=16)
        rows.append({
            "item_id": item_id,
            "tier": "fresh",
            "source": "mimic_iv",
            "source_record_id": str(record["hadm_id"]),
            "source_url": "https://physionet.org/content/mimiciv/3.1/",
            "source_license": "PhysioNet credentialed (D28)",
            "record_date": "",
            "split": "train",
            "template_id": TEMPLATE,
            "skill": "diagnosis",
            "qtype": "choice",
            "state": state,
            "question": "Which primary diagnosis does this presentation best match?",
            "options": [{"key": key, "label": option["title"], "description": None}
                        for key, option in zip(keys, options, strict=True)],
            "gold": gold_key,
            "meta": {"icd_code": code, "chapter": chapter, "n_options": n_options},
        })
        chapter_counts[chapter] += 1
        option_counts[str(n_options)] += 1
    if len(rows) < args.target:
        raise SystemExit(f"only {len(rows)} C1 rows built, target {args.target}: stop")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "generated_utc": utcnow(),
        "started_utc": started,
        "cell": "C1 many-option diagnosis",
        "template_id": TEMPLATE,
        "seed": args.seed,
        "target": args.target,
        "rows": len(rows),
        "path": str(args.out.relative_to(ROOT)),
        "sha256": sha256_file(args.out),
        "admissions_with_principal_icd10_train": len(admissions),
        "admissions_with_linked_note_train": len(frame),
        "drops": dict(sorted(drops.items())),
        "chapters": dict(sorted(chapter_counts.items())),
        "option_counts": dict(sorted(option_counts.items(), key=lambda kv: int(kv[0]))),
        "min_options": MIN_OPTIONS,
        "max_options": MAX_OPTIONS,
        "state_char_cap": STATE_CHAR_CAP,
        "notes": [
            "gold is the admission's principal_icd10 (structured field), never a teacher label",
            "only the MIMIC train patient split is read; dev/test patients and the audit suite are untouched",
            "distractors are ICD-10-CM titles from the same coarse chapter; option order is seeded",
        ],
    }
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(rows), "sha256": manifest["sha256"][:12], "drops": manifest["drops"],
                      "chapters": len(chapter_counts)}, indent=1))


if __name__ == "__main__":
    main()
