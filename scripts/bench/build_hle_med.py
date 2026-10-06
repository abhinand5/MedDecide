#!/usr/bin/env python
"""Build the HLE Biology/Medicine multiple-choice subset (loop task S4, supplementary only).

`cais/hle` is gated; its terms are accepted on the operator's HF account (verified: the dataset
loads). The subset is **supplementary**: it is reported beside MedDecide-Bench, never part of a
headline comparison, and **nothing is ever trained on it** (the S6 leakage check re-verifies
that no HLE item id reaches training data).

Selection, all recorded: `category == "Biology/Medicine"`, `answer_type == "multipleChoice"`,
**no image** (a text-only model cannot answer an image item, and the image is not fetched).
Options are parsed from the question text (`"A. ..."` lines, contiguous from A); the record's
`answer` letter is the gold. Anything that does not parse is dropped with a counted reason.

Usage:
    uv run python scripts/bench/build_hle_med.py --out data/bench/v0.2/supplementary
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from meddecide.bench.schema import QuestionType, Split, Tier, make_item
from meddecide.utils.io import write_json, write_jsonl
from meddecide.utils.provenance import utcnow

DATASET_ID = "cais/hle"
LICENSE = "mit"
CATEGORY = "Biology/Medicine"
ANSWER_TYPE = "multipleChoice"
OPTION_RE = re.compile(r"^\s*([A-Z])\.\s+(.+?)\s*$")


def parse_options(question: str) -> tuple[str, list[tuple[str, str]]]:
    """Split a question into ``(stem, [(letter, text), ...])``; empty list when it does not parse."""
    stem_lines: list[str] = []
    options: list[tuple[str, str]] = []
    for line in question.splitlines():
        match = OPTION_RE.match(line)
        if match:
            options.append((match.group(1), match.group(2)))
        elif options:
            # a wrapped option line belongs to the last option
            letter, text = options[-1]
            options[-1] = (letter, f"{text} {line.strip()}".strip())
        else:
            stem_lines.append(line)
    # options must start at A and be contiguous
    expected = [chr(ord("A") + i) for i in range(len(options))]
    if [letter for letter, _ in options] != expected:
        return "\n".join(stem_lines).strip(), []
    return "\n".join(stem_lines).strip(), options


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/bench/v0.2/supplementary"))
    parser.add_argument("--dataset", default=DATASET_ID)
    args = parser.parse_args()

    from datasets import load_dataset
    from huggingface_hub import HfApi

    revision = HfApi().dataset_info(args.dataset).sha
    ds = load_dataset(args.dataset, split="test")
    # read the columns through the arrow table: decoding the image column needs Pillow, and the
    # subset excludes image items anyway
    columns = {
        name: ds.data.column(name).to_pylist()
        for name in ("id", "question", "answer", "answer_type", "category", "raw_subject",
                     "image", "canary")
    }
    drops: Counter = Counter()
    rows: list[Any] = []
    for index in range(len(columns["id"])):
        if columns["category"][index] != CATEGORY:
            drops["category_not_biology_medicine"] += 1
            continue
        if columns["answer_type"][index] != ANSWER_TYPE:
            drops["answer_type_not_multiple_choice"] += 1
            continue
        if columns["image"][index]:
            drops["item_has_an_image"] += 1
            continue
        stem, options = parse_options(columns["question"][index])
        if len(options) < 2:
            drops["options_did_not_parse"] += 1
            continue
        gold = str(columns["answer"][index]).strip().upper()
        if gold not in [letter for letter, _ in options]:
            drops["answer_letter_not_among_options"] += 1
            continue
        if len(stem) < 20:
            drops["stem_too_short"] += 1
            continue
        rows.append(
            make_item(
                tier=Tier.FRESH,
                source="hle",
                source_record_id=str(columns["id"][index]),
                source_url=f"https://huggingface.co/datasets/{DATASET_ID}",
                source_license=LICENSE,
                record_date="2026-01-01",  # no per-item date in HLE; the release date is used
                split=Split.TEST,
                template_id="hle_med_mc_v1",
                skill="knowledge",
                qtype=QuestionType.CHOICE,
                state=stem,
                question="Which one of the following is correct?",
                options=[{"key": letter, "label": text} for letter, text in options],
                gold=gold,
                option_order_seed=0,
                meta={
                    "dataset": DATASET_ID,
                    "dataset_revision": revision,
                    "category": CATEGORY,
                    "raw_subject": columns["raw_subject"][index],
                    "answer_type": ANSWER_TYPE,
                    "n_options": len(options),
                    "has_canary": bool(columns["canary"][index]),
                    "supplementary": True,
                    "never_train_on": True,
                },
            )
        )

    args.out.mkdir(parents=True, exist_ok=True)
    path = write_jsonl(args.out / "hle_med.jsonl", rows)
    n_options = Counter(row.n_options for row in rows)
    manifest = {
        "generated_at_utc": utcnow(),
        "dataset": DATASET_ID,
        "dataset_revision": revision,
        "license": LICENSE,
        "selection": {
            "category": CATEGORY,
            "answer_type": ANSWER_TYPE,
            "excludes_images": True,
        },
        "n_items": len(rows),
        "n_options_histogram": dict(sorted(n_options.items())),
        "dropped": dict(sorted(drops.items())),
        "path": str(path),
        "role": "supplementary test only; never trained on (D13/D14 and ADVISORY section 7)",
        "checks": {
            "has_items": len(rows) > 0,
            "every_item_is_choice": all(row.qtype is QuestionType.CHOICE for row in rows),
            "every_gold_is_an_option_key": all(row.gold in row.option_keys for row in rows),
            "no_item_has_an_image": True,  # enforced by the selection, recorded for the reader
        },
    }
    manifest["verdict"] = "PASS" if all(manifest["checks"].values()) else "FAIL"
    write_json(args.out / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2)[:2000])
    return 0 if manifest["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
