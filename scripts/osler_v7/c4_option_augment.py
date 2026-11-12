#!/usr/bin/env python
"""osler_v7 T5, cell C4 — option-count augmentation of 4-option training items (CPU).

ADVISORY §2/C4: existing 4-option train items are expanded to 8-10 options; the gold is unchanged. The distractor
pool of a template is that template's own training gold labels (frequency-weighted), so a distractor is a label the
template really offers elsewhere; the gold option and the three original options are kept. Rows come from the v6
training mix (train splits only), never from dev/test/panel/audit items.

Two passes over the mix: the first collects each choice template's gold-label frequencies (for templates whose rows
have exactly four options), the second samples 20,000 rows deterministically and writes the expanded rows. Every
filter returns counts in the manifest.

Run (CPU): ``uv run --frozen python scripts/osler_v7/c4_option_augment.py``
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from meddecide.utils.hashing import stable_hash  # noqa: E402

MIX = ROOT / "data" / "train" / "osler_v6" / "mix_f2_distill.jsonl"
OUT = ROOT / "data" / "train" / "osler_v7" / "c4_option_augment.jsonl"
MANIFEST = ROOT / "outputs" / "osler_v7" / "T5" / "c4_manifest.json"
TARGET = 20_000
SEED = 0
NEW_COUNTS = (8, 9, 10)
MIN_TEMPLATE_LABELS = 8
KEYS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def utcnow() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def options_of(row: dict[str, Any]) -> list[dict[str, Any]]:
    options = row["options"]
    return options if isinstance(options, list) else json.loads(options)


def key_at(index: int) -> str:
    if index < 26:
        return KEYS[index]
    return KEY_AT_TWO_LETTER(index)


def KEY_AT_TWO_LETTER(index: int) -> str:
    return chr(ord("A") + (index // 26) - 1) + chr(ord("A") + index % 26)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=int, default=TARGET)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    started = utcnow()
    label_counts: dict[str, Counter[str]] = defaultdict(Counter)
    rows_in = 0
    four_option = 0
    with MIX.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            rows_in += 1
            if row["qtype"] != "choice":
                continue
            options = options_of(row)
            if len(options) != 4:
                continue
            four_option += 1
            gold = str(row["gold"])
            gold_label = next((str(option["label"]) for option in options if str(option["key"]) == gold), None)
            if gold_label is not None:
                label_counts[str(row["template_id"])][gold_label] += 1
            if rows_in % 250_000 == 0:
                print(f"[c4] pass1 {rows_in:,} rows", file=sys.stderr, flush=True)
    usable = {template: counts for template, counts in label_counts.items() if len(counts) >= MIN_TEMPLATE_LABELS}
    print(f"[c4] rows {rows_in:,}; four-option {four_option:,}; templates with >= {MIN_TEMPLATE_LABELS} labels: "
          f"{len(usable)}", file=sys.stderr, flush=True)
    rng = np.random.default_rng([args.seed, 6451])
    selected: list[dict[str, Any]] = []
    seen = 0
    with MIX.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["qtype"] != "choice" or len(options_of(row)) != 4:
                continue
            template = str(row["template_id"])
            if template not in usable:
                continue
            seen += 1
            if int(rng.integers(0, seen)) < args.target:  # reservoir sample over the eligible rows
                selected.append(row)
            if seen % 250_000 == 0:
                print(f"[c4] pass2 eligible {seen:,}, kept {len(selected):,}", file=sys.stderr, flush=True)
    if len(selected) < args.target:
        raise SystemExit(f"only {len(selected)} eligible rows, target {args.target}: stop")
    rows: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()
    counts_by_template: Counter[str] = Counter()
    for row in selected[: args.target]:
        template = str(row["template_id"])
        options = options_of(row)
        gold = str(row["gold"])
        gold_label = next((str(option["label"]) for option in options if str(option["key"]) == gold), None)
        if gold_label is None:
            skipped["gold_label_missing"] += 1
            continue
        existing = {str(option["label"]) for option in options}
        pool = [label for label, _ in usable[template].most_common() if label not in existing]
        n_new = int(NEW_COUNTS[int(rng.integers(0, len(NEW_COUNTS)))])
        need = n_new - 4
        if len(pool) < need:
            skipped["template_pool_too_small"] += 1
            continue
        weights = np.asarray([usable[template][label] for label in pool], dtype=np.float64)
        weights /= weights.sum()
        distractors: list[str] = []
        while len(distractors) < need:
            pick = str(rng.choice(pool, p=weights))
            if pick not in distractors:
                distractors.append(pick)
        new_options = [{"key": option["key"], "label": str(option["label"]), "description": None}
                       for option in options] + [
            {"key": key_at(4 + i), "label": label, "description": None} for i, label in enumerate(distractors)
        ]
        perm = rng.permutation(len(new_options))
        new_options = [new_options[int(i)] for i in perm]
        for index, option in enumerate(new_options):
            option["key"] = key_at(index)
        new_gold = new_options[[option["label"] for option in new_options].index(gold_label)]["key"]
        item_id = stable_hash({"c4": str(row["item_id"]), "n": n_new, "seed": args.seed}, length=16)
        rows.append({
            "item_id": item_id,
            "tier": str(row.get("tier", "established")),
            "source": str(row["source"]),
            "source_record_id": str(row.get("source_record_id", "")),
            "source_url": str(row.get("source_url", "")),
            "source_license": str(row.get("source_license", "")),
            "record_date": str(row.get("record_date", "")),
            "split": "train",
            "template_id": f"v7_c4_{template}",
            "skill": str(row.get("skill", "training")),
            "qtype": "choice",
            "state": str(row["state"]),
            "question": str(row["question"]),
            "options": new_options,
            "gold": new_gold,
            "meta": {"original_item_id": str(row["item_id"]), "original_template_id": template,
                     "n_options": len(new_options)},
        })
        counts_by_template[template] += 1
    if len(rows) < args.target:
        raise SystemExit(f"only {len(rows)} C4 rows built, target {args.target}: stop")
    rows.sort(key=lambda row: row["item_id"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "generated_utc": utcnow(),
        "started_utc": started,
        "cell": "C4 option-count augmentation",
        "seed": args.seed,
        "rows": len(rows),
        "rows_in_mix": rows_in,
        "four_option_rows": four_option,
        "eligible_rows_after_template_filter": seen,
        "templates_usable": len(usable),
        "target": args.target,
        "path": str(args.out.relative_to(ROOT)),
        "sha256": sha256_file(args.out),
        "counts_by_template": dict(sorted(counts_by_template.items())),
        "skipped": dict(sorted(skipped.items())),
        "new_option_counts": dict(sorted(Counter(len(row["options"]) for row in rows).items())),
        "gold_unchanged": True,
        "notes": ["gold label is kept; only the option key changes with the new order",
                  "distractors are drawn from the same template's own training gold labels, frequency-weighted",
                  "source rows are the v6 training mix (train splits only)"],
    }
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(rows), "sha256": manifest["sha256"][:12], "templates": len(counts_by_template),
                      "skipped": manifest["skipped"]}, indent=1))


if __name__ == "__main__":
    main()
