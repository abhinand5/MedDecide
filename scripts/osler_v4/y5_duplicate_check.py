"""Y5 check (ADVISORY §4 Y5): are item ids unique, and do repeated ids carry the same item?

Part 1, the anchor rows (block C1 and the broad replay): every row whose item_id repeats an earlier anchor row, counted by
source file pair, with the template-prefix and source census of the repeats and whether they share the first row's source
record id. The pool step deduplicates by normalised text, so a repeated id with different text is kept there; this part
measures how often that happens and why.

Part 2, every benchmark file under data/bench (any split): repeated ids inside each file, and ids that appear in a later file
with different text. The evaluation pairing (meddecide.train.blocks.pair_by_item) raises on duplicate ids, so this part states
whether any evaluation file is affected.

Part 3, every training file under data/train except dev.jsonl and this loop's own pool (data/train/osler_v4): repeated ids
inside each file, by template prefix. A mix whose trainer keys on ids would lose rows at a repeat; this part states whether any does.

Counts only; no item text is printed. Writes outputs/osler_v4/Y5/duplicate_check.json.
"""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from meddecide.panel.overlap import row_text_key
from meddecide.train.osler_v3 import is_anchor
from meddecide.utils.io import iter_jsonl, write_json

TRAIN_SOURCES = (Path("data/train/osler_v2/block_C1.jsonl"), Path("data/train/osler_v3/replay_broad_train.jsonl"))
TRAIN_ROOT = Path("data/train")
OWN_POOL_DIR = TRAIN_ROOT / "osler_v4"
BENCH = Path("data/bench")
OUT = Path("outputs/osler_v4/Y5/duplicate_check.json")


def anchor_report() -> dict[str, Any]:
    first: dict[str, tuple[int, str, str | None]] = {}  # item_id -> (source index, text key, source record id)
    repeats: Counter[str] = Counter()
    same_text: Counter[str] = Counter()
    census_prefix: Counter[str] = Counter()
    census_source: Counter[str] = Counter()
    same_record = 0
    for s_idx, source in enumerate(TRAIN_SOURCES):
        for row in iter_jsonl(source):
            if not is_anchor(str(row["template_id"])):
                continue
            item_id = str(row["item_id"])
            key = row_text_key(row)
            record = row.get("source_record_id")
            if item_id not in first:
                first[item_id] = (s_idx, key, record)
                continue
            first_source, first_key, first_record = first[item_id]
            pair = f"first in {TRAIN_SOURCES[first_source].name}; repeat in {source.name}"
            repeats[pair] += 1
            same_text[pair] += int(key == first_key)
            census_prefix[str(row["template_id"]).split("_")[0]] += 1
            census_source[str(row["source"])] += 1
            same_record += int(record == first_record)
    total = sum(repeats.values())
    identical = sum(same_text.values())
    return {
        "anchor_repeats": total,
        "repeats_with_identical_text": identical,
        "repeats_with_different_text": total - identical,
        "by_pair": {pair: {"repeats": repeats[pair], "identical_text": same_text[pair]} for pair in sorted(repeats)},
        "distinct_anchor_item_ids": len(first),
        "repeat_census_template_prefix": dict(census_prefix.most_common()),
        "repeat_census_source": dict(census_source.most_common()),
        "repeat_with_same_source_record_as_first": same_record,
    }


def bench_report() -> dict[str, Any]:
    first_text: dict[str, str] = {}  # item_id -> text key of its first occurrence in any benchmark file
    files: dict[str, dict[str, int]] = {}
    cross_file_conflicts = 0
    for path in sorted(BENCH.rglob("*.jsonl")):
        in_file: dict[str, str] = {}
        repeats = 0
        repeats_different = 0
        rows = 0
        for row in iter_jsonl(path):
            rows += 1
            item_id = str(row["item_id"])
            key = row_text_key(row)
            if item_id in in_file:
                repeats += 1
                repeats_different += int(in_file[item_id] != key)
            else:
                in_file[item_id] = key
            if item_id in first_text and first_text[item_id] != key:
                cross_file_conflicts += 1
            first_text.setdefault(item_id, key)
        files[str(path)] = {"rows": rows, "item_ids": len(in_file), "repeats": repeats,
                            "repeats_with_different_text": repeats_different}
    return {
        "files": files,
        "rows_total": sum(f["rows"] for f in files.values()),
        "within_file_repeats_total": sum(f["repeats"] for f in files.values()),
        "within_file_repeats_with_different_text_total": sum(f["repeats_with_different_text"] for f in files.values()),
        "rows_whose_id_has_different_text_in_an_earlier_file": cross_file_conflicts,
    }


def _train_file(path: Path) -> dict[str, Any]:
    seen: dict[str, str] = {}
    repeats: Counter[str] = Counter()
    different: Counter[str] = Counter()
    rows = 0
    for row in iter_jsonl(path):
        rows += 1
        item_id = str(row["item_id"])
        key = row_text_key(row)
        if item_id in seen:
            prefix = str(row["template_id"]).split("_")[0]
            repeats[prefix] += 1
            different[prefix] += int(seen[item_id] != key)
        else:
            seen[item_id] = key
    return {"rows": rows, "item_ids": len(seen), "repeats": sum(repeats.values()),
            "repeats_with_different_text": sum(different.values()), "repeats_by_template_prefix": dict(repeats.most_common()),
            "different_text_by_template_prefix": dict(different.most_common())}


def train_report() -> dict[str, Any]:
    files = sorted(p for p in TRAIN_ROOT.rglob("*.jsonl") if p.name != "dev.jsonl" and OWN_POOL_DIR not in p.parents)
    with ProcessPoolExecutor(max_workers=16) as pool:
        results = list(pool.map(_train_file, files))
    return {str(path): result for path, result in zip(files, results, strict=True)}


def main() -> None:
    anchor = anchor_report()
    bench = bench_report()
    train = train_report()
    write_json(OUT, {"anchor": anchor, "bench": bench, "train": train})
    print(f"anchor repeats {anchor['anchor_repeats']}; identical text {anchor['repeats_with_identical_text']}; "
          f"different text {anchor['repeats_with_different_text']}; distinct anchor item ids "
          f"{anchor['distinct_anchor_item_ids']}; bench within-file repeats {bench['within_file_repeats_total']} "
          f"(different text {bench['within_file_repeats_with_different_text_total']}); bench cross-file conflicts "
          f"{bench['rows_whose_id_has_different_text_in_an_earlier_file']}; training files {len(train)}, within-file "
          f"repeats {sum(t['repeats'] for t in train.values())}")


if __name__ == "__main__":
    main()
