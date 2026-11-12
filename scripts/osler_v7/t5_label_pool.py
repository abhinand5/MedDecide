#!/usr/bin/env python
"""osler_v7 T5 - the C1-C4 pool and its pplx-27B soft labels (the v7 teacher step).

ADVISORY §2/§3: the teacher is `perplexity-ai/pplx-decider-v1.2-27b` at the pinned revision, run locally through its
own code; it labels the C1-C4 pool (and the unlabelled KL pool, which is the same prompts without gold). The teacher
never sees a dev, test, panel or held-out item.

Subcommands:
  pool    CPU. Merge the four built cells (C1/C2/C3/C4), check each row is sendable exactly, deduplicate by normalised
          text key, order by a stable hash, and write `data/train/osler_v7/pool.jsonl` plus the unlabelled KL file
          (`pool_unlabelled.jsonl`, prompts only). Manifest with counts, SHA-256 and overlap checks against the T2
          dev and confirmation sets.
  parity  GPU. 40 CG v2 dev items through the single-request reference, the batched reference and the fast path; the
          path choice follows the Y5 rule (batched reference if argmax agrees on all 40, else fast, else single).
  label   GPU. The pool in order until --max-hours, appending `pool_probs.jsonl` (resumable by pool position).
  join    CPU. Attach the probabilities to the pool rows as `teacher_probs`; counts of invalid/constant rows; writes
          `labelled.jsonl` and its manifest.
  report  CPU. Throughput and count summary.

Run (GPU, after T4 has released the GPU; one job at a time)::

    source scripts/pod_env.sh && setsid nohup uv run --frozen --with pillow==12.3.0 --with "torchvision==0.29.0" \\
        python scripts/osler_v7/t5_label_pool.py label --max-hours 6 \\
        > outputs/osler_v7/T5/logs/label.log 2>&1 < /dev/null &
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from meddecide.distill import soft_labels as sl  # noqa: E402
from meddecide.utils.hashing import file_sha256  # noqa: E402
from meddecide.utils.io import iter_jsonl, write_json  # noqa: E402
from meddecide.utils.provenance import utcnow  # noqa: E402

CELLS = {
    "C1": ROOT / "data" / "train" / "osler_v7" / "c1_many_option.jsonl",
    "C2": ROOT / "data" / "train" / "osler_v7" / "c2_pairs.jsonl",
    "C3": ROOT / "data" / "train" / "osler_v7" / "c3_code_concept.jsonl",
    "C4": ROOT / "data" / "train" / "osler_v7" / "c4_option_augment.jsonl",
}
DEV = ROOT / "data" / "bench" / "osler_v7" / "dev_format_matched.jsonl"
CONFIRM = ROOT / "data" / "bench" / "osler_v7" / "confirm" / "general.jsonl"
POOL = ROOT / "data" / "train" / "osler_v7" / "pool.jsonl"
UNLABELLED = ROOT / "data" / "train" / "osler_v7" / "pool_unlabelled.jsonl"
PROBS = ROOT / "data" / "train" / "osler_v7" / "pool_probs.jsonl"
LABELLED = ROOT / "data" / "train" / "osler_v7" / "labelled.jsonl"
OUT = ROOT / "outputs" / "osler_v7" / "T5"
PARITY = OUT / "parity.json"
PROGRESS = OUT / "label_progress.json"
POOL_MANIFEST = OUT / "pool_manifest.json"
JOIN_MANIFEST = OUT / "join_manifest.json"
Y5 = ROOT / "scripts" / "osler_v4" / "y5_soft_labels.py"
CG = ROOT / "data" / "bench" / "osler_v4" / "cg_v2.jsonl"
PARITY_ITEMS = 40
CHUNK = 512


def y5() -> Any:
    """The Y5 driver module (its Decider loading and scoring helpers), loaded by path like the tests do."""
    spec = importlib.util.spec_from_file_location("y5_soft_labels", Y5)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["y5_soft_labels"] = module
    spec.loader.exec_module(module)
    return module


def text_keys(path: Path) -> set[str]:
    keys: set[str] = set()
    for row in iter_jsonl(path):
        keys.add(sl.overlap_keys(row)[0])
    return keys


def cmd_pool(_: argparse.Namespace) -> None:
    seen: set[str] = set()
    entries: list[tuple[str, str, dict[str, Any]]] = []
    counts: Counter[str] = Counter()
    for cell, path in CELLS.items():
        rows_in = 0
        for row in iter_jsonl(path):
            rows_in += 1
            counts[f"{cell}: rows in"] += 1
            _, why = sl.request_for_row(row)
            if why is not None:
                counts[f"{cell}: skipped ({why})"] += 1
                continue
            key = sl.overlap_keys(row)[0]
            if key in seen:
                counts[f"{cell}: duplicate text"] += 1
                continue
            seen.add(key)
            entries.append((sl.pool_order_key(key), cell, row))
        counts[f"{cell}: rows_in"] = rows_in
    entries.sort(key=lambda item: item[0])
    forbidden = text_keys(DEV) | text_keys(CONFIRM)
    collisions = sum(1 for _, _, row in entries if sl.overlap_keys(row)[0] in forbidden)
    if collisions:
        raise SystemExit(f"{collisions} pool rows share text with the T2 dev or confirmation set: stop")
    POOL.parent.mkdir(parents=True, exist_ok=True)
    with POOL.open("w", encoding="utf-8") as handle:
        for _, _, row in entries:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with UNLABELLED.open("w", encoding="utf-8") as handle:
        for _, _, row in entries:
            prompt = {key: row[key] for key in ("item_id", "qtype", "state", "question", "options", "template_id",
                                                "source", "split")}
            handle.write(json.dumps(prompt, ensure_ascii=False) + "\n")
    manifest = {
        "generated_utc": utcnow(),
        "teacher": sl.TEACHER,
        "cells": {cell: {"path": str(path.relative_to(ROOT)), "sha256": file_sha256(path)} for cell, path in CELLS.items()},
        "counts": dict(sorted(counts.items())),
        "rows": len(entries),
        "qtypes": dict(sorted(Counter(str(row["qtype"]) for _, _, row in entries).items())),
        "cells_kept": dict(sorted(Counter(cell for _, cell, _ in entries).items())),
        "pool_file": str(POOL.relative_to(ROOT)),
        "pool_sha256": file_sha256(POOL),
        "unlabelled_file": str(UNLABELLED.relative_to(ROOT)),
        "unlabelled_sha256": file_sha256(UNLABELLED),
        "overlap_with_t2_sets": 0,
        "order": "sl.pool_order_key (stable hash of the normalised-text key, seed 0)",
        "notes": ["the unlabelled KL pool is the same prompts without gold",
                  "the teacher request never includes the gold; labels are the teacher's answers"],
    }
    write_json(POOL_MANIFEST, manifest)
    print(json.dumps({"rows": manifest["rows"], "qtypes": manifest["qtypes"], "cells": manifest["cells_kept"],
                      "sha256": manifest["pool_sha256"][:12]}, indent=1))


def cmd_parity(_: argparse.Namespace) -> None:
    driver = y5()
    rows = [row for row in iter_jsonl(CG) if sl.request_for_row(row)[1] is None][:PARITY_ITEMS]
    if len(rows) != PARITY_ITEMS:
        raise SystemExit("fewer than 40 mappable CG v2 items")
    results: dict[str, Any] = {}
    timings: dict[str, float] = {}
    for path, batched in (("reference_single", False), ("reference_batched", True), ("fast_batched", True)):
        decider = driver.load_decider(path)
        started = time.time()
        scored = driver.score_rows(decider, rows, batched=batched)
        timings[path] = round(time.time() - started, 1)
        results[path] = [s.probs for s in scored]
        del decider
        driver.release()
    single = results["reference_single"]
    if any(probs is None for probs in single):
        raise SystemExit("the single-request reference failed on a parity item")
    comparisons = {name: sl.compare_scorings(single, results[name]) for name in ("reference_batched", "fast_batched")}
    chosen = "reference_single"
    for name in ("reference_batched", "fast_batched"):
        if comparisons[name]["argmax_agreement"] == PARITY_ITEMS:
            chosen = name
            break
    write_json(PARITY, {"teacher": sl.TEACHER, "items": PARITY_ITEMS, "timings_s": timings,
                        "comparisons": comparisons, "chosen_path": chosen, "generated_utc": utcnow()})
    print(json.dumps({"chosen_path": chosen, "comparisons": comparisons}, indent=1))


def read_done() -> set[int]:
    if not PROBS.exists():
        return set()
    return {int(json.loads(line)["pool_index"]) for line in PROBS.open(encoding="utf-8") if line.strip()}


def cmd_label(args: argparse.Namespace) -> None:
    driver = y5()
    chosen = json.loads(PARITY.read_text())["chosen_path"] if PARITY.exists() else args.path
    if args.path:
        chosen = args.path
    pool_sha = file_sha256(POOL)
    done = read_done()
    if done and PROGRESS.exists() and json.loads(PROGRESS.read_text()).get("pool_file_sha256") != pool_sha:
        raise SystemExit("the pool file changed since scoring started: stop")
    decider = driver.load_decider(chosen)
    deadline = time.time() + args.max_hours * 3600
    started = time.time()
    statuses: Counter[str] = Counter()
    tokens = 0
    batch: list[tuple[int, dict[str, Any]]] = []
    limit = args.limit or None

    def flush() -> None:
        nonlocal tokens
        rows = [row for _, row in batch]
        scored = driver.score_rows(decider, rows, batched=(chosen != "reference_single"))
        with PROBS.open("a", encoding="utf-8") as handle:
            for (index, row), result in zip(batch, scored, strict=True):
                if result.probs is None:
                    statuses[f"error: {result.error}"] += 1
                    record = {"pool_index": index, "item_id": str(row["item_id"]), "status": "error",
                              "error": result.error}
                else:
                    statuses["scored"] += 1
                    tokens += result.input_tokens
                    record = {"pool_index": index, "item_id": str(row["item_id"]), "status": "scored",
                              "template_id": row["template_id"], "qtype": row["qtype"], "probs": result.probs,
                              "input_tokens": result.input_tokens}
                handle.write(json.dumps(record) + "\n")
        batch.clear()
        elapsed = time.time() - started
        write_json(PROGRESS, {"path": chosen, "pool_file_sha256": pool_sha, "statuses": dict(statuses),
                              "elapsed_s": round(elapsed, 1), "items_per_s": round(statuses["scored"] / elapsed, 3),
                              "input_tokens_per_s": round(tokens / elapsed, 1), "utc": utcnow()})
        print(f"[{utcnow()}] {dict(statuses)} elapsed {elapsed / 3600:.2f} h", flush=True)

    with POOL.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if limit is not None and index >= limit:
                break
            if index in done:
                continue
            batch.append((index, json.loads(line)))
            if len(batch) == CHUNK:
                flush()
                if time.time() >= deadline:
                    break
    if batch:
        flush()
    print(f"[{utcnow()}] label stopped: {dict(statuses)}")


def cmd_join(_: argparse.Namespace) -> None:
    probs: dict[int, dict[str, Any]] = {}
    statuses: Counter[str] = Counter()
    sums_off = 0.0
    constant = 0
    for line in PROBS.open(encoding="utf-8"):
        record = json.loads(line)
        statuses[str(record["status"])] += 1
        if record["status"] == "scored":
            probs[int(record["pool_index"])] = record
    rows: list[dict[str, Any]] = []
    errors = 0
    for index, row in enumerate(iter_jsonl(POOL)):
        scored = probs.get(index)
        if scored is None:
            errors += 1
            continue
        values = [float(p) for p in scored["probs"]]
        if len(values) != len(row["options"]):
            raise SystemExit(f"pool row {index}: {len(values)} probabilities for {len(row['options'])} options")
        sums_off = max(sums_off, abs(sum(values) - 1.0))
        if len({round(value, 6) for value in values}) <= 1:
            constant += 1
        rows.append({**row, "teacher_probs": values, "teacher": sl.TEACHER})
    if errors:
        raise SystemExit(f"{errors} pool rows have no scored record: the join would be partial")
    with LABELLED.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "generated_utc": utcnow(),
        "teacher": sl.TEACHER,
        "rows": len(rows),
        "pool_sha256": file_sha256(POOL),
        "labelled_sha256": file_sha256(LABELLED),
        "statuses": dict(statuses),
        "max_probability_sum_error": sums_off,
        "constant_rows": constant,
        "path": json.loads(PARITY.read_text())["chosen_path"] if PARITY.exists() else None,
        "throughput": json.loads(PROGRESS.read_text()) if PROGRESS.exists() else None,
    }
    write_json(JOIN_MANIFEST, manifest)
    print(json.dumps({"rows": len(rows), "statuses": dict(statuses), "max_sum_error": sums_off,
                      "constant": constant, "sha256": manifest["labelled_sha256"][:12]}, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("pool").set_defaults(func=cmd_pool)
    sub.add_parser("parity").set_defaults(func=cmd_parity)
    label = sub.add_parser("label")
    label.add_argument("--max-hours", type=float, default=6.0)
    label.add_argument("--limit", type=int, default=0)
    label.add_argument("--path", default=None, choices=("reference_single", "reference_batched", "fast_batched"))
    label.set_defaults(func=cmd_label)
    sub.add_parser("join").set_defaults(func=cmd_join)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    gc.collect()
    args.func(args)


if __name__ == "__main__":
    main()
