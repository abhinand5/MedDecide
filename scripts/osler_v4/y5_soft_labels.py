"""Y5 (osler_v4 ADVISORY §4 Y5; D48): pplx-decider-v1.2-27b soft labels for the anchor pool, scored through the authors' code.

Subcommands (one GPU job at a time; detached and resumable; the mapping and the exclusions are in meddecide.distill):
  pool    CPU. Excludes every anchor row that shares text or a (source, record id) with any benchmark row of any split;
          drops rows that cannot be sent exactly and rows whose normalised text repeats an earlier row (all counted);
          orders the rest by a stable hash of the text key and writes the first POOL_CAP to
          data/train/osler_v4/y5_anchor_pool.jsonl. Scored records are keyed by pool position (item ids are not unique).
          Manifest: outputs/osler_v4/Y5/anchor_pool_manifest.json.
  parity  GPU. 40 CG v2 items (dev): the single-request reference (the evaluated implementation), the batched reference and
          the batched fast path. Scoring path: the batched reference if it agrees on argmax with the single-request reference
          on all 40 items (D39's bar); else the fast path if it does; else the single-request reference.
  dev     GPU. CG v2 and knowledge dev v2 with the chosen path: the teacher's own accuracy (a diagnostic, never a choice).
  anchor  GPU. The pool in order until --max-hours; appends outputs/osler_v4/Y5/anchor_probs.jsonl.
  join    CPU. Joins the scored pool rows into data/train/osler_v4/y5_anchor_soft.jsonl; re-checks the overlap (must be 0).
  report  CPU. Writes loops/osler_v4/teacher27b.md (aggregates only) from the JSON outputs.

Usage:
    source scripts/pod_env.sh && setsid nohup uv run --frozen --with pillow==12.3.0 --with "torchvision==0.29.*" \
        python scripts/osler_v4/y5_soft_labels.py anchor --max-hours 5 \
        > outputs/osler_v4/Y5/logs/anchor.log 2>&1 < /dev/null &
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from meddecide.distill import soft_labels as sl
from meddecide.train.osler_v3 import is_anchor
from meddecide.utils.hashing import file_sha256
from meddecide.utils.io import iter_jsonl, write_json
from meddecide.utils.provenance import gpu_name, utcnow

BENCH = Path("data/bench")
TRAIN_SOURCES = (Path("data/train/osler_v2/block_C1.jsonl"), Path("data/train/osler_v3/replay_broad_train.jsonl"))
POOL = Path("data/train/osler_v4/y5_anchor_pool.jsonl")
SOFT = Path("data/train/osler_v4/y5_anchor_soft.jsonl")
CG = Path("data/bench/osler_v4/cg_v2.jsonl")
KNOWLEDGE = Path("data/bench/osler_v2/knowledge_dev_v2.jsonl")
OUT = Path("outputs/osler_v4/Y5")
POOL_MANIFEST = OUT / "anchor_pool_manifest.json"
PARITY = OUT / "parity.json"
DEV_PROBS = OUT / "dev_probs.jsonl"
DEV_ACCURACY = OUT / "dev_accuracy.json"
ANCHOR_PROBS = OUT / "anchor_probs.jsonl"
ANCHOR_PROGRESS = OUT / "anchor_progress.json"
JOIN_MANIFEST = OUT / "join_manifest.json"
REPORT = Path("loops/osler_v4/teacher27b.md")
CHECKPOINT = Path(os.environ["HF_HOME"]) / "hub" / f"models--{sl.REPO.replace('/', '--')}" / "snapshots" / sl.REVISION
SOURCE_SRC = CHECKPOINT / "source" / "src"
POOL_CAP = 400_000  # the stable-hash prefix: the 27B's measured throughput cannot reach the whole pool in the timebox
PARITY_ITEMS = 40
CHUNK = 512
PATHS = ("reference_batched", "fast_batched", "reference_single")


@dataclass
class Scored:
    probs: list[float] | None
    input_tokens: int
    error: str | None


def teacher_package() -> Any:
    """The authors' package, from the pinned checkpoint (imported after its source directory is on the path)."""
    if str(SOURCE_SRC) not in sys.path:
        sys.path.insert(0, str(SOURCE_SRC))
    import pplx_decider

    return pplx_decider


def bench_keys() -> tuple[set[str], set[sl.RecordKey], dict[str, int]]:
    """Text and record keys of every benchmark row, any split. A row that cannot be keyed stops the run (no silent gaps)."""
    text_keys: set[str] = set()
    record_keys: set[sl.RecordKey] = set()
    files: dict[str, int] = {}
    for path in sorted(BENCH.rglob("*.jsonl")):
        count = 0
        for row in iter_jsonl(path):
            text, record = sl.overlap_keys(row)
            text_keys.add(text)
            if record is not None:
                record_keys.add(record)
            count += 1
        files[str(path)] = count
    return text_keys, record_keys, files


def cmd_pool(args: argparse.Namespace) -> None:
    text_keys, record_keys, bench_files = bench_keys()
    entries: list[tuple[str, int, int, str]] = []  # (order key, source index, line number, qtype)
    seen_text: set[str] = set()  # the pool is deduplicated by content; item ids are not unique (duplicate_check.json)
    seen_ids: set[str] = set()
    counts: Counter[str] = Counter()
    for s_idx, source in enumerate(TRAIN_SOURCES):
        for line_no, row in enumerate(iter_jsonl(source)):
            counts[f"rows: {source.name}"] += 1
            if not is_anchor(str(row["template_id"])):
                counts["not an anchor (domain template prefix)"] += 1
                continue
            reason = sl.overlap_reason(row, text_keys, record_keys)
            if reason is not None:
                counts[f"excluded: overlaps a benchmark row ({reason})"] += 1
                continue
            _, why = sl.request_for_row(row)
            if why is not None:
                counts[f"skipped: cannot be sent exactly ({why})"] += 1
                continue
            text_key = sl.overlap_keys(row)[0]
            if text_key in seen_text:
                counts["skipped: same text as an earlier pool row"] += 1
                continue
            seen_text.add(text_key)
            item_id = str(row["item_id"])
            if item_id in seen_ids:
                counts["kept: item id shared with an earlier pool row (different text)"] += 1
            seen_ids.add(item_id)
            entries.append((sl.pool_order_key(text_key), s_idx, line_no, str(row["qtype"])))
    entries.sort()
    selected = entries[:POOL_CAP]
    rank_of = {(s_idx, line_no): rank for rank, (_, s_idx, line_no, _) in enumerate(selected)}
    lines: list[str | None] = [None] * len(selected)
    for s_idx, source in enumerate(TRAIN_SOURCES):
        for line_no, row in enumerate(iter_jsonl(source)):
            rank = rank_of.get((s_idx, line_no))
            if rank is not None:
                lines[rank] = json.dumps(row, ensure_ascii=False)
    POOL.parent.mkdir(parents=True, exist_ok=True)
    with POOL.open("w", encoding="utf-8") as handle:
        for line in lines:
            if line is None:
                raise RuntimeError("a selected pool row was not found in its source: stop")
            handle.write(line + "\n")
    write_json(POOL_MANIFEST, {
        "teacher": sl.TEACHER,
        "order": "sl.pool_order_key (stable hash of the normalised-text key, seed 0)",
        "pool_key": "normalised text key (state, question, option labels); item ids are not unique",
        "pool_sources": {str(p): counts[f"rows: {p.name}"] for p in TRAIN_SOURCES},
        "bench_files": bench_files,
        "bench_text_keys": len(text_keys),
        "bench_record_keys": len(record_keys),
        "counts": dict(sorted(counts.items())),
        "eligible_unique": len(entries),
        "eligible_qtypes": dict(Counter(qtype for _, _, _, qtype in entries)),
        "pool_cap": POOL_CAP,
        "pool_rows_written": len(selected),
        "pool_qtypes_written": dict(Counter(qtype for _, _, _, qtype in selected)),
        "pool_file": str(POOL),
        "pool_file_sha256": file_sha256(POOL),
        "generated_utc": utcnow(),
    })
    print(f"[{utcnow()}] eligible {len(entries)}; wrote {len(selected)} rows to {POOL}")


def load_decider(path: str) -> Any:
    """The authors' Decider on the pinned checkpoint: the reference path, or the packed fast path."""
    decider_module = teacher_package()
    return decider_module.Decider.from_pretrained(CHECKPOINT, device="cuda", fast=(path == "fast_batched"))


def release() -> None:
    """Frees the GPU after the caller has dropped its own reference to a Decider (a callee cannot free the caller's)."""
    gc.collect()
    torch.cuda.empty_cache()


def _scored(row: dict[str, Any], response: dict[str, Any]) -> Scored:
    probs = sl.teacher_probabilities(row, response["answers"][sl.QUESTION_ID])
    return Scored(probs, int(response["usage"]["input_tokens"]), None)


def _score_one(decider: Any, request: dict[str, Any], row: dict[str, Any]) -> Scored:
    errors = teacher_package()
    try:
        response = decider.decide(request)
    except (errors.RequestError, errors.LengthError) as exc:
        return Scored(None, 0, type(exc).__name__)
    return _scored(row, response)


def score_rows(decider: Any, rows: list[dict[str, Any]], *, batched: bool) -> list[Scored]:
    """One answer per row, in row order. Rows must already map to a request (the callers check)."""
    requests = []
    for row in rows:
        request, why = sl.request_for_row(row)
        if request is None:
            raise RuntimeError(f"row without a request ({why}): callers must filter first")
        requests.append(request)
    if not batched:
        return [_score_one(decider, request, row) for request, row in zip(requests, rows, strict=True)]
    errors = teacher_package()
    try:
        responses = decider.decide(requests)
    except (errors.RequestError, errors.LengthError):
        # one bad request voids a list call: score each alone, so the failing row is isolated and counted
        return [_score_one(decider, request, row) for request, row in zip(requests, rows, strict=True)]
    return [_scored(row, response) for row, response in zip(rows, responses, strict=True)]


def mappable(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], Counter[str]]:
    """The rows that map to a request, in stable-hash order, and the counted reasons for the rest."""
    kept: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()
    for row in sorted(rows, key=lambda r: sl.pool_order_key(str(r["item_id"]))):
        _, why = sl.request_for_row(row)
        if why is None:
            kept.append(row)
        else:
            skipped[why] += 1
    return kept, skipped


def cmd_parity(args: argparse.Namespace) -> None:
    rows, skipped = mappable(list(iter_jsonl(CG)))
    rows = rows[:PARITY_ITEMS]
    if len(rows) != PARITY_ITEMS:
        raise SystemExit(f"only {len(rows)} mappable CG v2 items; {PARITY_ITEMS} are needed")
    timings: dict[str, float] = {}

    decider = load_decider("reference_single")
    start = time.time()
    single = score_rows(decider, rows, batched=False)
    timings["reference_single"] = time.time() - start
    start = time.time()
    batched = score_rows(decider, rows, batched=True)
    timings["reference_batched"] = time.time() - start
    temperature = json.loads((CHECKPOINT / "decision_config.json").read_text())["temperature"]
    del decider
    release()

    fast = load_decider("fast_batched")
    start = time.time()
    fast_scored = score_rows(fast, rows, batched=True)
    timings["fast_batched"] = time.time() - start
    del fast
    release()

    for name, scored in (("single", single), ("batched", batched), ("fast", fast_scored)):
        if any(s.probs is None for s in scored):
            raise SystemExit(f"{name} scoring failed on a parity item ({[s.error for s in scored if s.probs is None]})")
    ref = [s.probs for s in single]
    comparisons = {"reference_batched": sl.compare_scorings(ref, [s.probs for s in batched]),
                   "fast_batched": sl.compare_scorings(ref, [s.probs for s in fast_scored])}
    if comparisons["reference_batched"]["argmax_agree"] == PARITY_ITEMS:
        path = "reference_batched"
    elif comparisons["fast_batched"]["argmax_agree"] == PARITY_ITEMS:
        path = "fast_batched"
    else:
        path = "reference_single"
    write_json(PARITY, {
        "teacher": sl.TEACHER,
        "licence": "apache-2.0 (README front matter and LICENSE file of the pinned snapshot)",
        "temperature": temperature,
        "items": PARITY_ITEMS,
        "items_qtypes": dict(Counter(str(r["qtype"]) for r in rows)),
        "cg_skipped_before_selection": dict(skipped),
        "bar": "argmax agreement with the single-request reference on every item (D39)",
        "comparisons_vs_single_request": comparisons,
        "seconds": timings,
        "path": path,
        "gpu": gpu_name(),
        "generated_utc": utcnow(),
    })
    print(f"[{utcnow()}] parity: {comparisons}; scoring path = {path}")


def chosen_path() -> str:
    if not PARITY.exists():
        raise SystemExit(f"{PARITY} is missing: run the parity step first")
    path = json.loads(PARITY.read_text())["path"]
    if path not in PATHS:
        raise SystemExit(f"unknown scoring path {path!r}")
    return path


def cmd_dev(args: argparse.Namespace) -> None:
    path = chosen_path()
    cg_rows = list(iter_jsonl(CG))
    know_rows = list(iter_jsonl(KNOWLEDGE))
    membership: dict[str, set[str]] = {}
    by_id: dict[str, dict[str, Any]] = {}
    for name, rows in (("cg_v2", cg_rows), ("knowledge_dev_v2", know_rows)):
        for row in rows:
            item_id = str(row["item_id"])
            if item_id in by_id and sl.overlap_keys(by_id[item_id])[0] != sl.overlap_keys(row)[0]:
                raise SystemExit(f"item id {item_id} has two different texts across the dev sets: stop")
            by_id.setdefault(item_id, row)
            membership.setdefault(item_id, set()).add(name)
    rows, skipped = mappable(list(by_id.values()))
    decider = load_decider(path)
    scored_rows: list[tuple[dict[str, Any], Scored]] = []
    for start in range(0, len(rows), CHUNK):
        chunk = rows[start:start + CHUNK]
        for row, result in zip(chunk, score_rows(decider, chunk, batched=(path != "reference_single")), strict=True):
            scored_rows.append((row, result))
    del decider
    release()
    OUT.mkdir(parents=True, exist_ok=True)
    errors: Counter[str] = Counter()
    with DEV_PROBS.open("w", encoding="utf-8") as handle:
        for row, result in scored_rows:
            item_id = str(row["item_id"])
            if result.probs is None:
                errors[str(result.error)] += 1
                handle.write(json.dumps({"item_id": item_id, "status": "error", "error": result.error}) + "\n")
                continue
            handle.write(json.dumps({
                "item_id": item_id, "status": "scored", "sets": sorted(membership[item_id]),
                "qtype": row["qtype"], "template_id": row["template_id"], "gold": row["gold"], "probs": result.probs,
                "argmax_key": row["options"][sl.argmax_index(result.probs)]["key"],
                "input_tokens": result.input_tokens}) + "\n")

    def accuracy(set_name: str | None) -> dict[str, Any]:
        pairs = [(row, result.probs) for row, result in scored_rows
                 if result.probs is not None and (set_name is None or set_name in membership[str(row["item_id"])])]
        return sl.accuracy_by_qtype(pairs)

    write_json(DEV_ACCURACY, {
        "teacher": sl.TEACHER, "path": path, "diagnostic_only": True,
        "rows_in_union": len(by_id), "mappable": len(rows), "skipped_before_scoring": dict(skipped),
        "scoring_errors": dict(errors),
        "accuracy": {"union": accuracy(None), "cg_v2": accuracy("cg_v2"),
                     "knowledge_dev_v2": accuracy("knowledge_dev_v2")},
        "generated_utc": utcnow(),
    })
    print(f"[{utcnow()}] dev: {len(scored_rows)} rows scored with {path}")


def read_done_indices() -> set[int]:
    """Pool positions already scored. The records are keyed by position: item ids are not unique."""
    if not ANCHOR_PROBS.exists():
        return set()
    return {int(rec["pool_index"]) for rec in iter_jsonl(ANCHOR_PROBS)}


def cmd_anchor(args: argparse.Namespace) -> None:
    path = chosen_path()
    pool_sha = file_sha256(POOL)
    done = read_done_indices()
    if done and json.loads(ANCHOR_PROGRESS.read_text())["pool_file_sha256"] != pool_sha:
        raise SystemExit("the pool file changed since this scoring started: stop (its positions would not match)")
    decider = load_decider(path)
    deadline = time.time() + args.max_hours * 3600
    started = time.time()
    statuses: Counter[str] = Counter()
    tokens = 0
    seen_pool = 0
    batch: list[tuple[int, dict[str, Any]]] = []

    def flush(handle: Any) -> None:
        nonlocal tokens
        rows = [row for _, row in batch]
        scored = score_rows(decider, rows, batched=(path != "reference_single"))
        for (pool_index, row), result in zip(batch, scored, strict=True):
            if result.probs is None:
                statuses[f"error: {result.error}"] += 1
                record = {"pool_index": pool_index, "item_id": str(row["item_id"]), "status": "error",
                          "error": result.error}
            else:
                statuses["scored"] += 1
                tokens += result.input_tokens
                record = {"pool_index": pool_index, "item_id": str(row["item_id"]), "status": "scored",
                          "template_id": row["template_id"], "qtype": row["qtype"], "probs": result.probs,
                          "argmax_key": row["options"][sl.argmax_index(result.probs)]["key"],
                          "input_tokens": result.input_tokens}
            handle.write(json.dumps(record) + "\n")
        handle.flush()
        elapsed = time.time() - started
        write_json(ANCHOR_PROGRESS, {"path": path, "pool_file_sha256": pool_sha, "statuses": dict(statuses),
                                     "pool_rows_read": seen_pool,
                                     "elapsed_s": round(elapsed, 1), "items_per_s": round(statuses["scored"] / elapsed, 3),
                                     "input_tokens_per_s": round(tokens / elapsed, 1), "utc": utcnow()})
        print(f"[{utcnow()}] {dict(statuses)} elapsed {elapsed / 3600:.2f} h", flush=True)

    limit = args.limit or None
    with ANCHOR_PROBS.open("a", encoding="utf-8") as handle:
        for pool_index, row in enumerate(iter_jsonl(POOL)):
            if limit is not None and pool_index >= limit:
                break
            seen_pool += 1
            if pool_index in done:
                continue
            batch.append((pool_index, row))
            if len(batch) == CHUNK:
                flush(handle)
                batch.clear()
                if time.time() >= deadline:
                    break
        if batch:  # the trailing partial chunk, at the pool's end or at --limit (a time-budget stop leaves batch empty)
            flush(handle)
            batch.clear()
    print(f"[{utcnow()}] anchor stopped: {dict(statuses)}; pool rows read {seen_pool}")


def cmd_join(args: argparse.Namespace) -> None:
    records: dict[int, dict[str, Any]] = {}
    for rec in iter_jsonl(ANCHOR_PROBS):
        index = int(rec["pool_index"])
        if index in records:
            raise SystemExit(f"pool row {index} is scored twice: stop")
        records[index] = rec
    counts: Counter[str] = Counter()
    with SOFT.open("w", encoding="utf-8") as handle:
        for pool_index, row in enumerate(iter_jsonl(POOL)):
            rec = records.get(pool_index)
            if rec is None:
                counts["pool row not scored (time budget)"] += 1
                continue
            if rec["status"] != "scored":
                counts[f"not scored: {rec['error']}"] += 1
                continue
            if rec["item_id"] != str(row["item_id"]) or rec["qtype"] != row["qtype"] or rec["template_id"] != row["template_id"]:
                raise SystemExit(f"score record {pool_index} does not match its pool row: stop")
            handle.write(json.dumps(sl.soft_label_row(row, rec["probs"]), ensure_ascii=False) + "\n")
            counts["soft label written"] += 1
    text_keys, record_keys, _ = bench_keys()
    overlap = sum(1 for row in iter_jsonl(SOFT) if sl.overlap_reason(row, text_keys, record_keys) is not None)
    if overlap:
        raise SystemExit(f"{overlap} soft-label rows overlap a benchmark row: the distillation set is not clean: stop")
    write_json(JOIN_MANIFEST, {"teacher": sl.TEACHER, "counts": dict(sorted(counts.items())),
                               "soft_label_file": str(SOFT), "soft_label_file_sha256": file_sha256(SOFT),
                               "overlap_with_benchmark_rows": overlap, "generated_utc": utcnow()})
    print(f"[{utcnow()}] join: {dict(counts)}; overlap {overlap}")


def cmd_report(args: argparse.Namespace) -> None:
    pool = json.loads(POOL_MANIFEST.read_text())
    parity = json.loads(PARITY.read_text())
    dev = json.loads(DEV_ACCURACY.read_text()) if DEV_ACCURACY.exists() else None
    join = json.loads(JOIN_MANIFEST.read_text()) if JOIN_MANIFEST.exists() else None
    lines = ["# osler_v4 Y5 — pplx-decider-v1.2-27b soft labels (aggregates only)", "",
             f"Teacher: `{sl.TEACHER}`; licence: Apache-2.0 (pinned snapshot). Temperature (saved): {parity['temperature']}.", "",
             "## Parity (40 CG v2 items, dev)", "", "| comparison | argmax agreement | max abs diff | mean abs diff |",
             "|---|---|---|---|"]
    for name, cmp in parity["comparisons_vs_single_request"].items():
        lines.append(f"| {name} vs reference_single | {cmp['argmax_agree']}/{cmp['rows']} | {cmp['max_abs_diff']:.4g} "
                     f"| {cmp['mean_abs_diff']:.4g} |")
    lines += ["", f"Scoring path: `{parity['path']}`. Seconds: " + ", ".join(
        f"{k} {v:.1f}" for k, v in parity["seconds"].items()) + ".", ""]
    lines += ["## Anchor pool", "", f"Eligible unique anchor rows after exclusions: {pool['eligible_unique']}; written: "
              f"{pool['pool_rows_written']} (cap {pool['pool_cap']}). Counts:", ""]
    lines += [f"- {k}: {v}" for k, v in pool["counts"].items()]
    if dev is not None:
        lines += ["", "## Teacher accuracy on dev diagnostics (not used for any choice)", "",
                  "| set | qtype | items | accuracy |", "|---|---|---|---|"]
        for set_name, table in dev["accuracy"].items():
            for qtype, stats in table.items():
                lines.append(f"| {set_name} | {qtype} | {stats['items']} | {stats['accuracy']:.4f} |")
    if join is not None:
        lines += ["", "## Soft labels joined", "", f"Overlap with benchmark rows: {join['overlap_with_benchmark_rows']}.", ""]
        lines += [f"- {k}: {v}" for k, v in join["counts"].items()]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[{utcnow()}] wrote {REPORT}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("pool")
    sub.add_parser("parity")
    sub.add_parser("dev")
    anchor = sub.add_parser("anchor")
    anchor.add_argument("--max-hours", type=float, default=5.0)
    anchor.add_argument("--limit", type=int, default=0, help="read only the first N pool rows (smoke runs)")
    sub.add_parser("join")
    sub.add_parser("report")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    {"pool": cmd_pool, "parity": cmd_parity, "dev": cmd_dev, "anchor": cmd_anchor, "join": cmd_join,
     "report": cmd_report}[args.command](args)


if __name__ == "__main__":
    main()
