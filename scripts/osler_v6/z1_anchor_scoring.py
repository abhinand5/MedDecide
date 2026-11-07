"""Z1 (osler_v6 ADVISORY §2.2, D58): score the 400,000-item general anchor pool with the osler_v4 Y5 path.

The pool rule, the request mapping, the teacher call and the record format are osler_v4's
(`scripts/osler_v4/y5_soft_labels.py` + `meddecide.distill.soft_labels`). This module changes only what Z1 must
change: every output goes to `outputs/osler_v6/Z1/` (nothing is appended to osler_v4's files), and the resume file
is seeded from the 112,128 records osler_v4 already scored.

Positions are not reusable across the two pools: Y5 built its pool on 2026-10-21, before the osler_v4 y9 panel,
knowledge, MR and MIMIC evaluation files existed. The pool is therefore rebuilt against today's bench exclusions and
the old records are matched by normalised-text key (the pool key), then each record is verified against its new pool
row (item id, template, qtype, option count, probability sum). Old records that the new exclusion set removes are
dropped and counted by reason and by the bench file they now overlap.

Commands:
  pool     rebuild the pool with today's bench exclusions into `data/train/osler_v6/z1_anchor_pool.jsonl`
  reuse    seed `outputs/osler_v6/Z1/anchor_probs.jsonl` from osler_v4's Y5 records; re-check the pool exclusions
  parity   re-run the 40-item parity check and choose the scoring path (GPU)
  anchor   score the pool from the resume point until `--max-hours` (GPU; the caller detaches it)
  join     write `data/train/osler_v6/z1_anchor_soft.jsonl`; overlap with every bench row must be 0
  summary  aggregate counts for STATE, CLAIMS and SELF_AUDIT
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from meddecide.distill import soft_labels as sl
from meddecide.utils.hashing import file_sha256
from meddecide.utils.io import iter_jsonl, write_json
from meddecide.utils.provenance import utcnow

REPO_ROOT = Path(__file__).resolve().parents[2]
Y5_SCRIPT = REPO_ROOT / "scripts" / "osler_v4" / "y5_soft_labels.py"
Y5_OUT = REPO_ROOT / "outputs" / "osler_v4" / "Y5"
Y5_POOL = REPO_ROOT / "data" / "train" / "osler_v4" / "y5_anchor_pool.jsonl"

Z1 = REPO_ROOT / "outputs" / "osler_v6" / "Z1"
POOL = REPO_ROOT / "data" / "train" / "osler_v6" / "z1_anchor_pool.jsonl"
SOFT = REPO_ROOT / "data" / "train" / "osler_v6" / "z1_anchor_soft.jsonl"
POOL_COMPARE = Z1 / "pool_compare.json"
REUSE_MANIFEST = Z1 / "reuse_manifest.json"
EXCLUSION_CHECK = Z1 / "exclusion_check.json"
SUMMARY = Z1 / "summary.json"


def load_y5() -> Any:
    """osler_v4's Y5 module, loaded by path (it is a script, not an importable package)."""
    if "y5_soft_labels" in sys.modules:
        return sys.modules["y5_soft_labels"]
    spec = importlib.util.spec_from_file_location("y5_soft_labels", Y5_SCRIPT)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load the osler_v4 Y5 script at {Y5_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["y5_soft_labels"] = module
    spec.loader.exec_module(module)
    return module


def configured() -> Any:
    """The Y5 module with every path pointed at this loop's data and outputs."""
    y5 = load_y5()
    y5.BENCH = REPO_ROOT / "data" / "bench"
    y5.TRAIN_SOURCES = (
        REPO_ROOT / "data" / "train" / "osler_v2" / "block_C1.jsonl",
        REPO_ROOT / "data" / "train" / "osler_v3" / "replay_broad_train.jsonl",
    )
    y5.POOL = POOL
    y5.SOFT = SOFT
    y5.CG = REPO_ROOT / "data" / "bench" / "osler_v4" / "cg_v2.jsonl"
    y5.KNOWLEDGE = REPO_ROOT / "data" / "bench" / "osler_v2" / "knowledge_dev_v2.jsonl"
    y5.OUT = Z1
    y5.POOL_MANIFEST = Z1 / "anchor_pool_manifest.json"
    y5.PARITY = Z1 / "parity.json"
    y5.DEV_PROBS = Z1 / "dev_probs.jsonl"
    y5.DEV_ACCURACY = Z1 / "dev_accuracy.json"
    y5.ANCHOR_PROBS = Z1 / "anchor_probs.jsonl"
    y5.ANCHOR_PROGRESS = Z1 / "anchor_progress.json"
    y5.JOIN_MANIFEST = Z1 / "join_manifest.json"
    y5.REPORT = Z1 / "teacher27b.md"
    return y5


def bench_index() -> tuple[set[str], set[Any], dict[str, str], dict[Any, str], dict[str, int], list[str]]:
    """Text keys, record keys, and the file that owns each, over every benchmark row of every split."""
    text_keys: set[str] = set()
    record_keys: set[Any] = set()
    text_owner: dict[str, str] = {}
    record_owner: dict[Any, str] = {}
    files: dict[str, int] = {}
    for path in sorted((REPO_ROOT / "data" / "bench").rglob("*.jsonl")):
        name = str(path.relative_to(REPO_ROOT))
        count = 0
        for row in iter_jsonl(path):
            text, record = sl.overlap_keys(row)
            text_keys.add(text)
            text_owner.setdefault(text, name)
            if record is not None:
                record_keys.add(record)
                record_owner.setdefault(record, name)
            count += 1
        files[name] = count
    return text_keys, record_keys, text_owner, record_owner, files, sorted(files)


def repo_relative(name: str) -> str:
    """A manifest path relative to the repository root, whether it was written relative or absolute."""
    path = Path(name)
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return name


def cmd_pool(args: argparse.Namespace) -> None:
    """Rebuild the pool with today's exclusions; compare its sha256 with the pool osler_v4 scored."""
    y5 = configured()
    old_manifest = json.loads((Y5_OUT / "anchor_pool_manifest.json").read_text())
    old_sha = old_manifest["pool_file_sha256"]
    current_pool_sha = file_sha256(Y5_POOL)
    if current_pool_sha != old_sha:
        raise SystemExit(
            f"osler_v4's pool file sha256 {current_pool_sha} differs from its manifest {old_sha}: stop"
        )
    y5.cmd_pool(args)
    new_manifest = json.loads(y5.POOL_MANIFEST.read_text())
    new_sha = new_manifest["pool_file_sha256"]
    write_json(POOL_COMPARE, {
        "osler_v4_pool_file": str(Y5_POOL.relative_to(REPO_ROOT)),
        "osler_v4_pool_sha256": old_sha,
        "osler_v6_pool_file": str(POOL.relative_to(REPO_ROOT)),
        "osler_v6_pool_sha256": new_sha,
        "same_file": current_pool_sha == new_sha,
        "osler_v4_eligible_unique": old_manifest["eligible_unique"],
        "osler_v6_eligible_unique": new_manifest["eligible_unique"],
        "osler_v4_excluded_counts": {
            key: value for key, value in old_manifest["counts"].items() if key.startswith("excluded")
        },
        "osler_v6_excluded_counts": {
            key: value for key, value in new_manifest["counts"].items() if key.startswith("excluded")
        },
        "bench_files_scanned": len(new_manifest["bench_files"]),
        "bench_files_scanned_vs_y5": sorted(
            {repo_relative(name) for name in new_manifest["bench_files"]}
            ^ {repo_relative(name) for name in old_manifest["bench_files"]}
        ),
        "generated_utc": utcnow(),
    })
    print(f"[{utcnow()}] pool rebuilt: sha {new_sha} (osler_v4 {old_sha}; same={new_sha == old_sha})")


def cmd_reuse(args: argparse.Namespace) -> None:
    """Seed the Z1 resume file with osler_v4's scored records, matched by text key and verified row by row."""
    Z1.mkdir(parents=True, exist_ok=True)
    if not POOL.exists():
        raise SystemExit(f"{POOL} is missing: run the pool step first")
    if not POOL_COMPARE.exists():
        raise SystemExit(f"{POOL_COMPARE} is missing: run the pool step first")
    pool_sha = file_sha256(POOL)
    compare = json.loads(POOL_COMPARE.read_text())
    if pool_sha != compare["osler_v6_pool_sha256"]:
        raise SystemExit("the pool file changed after the pool step: stop")

    text_keys, record_keys, text_owner, record_owner, bench_files, _ = bench_index()
    old_index: dict[str, int] = {}
    for old_position, row in enumerate(iter_jsonl(Y5_POOL)):
        key, _ = sl.overlap_keys(row)
        if key in old_index:
            raise SystemExit(f"osler_v4 pool text key repeated at positions {old_index[key]} and {old_position}: stop")
        old_index[key] = old_position
    records: dict[int, dict[str, Any]] = {}
    for rec in iter_jsonl(Y5_OUT / "anchor_probs.jsonl"):
        position = int(rec["pool_index"])
        if position in records:
            raise SystemExit(f"osler_v4 pool position {position} is scored twice: stop")
        records[position] = rec
    if not records:
        raise SystemExit("no osler_v4 records found: stop")

    counts: Counter[str] = Counter()
    excluded_by_file: Counter[str] = Counter()
    new_keys: set[str] = set()
    matched: set[int] = set()
    max_prob_sum_error = 0.0
    with (Z1 / "anchor_probs.jsonl").open("w", encoding="utf-8") as handle:
        for new_position, row in enumerate(iter_jsonl(POOL)):
            key, _ = sl.overlap_keys(row)
            new_keys.add(key)
            reason = sl.overlap_reason(row, text_keys, record_keys)
            if reason is not None:
                raise SystemExit(f"new pool row {new_position} overlaps a bench row ({reason}): stop")
            old_position = old_index.get(key)
            if old_position is None:
                counts["new pool row without an osler_v4 pool row"] += 1
                continue
            rec = records.get(old_position)
            if rec is None:
                counts["osler_v4 pool row was not scored"] += 1
                continue
            if rec["status"] != "scored":
                counts[f"osler_v4 record not scored: {rec.get('error')}"] += 1
                continue
            for field in ("item_id", "template_id", "qtype"):
                if str(rec[field]) != str(row[field]):
                    raise SystemExit(
                        f"osler_v4 record {old_position} {field} {rec[field]!r} != pool row {new_position} "
                        f"{row[field]!r}: stop"
                    )
            probs = [float(p) for p in rec["probs"]]
            if len(probs) != len(row["options"]):
                raise SystemExit(
                    f"osler_v4 record {old_position} has {len(probs)} probabilities for "
                    f"{len(row['options'])} options: stop"
                )
            max_prob_sum_error = max(max_prob_sum_error, abs(sum(probs) - 1.0))
            handle.write(json.dumps({
                "pool_index": new_position,
                "item_id": str(row["item_id"]),
                "status": "scored",
                "template_id": row["template_id"],
                "qtype": row["qtype"],
                "probs": probs,
                "argmax_key": rec["argmax_key"],
                "input_tokens": int(rec["input_tokens"]),
                "reused_from_osler_v4_pool_index": old_position,
            }) + "\n")
            matched.add(old_position)
            counts["reused"] += 1
            counts["position preserved" if new_position == old_position else "position changed"] += 1

    dropped: Counter[str] = Counter()
    remaining = [position for position in records if position not in matched]
    # Why each unmatched osler_v4 *record* is gone: re-read the osler_v4 pool and test it against today's exclusions.
    for old_position, row in enumerate(iter_jsonl(Y5_POOL)):
        if old_position not in remaining:
            continue
        text_key, record_key = sl.overlap_keys(row)
        if text_key in new_keys:
            dropped["old record whose text is in the new pool but was unmatched"] += 1
            continue
        reason = sl.overlap_reason(row, text_keys, record_keys)
        if reason is None:
            dropped["old record outside the rebuilt 400,000 (no bench overlap)"] += 1
            continue
        owner = text_owner.get(text_key) or (
            record_owner.get(record_key) if record_key is not None else None
        )
        excluded_by_file[f"{owner} ({reason})"] += 1
        dropped[f"old record excluded by today's bench set ({reason})"] += 1

    scored = counts["reused"]
    write_json(Z1 / "anchor_progress.json", {
        "path": None,
        "pool_file_sha256": pool_sha,
        "statuses": {"scored": scored},
        "pool_rows_read": sum(1 for _ in iter_jsonl(POOL)),
        "elapsed_s": 0.0,
        "items_per_s": 0.0,
        "input_tokens_per_s": 0.0,
        "reused_from_osler_v4": scored,
        "utc": utcnow(),
    })
    write_json(REUSE_MANIFEST, {
        "teacher": sl.TEACHER,
        "match_rule": "normalised-text key (sl.overlap_keys[0]); old pool positions are not reused",
        "osler_v4_probs": str((Y5_OUT / "anchor_probs.jsonl").relative_to(REPO_ROOT)),
        "osler_v4_pool_file": str(Y5_POOL.relative_to(REPO_ROOT)),
        "osler_v4_pool_sha256": file_sha256(Y5_POOL),
        "z1_pool_file": str(POOL.relative_to(REPO_ROOT)),
        "z1_pool_sha256": pool_sha,
        "z1_pool_rows": len(new_keys),
        "osler_v4_records": len(records),
        "osler_v4_pool_rows_without_record": counts["osler_v4 pool row was not scored"],
        "counts": dict(sorted(counts.items())),
        "dropped_osler_v4_records": dict(sorted(dropped.items())),
        "dropped_osler_v4_records_by_bench_file": dict(sorted(excluded_by_file.items())),
        "checks": {
            "verification_mismatches": 0,
            "new_pool_rows_overlapping_bench": 0,
            "max_abs_probability_sum_error": round(max_prob_sum_error, 12),
            "bench_files_scanned": len(bench_files),
            "bench_text_keys": len(text_keys),
            "bench_record_keys": len(record_keys),
        },
        "generated_utc": utcnow(),
    })
    print(f"[{utcnow()}] reuse: {dict(counts)}; dropped {dict(dropped)}; by file {dict(excluded_by_file)}")


def cmd_parity(args: argparse.Namespace) -> None:
    """Re-run the 40-item parity check with osler_v4's implementation and choose the path."""
    y5 = configured()
    y5.cmd_parity(args)
    parity = json.loads(y5.PARITY.read_text())
    if parity["path"] not in y5.PATHS:
        raise SystemExit(f"parity chose an unknown path {parity['path']!r}: stop")
    print(f"[{utcnow()}] parity path {parity['path']}")


def cmd_anchor(args: argparse.Namespace) -> None:
    """Score the pool from the resume point; osler_v4's implementation and record format."""
    y5 = configured()
    y5.cmd_anchor(args)


def cmd_join(args: argparse.Namespace) -> None:
    """Join the scored records into the soft-label file; the overlap check inside must pass (0)."""
    y5 = configured()
    y5.cmd_join(args)
    join = json.loads(y5.JOIN_MANIFEST.read_text())
    if join["overlap_with_benchmark_rows"] != 0:
        raise SystemExit("soft-label rows overlap a benchmark row: stop")
    print(f"[{utcnow()}] join: {join['counts']}; overlap {join['overlap_with_benchmark_rows']}")


def cmd_summary(args: argparse.Namespace) -> None:
    """Aggregates for STATE, CLAIMS and SELF_AUDIT: scored totals, coverage, rate, qtypes."""
    pool_manifest = json.loads((Z1 / "anchor_pool_manifest.json").read_text())
    parity = json.loads((Z1 / "parity.json").read_text())
    progress = json.loads((Z1 / "anchor_progress.json").read_text())
    reuse = json.loads(REUSE_MANIFEST.read_text()) if REUSE_MANIFEST.exists() else {}
    records = list(iter_jsonl(Z1 / "anchor_probs.jsonl"))
    statuses: Counter[str] = Counter(str(rec["status"]) for rec in records)
    qtypes: Counter[str] = Counter(str(rec["qtype"]) for rec in records if rec["status"] == "scored")
    errors: Counter[str] = Counter(str(rec.get("error")) for rec in records if rec["status"] == "error")
    soft_rows = sum(1 for _ in iter_jsonl(SOFT)) if SOFT.exists() else None
    scored = statuses["scored"]
    summary = {
        "teacher": sl.TEACHER,
        "pool_rows": pool_manifest["pool_rows_written"],
        "pool_sha256": pool_manifest["pool_file_sha256"],
        "pool_eligible_unique": pool_manifest["eligible_unique"],
        "parity": {
            "items": parity["items"],
            "path": parity["path"],
            "argmax_agree_vs_single_reference": [
                {"comparison": name, "agree": cmp["argmax_agree"], "rows": cmp["rows"]}
                for name, cmp in parity["comparisons_vs_single_request"].items()
            ],
            "seconds": parity["seconds"],
        },
        "records": dict(sorted(statuses.items())),
        "scored_total": scored,
        "scored_reused_from_osler_v4": reuse.get("counts", {}).get("reused"),
        "scored_new": scored - int(reuse.get("counts", {}).get("reused", 0)),
        "coverage_of_pool": scored / pool_manifest["pool_rows_written"],
        "scored_qtypes": dict(sorted(qtypes.items())),
        "errors": dict(sorted(errors.items())),
        "anchor_progress": progress,
        "soft_label_file": str(SOFT.relative_to(REPO_ROOT)) if SOFT.exists() else None,
        "soft_label_rows": soft_rows,
        "generated_utc": utcnow(),
    }
    write_json(SUMMARY, summary)
    print(json.dumps(summary, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("pool")
    sub.add_parser("reuse")
    sub.add_parser("parity")
    anchor = sub.add_parser("anchor")
    anchor.add_argument("--max-hours", type=float, default=5.0)
    anchor.add_argument("--limit", type=int, default=0, help="read only the first N pool rows (smoke runs)")
    sub.add_parser("join")
    sub.add_parser("summary")
    args = parser.parse_args()
    (REPO_ROOT / "data" / "train" / "osler_v6").mkdir(parents=True, exist_ok=True)
    Z1.mkdir(parents=True, exist_ok=True)
    {"pool": cmd_pool, "reuse": cmd_reuse, "parity": cmd_parity, "anchor": cmd_anchor,
     "join": cmd_join, "summary": cmd_summary}[args.command](args)


if __name__ == "__main__":
    main()
