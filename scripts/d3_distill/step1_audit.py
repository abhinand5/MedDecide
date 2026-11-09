"""d3_distill step 1 audit (CPU): re-derive every headline label-set number from the raw files.

Reads the pool and the labels file, recomputes the distributions, sums, per-source/per-split counts, the teacher's
top-1 gold accuracy (a diagnostic of the labels, never a gate) and the file hashes, and writes a JSON plus a short
markdown audit. Nothing is taken from the labelling run's own manifest without re-deriving it.

Usage:
    python scripts/d3_distill/step1_audit.py --out outputs/d3_distill/step1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
POOL = REPO / "data" / "train" / "d3_distill" / "pool_200k.jsonl"
LABELS = REPO / "outputs" / "d3_distill" / "step1" / "labels.jsonl"
Y9 = REPO / "data" / "bench" / "osler_v4" / "y9"
Y9_SETS = ("knowledge", "panel", "mr_verdict")
TOL = 1e-3


def utcnow() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def y9_overlap(pool: Path) -> dict:
    """Exact normalised-text overlap between the pool and the frozen Y9 sets that are now on this machine."""
    from meddecide.panel.overlap import text_key

    ykeys: dict[str, set[str]] = {}
    items = 0
    for name in Y9_SETS:
        path = Y9 / f"{name}.jsonl"
        if not path.exists():
            continue
        for line in path.open(encoding="utf-8"):
            row = json.loads(line)
            items += 1
            key = text_key(row["state"], row["question"], [o["label"] for o in row["options"]])
            ykeys.setdefault(key, set()).add(name)
    matches, by_set = 0, Counter()
    for line in pool.open(encoding="utf-8"):
        row = json.loads(line)
        key = text_key(row["state"], row["question"], [o["label"] for o in row["options"]])
        if key in ykeys:
            matches += 1
            for name in ykeys[key]:
                by_set[name] += 1
    return {"items": items, "unique_keys": len(ykeys), "matches": matches, "by_set": dict(sorted(by_set.items())),
            "note": "exact normalised text key; the MIMIC suite, MedNLI test and CG v2 are still absent"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pool", type=Path, default=POOL)
    ap.add_argument("--labels", type=Path, default=LABELS)
    ap.add_argument("--out", type=Path, default=REPO / "outputs" / "d3_distill" / "step1")
    args = ap.parse_args()

    pool = {row["row_id"]: row for row in read_jsonl(args.pool)}
    labels = read_jsonl(args.labels)
    ids = [row["row_id"] for row in labels]
    duplicate_ids = len(ids) - len(set(ids))
    missing_pool = sum(1 for i in ids if i not in pool)
    sums, option_counts, constant = [], Counter(), 0
    invalid, correct = 0, 0
    by_source, by_split = Counter(), Counter()
    correct_by_source: Counter[str] = Counter()
    for row in labels:
        by_source[row["source"]] += 1
        by_split[row["split"]] += 1
        probs = row.get("teacher_probs")
        if not row.get("valid") or probs is None:
            invalid += 1
            continue
        if len(probs) != len(row["option_keys"]):
            invalid += 1
            continue
        if not all(math.isfinite(p) and p >= 0 for p in probs):
            invalid += 1
            continue
        total = sum(probs)
        sums.append(total)
        option_counts[len(probs)] += 1
        if len({round(p, 6) for p in probs}) == 1:
            constant += 1
        keys = row["option_keys"]
        best = keys[max(range(len(probs)), key=probs.__getitem__)]
        correct += int(best == row["gold"])
        correct_by_source[row["source"]] += int(best == row["gold"])
    accuracy = correct / len(labels) if labels else None
    audit = {
        "pool_rows": len(pool),
        "label_rows": len(labels),
        "duplicate_row_ids": duplicate_ids,
        "labels_without_pool_row": missing_pool,
        "pool_rows_without_label": len(set(pool) - set(ids)),
        "invalid": invalid,
        "invalid_rate": invalid / len(labels) if labels else None,
        "constant": constant,
        "sum_min": min(sums) if sums else None,
        "sum_max": max(sums) if sums else None,
        "all_sums_within_1e-3": all(abs(s - 1) <= TOL for s in sums),
        "option_counts": dict(sorted(option_counts.items())),
        "teacher_top1_gold_accuracy": accuracy,
        "teacher_top1_by_source": {
            s: {"correct": correct_by_source[s], "items": n, "accuracy": correct_by_source[s] / n}
            for s, n in sorted(by_source.items())
        },
        "per_source": dict(sorted(by_source.items())),
        "per_split": dict(sorted(by_split.items())),
        "y9_overlap": y9_overlap(args.pool),
        "pool_sha256": hashlib.sha256(args.pool.read_bytes()).hexdigest(),
        "labels_sha256": hashlib.sha256(args.labels.read_bytes()).hexdigest(),
        "finished_utc": utcnow(),
    }
    out = args.out if args.out.is_absolute() else REPO / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "audit.json").write_text(json.dumps(audit, indent=1), encoding="utf-8")
    lines = [
        "# d3_distill step 1 — SELF_AUDIT",
        "",
        f"Re-derived `{utcnow()}` from the raw pool and labels files; nothing taken from the run's own manifest.",
        "",
        f"- labels: {audit['label_rows']} rows, {audit['invalid']} invalid ({audit['invalid_rate']:.4%} of rows)",
        f"- duplicate row ids: {audit['duplicate_row_ids']}; labels without a pool row: {audit['labels_without_pool_row']}; "
        f"pool rows without a label: {audit['pool_rows_without_label']}",
        f"- sums: [{audit['sum_min']}, {audit['sum_max']}], all within 1e-3: {audit['all_sums_within_1e-3']}",
        f"- option counts: {audit['option_counts']}; constant distributions: {audit['constant']}",
        f"- teacher top-1 gold accuracy (diagnostic): {audit['teacher_top1_gold_accuracy']}",
        f"- per source: {audit['per_source']}",
        f"- per split: {audit['per_split']}",
        f"- frozen Y9 text overlap: {audit['y9_overlap']['matches']} matches over "
        f"{audit['y9_overlap']['items']} items ({audit['y9_overlap']['by_set']})",
        f"- pool sha256: `{audit['pool_sha256']}`; labels sha256: `{audit['labels_sha256']}`",
        "",
        "Suspicious/limits: the labels are the teacher's answers on train-side items only; the teacher's top-1 accuracy",
        "here is in-distribution and is not a benchmark number. The frozen-set overlap checks are NOT MEASURED (see",
        "STATE.md constraint 2).",
        "",
    ]
    (out / "SELF_AUDIT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({k: audit[k] for k in (
        "label_rows", "invalid", "invalid_rate", "constant", "sum_min", "sum_max",
        "all_sums_within_1e-3", "teacher_top1_gold_accuracy", "per_source")}, indent=1))


if __name__ == "__main__":
    main()
