"""d3_distill step 1 (GPU): label the pool with the local d3-ultra teacher through the batched runtime path.

One one-question request per row (the card's shape); the forwards are fused over rows by
``meddecide.distill.d3_batch`` exactly as in step 0. Resumable: rows already in the labels file are skipped, windows of
the pool are labelled together and appended in pool order, and the file is flushed per window.

Quality: every row's teacher answer must be a distribution over exactly its options (finite, non-negative, sums to 1
within 1e-3) and non-constant. Invalid rows are dropped and counted; an invalid rate above 2 % stops the run (PLAN.md
step 0 stop condition). No row drops silently: every drop has a reason count.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ.setdefault("TRITON_CACHE_AUTOTUNING", "1")

import torch
from transformers import AutoModel

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from meddecide.distill.d3_batch import (  # noqa: E402
    align_probabilities,
    distribution_ok,
    length_batches,
    non_constant,
    prepare_request,
    score_batch,
)
from meddecide.utils.hashing import file_sha256  # noqa: E402

POOL = REPO / "data" / "train" / "d3_distill" / "pool_200k.jsonl"
LABELS = REPO / "outputs" / "d3_distill" / "step1" / "labels.jsonl"
MANIFEST = REPO / "outputs" / "d3_distill" / "step1" / "label_manifest.json"
INVALID_STOP_RATE = 0.02


def utcnow() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def environment(snapshot: Path) -> dict:
    import transformers

    commit = subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    return {
        "git_commit": commit,
        "snapshot": str(snapshot),
        "revision": snapshot.name,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "offline": os.environ.get("HF_HUB_OFFLINE"),
    }


def iter_pool(pool: Path):
    with pool.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def done_ids(labels: Path) -> set[str]:
    if not labels.exists():
        return set()
    done = set()
    with labels.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                done.add(json.loads(line)["row_id"])
    return done


def label_window(model, rows: list[dict], *, max_rows: int, max_tokens: int) -> tuple[list[dict], dict]:
    """Label one window of pool rows; returns the label rows (with invalid reasons) and timing."""
    runtime = model.runtime
    t0 = time.time()
    prepared, failures = [], {}
    for row in rows:
        options = [(o["key"], o["label"]) for o in row["options"]]
        try:
            prepared.append(
                prepare_request(runtime, len(prepared), row["state"], row["question"], row["qtype"], options)
            )
        except ValueError as exc:
            failures[row["row_id"]] = {"valid": False, "error": "prepare", "reason": str(exc)[:300]}
    prepare_s = time.time() - t0
    # index -> pool row (prepared indices are positions in the kept list)
    kept = [row for row in rows if row["row_id"] not in failures]
    batches = length_batches(prepared, max_rows=max_rows, max_tokens=max_tokens)
    probs: list[list[float] | None] = [None] * len(prepared)
    forward_s = 0.0
    with runtime.request_temperature(1):
        for batch in batches:
            torch.cuda.synchronize()
            t1 = time.time()
            values = score_batch(runtime, batch)
            torch.cuda.synchronize()
            forward_s += time.time() - t1
            for row, value in zip(batch, values, strict=True):
                probs[row.index] = value
    out = []
    for i, row in enumerate(kept):
        options = [(o["key"], o["label"]) for o in row["options"]]
        try:
            aligned = align_probabilities(row["qtype"], options, probs[i]) if probs[i] is not None else None
        except ValueError:
            aligned = None
        valid = aligned is not None and distribution_ok(aligned, len(options)) and non_constant(aligned)
        out.append(
            {
                "row_id": row["row_id"],
                "source": row["source"],
                "template_id": row["template_id"],
                "qtype": row["qtype"],
                "split": row["split"],
                "option_keys": [o["key"] for o in row["options"]],
                "gold": row["gold"],
                "teacher_probs": aligned,
                "valid": bool(valid),
                "error": None if valid else ("invalid_distribution" if aligned is not None else "missing"),
                "input_tokens": prepared[i].length,
            }
        )
    for row in rows:
        if row["row_id"] in failures:
            out.append(
                {
                    "row_id": row["row_id"],
                    "source": row["source"],
                    "template_id": row["template_id"],
                    "qtype": row["qtype"],
                    "split": row["split"],
                    "option_keys": [o["key"] for o in row["options"]],
                    "gold": row["gold"],
                    "teacher_probs": None,
                    "valid": False,
                    **failures[row["row_id"]],
                }
            )
    # Keep the pool order inside the window.
    order = {row["row_id"]: i for i, row in enumerate(rows)}
    out.sort(key=lambda r: order[r["row_id"]])
    return out, {"prepare_s": prepare_s, "forward_s": forward_s, "rows": len(rows), "batches": len(batches)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--snapshot", type=Path, required=True)
    ap.add_argument("--pool", type=Path, default=POOL)
    ap.add_argument("--labels", type=Path, default=LABELS)
    ap.add_argument("--manifest", type=Path, default=MANIFEST)
    ap.add_argument("--max-rows", type=int, default=0, help="0: the whole pool")
    ap.add_argument("--window", type=int, default=4000)
    ap.add_argument("--max-rows-per-batch", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=32768)
    args = ap.parse_args()

    args.labels.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    done = done_ids(args.labels)
    todo = [row for row in iter_pool(args.pool) if row["row_id"] not in done]
    if args.max_rows:
        todo = todo[: args.max_rows]
    if not todo:
        print(f"[{utcnow()}] nothing to do: {len(done)} rows already labelled", flush=True)
        return
    print(f"[{utcnow()}] {len(done)} rows already labelled; {len(todo)} to label", flush=True)

    model_loaded_at = time.time()
    model = AutoModel.from_pretrained(
        str(args.snapshot.resolve()), trust_remote_code=True, dtype=torch.bfloat16, device="cuda:0"
    )
    load_s = time.time() - model_loaded_at
    env = environment(args.snapshot.resolve())
    torch.cuda.reset_peak_memory_stats()

    started = time.time()
    total, invalid, valid_rows = 0, 0, 0
    by_source: Counter[str] = Counter()
    by_split: Counter[str] = Counter()
    errors: Counter[str] = Counter()
    sum_min, sum_max = float("inf"), float("-inf")
    window_times = []
    with args.labels.open("a", encoding="utf-8") as fh:
        for start in range(0, len(todo), args.window):
            window = todo[start : start + args.window]
            rows, timing = label_window(
                model, window, max_rows=args.max_rows_per_batch, max_tokens=args.max_tokens
            )
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                total += 1
                if row["valid"]:
                    valid_rows += 1
                    by_source[row["source"]] += 1
                    by_split[row["split"]] += 1
                    sums = sum(row["teacher_probs"])
                    sum_min, sum_max = min(sum_min, sums), max(sum_max, sums)
                else:
                    invalid += 1
                    errors[f"{row['source']}:{row.get('error') or row.get('reason')}"] += 1
            fh.flush()
            window_times.append(timing)
            elapsed = time.time() - started
            rate = total / elapsed
            print(
                f"[{utcnow()}] {total}/{len(todo)} rows, {valid_rows} valid, {invalid} invalid ({invalid / total:.2%}), "
                f"{rate:.1f} rows/s; prepare {timing['prepare_s']:.1f}s forward {timing['forward_s']:.1f}s "
                f"(window {len(window)})",
                flush=True,
            )
            if invalid / total > INVALID_STOP_RATE:
                print(f"[{utcnow()}] STOP: invalid rate {invalid / total:.2%} is over the 2 % stop condition", flush=True)
                break
    wall = time.time() - started
    manifest = {
        "pool": str(args.pool),
        "labels": str(args.labels),
        "rows_total_in_file": len(done) + total,
        "rows_this_run": total,
        "rows_already_labelled": len(done),
        "valid": valid_rows,
        "invalid": invalid,
        "invalid_rate": invalid / total if total else None,
        "stopped": invalid / total > INVALID_STOP_RATE if total else False,
        "errors": dict(sorted(errors.items())),
        "by_source": dict(sorted(by_source.items())),
        "by_split": dict(sorted(by_split.items())),
        "sum_min": sum_min if valid_rows else None,
        "sum_max": sum_max if valid_rows else None,
        "load_s": round(load_s, 1),
        "wall_s": round(wall, 1),
        "rows_per_s": round(total / wall, 2) if wall else None,
        "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
        "windows": len(window_times),
        "max_rows_per_batch": args.max_rows_per_batch,
        "batch_note": "length-sorted left-padded batches; one one-question request per row; temperature 1.046",
        "finished_utc": utcnow(),
        **env,
    }
    # Hash and row count of the whole (possibly resumed) labels file, not just this run's window.
    if args.labels.exists():
        manifest["labels_sha256"] = file_sha256(args.labels)
        with args.labels.open(encoding="utf-8") as fh:
            manifest["labels_rows"] = sum(1 for line in fh if line.strip())
    args.manifest.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in (
        "rows_this_run", "valid", "invalid", "invalid_rate", "rows_per_s", "peak_gpu_gb", "stopped")}), flush=True)


if __name__ == "__main__":
    main()
