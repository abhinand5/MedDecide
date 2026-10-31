"""d1-2B on one frozen set (bench/d1-2b): one state and one question per call, through the card's System One API.

Reads the frozen set (the osler_v4 Y9 copies; each row carries item_id, template_id, qtype, options, gold, state, question),
writes one prediction row per item to outputs/d1_bench/<set>/predictions.jsonl (resumable: finished item ids are skipped)
and a manifest with the counts, the file hash and the provenance. Any failure stops the run: nothing is dropped silently.
Predictions hold option probabilities and gold keys: private to this machine, never committed.

Usage (one GPU job at a time; detached):
    setsid nohup uv run --frozen --no-sync python scripts/bench/d1_score.py --set knowledge \
        > outputs/d1_bench/logs/knowledge.log 2>&1 < /dev/null &
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from collections import Counter
from pathlib import Path

import torch
import transformers

from meddecide.model.d1 import QTYPES, REPO, REVISION, D1Model
from meddecide.utils.hashing import file_sha256
from meddecide.utils.io import iter_jsonl, write_json
from meddecide.utils.provenance import git_commit, gpu_name, utcnow

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = Path("/workspace/MedDecide/data/bench/osler_v4/y9")  # the frozen Y9 sets (gitignored data, main checkout)
OUT_ROOT = REPO_ROOT / "outputs" / "d1_bench"
# The frozen counts each set must have (the Y9 sets manifest and the task brief). A mismatch stops the run.
SETS = {"knowledge": 5_456, "mednli_test": 1_422, "panel": 4_353, "mr_verdict": 13_103, "mimic_suite": 4_253}


def _git(args: list[str]) -> str:
    return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--set", required=True, choices=sorted(SETS))
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT)
    parser.add_argument("--limit", type=int, default=0, help="score only the first N new items (smoke runs)")
    args = parser.parse_args()

    items_path = args.data_dir / f"{args.set}.jsonl"
    out_dir = args.out_root / args.set
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "predictions.jsonl"

    rows = list(iter_jsonl(items_path))
    if len(rows) != SETS[args.set]:
        raise SystemExit(f"{args.set}: {len(rows)} rows, expected {SETS[args.set]}")
    qtypes = Counter(str(r["qtype"]) for r in rows)
    unknown = set(qtypes) - set(QTYPES)
    if unknown:
        raise SystemExit(f"{args.set}: qtypes {sorted(unknown)} are not handled by the d1 runner")

    done: set[str] = set()
    if pred_path.exists():
        with pred_path.open(encoding="utf-8") as fh:
            done = {str(json.loads(line)["item_id"]) for line in fh if line.strip()}
    print(f"[{utcnow()}] {args.set}: {len(rows)} items, resume_rows={len(done)}", flush=True)

    loaded = D1Model.load()
    started = time.time()
    new = 0
    with pred_path.open("a", encoding="utf-8") as fh, torch.inference_mode():
        for row in rows:
            item_id = str(row["item_id"])
            if item_id in done:
                continue
            if args.limit and new >= args.limit:
                break
            options = [(str(o["key"]), str(o["label"])) for o in row["options"]]
            t0 = time.time()
            probs, meta = loaded.answer(row["state"], str(row["qtype"]), str(row["question"]), options)
            record = {"item_id": item_id, "set_name": args.set, "template_id": str(row["template_id"]),
                      "qtype": str(row["qtype"]), "option_keys": [k for k, _ in options], "gold": str(row["gold"]),
                      "status": "scored", "probs": [float(p) for p in probs],
                      "latency_s": round(time.time() - t0, 4), **meta}
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            new += 1
            if new % 500 == 0:
                fh.flush()
                rate = new / (time.time() - started)
                print(f"[{utcnow()}] {new} new rows, {rate:.1f} items/s", flush=True)
    total = len(done) + new
    complete = total == len(rows)
    manifest = {
        "task": "bench/d1-2b", "set": args.set, "utc": utcnow(), "git_commit": git_commit(REPO_ROOT),
        "branch": _git(["branch", "--show-current"]), "model": REPO, "revision": REVISION,
        "items_file": str(items_path), "items_sha256": file_sha256(items_path), "items": len(rows),
        "qtypes": dict(qtypes), "scored_total": total, "scored_new_this_run": new, "complete": complete,
        "limit": args.limit, "transformers": transformers.__version__, "torch": torch.__version__,
        "gpu": gpu_name(), "calibration_applied": loaded.calibrated, "wall_s_this_run": round(time.time() - started, 1),
        "command": "scripts/bench/d1_score.py --set " + args.set,
    }
    write_json(out_dir / "manifest.json", manifest)
    print(f"[{utcnow()}] done: scored_total={total} of {len(rows)} complete={complete}", flush=True)


if __name__ == "__main__":
    main()
