"""osler_v5 Z10 (ADVISORY §10 Z10; GPU; additional, labelled; no choice): Cloudflare Clef-Flash on the Z10 sets, through the authors' code.

Runs in envs/clef (the authors' stack; no meddecide import), like osler_v0's O2 runner scripts/osler/o2_run_clef.py, whose protocol it
keeps: load_release_model(snapshot), then systemone(model, processor, request) with one question per request, single order (the
authors' encoder sorts choice keys). The same rows as the zero-shot runner go in, and the rows come out in the O2 schema.

GATE (ADVISORY §10 Z10.2): this script refuses to start unless outputs/osler_v5/Z10/clef_local_check.json (written by
scripts/osler_v5/z10_clef_local_check.py) says local_only is true for this snapshot. Every Z10 set is gated, because the MR sets may
contain MIMIC-derived items and MedNLI is built from MIMIC-III notes. Without that record the Clef cells are NOT MEASURED.
The process also runs with HF_HUB_OFFLINE and TRANSFORMERS_OFFLINE set, so it cannot reach the Hub while it scores.

Outputs: outputs/osler_v5/Z10/clef_flash/<set>.jsonl (gitignored; resumable per set). Items the authors' limit rejects are written as
skipped rows with the reason, and nothing else is dropped.

Usage (GPU; one job; detached):
    source scripts/pod_env.sh && setsid nohup envs/clef/.venv/bin/python -I scripts/osler_v5/z10_clef.py --path <snapshot> \
        > outputs/osler_v5/Z10/logs/clef_flash.log 2>&1 < /dev/null &
A smoke run passes --limit N (N new items per set; delete the smoke files before the full run).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

REPO = Path(__file__).resolve().parents[2]
Y9_DIR = REPO / "data" / "bench" / "osler_v4" / "y9"
CHECK = REPO / "outputs" / "osler_v5" / "Z10" / "clef_local_check.json"
OUT = REPO / "outputs" / "osler_v5" / "Z10" / "clef_flash"
SETS = {
    "mr_verdict": "mr_verdict.jsonl",
    "mr_reported": "mr_reported.jsonl",
    "mimic_suite": "mimic_suite.jsonl",
    "mimic_heldout": "mimic_heldout.jsonl",
    "mednli_test": "mednli_test.jsonl",
    "panel": "panel.jsonl",
    "knowledge": "knowledge.jsonl",
}


def utcnow() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def check_gate(snapshot: Path) -> None:
    """Refuse to run unless the local-only check passed for this snapshot."""
    if not CHECK.exists():
        raise SystemExit(f"refused: {CHECK.relative_to(REPO)} does not exist (run z10_clef_local_check.py first)")
    record = json.loads(CHECK.read_text(encoding="utf-8"))
    if not (record.get("local_only") is True and record.get("revision_matches") is True):
        raise SystemExit("refused: the local-only check did not pass (Clef cells are NOT MEASURED)")
    if Path(record["snapshot"]).resolve() != snapshot:
        raise SystemExit("refused: the snapshot checked is not the snapshot given")


def iter_rows(path: Path):
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open(encoding="utf-8") as fh:
        return {json.loads(line)["item_id"] for line in fh if line.strip()}


def authors_question(item: dict) -> dict:
    if item["qtype"] == "choice":
        return {"type": "choice", "instructions": item["question"], "criteria": {o["key"]: o["label"] for o in item["options"]}}
    if item["qtype"] == "noul":
        return {"type": "noul", "instructions": item["question"]}
    return {"type": "score", "instructions": item["question"], "criteria": [o["label"] for o in item["options"]]}


def our_probs(item: dict, answer: dict) -> list[float]:
    """The authors' answer for one question, as a vector in our option order (as o2_run_clef.py)."""
    if item["qtype"] == "noul":
        p_true = float(answer["noul"])
        return [p_true, 1.0 - p_true]  # (yes, no)
    if item["qtype"] == "choice":
        return [float(answer["probabilities"][o["key"]]) for o in item["options"]]
    return [float(answer["probabilities"][str(i)]) for i in range(len(item["options"]))]


def prediction_row(item: dict, *, status: str, reason: str | None = None, probs: list[float] | None = None,
                   latency_s: float | None = None, extras: dict | None = None) -> dict:
    """The O2 row shape (meddecide.eval.o2_io.prediction_row), written here without importing meddecide."""
    return {
        "item_id": item["item_id"], "benchmark": item["benchmark"], "set_name": item["set_name"],
        "template_id": item["template_id"], "qtype": item["qtype"], "option_keys": [o["key"] for o in item["options"]],
        "gold": item["gold"], "status": status, "reason": reason, "probs": probs, "latency_s": latency_s,
        "prompt_tokens": None, "extras": extras or {},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--path", type=Path, required=True, help="the local snapshot directory of Cloudflare/clef-flash")
    parser.add_argument("--limit", type=int, default=0, help="score only the first N new items of each set (smoke runs)")
    args = parser.parse_args()
    snapshot = args.path.resolve()
    check_gate(snapshot)
    import torch

    sys.path.insert(0, str(snapshot))
    import joint_schema_model as authors  # the authors' module, from the snapshot

    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    model, processor = authors.load_release_model(snapshot, device="cuda")
    print(f"[{utcnow()}] clef_flash loaded in {time.time() - t0:.1f}s on {torch.cuda.get_device_name(0)} (offline)", flush=True)
    for set_name, file_name in SETS.items():
        out_path = OUT / f"{set_name}.jsonl"
        done = completed_ids(out_path)
        print(f"[{utcnow()}] {set_name}: resume_rows={len(done)} -> {out_path.relative_to(REPO)}", flush=True)
        n_new = 0
        started = time.time()
        with out_path.open("a", encoding="utf-8") as fh:
            for item in iter_rows(Y9_DIR / file_name):
                if item["item_id"] in done:
                    continue
                if args.limit and n_new >= args.limit:
                    break
                t1 = time.time()
                request = {"model": "clef_flash", "state": item["state"], "questions": {"q": authors_question(item)}}
                try:
                    response = authors.systemone(model, processor, request)
                except ValueError as exc:
                    if "exceed" not in str(exc) and "limit" not in str(exc):
                        raise
                    row = prediction_row(item, status="skipped", reason=f"authors' limit: {exc}"[:200])
                else:
                    probs = our_probs(item, response["answers"]["q"])
                    if not all(math.isfinite(p) for p in probs):
                        raise RuntimeError(f"non-finite probabilities for {item['item_id']}")
                    row = prediction_row(item, status="scored", probs=probs, latency_s=round(time.time() - t1, 4),
                                         extras={"answer_type": response["answers"]["q"]["type"], "z10_set": set_name})
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                fh.flush()
                n_new += 1
                if n_new % 500 == 0:
                    print(f"[{utcnow()}] {set_name}: {n_new} new rows, {n_new / (time.time() - started):.2f} items/s", flush=True)
        print(f"[{utcnow()}] {set_name}: done, new rows={n_new}", flush=True)
    print(f"[{utcnow()}] all sets done", flush=True)


if __name__ == "__main__":
    main()
