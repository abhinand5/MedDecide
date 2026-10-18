"""osler_v3 X3: generate claim-grounding and restyled-page candidates with the local Gemma (GPU; ADVISORY X3 steps 1-2; D43).

One candidate per substrate document (a stable order by document key), so the first N documents are the same in every run.
For each candidate the code draws the label (or the page class) first; the local Gemma writes; an independent verification call
(a different system prompt, temperature 0) classifies. Every candidate is written with its verdict, its keep flag and the reason
(every drop has one). The run is resumable: documents already in the output file are skipped.

The endpoint must be local (TEACHER_LOCAL=1 and a 127.0.0.1 base URL); a hosted endpoint is refused (D30, D43).

Usage (after the Gemma server of scripts/osler_v3 X3 is up on 127.0.0.1:8000; detached, one GPU job at a time):
    source scripts/pod_env.sh && TEACHER_BASE_URL=http://127.0.0.1:8000/v1 TEACHER_MODEL=google/gemma-4-31B-it-qat-w4a16-ct \
      TEACHER_LOCAL=1 TEACHER_THINKING=none TEACHER_EXTRA_BODY='{"chat_template_kwargs": {"enable_thinking": false}}' \
      setsid nohup uv run --frozen python scripts/osler_v3/x3_generate.py --task claims --split train --limit 300 \
      > outputs/osler_v3/X3/logs/claims_train_pilot.log 2>&1 < /dev/null &
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from meddecide.mix import clinclaim as cc
from meddecide.mix import restyle as rs
from meddecide.mix import substrate as sub
from meddecide.teacher.client import TeacherClient
from meddecide.utils.hashing import stable_hash

REPO = Path(__file__).resolve().parents[2]
SUBSTRATE = REPO / "data" / "gen" / "osler_v3" / "x3_substrate.jsonl"
GEN = REPO / "data" / "gen" / "osler_v3"
RESTYLE_SOURCE_CHARS = 6000  # the case text the restyle writer sees (the first characters of the document)
RETRIES = 2


def load_docs(split: str) -> list[dict[str, Any]]:
    docs = []
    with SUBSTRATE.open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            if row["split"] == split:
                docs.append(row)
    # a stable order: the same first N documents in every run and in the pilot
    return sorted(docs, key=lambda r: stable_hash({"order": r["key"]}, length=16))


def done_keys(path: Path) -> set[str]:
    """The documents already written with a result; an errored document is not done, so a resumed run retries it."""
    if not path.exists():
        return set()
    keys = set()
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                record = json.loads(line)
                if record.get("reason") != "error":
                    keys.add(record["key"])
    return keys


class Writer:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.lock = threading.Lock()
        path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, record: dict[str, Any]) -> None:
        with self.lock, self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def chat(client: TeacherClient, messages: list[dict[str, str]], *, temperature: float, max_tokens: int) -> str:
    last: Exception | None = None
    for attempt in range(RETRIES + 1):
        try:
            return client.chat(messages, temperature=temperature, max_tokens=max_tokens)
        except Exception as exc:  # the server may time out under load; the document is retried, then recorded as an error
            last = exc
            time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"chat failed after {RETRIES + 1} attempts: {last}")


def claim_record(client: TeacherClient, doc: dict[str, Any], seed: int) -> dict[str, Any]:
    passage = sub.passage(doc["text"], doc["key"])
    base = {"template": cc.TEMPLATE, "key": doc["key"], "source": doc["source"], "split": doc["split"],
            "doc_id": doc["doc_id"], "licence": doc["licence"], "record_url": doc["record_url"], "seed": seed}
    if passage is None:
        return {**base, "kept": False, "reason": "no_passage", "passage": None}
    label, kind = cc.draw(doc["key"], 0, seed, passage)
    writer_reply = chat(client, cc.writer_messages(passage, label, kind), temperature=0.7, max_tokens=120)
    claim = cc.clean_claim(writer_reply)
    verifier_reply = chat(client, cc.verifier_messages(passage, claim), temperature=0.0, max_tokens=8) if claim else ""
    verdict = cc.parse_verdict(verifier_reply)
    kept, reason = cc.keep(label, kind, claim, passage, verdict)
    return {**base, "label": label, "kind": kind, "passage": passage, "claim": claim, "verifier_reply": verifier_reply.strip(),
            "verdict": verdict, "kept": kept, "reason": reason,
            "copy_share": round(cc.copy_share(claim, passage), 4) if claim else None}


def restyle_record(client: TeacherClient, doc: dict[str, Any], seed: int) -> dict[str, Any]:
    case = str(doc["text"])[:RESTYLE_SOURCE_CHARS]
    page_class = rs.draw_class(doc["key"], 0, seed)
    base = {"template": rs.TEMPLATE, "key": doc["key"], "source": doc["source"], "split": doc["split"],
            "doc_id": doc["doc_id"], "licence": doc["licence"], "record_url": doc["record_url"], "seed": seed,
            "page_class": page_class}
    writer_reply = chat(client, rs.writer_messages(case, page_class), temperature=0.7, max_tokens=900)
    page = rs.clean_page(writer_reply)
    verifier_reply = chat(client, rs.classifier_messages(page), temperature=0.0, max_tokens=4) if page else ""
    letter = rs.parse_letter(verifier_reply)
    kept, reason = rs.keep(page_class, page, letter)
    return {**base, "page": page, "verifier_reply": verifier_reply.strip(), "verified_letter": letter, "kept": kept,
            "reason": reason, "words": len(page.split())}


def run(task: str, split: str, limit: int, concurrency: int, seed: int, out: Path) -> dict[str, int]:
    client = TeacherClient.from_env()
    if not client.local or "127.0.0.1" not in client.base_url:
        raise SystemExit("refusing to run: the X3 generator needs the local Gemma endpoint (TEACHER_LOCAL=1, 127.0.0.1)")
    docs = load_docs(split)[:limit]
    skip = done_keys(out)
    todo = [d for d in docs if d["key"] not in skip]
    writer = Writer(out)
    make = claim_record if task == "claims" else restyle_record
    counts: dict[str, int] = {"documents": len(docs), "already_done": len(docs) - len(todo), "written": 0, "errors": 0}
    started = time.time()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(make, client, doc, seed): doc for doc in todo}
        for future in as_completed(futures):
            doc = futures[future]
            try:
                record = future.result()
            except Exception as exc:  # recorded, never silently dropped
                record = {"template": cc.TEMPLATE if task == "claims" else rs.TEMPLATE, "key": doc["key"], "source": doc["source"],
                          "split": doc["split"], "doc_id": doc["doc_id"], "licence": doc["licence"],
                          "record_url": doc["record_url"], "seed": seed, "kept": False, "reason": "error",
                          "error": str(exc)[:300]}
                counts["errors"] += 1
            writer.write(record)
            counts["written"] += 1
            if counts["written"] % 100 == 0:
                rate = counts["written"] / max(1e-9, time.time() - started)
                print(f"{task}/{split}: {counts['written']} of {len(todo)} written, {rate:.2f} documents/s, "
                      f"errors {counts['errors']}", flush=True)
    counts["seconds"] = round(time.time() - started, 1)
    print(f"{task}/{split}: done {json.dumps(counts)}", flush=True)
    return counts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--task", required=True, choices=["claims", "restyle"])
    ap.add_argument("--split", required=True, choices=["train", "dev", "test"])
    ap.add_argument("--limit", type=int, required=True)
    ap.add_argument("--concurrency", type=int, default=48)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    name = {"claims": "x3_claims", "restyle": "x3_restyle"}[args.task]
    out = args.out or GEN / f"{name}_{args.split}.jsonl"
    run(args.task, args.split, args.limit, args.concurrency, args.seed, out)


if __name__ == "__main__":
    sys.exit(main())
