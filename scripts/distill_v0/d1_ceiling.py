"""D1: the hosted ``d1-ultra`` teacher ceiling on the frozen sets (distill_v0).

Sends every eligible frozen-set item to the pinned teacher once, in the item's own option order,
validates every answer, freezes the distributions to JSONL with a SHA-256, books every call in a
cost ledger under the loop's hard cap, and reports per-set accuracy against the frozen gold.

MIMIC-derived rows are **not** sent to the hosted teacher (AGENTS.md data rules; D60: MIMIC only
after the operator confirms the provider's BAA). They are refused by licence/source/set/metadata
markers and counted with a reason in the manifest; nothing is silently omitted.

These labels are an *evaluation* of the teacher on frozen sets. They are never attached to a
training row (GOAL §5 rule 1); the D2 basket builder excludes every frozen item by content hash.

Usage (CPU/network; one job; detached):
    source /workspace/.secrets.env && setsid nohup uv run --frozen python \
        scripts/distill_v0/d1_ceiling.py --out outputs/distill_v0/D1 \
        > outputs/distill_v0/D1/logs/run.log 2>&1 < /dev/null &
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import json
import os
import sys
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any

from meddecide.distill.d1_client import MAX_ATTEMPTS, credits, label_one
from meddecide.distill.d1_teacher import (
    CAP_USD,
    ENDPOINT,
    MODEL,
    PROB_TOLERANCE,
    CostLedger,
    TeacherRequest,
    file_sha256,
    hosted_guard,
    is_correct,
    load_frozen,
    request_for_item,
)
from meddecide.utils.provenance import git_commit, utcnow

REPO = Path(__file__).resolve().parents[2]
Y9 = REPO / "data" / "bench" / "osler_v4" / "y9"
FROZEN_SETS = (
    "knowledge",
    "panel",
    "mr_verdict",
    "mednli_test",
    "mr_reported",
    "mr_heldout_test",
    "mimic_suite",
    "pairs_test",
    "mimic_heldout",
)
MAX_TOTAL_ATTEMPTS = 6  # across resumes: a row that failed 6 times is not retried again
STOP_FILE = REPO / "outputs" / "distill_v0" / "STOP"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_jsonl_lenient(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file that may have a torn last line from a hard kill."""
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                break
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    tmp.replace(path)


def write_json(path: Path, payload: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def plan(limit: int | None, sets: tuple[str, ...],
         allow_protected: bool = False) -> tuple[list[tuple[str, dict[str, Any]]], dict[str, Any]]:
    """Read the frozen sets, apply the hosted guard and request rules, and return the sendable items."""
    sendable: list[tuple[str, dict[str, Any]]] = []
    report: dict[str, Any] = {"sets": {}, "frozen_files": {}}
    for name in sets:
        path = Y9 / f"{name}.jsonl"
        if not path.exists():
            raise SystemExit(f"missing frozen set {path}")
        items = read_jsonl(path)
        report["frozen_files"][name] = {"path": str(path.relative_to(REPO)), "items": len(items),
                                        "sha256": file_sha256(path)}
        refused: Counter[str] = Counter()
        unrequestable: Counter[str] = Counter()
        qtypes: Counter[str] = Counter()
        sent = 0
        for item in items:
            allowed, why = hosted_guard(item, allow_protected=allow_protected)
            if not allowed:
                refused[str(why)] += 1
                continue
            request, reason = request_for_item(item)
            if request is None:
                unrequestable[str(reason)] += 1
                continue
            qtypes[request.qtype] += 1
            if limit is not None and sent >= limit:
                continue
            sendable.append((name, item))
            sent += 1
        report["sets"][name] = {"items": len(items), "sendable": sent, "refused": dict(refused),
                                "unrequestable": dict(unrequestable), "qtypes": dict(qtypes)}
    return sendable, report


def score_sets(sendable: list[tuple[str, dict[str, Any]]],
               labels: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Per-set, per-qtype and per-template accuracy of the teacher's modal key against the frozen gold."""
    per_set: dict[str, Any] = {}
    for name, item in sendable:
        bucket = per_set.setdefault(name, {"items": 0, "labelled": 0, "correct": 0, "qtypes": {},
                                           "templates": {}, "gold": Counter()})
        bucket["items"] += 1
        row = labels.get(str(item["item_id"]))
        if row is None:
            continue
        if list(row["option_keys"]) != [str(option["key"]) for option in item["options"]]:
            raise SystemExit(f"{name} {item['item_id']}: frozen option keys do not match the label row")
        correct = int(is_correct(item, row["teacher_probs"]))
        bucket["labelled"] += 1
        bucket["correct"] += correct
        bucket["gold"][str(item["gold"])] += 1
        qtype = str(item["qtype"])
        q = bucket["qtypes"].setdefault(qtype, {"items": 0, "correct": 0})
        q["items"] += 1
        q["correct"] += correct
        template = str(item.get("template_id") or "")
        t = bucket["templates"].setdefault(template, {"items": 0, "correct": 0})
        t["items"] += 1
        t["correct"] += correct
    out: dict[str, Any] = {}
    for name, bucket in per_set.items():
        labelled = bucket["labelled"]
        templates = bucket["templates"]
        out[name] = {
            "items": bucket["items"],
            "labelled": labelled,
            "correct": bucket["correct"],
            "accuracy_micro": (bucket["correct"] / labelled) if labelled else None,
            "accuracy_template_macro": (sum(t["correct"] / t["items"] for t in templates.values()) / len(templates))
            if templates else None,
            "majority_baseline": (max(bucket["gold"].values()) / labelled) if labelled else None,
            "qtypes": {q: {"items": v["items"], "correct": v["correct"], "accuracy": v["correct"] / v["items"]}
                       for q, v in sorted(bucket["qtypes"].items())},
            "templates": {t: {"items": v["items"], "correct": v["correct"], "accuracy": v["correct"] / v["items"]}
                          for t, v in sorted(templates.items())},
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", default="outputs/distill_v0/D1")
    parser.add_argument("--cap", type=float, default=CAP_USD)
    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument("--limit", type=int, default=None, help="first N sendable items per set (smoke only)")
    parser.add_argument("--sets", default=",".join(FROZEN_SETS))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-protected", action="store_true",
                        help="send MIMIC-derived rows to the hosted teacher (operator authorization 2026-11-03)")
    parser.add_argument("--authorization", default="none",
                        help="the operator authorization recorded in the manifest when --allow-protected is set")
    args = parser.parse_args()
    out = (REPO / args.out) if not Path(args.out).is_absolute() else Path(args.out)
    (out / "logs").mkdir(parents=True, exist_ok=True)
    started = utcnow()
    clock = time.time()
    sets = tuple(part.strip() for part in args.sets.split(",") if part.strip())
    unknown = [name for name in sets if name not in FROZEN_SETS]
    if unknown:
        raise SystemExit(f"unknown frozen sets: {unknown}")

    sendable, report = plan(args.limit, sets, allow_protected=args.allow_protected)
    report["planned_sendable"] = len(sendable)
    report["limit_per_set"] = args.limit
    print(f"planned sendable items: {len(sendable)}", flush=True)
    for name, info in report["sets"].items():
        print(f"  {name}: items={info['items']} sendable={info['sendable']} refused={info['refused']} "
              f"unrequestable={info['unrequestable']}", flush=True)
    if args.dry_run:
        write_json(out / "plan.json", report)
        print("dry run: plan written", flush=True)
        return

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY is not set (source /workspace/.secrets.env)")
    balance = credits(api_key)
    print(f"balance: {balance}", flush=True)

    labels_path = out / "labels.jsonl"
    failures_path = out / "failures.jsonl"
    existing = load_frozen(labels_path) if args.resume else {}
    existing_failures = {str(row["item_id"]): row for row in read_jsonl_lenient(failures_path)} if args.resume else {}
    if args.resume:  # a killed run's incremental file is merged, never discarded
        for row in read_jsonl_lenient(out / "labels.partial.jsonl"):
            existing.setdefault(str(row["item_id"]), row)
    todo: list[tuple[str, dict[str, Any], TeacherRequest]] = []
    for name, item in sendable:
        item_id = str(item["item_id"])
        if item_id in existing:
            continue
        failures = existing_failures.get(item_id)
        if failures is not None and int(failures.get("attempts") or 0) >= MAX_TOTAL_ATTEMPTS:
            continue
        request, reason = request_for_item(item)
        if request is None:  # already counted in the plan; cannot happen
            raise SystemExit(f"{item_id} became unrequestable: {reason}")
        todo.append((name, item, request))
    print(f"to label: {len(todo)} (already frozen: {len(existing)})", flush=True)

    remaining = float(balance.get("remaining") or args.cap) if balance and "remaining" in balance else args.cap
    ledger = CostLedger(cap_usd=min(args.cap, remaining))
    if existing:  # a resumed run starts from the spend its frozen labels already booked
        ledger.seed(sum(float(row["cost"]) for row in existing.values()),
                    sum(int(row["input_tokens"]) for row in existing.values()), len(existing))
    fatal = threading.Event()
    new_labels: list[dict[str, Any]] = []
    new_failures: list[dict[str, Any]] = []
    stop_reason = "complete"
    chunk = max(args.threads * 2, 16)
    partial_path = out / "labels.partial.jsonl"

    def work(job: tuple[str, dict[str, Any], TeacherRequest]) -> tuple[str, TeacherRequest,
                                                                       dict[str, Any] | None,
                                                                       dict[str, Any] | None]:
        request = job[2]
        row, failure = label_one(api_key, request, ledger, fatal)
        return job[0], request, row, failure

    with futures.ThreadPoolExecutor(max_workers=args.threads) as pool, partial_path.open("w", encoding="utf-8") as handle:
        for base in range(0, len(todo), chunk):
            if STOP_FILE.exists():
                stop_reason = "stop_file"
                break
            if ledger.exhausted:
                stop_reason = "cap_reached"
                break
            for _name, _request, row, failure in pool.map(work, todo[base:base + chunk]):
                if row is not None:
                    new_labels.append(row)
                    handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                elif failure is not None:
                    new_failures.append(failure)
            handle.flush()
            if (base // chunk) % 20 == 0:
                print(f"progress: labelled={len(new_labels)} failed={len(new_failures)} calls={ledger.calls} "
                      f"spent=${ledger.spent_usd:.4f}", flush=True)
            if fatal.is_set():
                stop_reason = "auth_or_quota"
                break
    elapsed = round(time.time() - clock, 1)
    print(f"calls done: labelled={len(new_labels)} failed={len(new_failures)} spent=${ledger.spent_usd:.4f} "
          f"stop={stop_reason} elapsed={elapsed}s", flush=True)

    all_labels = {**existing, **{str(row["item_id"]): row for row in new_labels}}
    all_failures = {**existing_failures, **{str(row["item_id"]): row for row in new_failures}}
    write_jsonl(labels_path, [all_labels[key] for key in sorted(all_labels)])
    write_jsonl(failures_path, [all_failures[key] for key in sorted(all_failures)])
    if partial_path.exists():
        partial_path.unlink()

    ceiling = score_sets(sendable, all_labels)
    failure_reasons = Counter(str(row.get("reason")) for row in all_failures.values())
    manifest = {
        "task": "D1",
        "purpose": "teacher ceiling on frozen evaluation sets; these labels are NEVER attached to a training row",
        "model": MODEL,
        "endpoint": ENDPOINT,
        "provider": "Liquid (OpenRouter)",
        "git_commit": git_commit(REPO),
        "command": " ".join(sys.argv),
        "started_utc": started,
        "finished_utc": utcnow(),
        "wall_clock_seconds": elapsed,
        "prob_tolerance": PROB_TOLERANCE,
        "max_attempts": MAX_ATTEMPTS,
        "cap_usd": ledger.cap_usd,
        "allow_protected": bool(args.allow_protected),
        "operator_authorization": args.authorization if args.allow_protected else None,
        "cap_note": "calls are submitted in chunks; the cap can be overshot by at most one chunk of in-flight calls",
        "ledger": ledger.to_dict(),
        "balance_before": balance,
        "stop_reason": stop_reason,
        "counts": {
            "planned_sendable": len(sendable),
            "frozen_rows": len(all_labels),
            "failed_rows": len(all_failures),
            "failure_reasons": dict(sorted(failure_reasons.items())),
            "attempted_this_run": len(new_labels) + len(new_failures),
        },
        "sets": report["sets"],
        "frozen_files": report["frozen_files"],
        "labels_sha256": file_sha256(labels_path),
        "failures_sha256": file_sha256(failures_path),
        "ceiling": ceiling,
        "environment": {"hostname": os.uname().nodename, "python": sys.version.split()[0]},
    }
    write_json(out / "ceiling.json", ceiling)
    write_json(out / "manifest.json", manifest)
    summary = {"ledger": manifest["ledger"], "counts": manifest["counts"],
               "ceiling": {name: {"labelled": v["labelled"], "micro": v["accuracy_micro"],
                                  "macro": v["accuracy_template_macro"], "majority": v["majority_baseline"]}
                           for name, v in ceiling.items()}}
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
