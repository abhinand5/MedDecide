"""V4: teacher-written prose for the medical-record decisions (osler_v1 ADVISORY V4; D27, D30).

Two generators: ``mr_icd_support_prose_v1`` (ICD-10-CM support, one base and twin per patient id) and
``mr_indicator_prose_v1`` (clinical-indicator evidence, one base and twin per patient id). The per-patient work lives in
``meddecide.mr.prose_build``; this script only schedules it, enforces the guards and writes the results.

Guards (ADVISORY V4 and §3.8): at most 8 workers and 2 requests per second in total; HTTP 429 and 5xx retried with
exponential backoff; HTTP 401 or 403 halts the run; the spend ledger is checked before every submission (5-hour window $14,
week $36, loop $60); bulk requests pause during the teacher's peak hours; a streak of 25 failed requests halts the run.
Spend is recorded per patient id as a USD cost only. Teacher text goes only to data/gen (gitignored).

Modes:
    --mode pilot --gen icd --n 100                    first 100 pilot ids; stats to outputs/osler_v1/V4/pilot_<gen>.json
    --mode build --gen icd --start 0 --count 15000    resumable; verified items appended to data/gen/osler_v1/teacher/

Usage:
    source scripts/pod_env.sh && uv run --frozen python scripts/osler_v1/v4_teacher_build.py --mode pilot --gen icd
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from collections import Counter
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from functools import partial
from pathlib import Path
from typing import Any

from meddecide.mr.narrative import load_encounters, sample_encounters
from meddecide.mr.prose_build import PROCESS, PROSE_PREFIX, Outcome, process_narrative
from meddecide.teacher.budget import RateLimiter, SpendLedger
from meddecide.teacher.client import TeacherClient, is_peak
from meddecide.teacher.guarded import GuardedChat, Halt
from meddecide.utils.io import write_json
from meddecide.utils.provenance import git_commit, utcnow

REPO = Path(__file__).resolve().parents[2]
TEACHER_DIR = REPO / "data" / "gen" / "osler_v1" / "teacher"
V4_DIR = REPO / "outputs" / "osler_v1" / "V4"
LEDGER = V4_DIR / "spend_ledger.jsonl"
WORKERS = 8
REQUESTS_PER_S = 2.0
INFLIGHT = 16
CONSECUTIVE_ERRORS_TO_HALT = 25
FAILED_REASONS = ("write failed", "extract failed", "RuntimeError", "HTTPStatusError")


class Worker:
    """One thread's teacher client, so the usage meter is never shared between threads."""

    def __init__(self, limiter: RateLimiter, halt: Halt, process_fn: Callable[[str, Any], Outcome]) -> None:
        self.limiter = limiter
        self.halt = halt
        self.process_fn = process_fn
        self._local = threading.local()

    def guarded(self) -> tuple[TeacherClient, GuardedChat]:
        if not hasattr(self._local, "guarded"):
            teacher = TeacherClient.from_env()
            self._local.teacher = teacher
            self._local.guarded = GuardedChat(teacher, self.limiter, self.halt)
        return self._local.teacher, self._local.guarded

    def run(self, pid: str) -> Outcome:
        teacher, guarded = self.guarded()
        before = teacher.usage.as_dict()
        try:
            outcome = self.process_fn(pid, guarded)
        except Exception as exc:
            outcome = Outcome(pid, "error", [type(exc).__name__])
        after = teacher.usage.as_dict()
        outcome.calls = after["calls"] - before["calls"]
        outcome.prompt_tokens = after["prompt_tokens"] - before["prompt_tokens"]
        outcome.cached_tokens = after["cached_tokens"] - before["cached_tokens"]
        outcome.completion_tokens = after["completion_tokens"] - before["completion_tokens"]
        outcome.cost_usd = round(after["cost_usd"] - before["cost_usd"], 8)
        return outcome


def load_done(progress: Path) -> set[str]:
    if not progress.exists():
        return set()
    with progress.open(encoding="utf-8") as fh:
        return {json.loads(line)["pid"] for line in fh}


def run(gen: str, pids: list[str], out_items: Path, progress: Path, *, workers: int, rps: float,
        process_fn: Callable[[str, Any], Outcome]) -> dict[str, Any]:
    ledger = SpendLedger(LEDGER)
    halt = Halt()
    worker = Worker(RateLimiter(rps), halt, process_fn)
    verdicts: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    totals: Counter[str] = Counter()
    consecutive_errors = 0
    stop_reason = ""
    out_items.parent.mkdir(parents=True, exist_ok=True)
    progress.parent.mkdir(parents=True, exist_ok=True)

    def handle(outcome: Outcome) -> None:
        nonlocal consecutive_errors
        verdicts[outcome.verdict] += 1
        reasons.update(outcome.reasons)
        for key in ("calls", "prompt_tokens", "cached_tokens", "completion_tokens"):
            totals[key] += getattr(outcome, key)
        totals["cost_usd_micro"] += round(outcome.cost_usd * 1e6)
        if outcome.verdict == "accepted":
            with out_items.open("a", encoding="utf-8") as fh:
                for item in outcome.items:
                    fh.write(json.dumps(item.as_dict(), ensure_ascii=False) + "\n")
        ledger.record(outcome.cost_usd, label=f"{gen}:{outcome.verdict}")
        with progress.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"pid": outcome.pid, "verdict": outcome.verdict, "reasons": outcome.reasons,
                                 "calls": outcome.calls, "cost_usd": outcome.cost_usd}) + "\n")
        failed = any(r.startswith(FAILED_REASONS) for r in outcome.reasons)
        consecutive_errors = consecutive_errors + 1 if failed else 0
        n = sum(verdicts.values())
        if n % 100 == 0:
            print(f"{utcnow()} {gen}: {n} patient ids, {dict(verdicts)}, "
                  f"cost ${totals['cost_usd_micro'] / 1e6:.4f}", flush=True)

    inflight: dict[Future[Outcome], str] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for pid in pids:
            if halt.is_set():
                stop_reason = halt.reason
                break
            if consecutive_errors >= CONSECUTIVE_ERRORS_TO_HALT:
                halt.set(f"{consecutive_errors} failed requests in a row")
                stop_reason = halt.reason
                break
            while is_peak():
                time.sleep(60)  # bulk generation pauses in the teacher's peak hours (ADVISORY V4)
            check = ledger.check()
            if not check.ok:
                stop_reason = f"spend guard: {check.reason}"
                break
            while len(inflight) >= INFLIGHT:
                done, _ = wait(inflight, return_when=FIRST_COMPLETED)
                for future in done:
                    inflight.pop(future)
                    handle(future.result())
            inflight[pool.submit(worker.run, pid)] = pid
        for future in list(inflight):
            handle(future.result())
            inflight.pop(future)
    spend = ledger.check()
    return {
        "generator": gen,
        "patient_ids_requested": len(pids),
        "verdicts": dict(verdicts),
        "reasons": dict(reasons),
        "calls": totals["calls"],
        "prompt_tokens": totals["prompt_tokens"],
        "cached_tokens": totals["cached_tokens"],
        "completion_tokens": totals["completion_tokens"],
        "cost_usd": round(totals["cost_usd_micro"] / 1e6, 6),
        "ledger_window_usd": round(spend.window_usd, 6),
        "ledger_week_usd": round(spend.week_usd, 6),
        "ledger_loop_usd": round(spend.loop_usd, 6),
        "stop_reason": stop_reason or "completed",
        "workers": workers,
        "requests_per_s": rps,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=("pilot", "build"), required=True)
    parser.add_argument("--gen", choices=sorted(PROSE_PREFIX), required=True)
    parser.add_argument("--n", type=int, default=100, help="pilot: patient ids")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=0)
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--rps", type=float, default=REQUESTS_PER_S)
    args = parser.parse_args()
    prefix = PROSE_PREFIX[args.gen]
    process_fn: Callable[[str, Any], Outcome] = PROCESS.get(args.gen, PROCESS.get("icd"))
    if args.gen == "narrative":
        # one page per Synthea encounter (V5a extract); the pid carries the encounter id after the colon
        store = load_encounters(REPO / "data" / "synthea" / "osler_v1")
        ids = (sample_encounters(store, args.n, seed=0) if args.mode == "pilot"
               else sorted(store)[args.start: args.start + args.count])
        pids = [f"synthea-narr:{eid}" for eid in ids]
        process_fn = partial(process_narrative, store=store)
        print(f"{utcnow()} encounters loaded: {len(store)}; this run: {len(pids)}", flush=True)
    elif args.mode == "pilot":
        pids = [f"{prefix}-pilot-{k:07d}" for k in range(args.n)]
    else:
        if args.count < 1:
            raise SystemExit("--count must be at least 1 in build mode")
        pids = [f"{prefix}-{k:07d}" for k in range(args.start, args.start + args.count)]
    if args.mode == "pilot":
        out_items = TEACHER_DIR / f"{args.gen}_pilot.jsonl"
        progress = V4_DIR / f"{args.gen}_pilot_progress.jsonl"
    else:
        if args.count < 1:
            raise SystemExit("--count must be at least 1 in build mode")
        out_items = TEACHER_DIR / f"{args.gen}.jsonl"
        progress = V4_DIR / f"{args.gen}_progress.jsonl"
    done = load_done(progress)
    todo = [p for p in pids if p not in done]
    print(f"{utcnow()} {args.mode} {args.gen}: {len(todo)} to do ({len(done)} already done)", flush=True)
    started = time.time()
    summary = run(args.gen, todo, out_items, progress, workers=args.workers, rps=args.rps, process_fn=process_fn)
    summary.update({"mode": args.mode, "git_commit": git_commit(REPO), "written_utc": utcnow(),
                    "wall_clock_s": round(time.time() - started, 1), "pids_already_done": len(done),
                    "items_file": str(out_items.relative_to(REPO))})
    stats_path = V4_DIR / (f"pilot_{args.gen}.json" if args.mode == "pilot" else f"build_{args.gen}_stats.json")
    write_json(stats_path, summary)
    print(json.dumps({k: summary[k] for k in ("verdicts", "cost_usd", "stop_reason", "wall_clock_s")}), flush=True)
    sys.exit(0)


if __name__ == "__main__":
    main()
