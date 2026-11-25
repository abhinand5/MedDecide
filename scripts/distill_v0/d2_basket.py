"""D2: the teacher basket, the hosted ``d1-ultra`` label run, QA and the frozen label file (distill_v0).

Steps, in order:
1. compose osler_v4's final mix in memory (Y8 composer; the original composition hash is checked),
   drop ``fda_route_choice_v1`` whole (D56a; the corrected hash is checked), then drop the literal
   rule-lookup templates (FDA field lookups and the PubMed publication-type/check family) with counts
   and reasons;
2. select the pre-registered teacher basket T (GOAL §3.3) — the nine CT templates capped at 10,000
   rows each by stable-hash order;
3. check every T row against every frozen set by item id and by normalised text key (a hit is dropped
   with reason ``frozen_overlap``, never labelled);
4. label every sendable T row once with the pinned teacher, freeze the distributions to JSONL, book
   every call in the loop cost ledger under the $15 hard cap;
5. QA every label (schema, option order, probability sums, coverage, frozen/MIMIC exclusion) and write
   the label manifest with the SHA-256.

The training basket is keyed by the normalised **text key**, not the item id: Winogrande twins share an
item id while their state text differs (5,232 ids / 22,108 extra rows in the base mix), so an
id-keyed label file would collapse distinct rows. ``text_key`` is therefore an added field on the
D2 freeze rows (D1's ceiling labels keep the eight-field schema where the id is unique).

``--plan-only`` stops after step 3 and writes ``composition.json``. Interrupted runs resume from the
incremental label file; the partial ledger is booked as a resumed attempt so the cap stays loop-wide.

Usage (CPU/network; one job; detached):
    source /workspace/.secrets.env && setsid nohup uv run --frozen python \
        scripts/distill_v0/d2_basket.py --out outputs/distill_v0/D2 \
        > outputs/distill_v0/D2/logs/run.log 2>&1 < /dev/null &
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import json
import math
import os
import sys
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any

from meddecide.distill.d1_client import credits, label_one
from meddecide.distill.d1_teacher import (
    CAP_USD,
    MODEL,
    CostLedger,
    file_sha256,
    hosted_guard,
    request_for_item,
)
from meddecide.distill.loop_ledger import append_entry, prior_cost, read_entries
from meddecide.panel.overlap import row_text_key
from meddecide.train.distill_mix import (
    POOL_ROWS_EXPECTED,
    TEACHER_ROWS_EXPECTED,
    basket_shares,
    estimate_tokens,
    select_teacher_rows,
)
from meddecide.train.distill_pool import compose_pool
from meddecide.utils.provenance import git_commit, utcnow

REPO = Path(__file__).resolve().parents[2]
LOOP_LEDGER = REPO / "outputs" / "distill_v0" / "cost_ledger.jsonl"
STOP_FILE = REPO / "outputs" / "distill_v0" / "STOP"
MAX_TOTAL_ATTEMPTS = 6
MAX_CONSECUTIVE_FAILURES = 5  # ADVISORY §1.3: 5 consecutive failures within 10 minutes stops the labeler
LABEL_FIELDS = {"item_id", "template_id", "qtype", "option_keys", "teacher_probs", "teacher", "cost",
                "input_tokens", "text_key"}
FROZEN_FILES = (
    "data/bench/osler_v4/y9/knowledge.jsonl",
    "data/bench/osler_v4/y9/panel.jsonl",
    "data/bench/osler_v4/y9/mr_verdict.jsonl",
    "data/bench/osler_v4/y9/mednli_test.jsonl",
    "data/bench/osler_v4/y9/mr_reported.jsonl",
    "data/bench/osler_v4/y9/mr_heldout_test.jsonl",
    "data/bench/osler_v4/y9/mimic_suite.jsonl",
    "data/bench/osler_v4/y9/pairs_test.jsonl",
    "data/bench/osler_v4/y9/mimic_heldout.jsonl",
    "data/bench/osler_mimic_v1/mimic_audit_suite_v1.jsonl",
    "data/bench/osler_mimic_v1/mimic_heldout_v1.jsonl",
    "data/bench/osler_v2/knowledge_dev_v2.jsonl",
    "data/bench/osler_v4/fastdev_v4.jsonl",
    "data/bench/osler_v4/cg_v2.jsonl",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_jsonl_lenient(path: Path) -> list[dict[str, Any]]:
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


def write_json(path: Path, payload: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    tmp.replace(path)


def frozen_index() -> tuple[set[str], set[str], dict[str, Any]]:
    """Item ids and normalised text keys of every frozen set, with per-file counts and hashes."""
    ids: set[str] = set()
    texts: set[str] = set()
    files: dict[str, Any] = {}
    for relative in FROZEN_FILES:
        path = REPO / relative
        if not path.exists():
            raise SystemExit(f"missing frozen file {relative}")
        rows = read_jsonl(path)
        for row in rows:
            ids.add(str(row["item_id"]))
            texts.add(row_text_key(row))
        files[relative] = {"items": len(rows), "sha256": file_sha256(path)}
    return ids, texts, {"files": files, "item_ids": len(ids), "text_keys": len(texts)}


def overlap_check(rows: list[dict[str, Any]], keys: list[str], ids: set[str],
                  texts: set[str]) -> tuple[list[dict[str, Any]], list[str], dict[str, Any]]:
    """Drop every row whose item id or normalised text key is in a frozen set (reason ``frozen_overlap``)."""
    kept_rows: list[dict[str, Any]] = []
    kept_keys: list[str] = []
    reasons: Counter[str] = Counter()
    for row, key in zip(rows, keys, strict=True):
        hit_id = str(row["item_id"]) in ids
        hit_text = key in texts
        if hit_id or hit_text:
            reasons["item_id and text" if hit_id and hit_text else ("item_id" if hit_id else "text")] += 1
            continue
        kept_rows.append(row)
        kept_keys.append(key)
    report = {"reason": "frozen_overlap", "scanned": len(rows), "dropped": len(rows) - len(kept_rows),
              "dropped_by_reason": dict(sorted(reasons.items())), "kept": len(kept_rows)}
    return kept_rows, kept_keys, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", default="outputs/distill_v0/D2")
    parser.add_argument("--cap", type=float, default=CAP_USD)
    parser.add_argument("--threads", type=int, default=48)
    parser.add_argument("--limit", type=int, default=None, help="first N teacher rows (smoke only)")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-protected", action="store_true",
                        help="send MIMIC-derived rows (MedNLI and any other credentialed source) to the hosted teacher "
                             "(operator authorization 2026-11-03)")
    parser.add_argument("--authorization", default="none",
                        help="the operator authorization recorded in the plan and label manifest")
    args = parser.parse_args()
    out = (REPO / args.out) if not Path(args.out).is_absolute() else Path(args.out)
    (out / "logs").mkdir(parents=True, exist_ok=True)
    started = utcnow()
    clock = time.time()
    command = " ".join(sys.argv)

    print("composing the pool ...", flush=True)
    pool, pool_keys, composition = compose_pool()
    print(f"pool rows: {composition['pool_rows']} (expected {POOL_ROWS_EXPECTED}); "
          f"text sha {composition['pool_text_sha256'][:16]}...", flush=True)
    print(f"rule-lookup drop: {composition['rule_lookup_drop']['dropped_total']} rows, "
          f"matches_expected={composition['rule_lookup_drop']['matches_expected']}; "
          f"duplicate item ids: {composition['item_id_duplicates']['ids']} "
          f"({composition['item_id_duplicates']['rows']} rows)", flush=True)

    teacher_indices, teacher_report = select_teacher_rows(pool)
    teacher_rows = [pool[index] for index in teacher_indices]
    teacher_keys = [pool_keys[index] for index in teacher_indices]
    print(f"teacher basket: {len(teacher_rows)} rows (expected {TEACHER_ROWS_EXPECTED}); "
          f"matches_expected={teacher_report['matches_expected']}", flush=True)
    groups_by_key: dict[str, list[dict[str, Any]]] = {}
    for key, row in zip(teacher_keys, teacher_rows, strict=True):
        groups_by_key.setdefault(key, []).append(row)
    unique_keys = list(groups_by_key)
    unique_rows = [groups_by_key[key][0] for key in unique_keys]
    multiplicity = Counter(len(members) for members in groups_by_key.values())
    print(f"teacher unique text keys: {len(unique_keys)} (multiplicity histogram rows/key: {dict(sorted(multiplicity.items()))})",
          flush=True)
    ids, texts, frozen = frozen_index()
    unique_rows, unique_keys, overlap = overlap_check(unique_rows, unique_keys, ids, texts)
    print(f"frozen overlap: dropped {overlap['dropped']} of {overlap['scanned']} unique keys", flush=True)
    shares = basket_shares(pool, set(unique_keys))
    estimate = estimate_tokens(unique_rows)
    if args.limit is not None:
        unique_rows = unique_rows[: args.limit]
        unique_keys = unique_keys[: args.limit]
    pre_registered = {
        "teacher_rows": TEACHER_ROWS_EXPECTED,
        "realised_teacher_rows": teacher_report["selected_total"],
        "delta": teacher_report["selected_total"] - TEACHER_ROWS_EXPECTED,
        "cause": "the GOAL §3.3 per-template counts are stale for nine templates: the composed mix holds fewer rows "
                 "than the pre-registration lists (the two small CT templates have 6,318 and 4,237 rows, below the "
                 "10,000 cap; seven broad replay templates are smaller than listed). Every available row of each "
                 "listed template is in the basket; the shortfall is not topped up from other families.",
        "per_template_shortfall": {template: info["expected"] - info["selected"]
                                   for template, info in teacher_report["per_template"].items()
                                   if info["selected"] < info["expected"]},
        "shares_after": {"teacher": teacher_report["selected_total"] / len(pool)},
        "dose_band_ok": 0.38 <= teacher_report["selected_total"] / len(pool) <= 0.45,
    }
    plan = {
        "task": "D2 plan",
        "basket_deviation": pre_registered,
        "git_commit": git_commit(REPO),
        "command": command,
        "started_utc": started,
        "model": MODEL,
        "allow_protected": bool(args.allow_protected),
        "operator_authorization": args.authorization if args.allow_protected else None,
        "pool": composition,
        "teacher_basket": {**teacher_report, "unique_text_keys": len(unique_keys),
                           "rows_after_overlap": len(unique_rows), "limit": args.limit,
                           "multiplicity": {str(k): v for k, v in sorted(multiplicity.items())}},
        "overlap": overlap,
        "frozen": frozen,
        "shares": shares,
        "gold_only_rows": len(pool) - len(teacher_rows),
        "estimate": estimate,
        "checks": {
            "pool_matches_expected": composition["pool_matches_expected"],
            "rule_lookup_matches_expected": composition["rule_lookup_drop"]["matches_expected"],
            "basket_matches_pre_registration": teacher_report["counts_match_expected"],
            "basket_inside_dose_band": pre_registered["dose_band_ok"],
            "no_frozen_overlap": overlap["dropped"] == 0,
            "text_duplicates_recorded": True,
            "shares_in_band": (shares["teacher_share"] is not None and 0.38 <= shares["teacher_share"] <= 0.45
                               and shares["mimic_share"] is not None and 0.37 <= shares["mimic_share"] <= 0.42),
        },
        "counts": {"pool": len(pool), "teacher": len(teacher_rows), "teacher_unique_keys": len(unique_keys),
                   "gold_only": len(pool) - len(teacher_rows)},
    }
    write_json(out / "composition.json", plan)
    print(json.dumps({k: plan[k] for k in ("counts", "shares", "estimate", "checks")}, indent=2), flush=True)
    if args.plan_only:
        print(f"plan only: wrote {out / 'composition.json'}", flush=True)
        return

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY is not set (source /workspace/.secrets.env)")
    skipped: Counter[str] = Counter()
    skipped_rows: list[dict[str, Any]] = []
    sendable_rows: list[dict[str, Any]] = []
    sendable_keys: list[str] = []
    for row, key in zip(unique_rows, unique_keys, strict=True):
        allowed, why = hosted_guard(row, allow_protected=args.allow_protected)
        if not allowed:
            skipped[str(why)] += 1
            skipped_rows.append({"item_id": str(row["item_id"]), "text_key": key,
                                 "template_id": str(row["template_id"]), "reason": str(why)})
            continue
        request, reason = request_for_item(row)
        if request is None:
            skipped[str(reason)] += 1
            skipped_rows.append({"item_id": str(row["item_id"]), "text_key": key,
                                 "template_id": str(row["template_id"]), "reason": str(reason)})
            continue
        sendable_rows.append(row)
        sendable_keys.append(key)
    write_jsonl(out / "skipped.jsonl", sorted(skipped_rows, key=lambda row: row["text_key"]))
    print(f"sendable: {len(sendable_rows)} skipped: {dict(skipped)}", flush=True)

    labels_path = out / "labels.jsonl"
    failures_path = out / "failures.jsonl"
    partial_path = out / "labels.partial.jsonl"
    ledger_partial_path = out / "ledger.partial.json"
    entries = read_entries(LOOP_LEDGER)
    prior = prior_cost(entries)
    partial_mode = "w"
    if args.resume:
        existing = {str(row["text_key"]): row for row in read_jsonl_lenient(labels_path)}
        previous_partial = read_jsonl_lenient(partial_path)
        for row in previous_partial:
            existing.setdefault(str(row["text_key"]), row)
        if previous_partial:  # preserve every row of an interrupted attempt; the run appends
            write_jsonl(partial_path, sorted(previous_partial, key=lambda row: str(row["text_key"])))
            partial_mode = "a"
        existing_failures = {str(row["text_key"]): row for row in read_jsonl_lenient(failures_path)}
        if ledger_partial_path.exists():
            partial_ledger = json.loads(ledger_partial_path.read_text(encoding="utf-8"))
            resumed_spend = float(partial_ledger.get("this_task_spend_usd") or 0.0)
            if resumed_spend > 0:
                append_entry(LOOP_LEDGER, {"task": "D2", "attempt": "resumed-away", "cost_usd": resumed_spend,
                                           "calls": int(partial_ledger.get("calls") or 0),
                                           "input_tokens": int(partial_ledger.get("input_tokens") or 0),
                                           "note": "spend of an interrupted attempt, booked on resume"})
                prior += resumed_spend
    else:
        existing = {}
        existing_failures = {}
    balance = credits(api_key)
    print(f"balance: {balance}; loop spend before this run: ${prior:.4f}", flush=True)
    ledger = CostLedger(cap_usd=min(args.cap, prior + float(balance.get("remaining") or args.cap)))
    ledger.spent_usd = prior  # the ledger is loop-cumulative: the cap covers every earlier task
    todo = [(key, row) for key, row in zip(sendable_keys, sendable_rows, strict=True)
            if key not in existing
            and int((existing_failures.get(key) or {}).get("attempts") or 0) < MAX_TOTAL_ATTEMPTS]
    print(f"to label: {len(todo)} (already frozen: {len(existing)})", flush=True)

    fatal = threading.Event()
    stop_reason = "complete"
    new_labels: list[dict[str, Any]] = []
    new_failures: list[dict[str, Any]] = []
    consecutive_failures = 0
    first_failure_time = 0.0
    chunk = max(args.threads * 2, 16)
    last_progress = time.time()
    labelling_clock = time.time()  # the plan phase is not part of the labelling rate

    def work(job: tuple[str, dict[str, Any]]) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        key, row = job
        request, reason = request_for_item(row)
        if request is None:
            raise SystemExit(f"{key} became unrequestable: {reason}")
        return label_one(api_key, request, ledger, fatal, text_key=key)

    with futures.ThreadPoolExecutor(max_workers=args.threads) as pool_exec, partial_path.open(partial_mode, encoding="utf-8") as handle:
        for base in range(0, len(todo), chunk):
            if STOP_FILE.exists():
                stop_reason = "stop_file"
                break
            if ledger.spent_usd >= ledger.cap_usd:
                stop_reason = "cap_reached"
                break
            batch = todo[base:base + chunk]
            for label, failure in pool_exec.map(work, batch):
                if label is not None:
                    new_labels.append(label)
                    handle.write(json.dumps(label, ensure_ascii=False, sort_keys=True) + "\n")
                    consecutive_failures = 0
                elif failure is not None:
                    prior_attempts = int((existing_failures.get(str(failure["text_key"])) or {}).get("attempts") or 0)
                    new_failures.append({**failure, "attempts": prior_attempts + int(failure["attempts"])})
                    consecutive_failures += 1
                    if consecutive_failures == 1:
                        first_failure_time = time.time()
                    elif consecutive_failures >= MAX_CONSECUTIVE_FAILURES and time.time() - first_failure_time <= 600:
                        stop_reason = "api_unavailable"
                        fatal.set()
            handle.flush()
            write_json(ledger_partial_path, {"this_task_spend_usd": ledger.spent_usd - prior,
                                             "cumulative_spend_usd": ledger.spent_usd, "calls": ledger.calls,
                                             "input_tokens": ledger.input_tokens, "labelled": len(new_labels),
                                             "failed": len(new_failures), "updated_utc": utcnow()})
            if time.time() - last_progress >= 60:
                elapsed_so_far = time.time() - labelling_clock
                rate = len(new_labels) / elapsed_so_far if elapsed_so_far else 0.0
                remaining = len(todo) - len(new_labels) - len(new_failures)
                eta = remaining / rate / 60 if rate else 0
                print(f"progress: labelled={len(new_labels)} failed={len(new_failures)} "
                      f"spent_cum=${ledger.spent_usd:.4f} rate={rate:.1f}/s eta={eta:.0f}min", flush=True)
                last_progress = time.time()
            if fatal.is_set():
                break
    elapsed = round(time.time() - clock, 1)
    print(f"label run done: labelled={len(new_labels)} failed={len(new_failures)} "
          f"cum_spend=${ledger.spent_usd:.4f} stop={stop_reason} elapsed={elapsed}s", flush=True)

    all_labels = {**existing, **{str(row["text_key"]): row for row in new_labels}}
    all_failures = {**existing_failures, **{str(row["text_key"]): row for row in new_failures}}
    write_jsonl(labels_path, [all_labels[key] for key in sorted(all_labels)])
    write_jsonl(failures_path, [all_failures[key] for key in sorted(all_failures)])
    if partial_path.exists():
        partial_path.unlink()

    # --- QA ------------------------------------------------------------------------------------------------------------
    by_key = dict(zip(sendable_keys, sendable_rows, strict=True))
    bad_schema = [key for key, row in all_labels.items() if set(row) != LABEL_FIELDS]
    bad_teacher = [key for key, row in all_labels.items() if row["teacher"] != MODEL]
    bad_key = [key for key, row in all_labels.items() if str(row["text_key"]) != key]
    bad_order = [key for key, row in all_labels.items() if key in by_key
                 and list(row["option_keys"]) != [str(option["key"]) for option in by_key[key]["options"]]]
    bad_probs = [key for key, row in all_labels.items()
                 if len(row["teacher_probs"]) != len(row["option_keys"])
                 or any(not isinstance(p, (int, float)) or not math.isfinite(float(p)) or float(p) < 0
                        for p in row["teacher_probs"])
                 or abs(sum(row["teacher_probs"]) - 1.0) > 1e-6]
    labelled_keys = set(all_labels)
    frozen_hits = sorted({row["item_id"] for row in all_labels.values()} & ids)
    frozen_text_hits = sorted(labelled_keys & texts)
    mimic_labelled = sorted(key for key, row in all_labels.items() if str(row["template_id"]).startswith("mimic_"))
    missing = sorted(set(by_key) - labelled_keys - set(all_failures))
    stray = sorted(labelled_keys - set(by_key))
    cost = sum(float(row["cost"]) for row in new_labels)
    tokens = sum(int(row["input_tokens"]) for row in new_labels)
    per_template = Counter(str(row["template_id"]) for row in all_labels.values())
    qa = {
        "labels": len(all_labels),
        "failures": len(all_failures),
        "skipped": len(skipped_rows),
        "sendable": len(sendable_rows),
        "bad_schema": len(bad_schema),
        "bad_teacher_id": len(bad_teacher),
        "bad_text_key": len(bad_key),
        "bad_option_order": len(bad_order),
        "bad_probability_rows": len(bad_probs),
        "labelled_frozen_item_hits": len(frozen_hits),
        "labelled_frozen_text_hits": len(frozen_text_hits),
        "labelled_mimic_rows": len(mimic_labelled),
        "allow_protected": bool(args.allow_protected),
        "missing_rows": len(missing),
        "stray_rows": len(stray),
        "coverage_ok": len(all_labels) + len(all_failures) + len(skipped_rows) == len(unique_keys)
        and not missing and not stray,
        "unique_keys": len(unique_keys),
        "failure_reasons": dict(sorted(Counter(str(r.get("reason")) for r in all_failures.values()).items())),
        "per_template": dict(sorted(per_template.items())),
    }
    checks = {
        "schema": not bad_schema, "teacher_id": not bad_teacher, "text_key": not bad_key,
        "option_order": not bad_order, "probability_sums": not bad_probs,
        "no_frozen_overlap": not frozen_hits and not frozen_text_hits,
        "mimic_only_under_authorization": args.allow_protected or not mimic_labelled,
        "coverage": qa["coverage_ok"], "cost_sum": abs(cost - (ledger.spent_usd - prior)) <= 1e-6,
        "token_sum": tokens == ledger.input_tokens, "cap_respected": ledger.spent_usd <= ledger.cap_usd,
    }
    manifest = {
        "task": "D2 label run",
        "git_commit": git_commit(REPO),
        "command": command,
        "model": MODEL,
        "allow_protected": bool(args.allow_protected),
        "operator_authorization": args.authorization if args.allow_protected else None,
        "started_utc": started,
        "finished_utc": utcnow(),
        "wall_clock_seconds": elapsed,
        "stop_reason": stop_reason,
        "prior_loop_spend_usd": round(prior, 6),
        "this_task_spend_usd": round(ledger.spent_usd - prior, 6),
        "cumulative_spend_usd": round(ledger.spent_usd, 6),
        "cap_usd": ledger.cap_usd,
        "balance_before": balance,
        "qa": qa,
        "checks": checks,
        "label_file_cost_usd": round(sum(float(row["cost"]) for row in all_labels.values()), 9),
        "discarded_attempts_note": "loop-ledger D2 spend may exceed the frozen label file by the spend of attempts whose "
                                  "partial rows were discarded by an interrupted run; both figures are reported",
        "labels_sha256": file_sha256(labels_path),
        "failures_sha256": file_sha256(failures_path),
        "skipped_sha256": file_sha256(out / "skipped.jsonl"),
        "plan_sha256": file_sha256(out / "composition.json"),
        "environment": {"hostname": os.uname().nodename, "python": sys.version.split()[0]},
    }
    write_json(out / "label_manifest.json", manifest)
    if ledger_partial_path.exists():
        ledger_partial_path.unlink()  # deleted before the entry is booked, so a resumed run cannot double-book it
    append_entry(LOOP_LEDGER, {"task": "D2", "attempt": "final" if stop_reason == "complete" else stop_reason,
                               "cost_usd": round(ledger.spent_usd - prior, 6), "calls": ledger.calls,
                               "input_tokens": ledger.input_tokens, "labels": len(all_labels),
                               "stop_reason": stop_reason})
    print(json.dumps({"qa": qa, "checks": checks, "spend": {
        "prior": round(prior, 6), "this_task": round(ledger.spent_usd - prior, 6),
        "cumulative": round(ledger.spent_usd, 6), "cap": ledger.cap_usd}}, indent=2), flush=True)
    if not all(checks.values()):
        raise SystemExit(f"D2 QA failed: {[name for name, ok in checks.items() if not ok]}")


if __name__ == "__main__":
    main()
