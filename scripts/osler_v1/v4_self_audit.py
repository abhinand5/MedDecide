"""V4 self-audit (osler_v1 R6): re-derives the headline numbers of the teacher build from the raw files, in a fresh process.

Reads data/gen/osler_v1/teacher (the verified rows), outputs/osler_v1/V4 (progress files, spend ledgers, closure.json and
FINAL.json when they exist). Writes outputs/osler_v1/V4/SELF_AUDIT.md (counts and checks; no teacher text) and self_audit.json.
A check that fails is printed as FAIL and the script exits 1; nothing is edited to make a check pass (R8).

Usage:
    uv run --frozen python scripts/osler_v1/v4_self_audit.py
"""

from __future__ import annotations

import datetime as dt
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from meddecide.mr.prose_items import has_code_string
from meddecide.utils.io import write_json
from meddecide.utils.provenance import utcnow

REPO = Path(__file__).resolve().parents[2]
TEACHER = REPO / "data" / "gen" / "osler_v1" / "teacher"
V4 = REPO / "outputs" / "osler_v1" / "V4"
LEGACY = "not recorded (OpenCode Go, before V4-D7)"
HELD_OUT_INDICATOR = "aki_creatinine_rise"  # ADVISORY V4, ground rule 3: excluded from every training source
EXPECTED_MODEL_OPENCODE = "deepseek-v4.1-flash"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def ledger_totals(path: Path, now: dt.datetime) -> dict[str, float]:
    rows = read_jsonl(path)
    stamp = [dt.datetime.fromisoformat(r["utc"]) for r in rows]
    cost = [float(r["cost_usd"]) for r in rows]
    return {"rows": len(rows), "total_usd": round(sum(cost), 6),
            "last_5h_usd": round(sum(c for t, c in zip(stamp, cost, strict=True) if t >= now - dt.timedelta(hours=5)), 6),
            "last_7d_usd": round(sum(c for t, c in zip(stamp, cost, strict=True) if t >= now - dt.timedelta(days=7)), 6)}


def main() -> None:
    now = dt.datetime.now(dt.UTC)
    checks: list[dict[str, Any]] = []

    def check(name: str, ok: bool, detail: str) -> None:
        checks.append({"check": name, "result": "PASS" if ok else "FAIL", "detail": detail})

    rows = {name: read_jsonl(TEACHER / f"{name}.jsonl") for name in ("indicator", "narrative")}
    counts: dict[str, Any] = {}
    for name, items in rows.items():
        ids = [r["item_id"] for r in items]
        counts[name] = {
            "rows": len(items),
            "by_split": dict(Counter(r["split"] for r in items)),
            "by_role": dict(Counter(r.get("role", "") for r in items)),
            "by_endpoint": dict(Counter(r.get("teacher_endpoint") or LEGACY for r in items)),
            "by_model": dict(Counter(r.get("teacher_model") or EXPECTED_MODEL_OPENCODE + " (legacy)" for r in items)),
            "code_string_hits": sum(1 for r in items if has_code_string(r["state"]) or has_code_string(r["question"])),
            "held_out_indicator_rows": sum(HELD_OUT_INDICATOR in json.dumps(r) for r in items),
        }
        check(f"{name}: item ids are unique", len(set(ids)) == len(ids), f"{len(ids)} rows, {len(set(ids))} unique")
        check(f"{name}: no ICD code string in verified rows", counts[name]["code_string_hits"] == 0,
              f"{counts[name]['code_string_hits']} hits")
        check(f"{name}: held-out indicator excluded", counts[name]["held_out_indicator_rows"] == 0,
              f"{counts[name]['held_out_indicator_rows']} rows mention {HELD_OUT_INDICATOR}")
        by_pair: dict[str, set[str]] = defaultdict(set)
        for r in items:
            by_pair[str(r.get("pair_id") or r["item_id"])].add(r["split"])
        check(f"{name}: splits disjoint by pair", all(len(s) == 1 for s in by_pair.values()),
              f"{sum(1 for s in by_pair.values() if len(s) > 1)} pairs span two splits")

    progress: dict[str, Any] = {}
    for name in ("indicator", "narrative"):
        prog = read_jsonl(V4 / f"{name}_progress.jsonl")
        verdicts = Counter(r["verdict"] for r in prog)
        by_endpoint: dict[str, Counter] = defaultdict(Counter)
        cost_by_endpoint: dict[str, float] = defaultdict(float)
        for r in prog:
            key = r.get("endpoint") or LEGACY
            by_endpoint[key][r["verdict"]] += 1
            cost_by_endpoint[key] += float(r.get("cost_usd", 0.0))
        progress[name] = {"ids": len(prog), "verdicts": dict(verdicts),
                          "by_endpoint": {k: {"verdicts": dict(v), "cost_usd": round(cost_by_endpoint[k], 4)}
                                          for k, v in by_endpoint.items()}}
        pids = [r["pid"] for r in prog]
        check(f"{name}: one progress row per id", len(set(pids)) == len(pids), f"{len(pids)} rows, {len(set(pids))} ids")
        error_rows = verdicts.get("error", 0)
        check(f"{name}: no error verdicts", error_rows == 0, f"{error_rows} error verdicts (each is an id not retried on resume)")

    accepted_indicator = progress["indicator"]["verdicts"].get("accepted", 0)
    accepted_narrative = progress["narrative"]["verdicts"].get("accepted", 0)
    check("indicator: two rows (base and twin) per accepted id", counts["indicator"]["rows"] == 2 * accepted_indicator,
          f"{counts['indicator']['rows']} rows, 2 x {accepted_indicator} accepted ids")
    check("narrative: one row per accepted encounter", counts["narrative"]["rows"] == accepted_narrative,
          f"{counts['narrative']['rows']} rows, {accepted_narrative} accepted ids")

    ledgers = {"opencode_go": ledger_totals(V4 / "spend_ledger.jsonl", now),
               "deepseek_api": ledger_totals(V4 / "spend_ledger_deepseek.jsonl", now)}
    # the pilots wrote to the same OpenCode ledger, so every progress file (pilots included) is summed against the ledgers
    pilot_cost = sum(float(r.get("cost_usd", 0.0)) for path in sorted(V4.glob("*_progress.jsonl"))
                     if path.name.endswith("_pilot_progress.jsonl") for r in read_jsonl(path))
    progress_cost = sum(v["cost_usd"] for p in progress.values() for v in p["by_endpoint"].values()) + pilot_cost
    check("spend: progress-file cost (builds and pilots) equals the ledgers' total (OpenCode and DeepSeek)",
          abs(progress_cost - sum(v["total_usd"] for v in ledgers.values())) < 0.01,
          f"progress {progress_cost:.4f} USD (pilots {pilot_cost:.4f}), ledgers {sum(v['total_usd'] for v in ledgers.values()):.4f} USD")
    check("spend: OpenCode week stays at or under the $36 cap plus one in-flight batch",
          ledgers["opencode_go"]["last_7d_usd"] <= 36.0 + 0.5, f"{ledgers['opencode_go']['last_7d_usd']:.4f} USD in 7 days")
    check("spend: DeepSeek ledger stays under its $9.00 cap plus one in-flight batch",
          ledgers["deepseek_api"]["total_usd"] <= 9.0 + 0.5, f"{ledgers['deepseek_api']['total_usd']:.4f} USD")

    closure_path = V4 / "closure.json"
    if closure_path.exists():
        closure = json.loads(closure_path.read_text(encoding="utf-8"))
        check("closure.json rows equal the raw counts",
              closure["indicator"]["rows"] == counts["indicator"]["rows"]
              and closure["narrative"]["rows"] == counts["narrative"]["rows"],
              f"closure {closure['indicator']['rows']} / {closure['narrative']['rows']}")

    result = {"written_utc": utcnow(), "counts": counts, "progress": progress, "ledgers": ledgers, "checks": checks,
              "closure_present": closure_path.exists(), "final_present": (V4 / "FINAL.json").exists()}
    write_json(V4 / "self_audit.json", result)

    failed = [c for c in checks if c["result"] == "FAIL"]
    lines = ["# osler_v1 V4 — self-audit (R6)", "",
             f"Written {result['written_utc']} by `scripts/osler_v1/v4_self_audit.py` in a fresh process from the raw files. "
             "No teacher text is recorded here; defect types are recorded in the V4 STATE entry.", "",
             "| check | result | detail |", "|---|---|---|"]
    lines += [f"| {c['check']} | {c['result']} | {c['detail']} |" for c in checks]
    lines += ["", f"{len(checks) - len(failed)} PASS, {len(failed)} FAIL.", ""]
    (V4 / "SELF_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for c in checks:
        print(f"{c['result']}  {c['check']}  ({c['detail']})", flush=True)
    print(f"wrote {(V4 / 'SELF_AUDIT.md').relative_to(REPO)}; {len(failed)} FAIL", flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
