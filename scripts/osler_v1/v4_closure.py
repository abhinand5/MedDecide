"""V4 closure checks (osler_v1 ADVISORY V4 step 2): the verified counts, the ICD-code-string check and the O3 screen of the
teacher prose, from the verified items the build appended to data/gen/osler_v1/teacher/.

Writes outputs/osler_v1/V4/closure.json with counts and screen results only (no page text). The spot reads (ADVISORY V4 step 3)
are done separately, in a fresh process, and recorded as defect types in the SELF_AUDIT.

Usage (after the build has finished or has been stopped at the weekly guard):
    source scripts/pod_env.sh && uv run --frozen python scripts/osler_v1/v4_closure.py
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from meddecide.gen.screen import screen_generator
from meddecide.mr.prose_items import has_code_string
from meddecide.utils.io import write_json
from meddecide.utils.provenance import git_commit, utcnow

REPO = Path(__file__).resolve().parents[2]
TEACHER = REPO / "data" / "gen" / "osler_v1" / "teacher"
OUT = REPO / "outputs" / "osler_v1" / "V4" / "closure.json"


def read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def closure_for(name: str) -> dict[str, Any]:
    path = TEACHER / f"{name}.jsonl"
    rows = read_rows(path) if path.exists() else []
    code_hits = sum(1 for r in rows if has_code_string(r["state"]) or has_code_string(r["question"]))
    dev = [r for r in rows if r["split"] == "dev"]
    train = [r for r in rows if r["split"] == "train"]
    entry: dict[str, Any] = {
        "file": str(path.relative_to(REPO)), "rows": len(rows),
        "by_split": dict(Counter(r["split"] for r in rows)),
        "by_generator": dict(Counter(r["generator"] for r in rows)),
        "by_role": dict(Counter(r.get("role", "") for r in rows)),
        "code_string_hits": code_hits,
    }
    if dev and train:
        report = screen_generator(name, False, dev, train, "teacher train split")
        entry["screen"] = report.as_dict()
    else:
        entry["screen"] = {"status": "NOT MEASURED — no dev or no train rows"}
    return entry


def progress_counts(name: str) -> dict[str, Any]:
    """The verdicts of every id the build has processed (the progress file holds one row per id, across restarts)."""
    path = REPO / "outputs" / "osler_v1" / "V4" / f"{name}_progress.jsonl"
    rows = read_rows(path) if path.exists() else []
    return {"ids_processed": len(rows), "verdicts": dict(Counter(r["verdict"] for r in rows)),
            "cost_usd_in_progress_file": round(sum(float(r.get("cost_usd", 0.0)) for r in rows), 4)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--finalize", action="store_true",
                        help="write outputs/osler_v1/V4/FINAL.json, which releases the B2 and B3 queue")
    parser.add_argument("--reason", default="", help="why V4 is closed (for example: complete, or the weekly guard)")
    args = parser.parse_args()
    report = {"written_utc": utcnow(), "git_commit": git_commit(REPO),
              "indicator": closure_for("indicator"), "narrative": closure_for("narrative"),
              "progress": {"indicator": progress_counts("indicator"), "narrative": progress_counts("narrative")}}
    write_json(OUT, report)
    for name in ("indicator", "narrative"):
        e = report[name]
        print(name, e["rows"], e["by_split"], "code-string hits", e["code_string_hits"], "screen",
              e["screen"].get("passes", e["screen"].get("status")), flush=True)
    print("progress", report["progress"], flush=True)
    print(f"wrote {OUT.relative_to(REPO)}", flush=True)
    if args.finalize:
        if not args.reason:
            raise SystemExit("--finalize needs --reason")
        final = {"closed_utc": utcnow(), "reason": args.reason, "closure": str(OUT.relative_to(REPO)),
                 "indicator_pairs": report["indicator"]["rows"] // 2,
                 "narrative_pages": report["narrative"]["rows"]}
        write_json(REPO / "outputs" / "osler_v1" / "V4" / "FINAL.json", final)
        print(f"wrote FINAL.json: {final}", flush=True)


if __name__ == "__main__":
    main()
