"""osler_v3 R6 re-derivation (X1 and X2): recount the written source files in a fresh process and compare with the build reports.

Reads the JSONL files the builds wrote (data/train, data/bench) and the build.json and checks.json reports; prints one line per
check with the recomputed value, the recorded value and whether they agree. Aggregates only; no item text is printed.
Exits non-zero when any check disagrees.

Usage:  source scripts/pod_env.sh && uv run --frozen python scripts/osler_v3/recount_sources.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
X1 = REPO / "outputs" / "osler_v3" / "X1"
X2 = REPO / "outputs" / "osler_v3" / "X2"


def rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def check(name: str, recomputed, recorded) -> bool:
    ok = recomputed == recorded
    print(f"{'OK  ' if ok else 'FAIL'} {name}: recomputed {recomputed} | recorded {recorded}", flush=True)
    return ok


def main() -> int:
    ok = True
    x1 = read_json(X1 / "build.json")
    for split, path in (("train", REPO / x1["files"]["train"]), ("dev", REPO / x1["files"]["dev"]),
                        ("test", REPO / x1["files"]["test"])):
        data = rows(path)
        ok &= check(f"X1 {split} rows", len(data), x1["rows"][split])
        by_template = Counter(r["template_id"] for r in data)
        for template, golds in x1["gold_by_template_and_split"].items():
            recorded = sum(golds.get(split, {}).values()) if isinstance(golds.get(split), dict) else 0
            ok &= check(f"X1 {split} {template} items", by_template.get(template, 0), recorded)
        ok &= check(f"X1 {split} gold balance (grounded template, yes == no)", _balanced(data, "ragtruth_response_grounded_v1"),
                    True)
    checks = read_json(X1 / "checks.json")
    ok &= check("X1 train leakage hits", _hits(checks["leakage"]["train"]), 0)
    ok &= check("X1 dev leakage hits", _hits(checks["leakage"]["dev"]), 0)
    ok &= check("X1 test leakage hits", _hits(checks["leakage"]["test"]), 0)
    x2 = read_json(X2 / "build.json")
    data = rows(REPO / "data" / "train" / "osler_v3" / "replay_broad_train.jsonl")
    ok &= check("X2 rows", len(data), x2["rows"])
    by_source = Counter(r["source"] for r in data)
    for name, info in x2["sources"].items():
        ok &= check(f"X2 {name} rows", by_source.get(name, 0), info["kept_after_dedupe"])
    ok &= check("X2 leakage hits", _hits(x2["leakage"]), 0)
    ok &= check("X2 licences present on every row", all(r["source_license"] for r in data), True)
    return 0 if ok else 1


def _balanced(data: list[dict], template: str) -> bool:
    golds = Counter(r["gold"] for r in data if r["template_id"] == template)
    return len(set(golds.values())) <= 1


def _hits(counts: dict) -> int:
    return sum(v for k, v in counts.items() if k != "rows_checked" and k.startswith(("text:", "record:", "rows_")))


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    sys.exit(main())
