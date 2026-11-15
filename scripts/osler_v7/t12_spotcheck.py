#!/usr/bin/env python
"""osler_v7 T12 — five-claim spot-check (R6, CPU): re-derive five headline numbers from the raw artifacts.

Fresh process, no cached aggregates. Each entry recomputes a claim's value from prediction/evaluation files and compares
it with the value recorded in CLAIMS.md. Writes `outputs/osler_v7/T12/spotcheck.json` and a SELF_AUDIT table.

Run (CPU): ``uv run --frozen python scripts/osler_v7/t12_spotcheck.py``
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from meddecide.eval.composite import composite  # noqa: E402
from meddecide.utils.provenance import git_commit, utcnow  # noqa: E402

OUT = ROOT / "outputs" / "osler_v7" / "T12"
DEV = ROOT / "data" / "bench" / "osler_v7" / "dev_format_matched.jsonl"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def s_from_rows(rows: list[dict[str, Any]], origin: dict[str, str]) -> float:
    fast = [row for row in rows if origin[str(row["item_id"])] == "fastdev_v4"]
    cg = [row for row in rows if origin[str(row["item_id"])] == "cg_v2"]
    return composite(fast, cg).to_dict()["S"]


def main() -> None:
    dev = read_jsonl(DEV)
    origin_raw = {str(row["item_id"]): str(row["dev_origin"]) for row in dev}
    # T7/T9 predictions use the trainer's recomputed ids; T6/T8/T11 eval files use the raw ids where noted
    from meddecide.train.mix_items import mix_row_to_item
    origin_recomputed = {mix_row_to_item(row).item_id: str(row["dev_origin"]) for row in dev}

    checks = []
    def add(name: str, claimed: float, value: float) -> None:
        ok = abs(claimed - value) < 1e-6
        checks.append({"claim": name, "claimed": claimed, "recomputed": round(value, 8), "match": ok})

    # 1. T6 arm final S (C030)
    rows = read_jsonl(ROOT / "outputs/osler_v7/T6/arm/evals/step_013992/fast_dev.jsonl") + \
        read_jsonl(ROOT / "outputs/osler_v7/T6/arm/evals/step_013992/cg_v2.jsonl")
    add("C030 T6 final S", 0.81118996, s_from_rows(rows, origin_recomputed))
    # 2. T8 arm final S (C037), raw ids
    add("C037 T8 final S", 0.82604843, s_from_rows(read_jsonl(ROOT / "outputs/osler_v7/T8/arm/evals/step_013992/dev.jsonl"), origin_raw))
    # 3. T9 chosen scale S (C040, s=1)
    add("C040 T9 chosen S", 0.82604843,
        s_from_rows(read_jsonl(ROOT / "outputs/osler_v7/T9/predictions/scale_1.jsonl"), origin_recomputed))
    # 4. T10 9B G-b lower bound (C045)
    gate = json.loads((ROOT / "outputs/osler_v7/T10/gate.json").read_text(encoding="utf-8"))
    add("C045 T10 9B G-b ci_lo", 0.09266438, gate["models"]["9B"]["checks"]["G-b"]["ci_lo"])
    # 5. T11 final S (C049), raw ids
    add("C049 T11 final S", 0.85381978,
        s_from_rows(read_jsonl(ROOT / "outputs/osler_v7/T11/final/evals/step_063304/dev.jsonl"), origin_raw))

    result = {"generated_utc": utcnow(), "git_commit": git_commit(ROOT),
              "command": "uv run --frozen python scripts/osler_v7/t12_spotcheck.py",
              "checks": checks, "status": "PASS" if all(c["match"] for c in checks) else "FAIL"}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "spotcheck.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = ["# T12 five-claim spot-check (osler_v7)", "",
             f"Generated {result['generated_utc']} at `{result['git_commit']}`; command `{result['command']}`.", "",
             f"**Status: {result['status']}**.", "", "| claim | recorded | re-derived | match |", "|---|---|---|---|"]
    for c in checks:
        lines.append(f"| {c['claim']} | {c['claimed']} | {c['recomputed']} | {'yes' if c['match'] else '**NO**'} |")
    (OUT / "SELF_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"T12 spot-check {result['status']}: {len(checks)} claims")
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
