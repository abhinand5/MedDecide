#!/usr/bin/env python
"""osler_v7 T5 self-audit (R6, CPU): re-derive the pool and labelled-set headline numbers from the raw files.

Fresh process. Checks the pool file against its manifest (rows, SHA-256, per-cell counts, qtype counts, zero overlap
with the T2 dev/confirmation sets), then the labelled file against the pool (row ids, option counts, probability
lengths and sums, constant rows, teacher id) and the recorded throughput. Writes `audit.json` and `SELF_AUDIT.md`.

Run (CPU; after the label/join steps): ``uv run --frozen python scripts/osler_v7/t5_audit.py``
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from meddecide.distill import soft_labels as sl  # noqa: E402
from meddecide.utils.provenance import git_commit, utcnow  # noqa: E402

OUT = ROOT / "outputs" / "osler_v7" / "T5"
POOL = ROOT / "data" / "train" / "osler_v7" / "pool.jsonl"
LABELLED = ROOT / "data" / "train" / "osler_v7" / "labelled.jsonl"
DEV = ROOT / "data" / "bench" / "osler_v7" / "dev_format_matched.jsonl"
CONFIRM = ROOT / "data" / "bench" / "osler_v7" / "confirm" / "general.jsonl"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    pool_manifest = json.loads((OUT / "pool_manifest.json").read_text(encoding="utf-8"))
    join_manifest = json.loads((OUT / "join_manifest.json").read_text(encoding="utf-8"))
    checks: list[dict[str, Any]] = []
    problems: list[str] = []

    def check(name: str, recorded: Any, recomputed: Any) -> None:
        match = recorded == recomputed
        checks.append({"check": name, "recorded": recorded, "recomputed": recomputed, "match": match})
        if not match:
            problems.append(f"{name}: recorded {recorded!r} != recomputed {recomputed!r}")

    pool = read_jsonl(POOL)
    check("pool.rows", pool_manifest["rows"], len(pool))
    check("pool.sha256", pool_manifest["pool_sha256"], sha256_file(POOL))
    check("pool.qtypes", pool_manifest["qtypes"], dict(sorted(Counter(str(r["qtype"]) for r in pool).items())))
    check("pool.cells", pool_manifest["cells_kept"], dict(sorted(Counter(
        "C1" if r["template_id"].startswith("v7_c1") else "C2" if r["template_id"].startswith("v7_c2")
        else "C3" if r["template_id"].startswith("v7_c3") else "C4" if r["template_id"].startswith("v7_c4")
        else "other" for r in pool).items())))
    dev_keys = {sl.overlap_keys(row)[0] for row in read_jsonl(DEV)}
    confirm_keys = {sl.overlap_keys(row)[0] for row in read_jsonl(CONFIRM)}
    check("pool.overlap_with_t2_sets", 0,
          sum(1 for row in pool if sl.overlap_keys(row)[0] in (dev_keys | confirm_keys)))
    check("pool.unique_item_ids", len(pool), len({str(r["item_id"]) for r in pool}))
    check("pool.unique_text_keys", len(pool), len({sl.overlap_keys(r)[0] for r in pool}))

    labelled = read_jsonl(LABELLED)
    check("labelled.rows", len(pool), len(labelled))
    check("labelled.sha256", join_manifest["labelled_sha256"], sha256_file(LABELLED))
    check("labelled.teacher", "perplexity-ai/pplx-decider-v1.2-27b@cce0f3f80033e9d89d80831d5281fd72124aacce",
          Counter(str(row["teacher"]) for row in labelled).most_common(1)[0][0])
    max_sum_error = 0.0
    constant = 0
    for row in labelled:
        values = [float(p) for p in row["teacher_probs"]]
        if len(values) != len(row["options"]):
            problems.append(f"{row['item_id']}: probability length mismatch")
        max_sum_error = max(max_sum_error, abs(sum(values) - 1.0))
        if len({round(value, 6) for value in values}) <= 1:
            constant += 1
    check("labelled.max_probability_sum_error", join_manifest["max_probability_sum_error"], max_sum_error)
    check("labelled.constant_rows", join_manifest["constant_rows"], constant)
    check("labelled.statuses", join_manifest["statuses"], {"scored": len(labelled)})
    throughput = join_manifest.get("throughput") or {}
    check("labeled.items_per_s_recorded", True, float(throughput.get("items_per_s", 0)) > 0)
    audit = {
        "generated_utc": utcnow(),
        "git_commit": git_commit(ROOT),
        "command": "uv run --frozen python scripts/osler_v7/t5_audit.py",
        "checks": checks,
        "problems": problems,
        "status": "PASS" if not problems else "FAIL",
    }
    (OUT / "audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# T5 SELF_AUDIT — pool and pplx-27B labels (osler_v7)",
        "",
        f"Generated {audit['generated_utc']} at `{audit['git_commit']}` in a fresh process; command `{audit['command']}`.",
        "",
        f"**Status: {audit['status']}** — {len(checks)} checks, {len(problems)} problems.",
        "",
        f"Throughput recorded: {throughput.get('items_per_s')} items/s, {throughput.get('input_tokens_per_s')} input tokens/s.",
        "",
        "| check | recorded | re-derived | match |",
        "|---|---|---|---|",
    ]
    for row in checks:
        lines.append(
            f"| {row['check']} | `{row['recorded']}` | `{row['recomputed']}` | {'yes' if row['match'] else '**NO**'} |"
        )
    lines.extend(["", "## Problems", ""])
    lines.extend([f"- {problem}" for problem in problems] or ["None."])
    lines.extend(["", "## Notes", "",
                  "- The pool and labelled files stay gitignored; only aggregates are committed.",
                  "- The teacher's probabilities are attached in the row's option order (soft_labels mapping).",
                  ""])
    (OUT / "SELF_AUDIT.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"T5 audit {audit['status']}: {len(checks)} checks, {len(problems)} problems")
    if problems:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
